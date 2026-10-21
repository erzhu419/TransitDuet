#!/usr/bin/env python3
"""Read-only bound and crossed-day diagnostics of the consumed V173 cohort."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, zip_longest
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v173_confirmation as saved


read, write, require, stream = saved.read, saved.write, saved.require, saved.stream
SCOPES = ('clearance', 'hour13_moving_launch')
COMPARATORS = ('coarse', 'v167')
BASE = ('above', 'excess_s', 'upper_s', 'headroom_s', 'point_bias_s')
PAIRED = ('reference_above', 'upper_delta_s', 'lower_delta_s', 'contracted', 'equal', 'expanded',
    'new_miss', 'repaired_miss', 'both_miss', 'neither_miss', 'above_delta', 'excess_delta_s',
    'width_delta_s', 'lower_penalty_delta_s', 'upper_penalty_delta_s', 'score_delta_s')
METRICS = (*BASE, *(c+'_'+m for c in COMPARATORS for m in PAIRED))


def scopes(q):
    bits = saved.fit.scope_bits(q)
    return tuple(s for s, selected in zip(SCOPES, (bool(bits & 4),
        bool(bits & 16) and q['native_stage']['kind'] == 'moving-launch')) if selected)


def diagnostics(q, p, label):
    require(p['finite'] and p['reference_finite'] and p['secondary_finite'] and
        label['status'] == 'resolved', 'finite query lacks matched evidence')
    truth, point = label['full_eta_s'], q['point_s']
    lo, hi = p['bounds']
    above, excess = truth > hi, max(truth-hi, 0.)
    result = dict(above=float(above), excess_s=excess, upper_s=hi,
        headroom_s=hi-point, point_bias_s=point-truth)
    for comparator, (rl, ru) in zip(COMPARATORS, (p['reference_bounds'], p['secondary_bounds'])):
        missed, delta = truth > ru, hi-ru
        new, repaired = above and not missed, missed and not above
        require(not new or delta < 0., 'new upper miss without contraction')
        require(not repaired or delta > 0., 'repaired upper miss without expansion')
        width = (hi-lo)-(ru-rl)
        lower = 10.*(max(lo-truth, 0.)-max(rl-truth, 0.))
        upper = 10.*(excess-max(truth-ru, 0.))
        values = (float(missed), delta, lo-rl, float(delta < 0.), float(delta == 0.), float(delta > 0.),
            float(new), float(repaired), float(above and missed), float(not above and not missed),
            float(above)-float(missed), upper/10., width, lower, upper, width+lower+upper)
        result.update((comparator+'_'+m, v) for m, v in zip(PAIRED, values))
    return np.array([result[m] for m in METRICS])


def collapse(episodes):
    require(bool(episodes), 'empty matched cell; do not drop it')
    rows = np.array(list(episodes.values()))
    result = dict(zip(METRICS, (rows[:, 1:]/rows[:, :1]).mean(axis=0).tolist()))
    for c in COMPARATORS:
        get = lambda m: result[c+'_'+m]
        saved.fit.close(get('new_miss')-get('repaired_miss'), get('above_delta'))
        saved.fit.close(get('new_miss')+get('repaired_miss')+get('both_miss')+get('neither_miss'), 1.)
        saved.fit.close(get('score_delta_s'), get('width_delta_s')+
            get('lower_penalty_delta_s')+get('upper_penalty_delta_s'))
    return result


def scan_path(args, protocol, tables, actor, day, mode):
    name = f'{actor}_{day}_{mode}'
    original = read(args.v173_root/'cells'/(name+'.json'))
    require((original['execution_source'], original['map_source'], original['sampler_seed']) ==
        ('0151b7c719', saved.MAP_SOURCE, protocol['reservation']['sampler_seeds'][str(day)]), 'source/cohort differs')
    groups, extra = {s: {} for s in SCOPES}, {s: {} for s in SCOPES}
    sources = [stream(args.v173_root/'confirmation'/d/(name+'.jsonl.gz'))
        for d in ('queries', 'interval_predictions', 'residuals')]
    count = 0
    for q, p, residual in zip_longest(*sources):
        require(q is not None and p is not None and residual is not None, 'query/label stream lengths differ')
        require(q['query_key'] == p['query_key'] == residual['query_key'] and
            q['query_time_s'] == p['query_time_s'] == residual['query_time_s'] and
            (q['actor'], q['day'], q['mode']) == (actor, day, mode), 'query/label join differs')
        count += 1
        selected = scopes(q) if q['denominator_eligible'] else ()
        if not selected:
            continue
        expected = saved.prediction(q, *tables, protocol)
        expected['candidate_key'] = None if expected['candidate_key'] is None else list(expected['candidate_key'])
        require(expected == p, 'saved bounds differ from sealed lookup')
        label, episode = residual['label'], tuple(q['episode'])
        scored = saved.values(p, label)
        require(not p['finite'] or scored is not None, 'finite evidence missing; cannot select a partial cohort')
        vector = diagnostics(q, p, label) if p['finite'] else None
        for scope in selected:
            saved.add_episode(groups[scope], episode, p['finite'], scored)
            if vector is not None:
                row = extra[scope].setdefault(episode, np.zeros(1+len(METRICS)))
                row[0] += 1
                row[1:] += vector
    require(count == original['scoring_counts']['requests'], 'saved query count differs')
    cells = []
    for scope in SCOPES:
        reproduced = saved.collapse(groups[scope])
        old = original['cells'][scope]
        for field, value in reproduced.items():
            if value is None:
                require(old[field] is None, 'saved missing state differs: '+field)
            else:
                saved.fit.close(value, old[field])
        result = collapse(extra[scope])
        for field, old_field in (('above', 'above'), ('coarse_reference_above', 'reference_above'),
                ('v167_reference_above', 'secondary_above'), ('coarse_score_delta_s', 'score_delta_s'),
                ('v167_score_delta_s', 'secondary_score_delta_s')):
            saved.fit.close(result[field], reproduced[old_field])
        cells.append(dict(actor=actor, day=day, mode=mode, scope=scope, **result,
            eligible_queries=reproduced['eligible_queries'], finite_queries=reproduced['finite_queries'],
            finite_fraction=reproduced['finite_fraction'], endpoint_reproduced=True))
    return cells


def bucket(args, actor, mode, protocol):
    tables = saved.maps(args, actor, mode)
    rows = []
    for day in protocol['reservation']['confirmation_days']:
        rows.extend(scan_path(args, protocol, tables, actor, day, mode))
    destination = args.out_root/'cells'/f'{actor}_{mode}.json'
    write(destination, dict(execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], cells=rows))
    print('V174_BUCKET_COMPLETE', actor, mode, flush=True)
    return rows


def variation(matrix):
    grand = matrix.mean()
    actor, day = matrix.mean(axis=1)-grand, matrix.mean(axis=0)-grand
    interaction = matrix-grand-actor[:, None]-day[None, :]
    components = np.array([np.mean(actor**2), np.mean(day**2), np.mean(interaction**2)])
    total = float(np.mean((matrix-grand)**2))
    saved.fit.close(float(components.sum()), total)
    return dict(total_variance=total, actor_variance=float(components[0]), day_variance=float(components[1]),
        interaction_variance=float(components[2]),
        fractions=None if total == 0. else dict(zip(('actor', 'day', 'interaction'), (components/total).tolist())),
        actor_range=[float(matrix.mean(axis=1).min()), float(matrix.mean(axis=1).max())],
        day_range=[float(matrix.mean(axis=0).min()), float(matrix.mean(axis=0).max())])


def summarize(rows, actors, days):
    actors, days = sorted(actors), sorted(days)
    index = {(r['actor'], r['day']): r for r in rows}
    require(len(index) == len(rows) == len(actors)*len(days) and
        set(index) == set(product(actors, days)), 'incomplete/duplicate crossed grid')
    rng = np.random.RandomState(20260803)
    a = rng.randint(0, len(actors), size=(5000, len(actors)))
    d = rng.randint(0, len(days), size=(5000, len(days)))
    estimates, matrices = {}, {}
    for metric in METRICS:
        values = np.array([[index[actor, day][metric] for day in days] for actor in actors])
        require(np.isfinite(values).all(), 'missing metric: '+metric)
        draws = values[a[:, :, None], d[:, None, :]].mean(axis=(1, 2))
        estimates[metric] = dict(mean=float(values.mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist())
        matrices[metric] = values
    marginal_fields = ('above', 'point_bias_s', 'headroom_s', 'coarse_reference_above', 'v167_reference_above',
        'coarse_new_miss', 'coarse_repaired_miss', 'coarse_upper_delta_s', 'coarse_score_delta_s',
        'v167_new_miss', 'v167_repaired_miss', 'v167_upper_delta_s', 'v167_score_delta_s')
    by_day = [dict(day=day, **{m: float(matrices[m][:, i].mean()) for m in marginal_fields})
        for i, day in enumerate(days)]
    partitions = {m: variation(matrices[m]) for m in ('above', 'point_bias_s', 'headroom_s',
        'coarse_upper_delta_s', 'coarse_above_delta', 'coarse_score_delta_s',
        'v167_upper_delta_s', 'v167_above_delta', 'v167_score_delta_s')}
    return dict(cells=len(rows), endpoints_reproduced=all(r['endpoint_reproduced'] for r in rows),
        estimates=estimates, descriptive_variance=partitions, by_day=by_day)


def run(args):
    require(args.out_root not in (args.v173_root, args.v172_root, args.v167_root), 'output must be separate')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v174_*.py', '-v'], check=True)
    registration = read(args.v173_root/'registration.json')
    sealed = read(args.v172_root/'model.json')
    summary = read(args.v173_root/'summary/status.json')
    require(registration['sealed_model'] == sealed and registration['execution_source'] == '0151b7c719' and
        summary['status'] == 'v173_independent_confirmation_complete' and summary['paths'] == 768 and
        not summary['interval_fit'] and summary['map_updates'] == 0 and not summary['main_policy_changed'],
        'requires completed frozen V173 evidence')
    protocol = sealed['frozen_protocol']
    actors, days = protocol['reservation']['actors'], protocol['reservation']['confirmation_days']
    require(len(actors) == 12 and days == list(range(3152001, 3152033)), 'consumed cohort differs')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = []
        for actor, mode in product(actors, ('old', 'handoff')):
            existing = args.out_root/'cells'/f'{actor}_{mode}.json'
            if existing.exists():
                cached = read(existing)
                require(cached['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'cached source differs')
                rows.extend(cached['cells'])
            else:
                jobs.append(pool.submit(bucket, args, actor, mode, protocol))
        for job in as_completed(jobs):
            rows.extend(job.result())
    write(args.out_root/'details/actor_day_cells.json', rows)
    results = []
    for mode, scope in product(('old', 'handoff'), SCOPES):
        result = summarize([r for r in rows if (r['mode'], r['scope']) == (mode, scope)], actors, days)
        old = next(r for r in summary['readouts'] if (r['mode'], r['scope']) == (mode, scope))['estimates']
        for metric, original in (('above', 'above'), ('coarse_score_delta_s', 'score_delta_s'),
                ('v167_score_delta_s', 'secondary_score_delta_s')):
            actual, expected = result['estimates'][metric], old[original]
            saved.fit.close(actual['mean'], expected['mean'])
            for x, y in zip(actual['ci95'], expected['ci95']):
                saved.fit.close(x, y)
        results.append(dict(mode=mode, scope=scope, **result))
    require(read(args.v172_root/'model.json') == sealed, 'sealed header changed')
    write(args.out_root/'summary/status.json', dict(status='v174_readonly_tail_diagnostic_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], input_source='0151b7c719',
        paths=768, scopes=list(SCOPES), results=results, bootstrap_draws=5000, bootstrap_seed=20260803,
        interval_fit=False, map_updates=0, new_replays=0, new_training=0, main_policy_changed=False,
        interpretation='Descriptive analysis of consumed confirmation days, not a new admission gate.'))
    print('V174_READONLY_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v173-root', 'v172-root', 'v167-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    run(parser.parse_args())
