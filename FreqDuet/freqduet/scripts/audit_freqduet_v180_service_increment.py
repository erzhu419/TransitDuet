#!/usr/bin/env python3
"""Read original calibration archives: service-prefix information within fixed states."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, zip_longest
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v179_calibration_prefix as previous


prior, saved = previous.prior, previous.saved
read, write, require, close = previous.read, previous.write, previous.require, previous.close
STATE = ('remaining_stops', 'stage', 'horizon', 'direction', 'target_stop', 'clearance_elapsed_s')
PARTITIONS = dict(distance=STATE[:1], geometry_time=STATE,
    observed_support=(*STATE, 'key_days', 'native_support', 'native_survivors', 'support_age_s'))
TARGETS = ('total_residual_s', 'service_residual_s', 'below', 'above', 'score_s',
    'expansion_s', 'upper_saving_s', 'envelope_score_delta_s')
SIZE = 3+3*len(TARGETS)


def moment(x, y):
    y = np.asarray(y)
    return np.r_[1., x, x*x, y, y*y, x*y]


def covariance(values):
    mass, x, xx = values[..., :1], values[..., 1:2], values[..., 2:3]
    y, yy, xy = np.split(values[..., 3:], 3, axis=-1)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0.)
    vx = (xx-divide(x*x, mass)).sum(axis=-2)
    vy = (yy-divide(y*y, mass)).sum(axis=-2)
    cov = (xy-divide(x*y, mass)).sum(axis=-2)
    den = np.sqrt(np.maximum(vx, 0.)*np.maximum(vy, 0.))
    corr = np.divide(cov, den, out=np.full_like(cov, np.nan), where=den > 0.)
    return corr, cov, vx, vy


def read_path(args, actor, day, mode, tables, model, archive, old):
    require(day in previous.DAYS, 'not an original calibration day')
    root, name = args.v166_root/'calibration', f'{actor}_{day}_{mode}'
    receipt = read(root/'capture_cells'/(name+'.json'))
    require((receipt['execution_source'], receipt['seed'], receipt['day'], receipt['mode']) ==
        ('20d914af03', actor, day, mode) and not receipt['interval_fit'] and not receipt['main_policy_changed'],
        'capture differs')
    requests = archive['requests']
    lookup = {tuple(requests[i, 5:].astype(int)): i for i in np.flatnonzero(requests[:, 0] == day)}
    atoms, selected, count = {}, [], 0
    for q, r in zip_longest(*(previous.stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals'))):
        require(q is not None and r is not None and q['query_key'] == r['query_key'] and
            q['episode'] == r['episode'] and q['query_time_s'] == r['query_time_s'] and
            (q['actor'], q['day'], q['mode']) == (actor, day, mode), 'query/residual join differs')
        count += 1
        previous.add_first(atoms, q, r)
        if not q['denominator_eligible']:
            continue
        i = lookup.pop(tuple(q['query_key']))
        require(requests[i, 3] == q['point_s'] and requests[i, 4] == q['query_time_s'] and
            archive['episode_keys'][int(requests[i, 1])].tolist() == [day, *q['episode']], 'archive coordinates differ')
        scopes = prior.prior.scopes(q)
        if not scopes:
            continue
        p = saved.prediction(q, *tables, model['frozen_protocol'])
        require(p['finite'] == bool(archive['finite'][i]), 'finite mask changed')
        if p['finite']:
            require(p['bounds'] == archive['candidate'][i].tolist() and
                p['temporal_bounds'] == archive['bounds'][i, 0].tolist(), 'sealed bounds changed')
        entry = None if p['candidate_key'] is None else tables[0].get(tuple(p['candidate_key']))
        values = saved.values(p, r['label'])
        require(not p['finite'] or values is not None, 'finite label/reference missing')
        require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'late load was backfilled')
        diagnostic = prior.diagnostics(q, p, r, entry) if p['finite'] else None
        y = None if diagnostic is None else [diagnostic[prior.METRICS.index(k)] for k in
            ('point_bias_s', 'service_bias_s', 'below', 'above')]+[values[saved.METRICS.index('score_s')]]+[
                diagnostic[prior.METRICS.index(k)] for k in TARGETS[5:]]
        selected.append(dict(q=q, scopes=scopes, values=values, features=prior.features(q, entry), y=y, finite=p['finite']))
    require(not lookup and count == receipt['query_record_count'], 'query cohort incomplete')
    prefix = previous.Prefix(atoms)
    base = {s: {} for s in prior.SCOPES}
    moments = {s: defaultdict(dict) for s in prior.SCOPES}
    for v in sorted(selected, key=lambda v: (v['q']['query_time_s'], v['q']['clocks']['query_ready_order'])):
        q, f = v['q'], v['features']
        histories = prefix.query(q['query_time_s'], None if f['remaining_stops'] is None else int(f['remaining_stops']))
        for scope in v['scopes']:
            ep = tuple(q['episode'])
            row = base[scope].setdefault(ep, np.zeros(3+len(saved.METRICS)))
            row[:3] += (1, int(v['finite']), int(v['values'] is not None))
            if v['values'] is not None:
                row[3:] += v['values']
            if not v['finite']:
                continue
            for h, history in histories.items():
                if history['mean'] is None:
                    continue
                value = moment(history['mean'][1], v['y'])
                for level, fields in PARTITIONS.items():
                    key = h, level, tuple(f[k] for k in fields)
                    moments[scope][ep].setdefault(key, np.zeros(SIZE))[:] += value
    rows = []
    for scope in prior.SCOPES:
        endpoint = saved.collapse(base[scope], 0, 0)
        totals = defaultdict(lambda: np.zeros(SIZE))
        nf = endpoint['finite_episodes']
        for ep, groups in moments[scope].items():
            for key, values in groups.items():
                totals[key] += values/(base[scope][ep][1]*nf)
        rows.append(dict(actor=actor, day=day, mode=mode, scope=scope, endpoint=endpoint,
            groups=[dict(history=h, level=l, key=list(k), moments=v.tolist()) for (h, l, k), v in totals.items()]))
    for row in rows:
        match = next(r for r in old['cells'] if r['day'] == day and r['scope'] == row['scope'])
        for k, value in row['endpoint'].items():
            require(match['endpoint'][k] is None, 'missing endpoint changed') if value is None else close(value, match['endpoint'][k])
    return rows


def bucket(args, actor, mode):
    model = previous.parents(args)
    chunk = next(c for c in model['chunks'] if (c['actor'], c['mode']) == (actor, mode))
    archive = previous.atom_archive(args, chunk)
    tables = saved.map_tables(args.v176_root, args.v172_root, args.v167_root, actor, mode)
    old = read(args.v179_root/'cells'/f'{actor}_{mode}.json')
    rows = []
    for i, day in enumerate(previous.DAYS, 1):
        rows.extend(read_path(args, actor, day, mode, tables, model, archive, old))
        print('V180_CALIBRATION_DAY_READ', actor, mode, i, '/64', flush=True)
    write(args.out_root/'cells'/f'{actor}_{mode}.json', dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], rows=rows))


def interval(value, draws):
    return dict(mean=None if not np.isfinite(value) else float(value),
        ci95=None if not np.isfinite(draws).all() else np.percentile(draws, [2.5, 97.5]).tolist())


def summarize(rows, actors, days, history, level, reference):
    index = {(r['actor'], r['day']): r for r in rows}
    require(set(index) == set(product(actors, days)) and len(index) == len(rows), 'crossed grid differs')
    keys = sorted({tuple(g['key']) for r in rows for g in r['groups'] if (g['history'], g['level']) == (history, level)}, key=str)
    if not keys:
        return dict(history=history, level=level, support_mass=0., estimates=None)
    key_index = {k: i for i, k in enumerate(keys)}
    matrix = np.zeros((len(rows), len(keys), SIZE))
    for i, identity in enumerate(product(actors, days)):
        for g in index[identity]['groups']:
            if (g['history'], g['level']) == (history, level):
                matrix[i, key_index[tuple(g['key'])]] = g['moments']
    mean = matrix.mean(axis=0)
    weights = prior.bootstrap_weights(len(actors), len(days))
    point, cov, vx, vy = covariance(mean)
    pooled = covariance(mean.sum(axis=0, keepdims=True))
    boot, masses, retained = [], [], []
    # Bounded batches keep the fine fixed-state bootstrap off the large raw archives.
    for start in range(0, len(weights), 64):
        draws = (weights[start:start+64]@matrix.reshape(len(rows), -1)).reshape(-1, len(keys), SIZE)
        stats, covariance_draws, xvar, yvar = covariance(draws)
        total_xvar = covariance(draws.sum(axis=1, keepdims=True))[2]
        boot.append(np.stack((stats, covariance_draws), axis=-1))
        masses.extend(draws[:, :, 0].sum(axis=1))
        retained.extend(np.divide(xvar[:, 0], total_xvar[:, 0], out=np.full(len(draws), np.nan), where=total_xvar[:, 0] > 0.))
    boot = np.concatenate(boot)
    if level == 'distance':
        close(mean[:, 0].sum(), reference['support_mass'])
        for target, old_target in (('total_residual_s', 'total_residual'), ('below', 'lower_miss')):
            old = next(c for c in reference['correlations'] if c['signal'] == 'service' and c['target'] == old_target)
            actual = interval(point[TARGETS.index(target)], boot[:, TARGETS.index(target), 0])
            close(actual['mean'], old['correlation'])
            for a, b in zip(actual['ci95'], old['ci95']):
                close(a, b)
    close(cov[TARGETS.index('envelope_score_delta_s')], cov[TARGETS.index('expansion_s')]-cov[TARGETS.index('upper_saving_s')])
    grid = matrix[:, :, 0].reshape(len(actors), len(days), len(keys)) > 0.
    supported_days = grid.any(axis=0).sum(axis=0)
    support = {label: float(mean[mask, 0].sum()) for label, mask in
        (('1-8', supported_days < 9), ('9-18', (supported_days >= 9)&(supported_days < 19)),
         ('19-31', (supported_days >= 19)&(supported_days < 32)), ('32-47', (supported_days >= 32)&(supported_days < 48)),
         ('48-64', supported_days >= 48))}
    return dict(history=history, level=level, fields=PARTITIONS[level], strata=len(keys),
        support_mass=interval(mean[:, 0].sum(), np.array(masses)), support_mass_by_distinct_days=support,
        retained_prefix_variance=interval(vx[0]/pooled[2][0] if pooled[2][0] > 0. else np.nan, np.array(retained)),
        estimates={t: dict(correlation=interval(point[i], boot[:, i, 0]),
            covariance=interval(cov[i], boot[:, i, 1]), marginal_correlation=None if not np.isfinite(pooled[0][i]) else float(pooled[0][i]))
            for i, t in enumerate(TARGETS)})


def reduce_scope(args, mode, scope):
    model = previous.parents(args)
    actors, days = sorted(model['actors']), list(previous.DAYS)
    rows = []
    for actor in actors:
        value = read(args.out_root/'cells'/f'{actor}_{mode}.json')
        require(value['source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'source differs')
        rows.extend(r for r in value['rows'] if r['scope'] == scope)
    old = read(args.v179_root/'summary/status.json')
    require(old['execution_source'] == '354867e01f' and old['paths'] == 1536 and not old['confirmation_outcomes_read'], 'V179 differs')
    results = []
    for split, cohort in (('all64', days), ('first32', days[:32]), ('last32', days[32:])):
        selected = [r for r in rows if r['day'] in cohort]
        for h, level in product(previous.HISTORIES, PARTITIONS):
            reference = next(r for r in old['signals'] if (r['mode'], r['scope'], r['split'], r['history']) == (mode, scope, split, h))
            results.append(dict(mode=mode, scope=scope, split=split,
                **summarize(selected, actors, cohort, h, level, reference)))
        print('V180_CROSSED_SUMMARY', mode, scope, split, flush=True)
    return results


def main(args):
    model = previous.parents(args)
    if args.command == 'prepare':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
            '-p', 'test_v180_*.py', '-v'], check=True)
        actor, day, mode = model['actors'][0], previous.DAYS[0], 'old'
        chunk = next(c for c in model['chunks'] if (c['actor'], c['mode']) == (actor, mode))
        read_path(args, actor, day, mode, saved.map_tables(args.v176_root, args.v172_root, args.v167_root, actor, mode),
            model, previous.atom_archive(args, chunk), read(args.v179_root/'cells'/f'{actor}_{mode}.json'))
        write(args.out_root/'prepare/summary/status.json', dict(status='v180_preflight_passed', tests_passed=True, endpoints_reproduced=True))
    elif args.command == 'run':
        cases = list(product(model['actors'], ('old', 'handoff')))[args.start:args.end]
        require(len(cases) == args.end-args.start > 0 and args.end <= 24, 'invalid shard')
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            for job in as_completed([pool.submit(bucket, args, a, m) for a, m in cases]):
                job.result()
        write(args.out_root/'receipts'/f'{args.start}-{args.end}'/'status.json', dict(status='v180_readonly_shard_complete'))
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            for job in as_completed([pool.submit(reduce_scope, args, m, s) for m, s in product(('old', 'handoff'), prior.SCOPES)]):
                results.extend(job.result())
        require(read(args.v176_root/'model.json') == model, 'frozen header changed')
        write(args.out_root/'summary/status.json', dict(status='v180_calibration_service_increment_complete',
            source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], paths=1536, calibration_days=previous.DAYS,
            partitions=PARTITIONS, results=results, v179_correlations_reproduced=True,
            interval_fit=False, map_updates=0, confirmation_outcomes_read=False, confirmation_started=False,
            new_replays=0, new_training=0, main_policy_changed=False,
            interpretation='Fixed-state within-stratum associations, calibration only; no fitted predictor or control intervention.'))
        print('V180_CALIBRATION_SERVICE_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run', 'aggregate'))
    for name in ('out-root', 'v166-root', 'v166-audit-root', 'v176-root', 'v172-root', 'v167-root', 'v179-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    main(p.parse_args())
