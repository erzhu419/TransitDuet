#!/usr/bin/env python3
"""Confirm the frozen V176 envelope on all 64 V175-reserved demand days."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gzip
from itertools import product, zip_longest
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v173_confirmation as saved
import preregister_freqduet_v175_dayblock_tails as plan


fit, capture, entry = saved.fit, saved.capture, saved.entry
read, write, require, stream = saved.read, saved.write, saved.require, saved.stream
MAP_SOURCE = '36108d4f49'
CORE = ('coverage', 'below', 'above', 'width_s', 'score_s')
DIAGNOSTICS = ('upper_delta_s', 'new_above', 'repaired_above', 'lower_penalty_delta_s', 'upper_penalty_delta_s')
METRICS = (*saved.METRICS, *('temporal_'+m for m in CORE),
    'temporal_score_delta_s', 'temporal_width_delta_s', 'upper_exceedance_s',
    *(c+'_'+m for c in ('coarse', 'v167', 'v172') for m in DIAGNOSTICS))


def parents(args):
    base, service = entry.parents(args)
    protocol = read(args.v175_root/'protocol.json')
    model = read(args.v176_root/'model.json')
    temporal, old = read(args.v172_root/'model.json'), read(args.v167_root/'model.json')
    require(model['status'] == 'sealed' and model['execution_source'] == MAP_SOURCE and
        model['frozen_protocol'] == protocol and protocol['execution_source'] == '5f8cdf4296' and
        model['base_model_source'] == temporal['execution_source'] == 'a5b45df520' and
        model['base_model_root'] == str(args.v172_root) and
        old['execution_source'] == model['frozen_v167_source'] == 'e6fc851ec1' and
        old['frozen_protocol'] == base, 'requires frozen V176/V175 and three original references')
    require(model['confirmation_release_eligible'] and model['support_admission']['decision'] == 'pass' and
        not model['confirmation_started'] and not model['confirmation_outcomes_read'] and
        not model['collector_submitted'] and not model['main_policy_changed'], 'V176 release missing')
    r = protocol['reservation']
    require(r['calibration_days'] == model['calibration_days'] == list(plan.CALIBRATION) and
        r['confirmation_days'] == list(plan.CONFIRMATION) and r['actors'] == model['actors'] == base['reservation']['actors'] and
        r['modes'] == ['old', 'handoff'] and r['confirmation_paths'] == 1536 and
        all(r['sampler_seeds'][str(d)] == plan.SAMPLERS[d] for d in plan.CONFIRMATION) and
        not set(plan.CONFIRMATION) & set(r['calibration_days']+r['excluded_confirmation_days']+service['calibration_days']),
        'reserved cohort or sampler assignment differs')
    require(protocol['frozen'] == base['frozen'] and protocol['support'] == base['support'] and
        protocol['conditioning'] == temporal['frozen_protocol']['conditioning'] and
        protocol['inference']['scopes'] == list(plan.prior.SCOPES) and
        protocol['gates'] == temporal['frozen_protocol']['gates'], 'frozen prediction or admission design differs')
    return protocol, service, model, base


def cases(protocol):
    return tuple(product(protocol['reservation']['actors'], protocol['reservation']['confirmation_days']))


@lru_cache(maxsize=4)
def map_tables(envelope_root, temporal_root, old_root, actor, mode):
    new = read(envelope_root/'maps'/f'{actor}_{mode}.json')
    temporal = read(temporal_root/'maps'/f'{actor}_{mode}.json')
    old = read(old_root/'maps'/f'{actor}_{mode}.json')
    require((new['actor'], new['mode'], new['execution_source']) == (actor, mode, MAP_SOURCE) and
        (temporal['actor'], temporal['mode'], temporal['execution_source']) == (actor, mode, 'a5b45df520') and
        (old['actor'], old['mode'], old['execution_source']) == (actor, mode, 'e6fc851ec1') and
        new['reference'] == temporal['reference'] == old['reference'], 'sealed map/reference differs')
    return tuple({tuple(e['key']): e for e in rows}
        for rows in (new['candidate'], old['reference'], old['candidate'], temporal['candidate']))


def prediction(q, candidate, coarse, old, temporal, protocol):
    p = saved.prediction(q, candidate, coarse, old, protocol)
    key = p['candidate_key']
    record = temporal.get(key)
    bounds = fit.interval(q['point_s'], record, q['native_supported'])
    require(p['finite'] == (bounds is not None), 'V176/V172 finite mask differs')
    if p['finite']:
        require(p['bounds'][0] == bounds[0] and p['bounds'][1] >= bounds[1], 'frozen envelope geometry differs')
        if not plan.key_scopes(key, protocol['conditioning']['candidate_key']):
            require(p['bounds'] == list(bounds), 'non-target interval differs')
    p.update(map_source=MAP_SOURCE, candidate_key=None if key is None else list(key),
        temporal_finite=bounds is not None, temporal_bounds=None if bounds is None else list(bounds))
    return p


def values(p, label):
    if not p['temporal_finite']:
        return None
    primary = saved.values(p, label)
    temporal = saved.prior.values(dict(p, reference_finite=p['temporal_finite'], reference_bounds=p['temporal_bounds']), label)
    if primary is None or temporal is None:
        return None
    truth, (lo, hi) = label['full_eta_s'], p['bounds']
    extra = [max(truth-hi, 0.)]
    for rl, ru in (p['reference_bounds'], p['secondary_bounds'], p['temporal_bounds']):
        extra.extend((hi-ru, float(truth > hi and truth <= ru), float(truth > ru and truth <= hi),
            10.*(max(lo-truth, 0.)-max(rl-truth, 0.)), 10.*(max(truth-hi, 0.)-max(truth-ru, 0.))))
    return np.r_[primary, temporal[5:], extra]


def collapse(group, unresolved, missing):
    rows = np.asarray(list(group.values()))
    result = dict(eligible_episodes=len(rows), eligible_queries=0, finite_queries=0, finite_episodes=0,
        finite_fraction=None, finite_unresolved_n=unresolved, missing_reference_n=missing,
        **{m: None for m in METRICS})
    if len(rows):
        matched = rows[:, 2] > 0
        result.update(eligible_queries=int(rows[:, 0].sum()), finite_queries=int(rows[:, 1].sum()),
            finite_episodes=int(np.count_nonzero(rows[:, 1])), finite_fraction=float(np.mean(rows[:, 1]/rows[:, 0])))
        if matched.any() and not (unresolved or missing):
            result.update(zip(METRICS, (rows[matched, 3:]/rows[matched, 2, None]).mean(axis=0).tolist()))
    return result


def reduce_rows(rows):
    groups, flags = {s: {} for s in plan.prior.SCOPES}, {s: Counter() for s in plan.prior.SCOPES}
    counts = Counter()
    for q, p, label in rows:
        counts.update({'requests': 1, 'label/'+label['status']: 1, 'support/'+p['support_reason']: 1})
        if not q['denominator_eligible']:
            continue
        counts['eligible_requests'] += 1
        vector, bits = values(p, label), fit.scope_bits(q)
        scopes = [s for i, s in enumerate(fit.SCOPES) if bits & (1 << i)]
        if bits & 16 and q['native_stage']['kind'] == 'moving-launch':
            scopes.append('hour13_moving_launch')
        for scope in scopes:
            row = groups[scope].setdefault(tuple(q['episode']), np.zeros(3+len(METRICS)))
            row[:3] += [1, int(p['finite']), int(vector is not None)]
            if vector is not None:
                row[3:] += vector
            flags[scope].update(unresolved=int(p['finite'] and label['status'] != 'resolved'),
                missing=int(p['finite'] and not (p['reference_finite'] and p['secondary_finite'] and p['temporal_finite'])))
    return ({s: collapse(groups[s], flags[s]['unresolved'], flags[s]['missing']) for s in plan.prior.SCOPES}, dict(counts))


def checked_rows(root, name, path, base, protocol, actor, day, mode, tables, counts):
    ready_rows = list(stream(root/'ready_labels'/(name+'.jsonl.gz')))
    ready = {capture.key(r): r for r in ready_rows}
    ranks = {k: i for i, k in enumerate(ready)}
    raw = list(stream(root/'query_snapshots'/(name+'.jsonl.gz')))
    snapshots = {tuple(r['query_key']): r for r in raw}
    require(len(ready_rows) == len(ready) == len(raw) == len(snapshots) and set(ready) == set(snapshots),
        'ready snapshots incomplete')
    for k, r in ready.items():
        saved.prior.metadata.live_record(snapshots[k], r, ranks[k])
    sources = [stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'interval_predictions', 'residuals')]
    seen = set()
    for q, p, residual in zip_longest(*sources):
        require(q is not None and p is not None and residual is not None, 'confirmation artifact lengths differ')
        k = tuple(q['query_key'])
        require(k not in seen and p == prediction(q, *tables, protocol), 'query repeated or frozen lookup differs')
        seen.add(k)
        entry.audit.query_record(q, residual, path['trace'][k], snapshots[k], ready, ranks, base,
            actor, day, mode, path['cutoff'], counts)
        yield q, p, residual['label']
    require(len(seen) == len(path['trace']), 'confirmation trace cohort incomplete')
    counts['live_ready_snapshots'] = len(snapshots)


def registered(args):
    r = read(args.out_root/'registration.json')
    require(r['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'] and r['map_source'] == MAP_SOURCE,
        'requires same-source confirmation preparation')
    return r


def collect_path(args, protocol, service, base, actor, day, mode, sampler):
    root, name = args.out_root/'confirmation', f'{actor}_{day}_{mode}'
    final = args.out_root/'cells'/(name+'.json')
    if final.exists():
        old = read(final)
        require((old['actor'], old['day'], old['mode'], old['sampler_seed'], old['execution_source'], old['map_source']) ==
            (actor, day, mode, sampler, os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], MAP_SOURCE), 'stored confirmation differs')
        return old
    tables = map_tables(args.v176_root, args.v172_root, args.v167_root, actor, mode)
    receipt = capture.collect(root, args.runtime, actor, day, mode, sampler)
    require(receipt['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'rollout source differs')
    path, geometry, captured = capture.save_queries(root, args.runtime, actor, day, mode, service, base)
    destination = root/'interval_predictions'/(name+'.jsonl.gz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, 'wt', encoding='ascii', compresslevel=1) as output:
        for q in stream(root/'queries'/(name+'.jsonl.gz')):
            output.write(json.dumps(prediction(q, *tables, protocol), allow_nan=False)+'\n')
    captured.update(capture.attach_labels(root, name, path, geometry))
    counts = Counter()
    cells, scoring = reduce_rows(checked_rows(root, name, path, base, protocol, actor, day, mode, tables, counts))
    for field, value in captured.items():
        require(counts[field] == value, 'capture/readback counter differs: '+field)
    require(counts['queries'] == receipt['decisions'] and counts['live_ready_snapshots'] == receipt['live_query_snapshots'],
        'rollout/query cohort differs')
    result = dict(actor=actor, day=day, mode=mode, sampler_seed=sampler, cells=cells, counts=dict(counts),
        scoring_counts=scoring, map_source=MAP_SOURCE, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        bounds_saved_before_labels=True, interval_fit=False, map_updates=0, actor_updates=0,
        critic_updates=0, dual_updates=0, main_policy_changed=False)
    write(final, result)
    print('V177_CONFIRMATION_PATH_COMPLETE', actor, day, mode, flush=True)
    return result


def unconsumed(args):
    roots = set(plan.ledger.archives.result_roots(args.parent)) | {args.v176_root.parent.parent}
    collisions, receipts, names = [], 0, 0
    for root in sorted(roots):
        for directory, children, filenames in os.walk(root):
            path = Path(directory)
            children[:] = [d for d in children if d not in plan.ledger.SKIP and
                path/d != args.out_root.parent and len((path/d).relative_to(root).parts) <= 5]
            if path.name not in plan.ledger.RAW | plan.ledger.RECEIPTS | {'interval_predictions'}:
                continue
            children[:] = []
            for name in filenames:
                names += 1
                day = plan.ledger.filename_day(name)
                if day in plan.CONFIRMATION:
                    collisions.append(str(path/name))
                if path.name in plan.ledger.RECEIPTS and name.endswith('.json'):
                    receipts += 1
                    if plan.ledger.metadata.sampler_ids(read(path/name)) & set(plan.SAMPLERS.values()):
                        collisions.append(str(path/name))
    require(not collisions, 'reserved confirmation day/sampler already consumed: '+repr(collisions[:10]))
    return dict(roots=len(roots), artifact_names=names, sampler_receipts=receipts, collisions=[],
        scope='current raw artifact filenames and sampler receipts only; original V175 reservation retained')


def prepare(args):
    require(not (args.out_root/'registration.json').exists(), 'confirmation already prepared')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v177_*.py', '-v'], check=True)
    protocol, _, model, _ = parents(args)
    proof = read(args.v166_root/'preflight/summary/status.json')
    require(proof['status'] == 'v166_passive_equivalence_passed' and proof['execution_source'] == '20d914af03' and
        all(proof[k] for k in entry.EQUIVALENCE), 'unchanged passive collector proof missing')
    benchmark = read(args.runtime_proof_root/'benchmark_summary.json')
    training = read(args.runtime_proof_root/'training_equivalence/summary.json')
    require(benchmark['all_exact'] and training['matches_original_smoke'], 'runtime equivalence proof missing')
    link = args.out_root/'runtime_optimization_20261014'
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(args.runtime_proof_root.resolve(), target_is_directory=True)
    _, runner, _, _ = capture.collector.capture.pilot.joint.optimized_runtime(args.out_root/'confirmation')
    require(Path(sys.modules[runner.__module__].__file__).resolve().parent == args.runtime.resolve(), 'runtime import differs')
    for actor, mode in product(protocol['reservation']['actors'], ('old', 'handoff')):
        map_tables(args.v176_root, args.v172_root, args.v167_root, actor, mode)
    freshness = unconsumed(args)
    registration = dict(status='v177_confirmation_prepared', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        map_source=MAP_SOURCE, protocol_source='5f8cdf4296', reservation=protocol['reservation'],
        sealed_model=model, confirmation_paths=1536, interval_fit=False, map_updates=0,
        main_policy_changed=False, bootstrap_draws=5000, bootstrap_seed=20260803, freshness=freshness,
        runtime_proof_root=str(args.runtime_proof_root.resolve()))
    write(args.out_root/'registration.json', registration)
    write(args.out_root/'prepare/summary/status.json', {k: v for k, v in registration.items() if k != 'sealed_model'})
    print('V177_FROZEN_CONFIRMATION_ENTRY_READY', flush=True)


def pair(args, index):
    registered(args)
    protocol, service, _, base = parents(args)
    actor, day = cases(protocol)[index]
    for mode in ('old', 'handoff'):
        collect_path(args, protocol, service, base, actor, day, mode, protocol['reservation']['sampler_seeds'][str(day)])


def run(args):
    registered(args)
    protocol, _, _, _ = parents(args)
    require(0 <= args.start < args.end <= len(cases(protocol)) and 1 <= args.workers <= 8, 'invalid shard')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(pair, args, i) for i in range(args.start, args.end)]
        for done, job in enumerate(as_completed(jobs), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=len(jobs), unit='confirmation_actor_day')), flush=True)
    write(args.out_root/'receipts'/f'{args.start}-{args.end}'/'status.json',
        dict(status='v177_shard_complete', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
            map_source=MAP_SOURCE, start=args.start, end=args.end, confirmation_paths=2*(args.end-args.start),
            interval_fit=False, map_updates=0, main_policy_changed=False))
    print('V177_CONFIRMATION_SHARD_COMPLETE', args.start, args.end, flush=True)


def crossed(cells, actors, days):
    actors, days = sorted(actors), sorted(days)
    index = {(r['actor'], r['day']): r for r in cells}
    require(len(index) == len(cells) == len(actors)*len(days) and set(index) == set(product(actors, days)),
        'crossed grid missing or duplicated')
    rng = np.random.RandomState(20260803)
    a = rng.randint(0, len(actors), size=(5000, len(actors)))
    d = rng.randint(0, len(days), size=(5000, len(days)))
    estimates = {}
    for metric in ('finite_fraction', *METRICS):
        raw = [[index[actor, day][metric] for day in days] for actor in actors]
        if any(v is None for row in raw for v in row):
            estimates[metric] = dict(mean=None, ci95=None, lower95=None, upper95=None)
        else:
            matrix = np.asarray(raw, dtype=float)
            require(np.isfinite(matrix).all(), 'nonfinite crossed metric')
            draws = matrix[a[:, :, None], d[:, None, :]].mean(axis=(1, 2))
            estimates[metric] = dict(mean=float(matrix.mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist(),
                lower95=float(np.percentile(draws, 5.)), upper95=float(np.percentile(draws, 95.)))
    return dict(cells=len(cells), complete_actor_day_grid=all(r['eligible_episodes'] and r['finite_episodes'] for r in cells),
        finite_unresolved_n=sum(r['finite_unresolved_n'] for r in cells),
        missing_reference_n=sum(r['missing_reference_n'] for r in cells), estimates=estimates,
        coverage_lower95=estimates['coverage']['lower95'], below_upper95=estimates['below']['upper95'],
        above_upper95=estimates['above']['upper95'], finite_fraction_lower95=estimates['finite_fraction']['lower95'],
        score_delta_upper95=estimates['score_delta_s']['upper95'])


def readout(cells, actors, days, mode, scope):
    return dict(mode=mode, scope=scope, **crossed([c for c in cells if (c['mode'], c['scope']) == (mode, scope)], actors, days))


def aggregate(args):
    registration = registered(args)
    protocol, _, model, _ = parents(args)
    require(model == registration['sealed_model'], 'sealed V176 header changed')
    r = protocol['reservation']
    names = {f'{a}_{d}_{m}.json' for a, d in cases(protocol) for m in ('old', 'handoff')}
    require({p.name for p in (args.out_root/'cells').glob('*.json')} == names, 'confirmation matrix incomplete/contaminated')
    receipts = [read(p) for p in (args.out_root/'receipts').glob('*/status.json')]
    require(sorted(i for s in receipts for i in range(s['start'], s['end'])) == list(range(len(cases(protocol)))) and
        all(s['status'] == 'v177_shard_complete' and s['execution_source'] == registration['execution_source'] and
            s['map_source'] == MAP_SOURCE and not s['interval_fit'] and not s['map_updates'] for s in receipts), 'shard gaps/source differ')
    paths = [read(args.out_root/'cells'/name) for name in sorted(names)]
    for p in paths:
        require(p['execution_source'] == registration['execution_source'] and p['map_source'] == MAP_SOURCE and
            p['sampler_seed'] == r['sampler_seeds'][str(p['day'])] and p['bounds_saved_before_labels'] and
            not p['interval_fit'] and p['map_updates'] == 0 and not p['main_policy_changed'] and
            all(p[k+'_updates'] == 0 for k in ('actor', 'critic', 'dual')), 'confirmation freeze differs')
    cells = [dict(actor=p['actor'], day=p['day'], mode=p['mode'], scope=s, **p['cells'][s])
        for p in paths for s in plan.prior.SCOPES]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(readout, cells, r['actors'], r['confirmation_days'], m, s)
            for m, s in product(('old', 'handoff'), plan.prior.SCOPES)]
        readouts = [job.result() for job in jobs]
    decision = plan.prior.gate(readouts)
    write(args.out_root/'details/actor_day_cells.json', cells)
    for axis in ('actor', 'day'):
        rows = []
        for mode, scope in product(('old', 'handoff'), plan.prior.SCOPES):
            selected = [c for c in cells if (c['mode'], c['scope']) == (mode, scope)]
            for ident in sorted({c[axis] for c in selected}):
                group = [c for c in selected if c[axis] == ident]
                rows.append(dict(mode=mode, scope=scope, **{axis: ident}, cells=len(group),
                    **{m: None if any(c[m] is None for c in group) else float(np.mean([c[m] for c in group]))
                        for m in ('finite_fraction', *METRICS)}))
        write(args.out_root/'details'/('by_'+axis+'.json'), rows)
    require(read(args.v176_root/'model.json') == registration['sealed_model'], 'map changed during aggregation')
    write(args.out_root/'summary/status.json', dict(status='v177_independent_confirmation_complete',
        execution_source=registration['execution_source'], map_source=MAP_SOURCE, protocol_source='5f8cdf4296',
        confirmation_days=r['confirmation_days'], actors=r['actors'], modes=['old', 'handoff'], paths=len(paths),
        shards=len(receipts), counts=dict(sum((Counter(p['counts']) for p in paths), Counter())),
        scoring_counts=dict(sum((Counter(p['scoring_counts']) for p in paths), Counter())),
        readouts=readouts, gate=decision, bootstrap_draws=5000, bootstrap_seed=20260803,
        interval_fit=False, map_updates=0, v167_changed=False, v172_changed=False, v176_changed=False,
        v151_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1], new_training=0,
        main_policy_changed=False, next='Preserve all independent outcomes; no tuning or map updates on these days.'))
    print('V177_INDEPENDENT_CONFIRMATION_COMPLETE', decision['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'aggregate'))
    for name in ('out-root', 'runtime', 'runtime-proof-root', 'v176-root', 'v175-root', 'v172-root',
            'v167-root', 'v166-root', 'v165-root', 'v161-root', 'v151-root', 'v146-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--parent', type=Path, action='append', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'aggregate': aggregate}[args.command](args)
