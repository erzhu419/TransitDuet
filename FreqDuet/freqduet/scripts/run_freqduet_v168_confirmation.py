#!/usr/bin/env python3
"""Collect the reserved confirmation days and apply sealed V167 maps once."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, zip_longest
import gzip
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import audit_freqduet_v166_capture as metadata
import run_freqduet_v166_fresh_capture as entry
import seal_freqduet_v163_joint_intervals as fit
from freqduet_interval_validation import hierarchical_bootstrap


capture, protocol = entry.capture, fit.protocol
read, write, require, stream = entry.read, entry.write, entry.require, entry.stream
MAP_SOURCE = 'e6fc851ec1'
METRICS = ('coverage', 'below', 'above', 'width_s', 'score_s', 'reference_coverage',
    'reference_below', 'reference_above', 'reference_width_s', 'reference_score_s',
    'score_delta_s', 'width_delta_s')


def parents(args):
    sealed, service = entry.parents(args)
    model = read(args.v167_root/'model.json')
    require(model['status'] == 'sealed' and model['execution_source'] == MAP_SOURCE and
        model['frozen_protocol'] == sealed and model['calibration_days'] == sealed['reservation']['calibration_days'],
        'requires unchanged V167 seal and V165 protocol')
    require(model['confirmation_release_eligible'] and model['support_admission']['decision'] == 'pass' and
        not model['confirmation_started'] and not model['confirmation_outcomes_read'], 'V167 support release missing')
    r = sealed['reservation']
    require(r['confirmation_days'] == list(range(2852001, 2852033)) and
        [r['sampler_seeds'][str(d)] for d in r['confirmation_days']] == list(range(1600000032, 1600000064)) and
        not set(r['confirmation_days']) & set(r['calibration_days']+r['excluded_development_days']),
        'reserved confirmation assignment changed')
    return sealed, service, model


def cases(sealed):
    r = sealed['reservation']
    return tuple(product(r['actors'], r['confirmation_days']))


def maps(args, actor, mode):
    saved = read(args.v167_root/'maps'/f'{actor}_{mode}.json')
    require((saved['actor'], saved['mode'], saved['execution_source']) == (actor, mode, MAP_SOURCE), 'map source differs')
    return ({tuple(e['key']): e for e in saved['candidate']}, {tuple(e['key']): e for e in saved['reference']})


def prediction(q, candidate, reference, sealed):
    key = tuple(q['candidate_key']) if q['native_supported'] else None
    record = candidate.get(key)
    ref = reference.get(tuple(q['condition_values'][f] for f in sealed['comparator']['key']))
    bounds = fit.interval(q['point_s'], record, q['native_supported'])
    coarse = fit.interval(q['point_s'], ref, q['native_supported'])
    reason = ('ineligible' if not q['denominator_eligible'] else
        'native/'+q['native_stage']['reason'] if not q['native_supported'] else
        'unseen_key' if record is None else record['status'])
    return dict(query_key=q['query_key'], query_time_s=q['query_time_s'], map_source=MAP_SOURCE,
        finite=bounds is not None, reference_finite=coarse is not None,
        bounds=None if bounds is None else list(bounds), reference_bounds=None if coarse is None else list(coarse),
        support_reason=reason, calibration_days=None if record is None else record['day_count'])


def save_predictions(root, name, candidate, reference, sealed):
    destination = root/'interval_predictions'/(name+'.jsonl.gz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, 'wt', encoding='ascii', compresslevel=1) as handle:
        for q in stream(root/'queries'/(name+'.jsonl.gz')):
            handle.write(json.dumps(prediction(q, candidate, reference, sealed), allow_nan=False)+'\n')


def values(predicted, label):
    if not predicted['finite'] or not predicted['reference_finite'] or label['status'] != 'resolved':
        return None
    truth = label['full_eta_s']
    output = []
    for lo, hi in (predicted['bounds'], predicted['reference_bounds']):
        width = hi-lo
        output.extend((float(lo <= truth <= hi), float(truth < lo), float(truth > hi),
            width, width+10.*max(lo-truth, 0.)+10.*max(truth-hi, 0.)))
    return np.array([*output, output[4]-output[9], output[3]-output[8]])


def add_episode(group, episode, finite, scored):
    value = group.setdefault(tuple(episode), np.zeros(3+len(METRICS)))
    value[0] += 1
    value[1] += int(finite)
    if scored is not None:
        value[2] += 1
        value[3:] += scored


def collapse(group):
    rows = np.asarray(list(group.values()))
    result = dict(eligible_episodes=len(rows), eligible_queries=0, finite_queries=0,
        finite_episodes=0, matched_episodes=0, finite_fraction=None, **{m: None for m in METRICS})
    if not len(rows):
        return result
    matched = rows[:, 2] > 0
    result.update(eligible_queries=int(rows[:, 0].sum()), finite_queries=int(rows[:, 1].sum()),
        finite_episodes=int((rows[:, 1] > 0).sum()), matched_episodes=int(matched.sum()),
        finite_fraction=float(np.mean(rows[:, 1]/rows[:, 0])))
    if matched.any():
        result.update(zip(METRICS, np.mean(rows[matched, 3:]/rows[matched, 2, None], axis=0).tolist()))
    return result


def reduce_rows(rows):
    groups = {s: {} for s in protocol.SCOPES}
    flags = {s: Counter() for s in protocol.SCOPES}
    key_groups, key_flags, counts = {}, defaultdict(Counter), Counter()
    for q, predicted, label in rows:
        counts['requests'] += 1
        counts['label/'+label['status']] += 1
        counts['support/'+predicted['support_reason']] += 1
        if not q['denominator_eligible']:
            continue
        counts['eligible_requests'] += 1
        scored = values(predicted, label)
        unresolved = int(predicted['finite'] and label['status'] != 'resolved')
        missing = int(predicted['finite'] and not predicted['reference_finite'])
        bits = fit.scope_bits(q)
        for i, scope in enumerate(protocol.SCOPES):
            if bits & (1 << i):
                add_episode(groups[scope], q['episode'], predicted['finite'], scored)
                flags[scope].update(finite_unresolved_n=unresolved, missing_reference_n=missing)
        if q['native_supported']:
            key = json.dumps(q['candidate_key'], separators=(',', ':'))
            add_episode(key_groups.setdefault(key, {}), q['episode'], predicted['finite'], scored)
            key_flags[key].update(finite_unresolved_n=unresolved, missing_reference_n=missing)
    def finish(group, flags):
        result = dict(collapse(group), finite_unresolved_n=flags['finite_unresolved_n'],
            missing_reference_n=flags['missing_reference_n'])
        if result['finite_unresolved_n'] or result['missing_reference_n']:
            result.update({m: None for m in METRICS})
        return result
    return ({s: finish(groups[s], flags[s]) for s in protocol.SCOPES},
        {k: finish(g, key_flags[k]) for k, g in key_groups.items()}, dict(counts))


def checked_rows(root, name, path, sealed, actor, day, mode, candidate, reference, counts):
    ready_rows = list(stream(root/'ready_labels'/(name+'.jsonl.gz')))
    ready = {capture.key(r): r for r in ready_rows}
    ranks = {k: i for i, k in enumerate(ready)}
    raw = list(stream(root/'query_snapshots'/(name+'.jsonl.gz')))
    snapshots = {tuple(r['query_key']): r for r in raw}
    require(len(ready_rows) == len(ready) == len(raw) == len(snapshots) and set(ready) == set(snapshots),
        'incomplete confirmation ready snapshots')
    for k, r in ready.items():
        metadata.live_record(snapshots[k], r, ranks[k])
    sources = [stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'interval_predictions', 'residuals')]
    seen = set()
    for q, predicted, residual in zip_longest(*sources):
        require(q is not None and predicted is not None and residual is not None, 'confirmation artifact lengths differ')
        k = tuple(q['query_key'])
        require(k not in seen, 'duplicate confirmation query')
        seen.add(k)
        require(predicted == prediction(q, candidate, reference, sealed), 'saved bounds differ from frozen lookup')
        entry.audit.query_record(q, residual, path['trace'][k], snapshots[k], ready, ranks, sealed,
            actor, day, mode, path['cutoff'], counts)
        yield q, predicted, residual['label']
    require(len(seen) == len(path['trace']), 'confirmation trace cohort incomplete')
    counts['live_ready_snapshots'] = len(snapshots)


def registered(args):
    r = read(args.out_root/'registration.json')
    require(r['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'] and r['map_source'] == MAP_SOURCE,
        'requires same-source confirmation preparation')
    return r


def collect_path(args, sealed, service, actor, day, mode, sampler):
    root, name = args.out_root/'confirmation', f'{actor}_{day}_{mode}'
    final = args.out_root/'cells'/(name+'.json')
    if final.exists():
        old = read(final)
        require((old['actor'], old['day'], old['mode'], old['sampler_seed'], old['execution_source'], old['map_source']) ==
            (actor, day, mode, sampler, os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], MAP_SOURCE), 'stored confirmation differs')
        return old
    candidate, reference = maps(args, actor, mode)
    receipt = capture.collect(root, args.runtime, actor, day, mode, sampler)
    require(receipt['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'rollout source differs')
    path, geometry, captured = capture.save_queries(root, args.runtime, actor, day, mode, service, sealed)
    save_predictions(root, name, candidate, reference, sealed)
    captured.update(capture.attach_labels(root, name, path, geometry))
    counts = Counter()
    cells, keys, scoring = reduce_rows(checked_rows(root, name, path, sealed, actor, day, mode, candidate, reference, counts))
    for field, value in captured.items():
        require(counts[field] == value, 'confirmation capture/readback counter differs: '+field)
    require(counts['queries'] == receipt['decisions'] and counts['live_ready_snapshots'] == receipt['live_query_snapshots'],
        'confirmation rollout/query cohort differs')
    result = dict(actor=actor, day=day, mode=mode, sampler_seed=sampler, counts=dict(counts),
        scoring_counts=scoring, cells=cells, key_stats=keys, map_source=MAP_SOURCE,
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], bounds_saved_before_labels=True,
        interval_fit=False, map_updates=0, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    write(final, result)
    print('V168_CONFIRMATION_PATH_COMPLETE', actor, day, mode, flush=True)
    return result


def prepare(args):
    require(not (args.out_root/'registration.json').exists(), 'confirmation entry already prepared')
    for pattern in ('test_v161_joint_protocol.py', 'test_v168_*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    sealed, _, model = parents(args)
    proof = read(args.v166_root/'preflight/summary/status.json')
    require(proof['status'] == 'v166_passive_equivalence_passed' and proof['execution_source'] == '20d914af03' and
        all(proof[k] for k in entry.EQUIVALENCE), 'passive capture admission missing')
    for actor in sealed['reservation']['actors']:
        for mode in protocol.MODES:
            maps(args, actor, mode)
    registration = dict(status='v168_confirmation_prepared', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        map_source=MAP_SOURCE, protocol_source='4a6d783667', reservation=sealed['reservation'],
        sealed_model=model, confirmation_paths=768, interval_fit=False, map_updates=0,
        main_policy_changed=False, bootstrap_draws=5000, bootstrap_seed=20260803)
    write(args.out_root/'registration.json', registration)
    write(args.out_root/'prepare/summary/status.json', {k: v for k, v in registration.items() if k != 'sealed_model'})
    print('V168_FROZEN_CONFIRMATION_ENTRY_READY', flush=True)


def pair(args, index):
    registered(args)
    sealed, service, _ = parents(args)
    actor, day = cases(sealed)[index]
    sampler = sealed['reservation']['sampler_seeds'][str(day)]
    for mode in protocol.MODES:
        collect_path(args, sealed, service, actor, day, mode, sampler)
    return index


def run(args):
    registered(args)
    sealed, _, _ = parents(args)
    require(0 <= args.start < args.end <= len(cases(sealed)) and 1 <= args.workers <= 8, 'invalid confirmation shard')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(pair, args, i) for i in range(args.start, args.end)]
        for done, job in enumerate(as_completed(jobs), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=len(jobs), unit='confirmation_actor_day')), flush=True)
    write(args.out_root/'shards'/f'{args.start}-{args.end}.json', dict(status='v168_confirmation_shard_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], map_source=MAP_SOURCE,
        start=args.start, end=args.end, confirmation_paths=2*(args.end-args.start), interval_fit=False, map_updates=0))
    print('V168_CONFIRMATION_SHARD_COMPLETE', args.start, args.end, flush=True)


def crossed(cells, actors, days):
    expected = {(a, d) for a in actors for d in days}
    identities = [(r['actor'], r['day']) for r in cells]
    require(len(identities) == len(expected) and set(identities) == expected, 'confirmation crossed grid differs')
    estimates = {}
    for metric in ('finite_fraction', *METRICS):
        vals = [r[metric] for r in cells]
        if any(v is None for v in vals):
            estimates[metric] = dict(mean=None, ci95=None, lower95=None, upper95=None)
            continue
        frame = pd.DataFrame([dict(train_seed=r['actor'], eval_seed=r['day'], **{metric: r[metric]}) for r in cells])
        draws = hierarchical_bootstrap(frame, metric, draws=5000)
        estimates[metric] = dict(mean=float(np.mean(vals)), ci95=np.percentile(draws, [2.5, 97.5]).tolist(),
            lower95=float(np.percentile(draws, 5.)), upper95=float(np.percentile(draws, 95.)))
    return dict(complete_actor_day_grid=all(r['eligible_episodes'] > 0 and r['finite_episodes'] > 0 for r in cells),
        finite_unresolved_n=sum(r['finite_unresolved_n'] for r in cells),
        missing_reference_n=sum(r['missing_reference_n'] for r in cells), cells=len(cells), estimates=estimates,
        coverage_lower95=estimates['coverage']['lower95'], below_upper95=estimates['below']['upper95'],
        above_upper95=estimates['above']['upper95'], finite_fraction_lower95=estimates['finite_fraction']['lower95'],
        score_delta_upper95=estimates['score_delta_s']['upper95'])


def aggregate(args):
    registration = registered(args)
    sealed, _, model = parents(args)
    require(model == registration['sealed_model'], 'V167 sealed model changed during confirmation')
    r = sealed['reservation']
    names = {f'{a}_{d}_{m}.json' for a, d in cases(sealed) for m in protocol.MODES}
    require({p.name for p in (args.out_root/'cells').glob('*.json')} == names, 'confirmation matrix incomplete/contaminated')
    shards = [read(p) for p in (args.out_root/'shards').glob('*.json')]
    require(sorted(i for s in shards for i in range(s['start'], s['end'])) == list(range(len(cases(sealed)))) and
        all(s['status'] == 'v168_confirmation_shard_complete' and s['map_source'] == MAP_SOURCE and
            s['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'] and not s['interval_fit'] and
            s['map_updates'] == 0 for s in shards), 'confirmation shard gaps/source differ')
    paths = [read(args.out_root/'cells'/n) for n in sorted(names)]
    for p in paths:
        require(p['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'] and p['map_source'] == MAP_SOURCE and
            p['sampler_seed'] == r['sampler_seeds'][str(p['day'])] and p['bounds_saved_before_labels'] and
            not p['interval_fit'] and not p['main_policy_changed'] and p['map_updates'] == 0 and
            all(p[k+'_updates'] == 0 for k in ('actor', 'critic', 'dual')), 'confirmation source/freeze differs')
    readouts, flattened = [], []
    for mode in protocol.MODES:
        for scope in protocol.SCOPES:
            selected = [dict(actor=p['actor'], day=p['day'], mode=mode, scope=scope, **p['cells'][scope])
                for p in paths if p['mode'] == mode]
            flattened.extend(selected)
            readouts.append(dict(mode=mode, scope=scope, **crossed(selected, r['actors'], r['confirmation_days'])))
    decision = protocol.gate(readouts)
    (args.out_root/'details').mkdir(parents=True, exist_ok=True)
    write(args.out_root/'details/actor_day_cells.json', flattened)
    for axis in ('actor', 'day'):
        marginal = []
        for mode in protocol.MODES:
            for scope in protocol.SCOPES:
                selected = [c for c in flattened if (c['mode'], c['scope']) == (mode, scope)]
                for ident in sorted({c[axis] for c in selected}):
                    group = [c for c in selected if c[axis] == ident]
                    marginal.append(dict(mode=mode, scope=scope, **{axis: ident}, cells=len(group),
                        **{m: None if any(c[m] is None for c in group) else float(np.mean([c[m] for c in group]))
                            for m in ('finite_fraction', *METRICS)}))
        write(args.out_root/'details'/('by_'+axis+'.json'), marginal)
    keys = defaultdict(list)
    for p in paths:
        for key, stats in p['key_stats'].items():
            keys[key].append(dict(day=p['day'], **stats))
    write(args.out_root/'details/candidate_keys.json', {key: dict(days=len(rows),
        requests=sum(x['eligible_queries'] for x in rows), finite=sum(x['finite_queries'] for x in rows),
        coverage=None if any(x['coverage'] is None for x in rows) else float(np.mean([x['coverage'] for x in rows])))
        for key, rows in keys.items()})
    summary = dict(status='v168_independent_confirmation_complete', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        map_source=MAP_SOURCE, protocol_source='4a6d783667', confirmation_days=r['confirmation_days'],
        actors=r['actors'], modes=list(protocol.MODES), paths=len(paths), shards=len(shards),
        counts=dict(sum((Counter(p['counts']) for p in paths), Counter())),
        scoring_counts=dict(sum((Counter(p['scoring_counts']) for p in paths), Counter())),
        readouts=readouts, gate=decision, bootstrap_draws=5000, bootstrap_seed=20260803,
        interval_fit=False, map_updates=0, v151_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1], new_training=0, main_policy_changed=False,
        next='Retain this independent result; no tuning or model updates on confirmation outcomes.')
    write(args.out_root/'summary/status.json', summary)
    print('V168_INDEPENDENT_CONFIRMATION_COMPLETE', decision['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'aggregate'))
    for name in ('out-root', 'runtime', 'v167-root', 'v166-root', 'v165-root', 'v161-root', 'v151-root', 'v146-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'aggregate': aggregate}[args.command](args)
