#!/usr/bin/env python3
"""Confirm the frozen V172 map on the 32 V171-reserved demand days once."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from itertools import product, zip_longest
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import run_freqduet_v168_confirmation as prior
import preregister_freqduet_v171_temporal_tails as plan


fit, capture, entry = prior.fit, prior.capture, prior.entry
read, write, require, stream = prior.read, prior.write, prior.require, prior.stream
MAP_SOURCE = 'a5b45df520'
METRICS = (*prior.METRICS, 'secondary_coverage', 'secondary_below', 'secondary_above',
    'secondary_width_s', 'secondary_score_s', 'secondary_score_delta_s', 'secondary_width_delta_s')


def parents(args):
    base, service = entry.parents(args)
    protocol = read(args.v171_root/'protocol.json')
    model = read(args.v172_root/'model.json')
    old = read(args.v167_root/'model.json')
    require(model['status'] == 'sealed' and model['execution_source'] == MAP_SOURCE and
        model['frozen_protocol'] == protocol and protocol['execution_source'] == 'f5d152f0bc' and
        old['execution_source'] == model['frozen_v167_source'] == 'e6fc851ec1' and
        old['frozen_protocol'] == base and model['frozen_v167_root'] == str(args.v167_root), 'frozen parents differ')
    require(model['confirmation_release_eligible'] and model['support_admission']['decision'] == 'pass' and
        not model['confirmation_started'] and not model['confirmation_outcomes_read'], 'V172 support release missing')
    r = protocol['reservation']
    require(r['calibration_days'] == model['calibration_days'] == list(range(2951001, 2951065)) and
        r['actors'] == model['actors'] == base['reservation']['actors'] and
        r['confirmation_days'] == list(plan.CONFIRMATION) and r['modes'] == ['old', 'handoff'] and
        all(r['sampler_seeds'][str(d)] == plan.SAMPLERS[d] for d in plan.CONFIRMATION) and
        not set(plan.CONFIRMATION) & set(r['calibration_days']+service['calibration_days']), 'reserved cohort differs')
    require(protocol['frozen'] == base['frozen'] and protocol['support'] == base['support'] and
        protocol['conditioning']['candidate_key'] == base['conditioning']['candidate_key']+['query_time_stratum'] and
        protocol['inference']['scopes'] == list(plan.SCOPES), 'frozen method differs')
    return protocol, service, model, base


def cases(protocol):
    return tuple(product(protocol['reservation']['actors'], protocol['reservation']['confirmation_days']))


def maps(args, actor, mode):
    new = read(args.v172_root/'maps'/f'{actor}_{mode}.json')
    old = read(args.v167_root/'maps'/f'{actor}_{mode}.json')
    require((new['actor'], new['mode'], new['execution_source']) == (actor, mode, MAP_SOURCE) and
        (old['actor'], old['mode'], old['execution_source']) == (actor, mode, 'e6fc851ec1') and
        new['reference'] == old['reference'], 'sealed maps/reference differ')
    return tuple({tuple(e['key']): e for e in rows} for rows in (new['candidate'], old['reference'], old['candidate']))


def prediction(q, candidate, reference, secondary, protocol):
    period = plan.STRATA[int(plan.strata(np.array([q['query_time_s']]))[0])]
    key = (*q['candidate_key'], period) if q['native_supported'] else None
    record = candidate.get(key)
    ref = reference.get(tuple(q['condition_values'][f] for f in protocol['comparator']['key']))
    old = secondary.get(tuple(q['candidate_key'])) if q['native_supported'] else None
    bounds = [fit.interval(q['point_s'], e, q['native_supported']) for e in (record, ref, old)]
    reason = ('ineligible' if not q['denominator_eligible'] else
        'native/'+q['native_stage']['reason'] if not q['native_supported'] else
        'unseen_key' if record is None else record['status'])
    return dict(query_key=q['query_key'], query_time_s=q['query_time_s'], candidate_key=key,
        map_source=MAP_SOURCE, finite=bounds[0] is not None, reference_finite=bounds[1] is not None,
        secondary_finite=bounds[2] is not None, bounds=None if bounds[0] is None else list(bounds[0]),
        reference_bounds=None if bounds[1] is None else list(bounds[1]),
        secondary_bounds=None if bounds[2] is None else list(bounds[2]), support_reason=reason,
        calibration_days=None if record is None else record['day_count'])


def save_predictions(root, name, tables, protocol):
    destination = root/'interval_predictions'/(name+'.jsonl.gz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, 'wt', encoding='ascii', compresslevel=1) as output:
        for q in stream(root/'queries'/(name+'.jsonl.gz')):
            output.write(json.dumps(prediction(q, *tables, protocol), allow_nan=False)+'\n')


def values(predicted, label):
    if not predicted['secondary_finite']:
        return None
    primary = prior.values(predicted, label)
    secondary = prior.values(dict(predicted, reference_finite=predicted['secondary_finite'],
        reference_bounds=predicted['secondary_bounds']), label)
    return None if primary is None or secondary is None else np.r_[primary, secondary[5:]]


def add_episode(group, episode, finite, scored):
    row = group.setdefault(tuple(episode), np.zeros(3+len(METRICS)))
    row[:3] += [1, int(finite), int(scored is not None)]
    if scored is not None:
        row[3:] += scored


def collapse(group, unresolved=0, missing=0):
    rows = np.asarray(list(group.values()))
    result = dict(eligible_episodes=len(rows), eligible_queries=0, finite_queries=0, finite_episodes=0,
        finite_fraction=None, finite_unresolved_n=unresolved, missing_reference_n=missing,
        **{m: None for m in METRICS})
    if len(rows):
        matched = rows[:, 2] > 0
        result.update(eligible_queries=int(rows[:, 0].sum()), finite_queries=int(rows[:, 1].sum()),
            finite_episodes=int((rows[:, 1] > 0).sum()), finite_fraction=float(np.mean(rows[:, 1]/rows[:, 0])))
        if matched.any() and not (unresolved or missing):
            result.update(zip(METRICS, np.mean(rows[matched, 3:]/rows[matched, 2, None], axis=0).tolist()))
    return result


def reduce_rows(rows):
    groups, flags, counts = {s: {} for s in plan.SCOPES}, {s: Counter() for s in plan.SCOPES}, Counter()
    for q, predicted, label in rows:
        counts.update({'requests': 1, 'label/'+label['status']: 1, 'support/'+predicted['support_reason']: 1})
        if not q['denominator_eligible']:
            continue
        counts['eligible_requests'] += 1
        scored = values(predicted, label)
        bits = fit.scope_bits(q)
        selected = [s for i, s in enumerate(fit.SCOPES) if bits & (1 << i)]
        if bits & 16 and q['native_stage']['kind'] == 'moving-launch':
            selected.append('hour13_moving_launch')
        for scope in selected:
            add_episode(groups[scope], q['episode'], predicted['finite'], scored)
            flags[scope].update(unresolved=int(predicted['finite'] and label['status'] != 'resolved'),
                missing=int(predicted['finite'] and not (predicted['reference_finite'] and predicted['secondary_finite'])))
    return ({s: collapse(groups[s], flags[s]['unresolved'], flags[s]['missing']) for s in plan.SCOPES}, dict(counts))


def checked_rows(root, name, path, base, protocol, actor, day, mode, tables, counts):
    ready_rows = list(stream(root/'ready_labels'/(name+'.jsonl.gz')))
    ready = {capture.key(r): r for r in ready_rows}
    ranks = {k: i for i, k in enumerate(ready)}
    raw = list(stream(root/'query_snapshots'/(name+'.jsonl.gz')))
    snapshots = {tuple(r['query_key']): r for r in raw}
    require(len(ready_rows) == len(ready) == len(raw) == len(snapshots) and set(ready) == set(snapshots),
        'incomplete confirmation snapshots')
    for k, r in ready.items():
        prior.metadata.live_record(snapshots[k], r, ranks[k])
    sources = [stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'interval_predictions', 'residuals')]
    seen = set()
    for q, predicted, residual in zip_longest(*sources):
        require(q is not None and predicted is not None and residual is not None, 'confirmation artifact lengths differ')
        k = tuple(q['query_key'])
        require(k not in seen, 'duplicate confirmation query')
        seen.add(k)
        expected = prediction(q, *tables, protocol)
        expected['candidate_key'] = None if expected['candidate_key'] is None else list(expected['candidate_key'])
        require(predicted == expected, 'saved bounds differ from frozen temporal lookup')
        entry.audit.query_record(q, residual, path['trace'][k], snapshots[k], ready, ranks, base,
            actor, day, mode, path['cutoff'], counts)
        yield q, predicted, residual['label']
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
            (actor, day, mode, sampler, os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], MAP_SOURCE), 'stored path differs')
        return old
    tables = maps(args, actor, mode)
    receipt = capture.collect(root, args.runtime, actor, day, mode, sampler)
    require(receipt['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'rollout source differs')
    # Preserve the audited raw query schema; only interval lookup adds the time stratum.
    path, geometry, captured = capture.save_queries(root, args.runtime, actor, day, mode, service, base)
    save_predictions(root, name, tables, protocol)
    captured.update(capture.attach_labels(root, name, path, geometry))
    counts = Counter()
    cells, scoring = reduce_rows(checked_rows(root, name, path, base, protocol, actor, day, mode, tables, counts))
    for field, value in captured.items():
        require(counts[field] == value, 'capture/readback counter differs: '+field)
    require(counts['queries'] == receipt['decisions'] and counts['live_ready_snapshots'] == receipt['live_query_snapshots'],
        'confirmation rollout/query cohort differs')
    result = dict(actor=actor, day=day, mode=mode, sampler_seed=sampler, cells=cells, counts=dict(counts),
        scoring_counts=scoring, map_source=MAP_SOURCE, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        bounds_saved_before_labels=True, interval_fit=False, map_updates=0, actor_updates=0,
        critic_updates=0, dual_updates=0, main_policy_changed=False)
    write(final, result)
    print('V173_CONFIRMATION_PATH_COMPLETE', actor, day, mode, flush=True)
    return result


def prepare(args):
    require(not (args.out_root/'registration.json').exists(), 'confirmation already prepared')
    for pattern in ('test_v168_*.py', 'test_v171_*.py', 'test_v173_*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    protocol, _, model, _ = parents(args)
    proof = read(args.v166_root/'preflight/summary/status.json')
    require(proof['status'] == 'v166_passive_equivalence_passed' and proof['execution_source'] == '20d914af03' and
        all(proof[k] for k in entry.EQUIVALENCE), 'unchanged passive capture proof missing')
    for actor in protocol['reservation']['actors']:
        for mode in ('old', 'handoff'):
            maps(args, actor, mode)
    registration = dict(status='v173_confirmation_prepared', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        map_source=MAP_SOURCE, protocol_source='f5d152f0bc', reservation=protocol['reservation'],
        sealed_model=model, confirmation_paths=768, interval_fit=False, map_updates=0,
        main_policy_changed=False, bootstrap_draws=5000, bootstrap_seed=20260803)
    write(args.out_root/'registration.json', registration)
    write(args.out_root/'prepare/summary/status.json', {k: v for k, v in registration.items() if k != 'sealed_model'})
    print('V173_FROZEN_CONFIRMATION_ENTRY_READY', flush=True)


def pair(args, index):
    registered(args)
    protocol, service, _, base = parents(args)
    actor, day = cases(protocol)[index]
    sampler = protocol['reservation']['sampler_seeds'][str(day)]
    for mode in ('old', 'handoff'):
        collect_path(args, protocol, service, base, actor, day, mode, sampler)


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
        dict(status='v173_shard_complete', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
            map_source=MAP_SOURCE, start=args.start, end=args.end, confirmation_paths=2*(args.end-args.start),
            interval_fit=False, map_updates=0, main_policy_changed=False))
    print('V173_CONFIRMATION_SHARD_COMPLETE', args.start, args.end, flush=True)


def crossed(cells, actors, days):
    expected = {(a, d) for a in actors for d in days}
    require(len(cells) == len(expected) and {(r['actor'], r['day']) for r in cells} == expected, 'crossed grid differs')
    estimates = {}
    for metric in ('finite_fraction', *METRICS):
        vals = [r[metric] for r in cells]
        if any(v is None for v in vals):
            estimates[metric] = dict(mean=None, ci95=None, lower95=None, upper95=None)
        else:
            frame = pd.DataFrame([dict(train_seed=r['actor'], eval_seed=r['day'], **{metric: r[metric]}) for r in cells])
            draws = prior.hierarchical_bootstrap(frame, metric, draws=5000)
            estimates[metric] = dict(mean=float(np.mean(vals)), ci95=np.percentile(draws, [2.5, 97.5]).tolist(),
                lower95=float(np.percentile(draws, 5.)), upper95=float(np.percentile(draws, 95.)))
    return dict(complete_actor_day_grid=all(r['eligible_episodes'] and r['finite_episodes'] for r in cells),
        finite_unresolved_n=sum(r['finite_unresolved_n'] for r in cells),
        missing_reference_n=sum(r['missing_reference_n'] for r in cells), cells=len(cells), estimates=estimates,
        coverage_lower95=estimates['coverage']['lower95'], below_upper95=estimates['below']['upper95'],
        above_upper95=estimates['above']['upper95'], finite_fraction_lower95=estimates['finite_fraction']['lower95'],
        score_delta_upper95=estimates['score_delta_s']['upper95'])


def calibration_moving(args, model):
    rows = []
    field = model['frozen_protocol']['conditioning']['candidate_key'].index('native_stage_kind')
    for chunk in model['chunks']:
        table = read(args.v172_root/chunk['map_file'])
        with np.load(args.v172_root/chunk['atom_file']) as z:
            q, ri, keys = z['requests'], z['atom_request'], z['candidate_key']
            finite, reference, secondary = z['finite_candidate'], z['finite_reference'], z['finite_frozen_v167']
            bounds, secondary_bounds = z['bounds'], z['frozen_v167_bounds']
            moving = np.zeros(len(q), dtype=bool)
            moving[ri] = np.array([e['key'][field] == 'moving-launch' for e in table['candidate']])[keys]
            selected = (q[:, 4]//3600 == 13) & moving
            truth = np.full(len(q), np.nan)
            truth[ri] = q[ri, 3]-z['residual_rsd'].sum(axis=1)
            saved_cells = {c['day']: c for c in table['cells'] if c['scope'] == 'hour13_moving_launch'}
            for day in model['calibration_days']:
                group = {}
                for i in np.flatnonzero(selected & (q[:, 0] == day)):
                    p = dict(finite=bool(finite[i]), reference_finite=bool(reference[i]),
                        secondary_finite=bool(secondary[i]), bounds=bounds[i, 0],
                        reference_bounds=bounds[i, 1], secondary_bounds=secondary_bounds[i])
                    add_episode(group, [int(q[i, 1])], p['finite'], values(p, dict(status='resolved', full_eta_s=truth[i])))
                cell = collapse(group)
                require(cell['eligible_queries'] == saved_cells[day]['eligible_queries'] and
                    cell['finite_queries'] == saved_cells[day]['finite_queries'] and
                    cell['finite_fraction'] == saved_cells[day]['finite_fraction'], 'calibration moving mask differs')
                rows.append(dict(actor=chunk['actor'], day=day, mode=chunk['mode'], scope='hour13_moving_launch', **cell))
    return rows


def migration(calibration, confirmation, actors, calibration_days, confirmation_days):
    rng = np.random.RandomState(20260803)
    a = rng.randint(0, len(actors), size=(5000, len(actors)))
    c = rng.randint(0, len(calibration_days), size=(5000, len(calibration_days)))
    f = rng.randint(0, len(confirmation_days), size=(5000, len(confirmation_days)))
    result = {}
    for metric in ('finite_fraction', *METRICS):
        arrays = []
        for rows, days in ((calibration, calibration_days), (confirmation, confirmation_days)):
            require(len(rows) == len(actors)*len(days) and len({(r['actor'], r['day']) for r in rows}) == len(rows),
                'migration crossed grid differs')
            frame = pd.DataFrame(rows).pivot(index='actor', columns='day', values=metric).reindex(index=actors, columns=days)
            arrays.append(frame.to_numpy(dtype=float))
        if not all(np.isfinite(v).all() for v in arrays):
            result[metric] = dict(delta=None, ci95=None)
        else:
            draws = arrays[1][a[:, :, None], f[:, None, :]].mean(axis=(1, 2))-arrays[0][a[:, :, None], c[:, None, :]].mean(axis=(1, 2))
            result[metric] = dict(delta=float(arrays[1].mean()-arrays[0].mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist())
    return result


def aggregate(args):
    registration = registered(args)
    protocol, _, model, _ = parents(args)
    require(model == registration['sealed_model'], 'sealed V172 model changed')
    r = protocol['reservation']
    names = {f'{a}_{d}_{m}.json' for a, d in cases(protocol) for m in ('old', 'handoff')}
    require({p.name for p in (args.out_root/'cells').glob('*.json')} == names, 'confirmation matrix incomplete/contaminated')
    receipts = [read(p) for p in (args.out_root/'receipts').glob('*/status.json')]
    require(sorted(i for s in receipts for i in range(s['start'], s['end'])) == list(range(len(cases(protocol)))) and
        all(s['status'] == 'v173_shard_complete' and s['map_source'] == MAP_SOURCE and
            s['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'] and not s['interval_fit'] and
            s['map_updates'] == 0 for s in receipts), 'shard gaps/source differ')
    paths = [read(args.out_root/'cells'/n) for n in sorted(names)]
    for p in paths:
        require(p['execution_source'] == registration['execution_source'] and p['map_source'] == MAP_SOURCE and
            p['sampler_seed'] == r['sampler_seeds'][str(p['day'])] and p['bounds_saved_before_labels'] and
            not p['interval_fit'] and p['map_updates'] == 0 and not p['main_policy_changed'] and
            all(p[k+'_updates'] == 0 for k in ('actor', 'critic', 'dual')), 'confirmation freeze differs')
    flattened = [dict(actor=p['actor'], day=p['day'], mode=p['mode'], scope=s, **p['cells'][s])
        for p in paths for s in plan.SCOPES]
    readouts = [dict(mode=m, scope=s, **crossed([c for c in flattened if (c['mode'], c['scope']) == (m, s)],
        r['actors'], r['confirmation_days'])) for m in ('old', 'handoff') for s in plan.SCOPES]
    decision = plan.gate(readouts)
    write(args.out_root/'details/actor_day_cells.json', flattened)
    margins = []
    for axis in ('actor', 'day'):
        rows = []
        for m in ('old', 'handoff'):
            for s in plan.SCOPES:
                selected = [c for c in flattened if (c['mode'], c['scope']) == (m, s)]
                for ident in sorted({c[axis] for c in selected}):
                    group = [c for c in selected if c[axis] == ident]
                    rows.append(dict(mode=m, scope=s, **{axis: ident}, cells=len(group),
                        **{metric: None if any(c[metric] is None for c in group) else float(np.mean([c[metric] for c in group]))
                            for metric in ('finite_fraction', *METRICS)}))
                summary = dict(axis=axis, mode=m, scope=s, groups=len({c[axis] for c in selected}), metrics={})
                for metric in ('finite_fraction', 'coverage', 'below', 'above', 'score_delta_s', 'secondary_score_delta_s'):
                    vals = [c[metric] for c in rows if (c['mode'], c['scope']) == (m, s)]
                    summary['metrics'][metric] = dict(complete=all(v is not None for v in vals),
                        min=None if any(v is None for v in vals) else min(vals),
                        max=None if any(v is None for v in vals) else max(vals))
                margins.append(summary)
        write(args.out_root/'details'/('by_'+axis+'.json'), rows)
    cal = calibration_moving(args, model)
    write(args.out_root/'details/calibration_moving.json', cal)
    migrations = []
    for m in ('old', 'handoff'):
        c = [v for v in cal if v['mode'] == m]
        f = [v for v in flattened if (v['mode'], v['scope']) == (m, 'hour13_moving_launch')]
        migrations.append(dict(mode=m, calibration_in_sample=crossed(c, r['actors'], r['calibration_days']),
            confirmation_minus_calibration=migration(c, f, r['actors'], r['calibration_days'], r['confirmation_days'])))
    require(read(args.v172_root/'model.json') == registration['sealed_model'], 'map header changed during aggregation')
    write(args.out_root/'summary/status.json', dict(status='v173_independent_confirmation_complete',
        execution_source=registration['execution_source'], map_source=MAP_SOURCE, protocol_source='f5d152f0bc',
        confirmation_days=r['confirmation_days'], actors=r['actors'], modes=['old', 'handoff'], paths=len(paths),
        shards=len(receipts), counts=dict(sum((Counter(p['counts']) for p in paths), Counter())),
        scoring_counts=dict(sum((Counter(p['scoring_counts']) for p in paths), Counter())),
        readouts=readouts, gate=decision, marginal_diagnostics=margins, moving_migration=migrations,
        bootstrap_draws=5000, bootstrap_seed=20260803, interval_fit=False, map_updates=0, v167_changed=False,
        v151_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1], new_training=0,
        main_policy_changed=False, next='Preserve independent results; no tuning or map updates on these days.'))
    print('V173_INDEPENDENT_CONFIRMATION_COMPLETE', decision['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'aggregate'))
    for name in ('out-root', 'runtime', 'v172-root', 'v171-root', 'v167-root', 'v166-root', 'v165-root',
            'v161-root', 'v151-root', 'v146-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'aggregate': aggregate}[args.command](args)
