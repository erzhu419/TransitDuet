#!/usr/bin/env python3
"""Apply sealed V184 once to V183's 32 reserved confirmation days; never fit."""

import argparse
from collections import Counter, deque
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

import seal_freqduet_v184_shared_lower as sealed
import run_freqduet_v177_confirmation as saved


read, write, require, stream = saved.read, saved.write, saved.require, saved.stream
capture, entry, fit = saved.capture, saved.entry, saved.fit
MAP_SOURCE = '2447e29d07'
DAYS, SAMPLERS = sealed.reservation.DAYS, sealed.reservation.SAMPLERS
SCOPES = sealed.temporal.SCOPES
CORE = ('coverage', 'below', 'above', 'width_s', 'score_s')
METRICS = (*saved.saved.prior.METRICS, 'lower_delta_s', 'zero_cap', 'upper_cap', 'zero_width')


def parents(args):
    base, service = entry.parents(args)
    _, protocol, support, reserved = sealed.parents(args)
    model = read(args.v184_root/'model.json')
    summary = read(args.v184_root/'summary/status.json')
    require(model['status'] == 'sealed' and model['source'] == summary['source'] == MAP_SOURCE and
        model['frozen_protocol'] == protocol and model['confirmation_reservation'] == reserved and
        model['base_model_root'] == str(args.v172_root) and model['support_admission']['decision'] == 'pass' and
        summary['status'] == 'v184_shared_lower_sealed_calibration_only' and
        summary['fitted_parameter_values'] == 48 and all(summary['invariants'].values()) and
        not model['confirmation_started'] and not model['confirmation_outcomes_read'] and
        not model['main_policy_changed'], 'requires unchanged admitted V184/V182/V183')
    require(protocol['frozen'] == base['frozen'] and protocol['inference']['scopes'] == list(SCOPES) and
        not set(DAYS) & set(service['calibration_days']+model['calibration_days']), 'point/cohort/inference changed')
    return base, service, protocol, model, reserved, summary['parameters']


def cases(reserved):
    return tuple(product(reserved['actors'], reserved['confirmation_days']))


class CompletedPrefix:
    """Replay completion events; calculate component residuals only after availability."""
    def __init__(self, queries, ready, residual):
        self.events, seen = [], set()
        for q in sorted(queries, key=lambda q: (q['query_time_s'], q['clocks']['query_ready_order'])):
            if not q['denominator_eligible'] or q['prediction_coarse'] is None:
                continue
            trip, bus, stop, direction = q['episode']
            target = ready.get((trip, stop, bool(direction)))
            if target is None or target['bus_id'] != bus or target['ready_time_s'] < q['query_time_s']:
                continue
            episode = tuple(q['episode'])
            if episode not in seen:
                seen.add(episode)
                self.events.append((target['ready_time_s'], (q['query_time_s'], q['clocks']['query_ready_order']), q))
        self.events.sort(key=lambda e: (e[0], e[1]))
        self.index, self.pool, self.total, self.residual = 0, deque(), np.zeros(3), residual

    def query(self, now):
        while self.index < len(self.events) and self.events[self.index][0] < now:
            available, _, q = self.events[self.index]
            errors = np.asarray(self.residual(q, now), dtype=float)
            require(errors.shape == (3,) and np.isfinite(errors).all(), 'incomplete completed-prefix vector')
            self.pool.append((available, errors))
            self.total += errors
            self.index += 1
        while self.pool and self.pool[0][0] < now-3600.:
            self.total -= self.pool.popleft()[1]
        require(not self.pool or now-3600. <= self.pool[0][0] <= self.pool[-1][0] < now, 'future/out-of-window prefix')
        return dict(n=len(self.pool), latest=None if not self.pool else self.pool[-1][0],
            mean=None if len(self.pool) < 9 else self.total.copy()/len(self.pool))


def completed_residual(q, now, path, geometry):
    row = path['trace'][tuple(q['query_key'])]
    label = capture.extraction.label_record(path, row)
    require(label['status'] == 'resolved' and label['available_time_s'] < now, 'prefix target not strictly completed')
    joint = q['inseparable_current_ready_key']
    actual = capture.accounting.coarse(capture.accounting.parts.actual_parts(path, row,
        geometry[bool(row['direction'])], None if joint is None else tuple(joint)))
    errors = np.array([q['prediction_coarse'][c]-actual[c] for c in ('running', 'service', 'residence')])
    capture.close(float(errors.sum()), q['point_s']-label['full_eta_s'], 'completed vector identity differs')
    return errors


@lru_cache(maxsize=4)
def tables(root, base_root, actor, mode):
    correction = read(root/'maps'/f'{actor}_{mode}.json')
    baseline = read(base_root/'maps'/f'{actor}_{mode}.json')
    require((correction['source'], correction['actor'], correction['mode']) == (MAP_SOURCE, actor, mode) and
        (baseline['execution_source'], baseline['actor'], baseline['mode']) == ('a5b45df520', actor, mode) and
        all(e['day_count'] >= 19 for e in baseline['candidate'] if e['status'] == 'finite'), 'map/source/support changed')
    return correction['parameters'], {tuple(e['key']): e for e in baseline['candidate']}


def prediction(q, parameters, baseline, history):
    period = sealed.temporal.STRATA[int(sealed.temporal.strata(np.array([q['query_time_s']]))[0])]
    key = (*q['candidate_key'], period) if q['native_supported'] else None
    old = baseline.get(key)
    bounds = fit.interval(q['point_s'], old, q['native_supported'])
    target = q['condition_values']['query_phase'] == 'clearance' or (
        q['observable']['hour'] == 13 and q['native_stage']['kind'] == 'moving-launch')
    sign = -1 if history['n'] is None else sealed.plan.prior.service_sign(history)
    reason = ('ineligible' if not q['denominator_eligible'] else 'native/'+q['native_stage']['reason']
        if not q['native_supported'] else 'unseen_key' if old is None else old['status'])
    updated, delta = bounds, None
    if bounds is not None and target:
        chosen = None if sign < 0 else next((e for e in parameters if e['sign'] == sealed.plan.SIGNS[sign]), None)
        if sign < 0:
            updated, reason = None, 'prefix_abstention'
        elif chosen is None or chosen['status'] != 'finite' or chosen['day_count'] < 19:
            updated, reason = None, 'parameter_abstention'
        else:
            delta = chosen['delta_s']
            updated = (min(bounds[1], max(0., bounds[0]-delta)), bounds[1])
    mean = history['mean']
    return dict(query_key=q['query_key'], query_time_s=q['query_time_s'], map_source=MAP_SOURCE,
        candidate_key=None if key is None else list(key), finite=updated is not None,
        reference_finite=bounds is not None, bounds=None if updated is None else list(updated),
        reference_bounds=None if bounds is None else list(bounds), support_reason=reason, target=target,
        prefix_n=history['n'], prefix_latest_s=history['latest'],
        prefix_service_mean_s=None if mean is None else float(mean[1]),
        sign=None if sign < 0 else sealed.plan.SIGNS[sign], delta_s=delta)


def predictions(queries, path, geometry, parameter_table, baseline):
    prefix = CompletedPrefix(queries, path['ready'], lambda q, now: completed_residual(q, now, path, geometry))
    for q in sorted(queries, key=lambda q: (q['query_time_s'], q['clocks']['query_ready_order'])):
        history = prefix.query(q['query_time_s']) if q['denominator_eligible'] else dict(n=None, latest=None, mean=None)
        yield q, prediction(q, parameter_table, baseline, history)


def values(p, label):
    primary = saved.saved.prior.values(p, label)
    if primary is None:
        return None
    lo, hi = p['bounds']
    old = p['reference_bounds'][0]
    raw = old if p['delta_s'] is None else old-p['delta_s']
    return np.r_[primary, lo-old, float(raw < 0.), float(raw > hi), float(lo == hi)]


def reduce_rows(rows):
    groups, own, flags, counts = ({s: {} for s in SCOPES}, {s: {} for s in SCOPES},
        {s: Counter() for s in SCOPES}, Counter())
    for q, p, label in rows:
        counts.update({'requests': 1, 'label/'+label['status']: 1, 'support/'+p['support_reason']: 1})
        if not q['denominator_eligible']:
            continue
        counts['eligible_requests'] += 1
        vector = values(p, label)
        bits = fit.scope_bits(q)
        scopes = [s for i, s in enumerate(fit.SCOPES) if bits & (1 << i)]
        if bits & 16 and q['native_stage']['kind'] == 'moving-launch':
            scopes.append('hour13_moving_launch')
        for scope in scopes:
            row = groups[scope].setdefault(tuple(q['episode']), np.zeros(3+len(METRICS)))
            row[:3] += (1, int(p['finite']), int(vector is not None))
            if vector is not None:
                row[3:] += vector
            ref = own[scope].setdefault(tuple(q['episode']), np.zeros(3+len(CORE)))
            matched = p['reference_finite'] and label['status'] == 'resolved'
            ref[:3] += (1, int(p['reference_finite']), int(matched))
            if matched:
                lo, hi = p['reference_bounds']
                y = label['full_eta_s']
                ref[3:] += (lo <= y <= hi, y < lo, y > hi, hi-lo,
                    hi-lo+10.*max(lo-y, 0.)+10.*max(y-hi, 0.))
            flags[scope].update(unresolved=int(p['finite'] and label['status'] != 'resolved'),
                missing=int(p['finite'] and not p['reference_finite']),
                own_unresolved=int(p['reference_finite'] and label['status'] != 'resolved'))
    def collapse(rows, metrics, unresolved, missing=0):
        result = dict(eligible_episodes=len(rows), eligible_queries=0, finite_queries=0, finite_episodes=0,
            finite_fraction=None, finite_unresolved_n=unresolved, missing_reference_n=missing,
            **{m: None for m in metrics})
        if rows:
            a = np.array(list(rows.values()))
            matched = a[:, 2] > 0
            result.update(eligible_queries=int(a[:, 0].sum()), finite_queries=int(a[:, 1].sum()),
                finite_episodes=int((a[:, 1] > 0).sum()), finite_fraction=float(np.mean(a[:, 1]/a[:, 0])))
            if matched.any() and not (unresolved or missing):
                result.update(zip(metrics, (a[matched, 3:]/a[matched, 2, None]).mean(axis=0).tolist()))
        return result
    cells = {}
    for scope in SCOPES:
        cells[scope] = collapse(groups[scope], METRICS, flags[scope]['unresolved'], flags[scope]['missing'])
        original = collapse(own[scope], CORE, flags[scope]['own_unresolved'])
        cells[scope].update({'v172_own_'+k: original[k] for k in ('finite_fraction', *CORE)})
        cells[scope]['v172_own_unresolved_n'] = original['finite_unresolved_n']
    return cells, dict(counts)


def registered(args):
    result = read(args.out_root/'registration.json')
    require(result['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'] and result['map_source'] == MAP_SOURCE,
        'requires same-source V185 preparation')
    return result


def check_parameters(parameters, reserved, actor, mode):
    expected = [{k: v for k, v in p.items() if k not in ('actor', 'mode')}
        for p in reserved['frozen_parameters'] if (p['actor'], p['mode']) == (actor, mode)]
    require(parameters == expected, 'sealed parameter values changed')


def collect_path(args, registration, service, base, actor, day, mode):
    root, name = args.out_root/'confirmation', f'{actor}_{day}_{mode}'
    final = args.out_root/'cells'/(name+'.json')
    if final.exists():
        result = read(final)
        require((result['actor'], result['day'], result['mode'], result['sampler_seed'], result['execution_source']) ==
            (actor, day, mode, SAMPLERS[day], registration['execution_source']), 'stored path assignment differs')
        return result
    parameters, baseline = tables(args.v184_root, args.v172_root, actor, mode)
    check_parameters(parameters, registration, actor, mode)
    receipt = capture.collect(root, args.runtime, actor, day, mode, SAMPLERS[day])
    require(receipt['execution_source'] == registration['execution_source'], 'rollout source differs')
    path, geometry, captured = capture.save_queries(root, args.runtime, actor, day, mode, service, base)
    queries = list(stream(root/'queries'/(name+'.jsonl.gz')))
    predicted = {tuple(q['query_key']): p for q, p in predictions(queries, path, geometry, parameters, baseline)}
    destination = root/'interval_predictions'/(name+'.jsonl.gz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, 'wt', encoding='ascii', compresslevel=1) as output:
        for q in queries:
            output.write(json.dumps(predicted[tuple(q['query_key'])], allow_nan=False)+'\n')
    captured.update(capture.attach_labels(root, name, path, geometry))
    ready = {capture.key(r): r for r in stream(root/'ready_labels'/(name+'.jsonl.gz'))}
    ranks = {k: i for i, k in enumerate(ready)}
    raw = {tuple(r['query_key']): r for r in stream(root/'query_snapshots'/(name+'.jsonl.gz'))}
    require(set(raw) == set(ready) and len(raw) == receipt['live_query_snapshots'], 'snapshot identities incomplete')
    for k, r in ready.items():
        saved.saved.prior.metadata.live_record(raw[k], r, ranks[k])
    replay = {tuple(q['query_key']): p for q, p in predictions(queries, path, geometry, parameters, baseline)}
    counts = Counter()
    def rows():
        for q, p, r in zip_longest(stream(root/'queries'/(name+'.jsonl.gz')),
                stream(destination), stream(root/'residuals'/(name+'.jsonl.gz'))):
            require(q is not None and p is not None and r is not None, 'artifact lengths differ')
            k = tuple(q['query_key'])
            require(p == replay.pop(k), 'saved query/prefix/frozen bounds differ')
            entry.audit.query_record(q, r, path['trace'][k], raw[k], ready, ranks, base,
                actor, day, mode, path['cutoff'], counts)
            yield q, p, r['label']
    cells, scoring = reduce_rows(rows())
    require(not replay and counts['queries'] == receipt['decisions'] == len(queries), 'query cohort incomplete')
    for k, v in captured.items():
        require(counts[k] == v, 'capture/readback counter differs: '+k)
    counts['live_ready_snapshots'] = len(raw)
    require(read(args.v184_root/'maps'/f'{actor}_{mode}.json')['parameters'] == parameters, 'frozen map changed')
    result = dict(actor=actor, day=day, mode=mode, sampler_seed=SAMPLERS[day], cells=cells,
        counts=dict(counts), scoring_counts=scoring, execution_source=registration['execution_source'], map_source=MAP_SOURCE,
        bounds_saved_before_labels=True, strict_completed_prefix_checked=True, interval_fit=False, map_updates=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    write(final, result)
    print('V185_CONFIRMATION_PATH_COMPLETE', actor, day, mode, flush=True)
    return result


def preflight(args):
    proof = read(args.v166_root/'preflight/summary/status.json')
    require(proof['status'] == 'v166_passive_equivalence_passed' and all(proof[k] for k in entry.EQUIVALENCE),
        'passive collector equivalence missing')
    checked = 0
    for mode in ('old', 'handoff'):
        root, actor, day = args.v166_root/'preflight_paths', 12011, 2451001
        name = f'{actor}_{day}_{mode}'
        geometry = capture.archived.native.model.load_geometry(args.runtime)
        path, _ = capture.archived.prepare(root, actor, day, mode, geometry)
        queries = list(stream(root/'queries'/(name+'.jsonl.gz')))
        atoms = {}
        for q, r in zip_longest(queries, stream(root/'residuals'/(name+'.jsonl.gz'))):
            require(q is not None and r is not None and q['query_key'] == r['query_key'], 'preflight join differs')
            sealed.previous.add_first(atoms, q, r)
        reference = sealed.previous.Prefix(atoms)
        current = CompletedPrefix(queries, path['ready'], lambda q, now: completed_residual(q, now, path, geometry))
        for q in sorted(queries, key=lambda q: (q['query_time_s'], q['clocks']['query_ready_order'])):
            if not q['denominator_eligible']:
                continue
            a, b = current.query(q['query_time_s']), reference.query(q['query_time_s'], None)['global']
            require(a['n'] == b['n'] and a['latest'] == b['latest'] and
                ((a['mean'] is None and b['mean'] is None) or
                    (a['mean'] is not None and b['mean'] is not None and np.array_equal(a['mean'], b['mean']))),
                'causal completion replay differs from frozen V179 first-query prefix')
            checked += 1
    return checked


def prepare(args):
    require(not (args.out_root/'registration.json').exists(), 'confirmation already prepared')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v185_*.py', '-v'], check=True)
    base, _, protocol, model, reserved, parameters = parents(args)
    proof = read(args.runtime_proof_root/'benchmark_summary.json')
    training = read(args.runtime_proof_root/'training_equivalence/summary.json')
    require(proof['all_exact'] and training['matches_original_smoke'], 'optimized runtime equivalence missing')
    link = args.out_root/'runtime_optimization_20261014'
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(args.runtime_proof_root.resolve(), target_is_directory=True)
    _, runner, _, _ = capture.collector.capture.pilot.joint.optimized_runtime(args.out_root/'confirmation')
    require(Path(sys.modules[runner.__module__].__file__).resolve().parent == args.runtime.resolve(), 'runtime import differs')
    for actor, mode in product(reserved['actors'], reserved['modes']):
        require(tables(args.v184_root, args.v172_root, actor, mode)[0] ==
            [{k: v for k, v in p.items() if k not in ('actor', 'mode')} for p in parameters
                if (p['actor'], p['mode']) == (actor, mode)], 'seal parameter readback differs')
    checked = preflight(args)
    # Existing reservations are legitimate; only actual artifacts/receipts consume this cohort.
    ledger = sealed.reservation.ledger
    roots = set(ledger.archives.result_roots(args.parent)) | {args.v184_root.parent.parent}
    names = receipts = 0
    for root in roots:
        for directory, children, files in os.walk(root):
            path = Path(directory)
            children[:] = [d for d in children if d not in ledger.SKIP and path/d != args.out_root.parent and
                len((path/d).relative_to(root).parts) <= 5]
            if path.name not in ledger.RAW | ledger.RECEIPTS | {'interval_predictions'}:
                continue
            children[:] = []
            for name in files:
                names += 1
                require(ledger.filename_day(name) not in DAYS, 'reserved demand day already consumed: '+str(path/name))
                if path.name in ledger.RECEIPTS and name.endswith('.json'):
                    receipts += 1
                    require(not ledger.metadata.sampler_ids(read(path/name)) & set(SAMPLERS.values()),
                        'reserved sampler already consumed: '+str(path/name))
    registration = dict(status='v185_confirmation_prepared', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        map_source=MAP_SOURCE, frozen_protocol=protocol, sealed_model=model, reservation=reserved,
        frozen_parameters=parameters, confirmation_paths=768, interval_fit=False, map_updates=0,
        main_policy_changed=False, preflight_queries=checked, bootstrap_draws=5000, bootstrap_seed=20260803,
        freshness=dict(roots=len(roots), artifact_names=names, sampler_receipts=receipts, collisions=0))
    write(args.out_root/'registration.json', registration)
    write(args.out_root/'prepare/summary/status.json', {k: v for k, v in registration.items()
        if k not in ('frozen_protocol', 'sealed_model', 'reservation', 'frozen_parameters')})
    print('V185_ENTRY_READY_NO_CONFIRMATION_COLLECTED', checked, flush=True)


def pair(args, registration, service, base, index):
    actor, day = cases(registration['reservation'])[index]
    for mode in ('old', 'handoff'):
        collect_path(args, registration, service, base, actor, day, mode)


def run(args):
    registration = registered(args)
    base, service, _, model, _, _ = parents(args)
    require(model == registration['sealed_model'] and 0 <= args.start < args.end <= 384 and 1 <= args.workers <= 8,
        'invalid shard/frozen model')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        for i, job in enumerate(as_completed([pool.submit(pair, args, registration, service, base, index)
                for index in range(args.start, args.end)]), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=i, total=args.end-args.start, unit='confirmation_actor_day')), flush=True)
    require(parents(args)[3] == model, 'frozen model changed')
    write(args.out_root/'receipts'/f'{args.start}-{args.end}'/'status.json', dict(status='v185_shard_complete',
        execution_source=registration['execution_source'], map_source=MAP_SOURCE, start=args.start, end=args.end,
        confirmation_paths=2*(args.end-args.start), interval_fit=False, map_updates=0, main_policy_changed=False))


def crossed(cells, actors, days):
    actors, days = sorted(actors), sorted(days)
    index = {(r['actor'], r['day']): r for r in cells}
    require(len(index) == len(cells) == len(actors)*len(days) and set(index) == set(product(actors, days)), 'crossed grid incomplete')
    rng = np.random.RandomState(20260803)
    a, d = rng.randint(0, len(actors), (5000, len(actors))), rng.randint(0, len(days), (5000, len(days)))
    estimates = {}
    for metric in ('finite_fraction', *METRICS, 'v172_own_finite_fraction', *('v172_own_'+m for m in CORE)):
        raw = [[index[actor, day][metric] for day in days] for actor in actors]
        if any(v is None for row in raw for v in row):
            estimates[metric] = dict(mean=None, ci95=None, lower95=None, upper95=None)
        else:
            matrix = np.array(raw, dtype=float)
            require(np.isfinite(matrix).all(), 'nonfinite crossed metric')
            draws = matrix[a[:, :, None], d[:, None, :]].mean(axis=(1, 2))
            estimates[metric] = dict(mean=float(matrix.mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist(),
                lower95=float(np.percentile(draws, 5.)), upper95=float(np.percentile(draws, 95.)))
    return dict(complete_actor_day_grid=all(r['eligible_episodes'] and r['finite_episodes'] for r in cells),
        finite_unresolved_n=sum(r['finite_unresolved_n'] for r in cells), missing_reference_n=sum(r['missing_reference_n'] for r in cells),
        cells=len(cells), estimates=estimates, coverage_lower95=estimates['coverage']['lower95'],
        below_upper95=estimates['below']['upper95'], above_upper95=estimates['above']['upper95'],
        finite_fraction_lower95=estimates['finite_fraction']['lower95'], score_delta_upper95=estimates['score_delta_s']['upper95'])


def aggregate(args):
    registration = registered(args)
    _, _, protocol, model, reserved, parameters = parents(args)
    require(model == registration['sealed_model'] and parameters == registration['frozen_parameters'], 'sealed map changed')
    names = {f'{a}_{d}_{m}.json' for a, d in cases(reserved) for m in ('old', 'handoff')}
    require({p.name for p in (args.out_root/'cells').glob('*.json')} == names, 'confirmation matrix incomplete/contaminated')
    receipts = [read(p) for p in (args.out_root/'receipts').glob('*/status.json')]
    require(sorted(i for r in receipts for i in range(r['start'], r['end'])) == list(range(384)) and
        all(r['status'] == 'v185_shard_complete' and r['execution_source'] == registration['execution_source'] and
            r['map_source'] == MAP_SOURCE and not r['interval_fit'] and not r['map_updates'] for r in receipts), 'shard gap/source differs')
    paths = [read(args.out_root/'cells'/n) for n in sorted(names)]
    for p in paths:
        require(p['execution_source'] == registration['execution_source'] and p['map_source'] == MAP_SOURCE and
            p['sampler_seed'] == SAMPLERS[p['day']] and p['bounds_saved_before_labels'] and p['strict_completed_prefix_checked'] and
            not p['interval_fit'] and not p['map_updates'] and not p['main_policy_changed'] and
            all(p[k+'_updates'] == 0 for k in ('actor', 'critic', 'dual')), 'confirmation freeze differs')
    cells = [dict(actor=p['actor'], day=p['day'], mode=p['mode'], scope=s, **p['cells'][s]) for p in paths for s in SCOPES]
    readouts = [dict(mode=m, scope=s, **crossed([c for c in cells if (c['mode'], c['scope']) == (m, s)],
        reserved['actors'], list(DAYS))) for m, s in product(('old', 'handoff'), SCOPES)]
    decision = sealed.temporal.gate(readouts)
    write(args.out_root/'details/actor_day_cells.json', cells)
    for axis in ('actor', 'day'):
        margins = []
        for mode, scope in product(('old', 'handoff'), SCOPES):
            rows = [r for r in cells if (r['mode'], r['scope']) == (mode, scope)]
            for identity in sorted({r[axis] for r in rows}):
                group = [r for r in rows if r[axis] == identity]
                margins.append(dict(mode=mode, scope=scope, **{axis: identity}, cells=len(group),
                    **{m: None if any(r[m] is None for r in group) else float(np.mean([r[m] for r in group]))
                        for m in ('finite_fraction', *METRICS)}))
        write(args.out_root/'details'/('by_'+axis+'.json'), margins)
    require(parents(args)[3] == model, 'model changed during aggregation')
    write(args.out_root/'summary/status.json', dict(status='v185_independent_confirmation_complete',
        execution_source=registration['execution_source'], map_source=MAP_SOURCE, protocol_source='a57afaa337',
        reservation_source='1ce0cfc596', actors=reserved['actors'], confirmation_days=list(DAYS),
        modes=['old', 'handoff'], paths=len(paths), shards=len(receipts), readouts=readouts, gate=decision,
        counts=dict(sum((Counter(p['counts']) for p in paths), Counter())),
        scoring_counts=dict(sum((Counter(p['scoring_counts']) for p in paths), Counter())),
        bootstrap_draws=5000, bootstrap_seed=20260803, interval_fit=False, map_updates=0,
        v184_changed=False, v172_changed=False, v151_changed=False, v146_changed=False,
        window_s=3600., tails=[.1, .1], confirmation_started=True, new_rollouts=len(paths),
        new_training=0, main_policy_changed=False,
        next='Preserve all confirmation outcomes; no fitting, tuning, cohort replacement or controller switch.'))
    print('V185_INDEPENDENT_CONFIRMATION_COMPLETE', decision['decision'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'aggregate'))
    for name in ('out-root', 'runtime', 'runtime-proof-root', 'v184-root', 'v183-root', 'v182-root',
            'v172-root', 'v166-root', 'v166-audit-root', 'v165-root', 'v161-root', 'v151-root', 'v146-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--parent', type=Path, action='append', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'aggregate': aggregate}[args.command](args)
