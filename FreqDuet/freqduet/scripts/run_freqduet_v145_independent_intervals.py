#!/usr/bin/env python3
"""Collect fresh frozen-control paths and evaluate unchanged V140--V142 shadows."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_freqduet_v137_passive_ready_exposure as capture
import audit_freqduet_v141_ready_eta as points
from audit_freqduet_v143_interval_support import support_context
from freqduet_interval_validation import strata, crossed
from freqduet_rank_interval import rank_interval


ACTORS = capture.frozen.TRAIN_SEEDS
DAYS = tuple(range(2451001, 2451033))
SAMPLERS = {d: 1450000000+i for i, d in enumerate(DAYS)}
MODES = ('old', 'handoff')
ITEMS = tuple(itertools.product(ACTORS, DAYS, MODES))
SUPPORT = ('no_completed_history', 'prefix_less_than_nine', 'window_less_than_nine',
    'elapsed_survivor_depletion', 'finite')
read, write, key, close = points.read, points.write, points.key, points.close


def registration():
    return dict(stage='v145_independent_interval_validation', archived_actors=list(ACTORS),
        demand_days=list(DAYS), sampler_seeds={str(d): SAMPLERS[d] for d in DAYS},
        modes=list(MODES), paths=len(ITEMS), checkpoint_ep=199,
        window_s=3600., lower_tail=.1, upper_tail=.1, rank_source='418a3f001a',
        point_sources=['4df22f4fa6', '9adf0bc09c'], bootstrap_draws=5000,
        bootstrap_source='existing hierarchical_bootstrap RandomState(20260803)',
        coverage_target=.8, lower_confidence_percentile=5.,
        main_policy_changed=False, actor_updates=0, critic_updates=0, dual_updates=0)


def check_prior_usage(archive, scripts):
    prior_days = set()
    for row in archive['metadata']:
        prior_days.update(row['days'])
    for row in archive['bundles']:
        if row['identity'] is not None:
            prior_days.add(row['identity'][1])
    if prior_days & set(DAYS):
        raise AssertionError('registered demand day is already in server archives')
    # Prior generators use literal day ranges; check those without enumerating caches.
    for path in scripts.glob('*.py'):
        if path.name == Path(__file__).name:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in DAYS:
                raise AssertionError(f'new demand day already occurs in {path.name}:{node.lineno}')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'range':
                if node.args and all(isinstance(a, ast.Constant) and isinstance(a.value, int) for a in node.args):
                    if any(d in range(*(a.value for a in node.args)) for d in DAYS):
                        raise AssertionError(f'new demand day overlaps prior generator {path.name}:{node.lineno}')
    return dict(archive_days_checked=len(prior_days), prior_script_ranges_checked=True)


def register(args):
    target = args.out_root/'protocol.json'
    plan = registration()
    if target.exists():
        old = read(target)
        if any(old[k] != v for k, v in plan.items()):
            raise AssertionError('registered independent validation protocol changed')
        return old
    evidence = check_prior_usage(read(args.v144_root/'details/admission.json'), Path(__file__).parent)
    for seed in ACTORS:
        root = capture.pilot.joint.TRAIN_ROOT/'logs'/f'{capture.pilot.joint.CURRENT}_seed{seed}'/'checkpoints'
        for name in ('runner_ep199.pt', 'lower_ep199.pt', 'upper_ep199.pt'):
            if not (root/name).is_file():
                raise AssertionError(f'archived deployment is missing {root/name}')
    plan.update(prior_usage=evidence, registration_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])
    write(target, plan)
    print('V145_REGISTERED_FRESH_DAYS', len(DAYS), len(ITEMS), flush=True)
    return plan


def collect(root, runtime, seed, day, mode, sampler):
    name = f'{seed}_{day}_{mode}'
    receipt = root/'paths'/(name+'.json')
    if receipt.exists():
        old = read(receipt)
        if (old['seed'], old['day'], old['mode'], old['sampler_seed']) != (seed, day, mode, sampler):
            raise AssertionError('stored path differs from registered path')
        return old
    rule = capture.ReadyExposureAudit(seed, capture.frozen.frozen.prior.model.load_geometry(runtime), mode)
    credit = capture.pilot.CreditObserver()
    result, _, _, _ = capture.pilot.rollout(root, seed, day, 'stochastic', sampler,
        [0.]*4, 'v145_'+name, probability_rule=rule, trace_observer=rule, credit_observer=credit)
    if not result['base_actors_unchanged'] or result['training_updates'] or result['dual_updates']:
        raise AssertionError('fresh data collection changed the frozen deployment')
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    values = {k: result[k] for k in capture.frozen.original.frozen.prior.METRICS}
    values.update({k: rule.terminal[k] for k in ('inclusive_terminal_delay_s', 'qualification_shift_s')})
    rows = []
    for r in rule.decisions:
        k = key(r)
        row = dict(r, **rule.queries[k])
        row.update(holding_person_s=capture.holding_exposure(row, rule.departure_exposure[k]),
            holding_load=rule.departure_exposure[k]['holding_load'])
        if not row['prediction_time_s'] == row['evidence']['ready_time_s'] <= row['policy_time_s']:
            raise AssertionError('collector changed the ready/policy forecast clocks')
        rows.append(row)
    rule.save(root/'records'/(name+'.npz'))
    capture.identity.write_trace(root/'traces'/(name+'.jsonl.gz'), rows)
    capture.identity.write_trace(root/'ready_labels'/(name+'.jsonl.gz'),
        [dict(trip_id=k[0], station_id=k[1], direction=k[2], **v) for k, v in rule.ready.items()])
    write(root/'terminal'/(name+'.json'), dict(plans=rule.plans, dispatches=rule.dispatches,
        returns=list(rule.returns.values()), reference_clocks=rule.reference_clocks))
    record = dict(seed=seed, day=day, mode=mode, sampler_seed=sampler, values=values,
        contract=rule.contract, decisions=len(rows), ready_labels=len(rule.ready),
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    write(receipt, record)
    print('V145_FROZEN_PATH_COLLECTED', seed, day, mode, flush=True)
    return record


def stage_request(path, forecast, row, geometry):
    if 'current_stop_key' not in forecast:
        return None
    current = tuple(forecast['current_stop_key'])
    now = row['evidence']['ready_time_s']
    start = path['arrivals'][current]['time_s']
    route = geometry[current[2]]
    origin, target = route['index'][current[1]], route['index'][row['station_id']]
    if start >= now or origin > target:
        return None
    ready = path['ready'].get(current)
    if ready is None or ready['ready_time_s'] >= now:
        return 'service', current, start
    command = path['trace'].get(current)
    if origin >= target or (command is not None and command['policy_time_s'] < now):
        return None
    return 'ready', current, ready['ready_time_s']


def duration_label(path, current, row, kind, now):
    ready = path['ready'].get(current)
    if ready is None:
        return 'missing_same_vehicle_ready', None
    if ready['bus_id'] != row['evidence']['predicted_follower_bus_id']:
        return 'same_vehicle_identity_mismatch', None
    end = ready['ready_time_s'] if kind == 'service' else path['departures'].get(current, {}).get('time_s')
    if end is None:
        return 'missing_departure_tail', None
    if end < now:
        return 'stage_already_finished', None
    return 'resolved', end-now


def evaluate_path(root, runtime, seed, day, mode):
    path = points.prior.load_path(root, root, seed, day, mode)
    geometry = points.model.load_geometry(runtime)
    history, services, indexed = points.frozen.prepare(path, geometry)
    durations = points.ServiceResiduals([(k[2], k[1], r['time_s'],
        r['time_s']-path['ready'][k]['ready_time_s']) for k, r in path['departures'].items() if k in path['ready']])
    histories = dict(service=services, ready=durations)
    intervals, forecasts, statuses = [], [], Counter()
    for k, row in path['trace'].items():
        e, now = row['evidence'], row['evidence']['ready_time_s']
        v140 = points.frozen.predict(path, row, indexed, geometry, history, services)
        v141 = dict(applied=False, status='unchanged_v140')
        if v140['status'] == 'ready_without_prior_command':
            current = tuple(v140['current_stop_key'])
            route = geometry[current[2]]
            v141 = points.predict_ready(history, durations, route, current[2], route['index'][current[1]],
                route['index'][row['station_id']], now, path['ready'][current]['ready_time_s'])
        old = v140['gap_s'] if v140['applied'] else row['online']['gap_s']
        new = v141['gap_s'] if v141['applied'] else old
        forecasts.append(dict(query=list(k), time_s=now, native_s=row['online']['gap_s'],
            v140_s=old, v141_s=new, v140_status=v140['status'], v141_status=v141['status']))
        request = stage_request(path, v140, row, geometry)
        statuses[v140['status']] += 1
        if request is None:
            continue
        kind, current, start = request
        support, samples = support_context(histories[kind], current[2], current[1], now, now-start)
        bounds = rank_interval(np.asarray([d-(now-start) for _, d in samples]))
        if samples and max(t for t, _ in samples) >= now:
            raise AssertionError('interval used a simultaneous or unfinished label')
        # Outcomes are attached after eligibility, frozen points and bounds are fixed.
        label_status, truth = duration_label(path, current, row, kind, now)
        scored = bounds['finite'] and label_status == 'resolved'
        intervals.append(dict(query=list(k), episode=list(current), kind=kind, time_s=now, hour=int(now//3600),
            rank=bounds, support_reason=support['reason'], pool_n=support['pool_n'],
            survivor_n=support['survivor_n'], v140_s=old, v141_s=new,
            label_status=label_status, truth_s=truth,
            covered=None if not scored else bounds['lower_s'] <= truth <= bounds['upper_s'],
            below=None if not scored else truth < bounds['lower_s'],
            above=None if not scored else truth > bounds['upper_s']))
    name = f'{seed}_{day}_{mode}'
    capture.identity.write_trace(root/'intervals'/(name+'.jsonl.gz'), intervals)
    capture.identity.write_trace(root/'shadows'/(name+'.jsonl.gz'), forecasts)
    result = dict(seed=seed, day=day, mode=mode, all_queries=len(forecasts), eligible_requests=len(intervals),
        statuses=dict(statuses), strata=strata(intervals), point_parameters_changed=False,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    write(root/'validation_cells'/(name+'.json'), result)
    return intervals, forecasts, result


def smoke(args):
    register(args)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v14[01235]_*.py', '-v'], check=True)
    seed, day, root = 12071, 2351015, args.out_root/'smoke_runs'
    for mode in MODES:
        current = collect(root, args.runtime, seed, day, mode, 187000000+seed+100*day)
        old = read(args.v137_root/'paths'/f'{seed}_{day}_{mode}.json')
        capture.pilot.joint.assert_same_state(old['values'], current['values'], 'V145_exact_consumed_endpoints')
        capture.passive.assert_arrays_and_actions(args.v137_root/'records'/f'{seed}_{day}_{mode}.npz',
            root/'records'/f'{seed}_{day}_{mode}.npz', capture.identity.read_trace(root/'traces'/f'{seed}_{day}_{mode}.jsonl.gz'))
        capture.pilot.joint.assert_same_state(read(args.v135_root/'terminal'/f'{seed}_{day}_{mode}.json'),
            read(root/'terminal'/f'{seed}_{day}_{mode}.json'), 'V145_exact_terminal_clocks')
        intervals, forecasts, _ = evaluate_path(root, args.runtime, seed, day, mode)
        lookup = {tuple(r['query']): r for r in forecasts}
        for version in ('v140', 'v141'):
            saved = read(getattr(args, version+'_root')/'details'/f'{seed}_{day}_{mode}.json')['rows']
            for r in saved:
                close(lookup[key(r)][version+'_s'], r['shadow_s'], 'frozen '+version+' point forecast')
        expected = read(args.v142_root/'details'/f'{seed}_{day}_{mode}.json')['intervals']
        for r in expected:
            matches = [v for v in intervals if v['kind'] == r['kind'] and
                v['episode'] == r['episode'][-3:] and v['time_s'] == r['time_s']]
            if not matches or any(v['rank'] != r['rank'] for v in matches):
                raise AssertionError('independent interval code changed a saved V142 bound')
    write(args.out_root/'preflight/done.json', dict(status='consumed_actions_clocks_points_and_bounds_exact',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], prospective_days_read=False))
    print('V145_PREFLIGHT_COMPLETE', flush=True)


def run_one(root, runtime, index):
    seed, day, mode = ITEMS[index]
    final = root/'validation_cells'/f'{seed}_{day}_{mode}.json'
    if not final.exists():
        collect(root, runtime, seed, day, mode, SAMPLERS[day])
        evaluate_path(root, runtime, seed, day, mode)
    return index


def require_preflight(args):
    plan = read(args.out_root/'protocol.json')
    if any(plan[k] != v for k, v in registration().items()):
        raise AssertionError('registered evaluation protocol changed')
    if read(args.out_root/'preflight/done.json')['execution_source'] != os.environ['FREQDUET_DIAGNOSTIC_COMMIT']:
        raise AssertionError('collection source differs from tested source')


def run_batch(args):
    require_preflight(args)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        pending = [pool.submit(run_one, args.out_root, args.runtime, i) for i in range(args.start, args.end)]
        for n, task in enumerate(as_completed(pending), 1):
            task.result()
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=n, total=args.end-args.start,
                unit='frozen_path', elapsed_seconds=time.monotonic()-started)), flush=True)
    write(args.out_root/'shards'/f'{args.start}-{args.end}.json', dict(status='complete', start=args.start, end=args.end))


def aggregate(args):
    require_preflight(args)
    started = time.monotonic()
    paths = [args.out_root/'validation_cells'/f'{s}_{d}_{m}.json' for s, d, m in ITEMS]
    while not all(p.exists() for p in paths):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=sum(p.exists() for p in paths),
            total=len(paths), unit='validation_cell', elapsed_seconds=time.monotonic()-started)), flush=True)
        time.sleep(30)
    cells = [read(p) for p in paths]
    if [(c['seed'], c['day'], c['mode']) for c in cells] != list(ITEMS):
        raise AssertionError('validation cells differ from frozen grid')
    summaries = {}
    for mode in MODES:
        chosen = [c for c in cells if c['mode'] == mode]
        summaries[mode] = {}
        for kind in ('service', 'ready'):
            entry = {}
            for scope in ('all', 'hour13', 'rest'):
                entry[scope] = crossed([dict(seed=c['seed'], day=c['day'], stats=c['strata'][kind][scope])
                    for c in chosen], ACTORS, DAYS)
            entry['by_support'] = {s: crossed([dict(seed=c['seed'], day=c['day'],
                stats=c['strata'][kind]['by_support'][s]) for c in chosen], ACTORS, DAYS) for s in SUPPORT}
            summaries[mode][kind] = entry
    passed = all(summaries[m][k]['all']['coverage_confirmed'] for m in MODES for k in ('service', 'ready'))
    write(args.out_root/'summary/status.json', dict(status='v145_independent_interval_validation_complete',
        registration=read(args.out_root/'protocol.json'), summaries=summaries,
        primary_finite_coverage_confirmed=passed, validation_run=True, new_rollouts=len(ITEMS),
        new_demand_days=len(DAYS), actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, controller_admitted=False,
        limitations='Fixed-actor simulator demand-day validation of current-stage intervals, not full ETA '
            'or control efficacy. Missing support/outcomes are not zero-filled. Partial actor/day support '
            'has no crossed confidence interval; unresolved finite outcomes prevent coverage admission. '
            'Marginal coverage does not validate hour13 or low-support cases. No window/tail or point tuning.'))
    print('V145_INDEPENDENT_VALIDATION_COMPLETE', passed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('register', 'smoke', 'run', 'aggregate'))
    for name in ('out-root', 'runtime', 'v135-root', 'v137-root', 'v140-root', 'v141-root', 'v142-root', 'v144-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(ITEMS))
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    {'register': register, 'smoke': smoke, 'run': run_batch, 'aggregate': aggregate}[args.command](args)
