#!/usr/bin/env python3
"""V161 calibration collection only; record queries before attaching outcomes."""

import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import itertools
import json
from math import isfinite
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v152_component_truth as accounting
import complete_freqduet_v160_metadata as metadata
import extract_freqduet_v159_joint_residuals as extraction
import preregister_freqduet_v161_joint_intervals as protocol
import run_freqduet_v145_independent_intervals as collector
import run_freqduet_v154_running_prior as archived


read, write, require = metadata.read, metadata.write, metadata.require
close, key, v151 = collector.close, collector.key, accounting.candidate
MODES = protocol.MODES


def horizon_band(point):
    if point is None or not isfinite(point) or point < 0.:
        return None
    return '[0,900]' if point <= 900. else '(900,1800]' if point <= 1800. else '(1800,infinity)'


def snapshot(bus, buses, ready, arrivals, departures, launches, now):
    k = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
    e = ready[k]
    follower = next((b for b in buses if int(b.bus_id) == e['predicted_follower_bus_id']), None)
    gap = bus.pre_action_follower_departure_gap
    gap = None if gap is None else float(gap)
    valid = str(e['follower_source']).lower().startswith('same_time_avl_') and gap is not None and isfinite(gap) and gap >= 0.
    return dict(query_key=k, query_time_s=float(now), ready_callback_order=len(ready)-1,
        evidence=deepcopy(e), follower_gap_prediction_s=gap, follower_evidence_valid=valid,
        own_progress_m=float(bus.travel_distance), own_speed_mps=float(bus.current_speed),
        follower=None if follower is None else dict(bus_id=int(follower.bus_id), trip_id=int(follower.trip_id),
            direction=bool(follower.direction), progress_m=float(follower.travel_distance),
            speed_mps=float(follower.current_speed), state=follower.state.name),
        events_seen=dict(arrivals=len(arrivals), departures=len(departures), ready=len(ready), launches=len(launches),
            last_arrival_s=None if not arrivals else arrivals[-1]['time_s'],
            last_departure_s=None if not departures else departures[-1]['time_s']),
        query_load=None, query_load_available=False)


class QueryAudit(collector.capture.ReadyExposureAudit):
    def __init__(self, *args):
        super().__init__(*args)
        self.ready_queries = {}

    def attach(self, runner):
        super().attach(runner)
        self.query_prepare = self.bus_type._prepare_for_action

        def prepare(bus, now, buses, debug):
            result = self.query_prepare(bus, now, buses, debug)
            k = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
            self.ready_queries[k] = snapshot(bus, buses, self.ready, self.arrivals,
                self.departures, self.launches, now)
            return result

        self.bus_type._prepare_for_action = prepare

    def detach(self):
        self.bus_type._prepare_for_action = self.query_prepare
        super().detach()


def validate_snapshot(raw, row, ready_rank):
    now, e, context = row['prediction_time_s'], row['evidence'], row['context']
    require(raw['query_key'] == list(key(row)), 'query snapshot identity changed')
    require(raw['query_time_s'] == now == e['ready_time_s'] <= row['policy_time_s'], 'query/policy clocks differ')
    require(raw['ready_callback_order'] == ready_rank, 'ready callback order changed')
    require(raw['follower_evidence_valid'] == context['follower_evidence_valid'], 'query follower validity changed')
    require(raw['follower_gap_prediction_s'] == e['follower_gap_prediction_s'] ==
        context['follower_departure_gap_s'], 'ready-cached follower gap changed before policy')
    require(raw['evidence']['follower_source'] == e['follower_source'] ==
        context['follower_evidence_source'], 'ready-cached follower source changed before policy')
    require(raw['query_load'] is None and not raw['query_load_available'], 'late load backfilled')
    for name in ('last_arrival_s', 'last_departure_s'):
        t = raw['events_seen'][name]
        require(t is None or t <= now, 'query snapshot contains a future event')


def collect(root, runtime, seed, day, mode, sampler):
    name = f'{seed}_{day}_{mode}'
    receipt = root/'paths'/(name+'.json')
    if receipt.exists():
        old = read(receipt)
        require((old['seed'], old['day'], old['mode'], old['sampler_seed']) ==
            (seed, day, mode, sampler), 'stored path differs from assigned calibration path')
        require((root/'query_snapshots'/(name+'.jsonl.gz')).is_file(), 'collected path lacks ready snapshots')
        return old
    rule = QueryAudit(seed, archived.native.model.load_geometry(runtime), mode)
    credit = collector.capture.pilot.CreditObserver()
    result, _, _, _ = collector.capture.pilot.rollout(root, seed, day, 'stochastic', sampler,
        [0.]*4, 'v145_'+name, probability_rule=rule, trace_observer=rule, credit_observer=credit)
    require(result['base_actors_unchanged'] and not result['training_updates'] and not result['dual_updates'],
        'query instrumentation changed frozen deployment')
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    values = {k: result[k] for k in collector.capture.frozen.original.frozen.prior.METRICS}
    values.update({k: rule.terminal[k] for k in ('inclusive_terminal_delay_s', 'qualification_shift_s')})
    rows = []
    for r in rule.decisions:
        k = key(r)
        row = dict(r, **rule.queries[k])
        row.update(holding_person_s=collector.capture.holding_exposure(row, rule.departure_exposure[k]),
            holding_load=rule.departure_exposure[k]['holding_load'])
        rows.append(row)
    rule.save(root/'records'/(name+'.npz'))
    emit = collector.capture.identity.write_trace
    emit(root/'traces'/(name+'.jsonl.gz'), rows)
    emit(root/'ready_labels'/(name+'.jsonl.gz'),
        [dict(trip_id=k[0], station_id=k[1], direction=k[2], **v) for k, v in rule.ready.items()])
    emit(root/'query_snapshots'/(name+'.jsonl.gz'), rule.ready_queries.values())
    write(root/'terminal'/(name+'.json'), dict(plans=rule.plans, dispatches=rule.dispatches,
        returns=list(rule.returns.values()), reference_clocks=rule.reference_clocks))
    record = dict(seed=seed, day=day, mode=mode, sampler_seed=sampler, values=values,
        contract=rule.contract, decisions=len(rows), ready_labels=len(rule.ready),
        live_query_snapshots=len(rule.ready_queries), actor_updates=0, critic_updates=0,
        dual_updates=0, main_policy_changed=False, execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])
    write(receipt, record)
    return record


def stage_support(path, row, p, status, objects):
    if p is None:
        return dict(kind='opaque', support_class=None, reason='opaque', current_key=None)
    now, direction = p['now'], p['direction']
    current = extraction.audit.stats.follower(row)[0], p['route']['ids'][p['origin']], direction
    motion = objects[-1][current[0]].motion
    i, j = bisect_left(motion.dt, now), bisect_left(motion.at, now)
    pending = bool(j and (not i or motion.at[j-1] > motion.dt[i-1]))
    same_ready = status == 'observed_same_clock_ready'
    if not pending and not same_ready:
        return dict(kind='moving-launch', support_class='not_applicable', reason='not_applicable', current_key=None)
    arrival = path['arrivals'][current]
    ready = path['ready'][current]['ready_time_s']
    observed = ready < now or (ready == now and same_ready)
    command = path['trace'].get(current)
    if observed and command is not None and command['policy_time_s'] < now:
        return dict(kind='known-command', support_class='not_applicable', reason='not_applicable',
            current_key=current, command_time_s=command['policy_time_s'], ready_time_s=ready)
    kind = 'ready-residence' if observed else 'service'
    start = ready if observed else arrival['time_s']
    history = objects[2] if observed else objects[1]
    context, _ = collector.support_context(history, direction, current[1], now, now-start)
    support_class = None if context['reason'] != 'finite' else (
        'finite_historical_fallback' if context['historical_fallback'] else 'finite_window')
    return dict(context, kind=kind, support_class=support_class, current_key=current, start_time_s=start)


def query_packet(path, row, geometry, objects, law, phases, raw, seed, day, mode):
    now, k = row['prediction_time_s'], key(row)
    validate_snapshot(raw, row, path['_observed_ready'].rank[k])
    law.calls.clear()
    point, detail = v151.forecast(path, row, geometry, objects, phases, law, True)
    calls = deepcopy(law.calls)
    if point is None:
        point = archived.fallback(path, row, geometry, objects, phases)
    p, status, _ = v151.position(path, row, objects[-1], geometry, objects, law, True)
    packet, joint, packet_status = accounting.structured_packet(path, row, geometry, objects, phases, law, True)
    if packet is not None:
        close(sum(packet.values()), point, 'V151 packet and point differ')
    stage = stage_support(path, row, p, status, objects)
    native = extraction.completed_support(p, objects[0])
    for stop in native:
        require(all(v is None or v['latest'] < now for v in stop['components'].values()), 'unfinished native support')
    for call in calls:
        support = call['support']
        require(support is None or support['latest_current_s'] is None or support['latest_current_s'] < now,
            'unfinished V151 service support')
    if stage['kind'] == 'ready-residence' and p is not None and p['origin'] < p['target']:
        residence = metadata.residence_support(objects[2], (p['direction'], stage['current_key'][1]), now, stage['elapsed_s'])
        close(residence['mean_remaining_s'], p['initial_s'], 'query residence support differs')
    else:
        residence = None
    observable = extraction.observable(row, p, status, path['cutoff'])
    observable.pop('query_load', None)
    band = horizon_band(point)
    eligible = raw['follower_evidence_valid'] and band is not None
    condition = dict(actor=seed, mode=mode, query_phase='clearance' if now >= path['cutoff'] else 'active',
        predicted_target_phase=None if band is None else 'clearance' if now+point >= path['cutoff'] else 'active',
        direction=k[2], target_stop=k[1], position_status=status, native_stage_kind=stage['kind'],
        native_support_class=stage['support_class'], predicted_horizon_band=band)
    supported = eligible and packet is not None and stage['support_class'] is not None
    return dict(actor=seed, day=day, mode=mode, query_key=k, query_time_s=now,
        predicted_follower=extraction.audit.stats.follower(row),
        episode=(*extraction.audit.stats.follower(row), *k[1:]), point_s=point,
        denominator_eligible=eligible, native_supported=supported,
        condition_values=condition, candidate_key=None,
        observable=observable, native_stage=stage,
        prediction_parts=None if packet is None else packet,
        prediction_coarse=None if packet is None else accounting.coarse(packet),
        inseparable_current_ready_key=joint, packet_status=packet_status,
        support=dict(service=calls, completed_components=native, current_residence=residence, v151_detail=detail),
        clocks=dict(ready_snapshot_time_s=now, query_ready_order=raw['ready_callback_order'],
            policy_snapshot_time_s=row['policy_time_s'], load_sample_time_s=row['policy_time_s'],
            query_load=None, query_load_available=False),
        state_origin='live ready snapshot; frozen V151 replay of strict completed prefix before target labels')


def save_queries(root, runtime, seed, day, mode, service_model, sealed):
    name = f'{seed}_{day}_{mode}'
    geometry = archived.native.model.load_geometry(runtime)
    path, objects = archived.prepare(root, seed, day, mode, geometry)
    law = extraction.RecordedService(service_model['actors'][str(seed)], path['arrivals'], path['ready'], path['cutoff'])
    phases = v151.frozen.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
    snapshots = {tuple(r['query_key']): r for r in metadata.stream(root/'query_snapshots'/(name+'.jsonl.gz'))}
    require(len(snapshots) == len(path['ready']), 'raw snapshots miss action-ready callbacks')
    counts = Counter()
    destination = root/'queries'/(name+'.jsonl.gz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Finish the entire query artifact before the outcome pass can attach labels.
    with gzip.open(destination, 'wt', encoding='ascii', compresslevel=1) as output:
        for k, row in path['trace'].items():
            q = query_packet(path, row, geometry, objects, law, phases, snapshots[k], seed, day, mode)
            q['candidate_key'] = [q['condition_values'][f] for f in sealed['conditioning']['candidate_key']] if q['native_supported'] else None
            output.write(json.dumps(q, allow_nan=False, separators=(',', ':'))+'\n')
            counts['queries'] += 1
            counts['denominator_eligible'] += int(q['denominator_eligible'])
            counts['native_supported'] += int(q['native_supported'])
            counts['native_reason/'+q['native_stage']['reason']] += 1
    return path, geometry, counts


def attach_labels(root, name, path, geometry):
    counts = Counter()
    destination = root/'residuals'/(name+'.jsonl.gz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, 'wt', encoding='ascii', compresslevel=1) as output:
        for q in metadata.stream(root/'queries'/(name+'.jsonl.gz')):
            row = path['trace'][tuple(q['query_key'])]
            label = extraction.label_record(path, row)
            actual, errors = None, None
            if label['status'] == 'resolved' and q['prediction_coarse'] is not None:
                joint = q['inseparable_current_ready_key']
                actual = accounting.coarse(accounting.parts.actual_parts(path, row, geometry[bool(row['direction'])],
                    None if joint is None else tuple(joint)))
                errors = {c: q['prediction_coarse'][c]-actual[c] for c in ('running', 'service', 'residence')}
                close(sum(actual.values()), label['full_eta_s'], 'joint truth does not sum to target ETA')
                close(sum(errors.values()), q['point_s']-label['full_eta_s'], 'joint signed residual identity differs')
            counts['label/'+label['status']] += 1
            counts['joint_resolved'] += int(errors is not None)
            counts['eligible_unresolved'] += int(q['denominator_eligible'] and label['status'] != 'resolved')
            output.write(json.dumps(dict(query_key=q['query_key'], episode=q['episode'],
                query_time_s=q['query_time_s'], label=label, truth_coarse=actual, residual_s=errors),
                allow_nan=False, separators=(',', ':'))+'\n')
    return counts


def parent_protocol(args):
    sealed = read(args.v161_root/'protocol.json')
    require(sealed['status'] == 'preregistered_not_collected' and sealed['execution_source'] == '09553bc920',
        'requires unchanged sealed V161 protocol')
    service = read(args.v151_root/'model.json')
    require((service['status'], service['execution_source']) == ('sealed', '1b783ea8f3'), 'V151 prior changed')
    old = read(args.v146_root/'model.json')
    require((old['status'], old['lower_extension_s'], old['upper_extension_s']) == ('sealed', 3., 0.), 'V146 changed')
    require(not set(service['calibration_days']) & set(sealed['reservation']['calibration_days']),
        'new calibration overlaps frozen point prior days')
    return sealed, service


def calibration_cases(sealed):
    r = sealed['reservation']
    return tuple(itertools.product(r['actors'], r['calibration_days']))


def path_case(args, seed, day, mode, sampler, *, root):
    sealed, service = parent_protocol(args)
    final = root/'capture_cells'/f'{seed}_{day}_{mode}.json'
    if final.exists():
        old = read(final)
        require(old['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'capture source changed')
        return old
    receipt = collect(root, args.runtime, seed, day, mode, sampler)
    path, geometry, counts = save_queries(root, args.runtime, seed, day, mode, service, sealed)
    counts.update(attach_labels(root, f'{seed}_{day}_{mode}', path, geometry))
    result = dict(seed=seed, day=day, mode=mode, sampler_seed=sampler, counts=dict(counts),
        query_record_count=counts['queries'], live_ready_snapshots=receipt['live_query_snapshots'],
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], all_queries_recorded_before_target_labels=True,
        interval_fit=False, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    write(final, result)
    print('V162_CAPTURE_PATH_COMPLETE', seed, day, mode, flush=True)
    return result


def preflight(args):
    sealed, _ = parent_protocol(args)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v162_*.py', '-v'], check=True)
    receipts = []
    for mode in MODES:
        seed, day, sampler = 12011, 2451001, 1450000000
        root = args.out_root/'preflight_paths'
        captured = path_case(args, seed, day, mode, sampler, root=root)
        name = f'{seed}_{day}_{mode}'
        old = read(args.archive_root/'paths'/(name+'.json'))
        new = read(root/'paths'/(name+'.json'))
        collector.capture.pilot.joint.assert_same_state(old['values'], new['values'], 'V162_exact_original_endpoints')
        collector.capture.pilot.joint.assert_same_state(old['contract'], new['contract'], 'V162_exact_original_contract')
        collector.capture.pilot.joint.assert_same_state(read(args.archive_root/'terminal'/(name+'.json')),
            read(root/'terminal'/(name+'.json')), 'V162_exact_terminal_clocks')
        rows = list(metadata.stream(root/'traces'/(name+'.jsonl.gz')))
        collector.capture.passive.assert_arrays_and_actions(args.archive_root/'records'/(name+'.npz'),
            root/'records'/(name+'.npz'), rows)
        collector.capture.identity.assert_decisions(list(metadata.stream(args.archive_root/'traces'/(name+'.jsonl.gz'))), rows)
        queries = {tuple(q['query_key']): q for q in metadata.stream(root/'queries'/(name+'.jsonl.gz'))}
        comparisons = 0
        for saved in metadata.stream(args.v159_root/'residuals'/(name+'.jsonl.gz')):
            if saved['forecasts'] is not None:
                q = queries[tuple(saved['query_key'])]
                close(q['point_s'], saved['forecasts']['v151']['point_s'], 'frozen V151 point differs')
                if saved['components'] is not None:
                    for c, value in saved['components']['v151']['prediction_s'].items():
                        close(q['prediction_coarse'][c], value, 'frozen V151 component differs')
                comparisons += 1
        receipts.append(dict(mode=mode, queries=captured['query_record_count'], v151_forecasts_reproduced=comparisons))
    write(args.out_root/'preflight/summary/status.json', dict(status='v162_capture_preflight_passed',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], consumed_paths=receipts,
        original_endpoints_actions_arrays_terminal_clocks_exact=True, query_and_support_clocks_checked=True,
        frozen_v151_forecasts_and_components_exact=True, new_calibration_days_collected=0,
        confirmation_collected=0, interval_fit=False, main_policy_changed=False))
    print('V162_PREFLIGHT_PASSED_CALIBRATION_NOT_STARTED', flush=True)


def require_preflight(args):
    proof = read(args.out_root/'preflight/summary/status.json')
    require((proof['status'], proof['execution_source']) == ('v162_capture_preflight_passed',
        os.environ['FREQDUET_DIAGNOSTIC_COMMIT']), 'requires successful same-source capture preflight')


def pair_case(args, index):
    sealed, _ = parent_protocol(args)
    seed, day = calibration_cases(sealed)[index]
    require(day not in sealed['reservation']['confirmation_days'], 'confirmation collection is not authorized')
    sampler = sealed['reservation']['sampler_seeds'][str(day)]
    for mode in MODES:
        path_case(args, seed, day, mode, sampler, root=args.out_root/'calibration')
    return index


def run_batch(args):
    require_preflight(args)
    cases = calibration_cases(parent_protocol(args)[0])
    require(0 <= args.start < args.end <= len(cases), 'invalid calibration shard')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(pair_case, args, i) for i in range(args.start, args.end)]
        for done, job in enumerate(as_completed(jobs), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=done, total=len(jobs), unit='calibration_actor_day')), flush=True)
    write(args.out_root/'shards'/f'{args.start}-{args.end}.json', dict(status='v162_calibration_shard_complete',
        start=args.start, end=args.end, calibration_paths=2*(args.end-args.start),
        confirmation_paths=0, interval_fit=False))


def aggregate(args):
    require_preflight(args)
    sealed, _ = parent_protocol(args)
    rows = [read(args.out_root/'calibration/capture_cells'/f'{s}_{d}_{m}.json')
        for s, d in calibration_cases(sealed) for m in MODES]
    counts = dict(sum((Counter(r['counts']) for r in rows), Counter()))
    write(args.out_root/'summary/status.json', dict(status='v162_calibration_capture_complete_no_fit',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], calibration_days=sealed['reservation']['calibration_days'],
        actors=sealed['reservation']['actors'], modes=list(MODES), paths=len(rows), counts=counts,
        confirmation_paths=0, interval_fit=False, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, v151_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1],
        next='Audit saved calibration metadata and support before any interval fit; confirmation remains unopened.'))
    print('V162_ALL_CALIBRATION_CAPTURED_NO_INTERVAL_FIT', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'run', 'aggregate'))
    for name in ('out-root', 'runtime', 'v161-root', 'v151-root', 'v146-root', 'archive-root', 'v159-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    {'preflight': preflight, 'run': run_batch, 'aggregate': aggregate}[args.command](args)
