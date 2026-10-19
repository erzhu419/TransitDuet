#!/usr/bin/env python3
"""Exactly replay six consumed V135 pairs; label ready truth and exposure."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import json
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
import subprocess
import sys

import numpy as np

import run_freqduet_v135_handoff_confirmation as frozen
import run_freqduet_v132_passive_terminal_trace as passive
from freqduet_system_credit import SystemCreditStream


pilot, identity = frozen.pilot, passive.identity
ITEMS = ((12071, 2351015), (12143, 2351015), (933059, 2351002),
    (12071, 2351007), (12037, 2351004), (12109, 2351014))
CASES = tuple((s, d, m) for s, d in ITEMS for m in frozen.MODES)
BIN_S = 900.


def read(path):
    return json.loads(path.read_text())


def close(a, b, label):
    if not np.isclose(a, b, rtol=0, atol=1e-7):
        raise AssertionError(f'{label}: {a} != {b}')


class ExposureTimeline(SystemCreditStream):
    def __init__(self):
        super().__init__()
        self.bands = defaultdict(lambda: np.zeros(5))

    def advance(self, start_s, dt_s, *rates):
        super().advance(start_s, dt_s, *rates)
        stop, now = start_s+dt_s, start_s
        while now < stop:
            band = int(now//BIN_S)
            end = min(stop, (band+1)*BIN_S)
            self.bands[band] += (end-now)*np.asarray(rates)
            now = end


class ReadyExposureAudit(frozen.ConfirmationAudit):
    def __init__(self, seed, geometry, mode):
        super().__init__(seed, geometry, mode)
        self.mode, self.queries, self.departure_exposure = mode, {}, {}

    def attach(self, runner):
        super().attach(runner)
        self.exposure_departure = self.recorder.record_departure

        def departure(recorder, station_id, direction, departure_time_s, trip_id):
            result = self.exposure_departure(recorder, station_id, direction, departure_time_s, trip_id)
            bus = self.buses[int(trip_id)]
            key = int(trip_id), int(station_id), bool(direction)
            self.departure_exposure[key] = dict(person_counter_s=float(bus.episode_hold_person_seconds),
                holding_load=int(bus._active_holding_load))
            return result

        self.recorder.record_departure = departure

    def detach(self):
        self.recorder.record_departure = self.exposure_departure
        super().detach()

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        key = passive.key(self.decisions[-1])
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        online = self.online[key]
        original = online
        if self.mode == 'handoff':
            forecast_key = frozen.frozen.prior.forecast_key(self.evidence[key], key[2], key[1])
            original = self.stream.original_predictions[forecast_key]
        self.queries[key] = deepcopy(dict(evidence=self.evidence[key], online=online,
            original_prediction=original, context=asdict(context),
            query_load=len(bus.passengers), person_counter_before_s=float(bus.episode_hold_person_seconds)))
        return law


def holding_exposure(row, end):
    exposed = end['person_counter_s']-row['person_counter_before_s']
    close(exposed, row['executed_s']*row['query_load'], 'counter-based stop passenger holding')
    return exposed


def ready_label(row, ready):
    e = row['evidence']
    key = e['predicted_follower_trip_id'], int(row['station_id']), bool(row['direction'])
    if key not in ready:
        return dict(resolved=False, reason='same_trip_ready_missing')
    label = ready[key]
    if label['bus_id'] != e['predicted_follower_bus_id']:
        return dict(resolved=False, reason='same_trip_bus_mismatch')
    gap = label['ready_time_s']-e['ready_time_s']
    if gap < 0:
        return dict(resolved=False, reason='same_follower_already_ready', same_ready_gap_s=gap)
    old, new = row['original_prediction']['gap_s'], row['online']['gap_s']
    return dict(resolved=True, same_ready_gap_s=gap, original_error_s=old-gap,
        handoff_error_s=new-gap, original_abs_error_s=abs(old-gap), handoff_abs_error_s=abs(new-gap))


def pair_loss_change(forward, follower, old_elapsed, new_elapsed):
    def loss(t):
        return (forward+t)**2+(follower-t)**2
    return loss(new_elapsed)-loss(old_elapsed)


def replay(args, case):
    seed, day, mode = case
    name = f'{seed}_{day}_{mode}'
    receipt_path = args.out_root/'paths'/(name+'.json')
    if receipt_path.exists():
        return read(receipt_path)
    rule = ReadyExposureAudit(seed, frozen.frozen.prior.model.load_geometry(args.runtime_root), mode)
    credit = pilot.CreditObserver()
    credit.stream = ExposureTimeline()
    result, _, _, _ = pilot.rollout(args.out_root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, 'v137_'+name, probability_rule=rule, trace_observer=rule, credit_observer=credit)
    # Open consumed future artifacts only after the unchanged live replay ends.
    old_native = read(args.v135_root/'ready'/('v135_'+name+'.json'))
    pilot.joint.assert_same_state(old_native, {k: result[k] for k in old_native}, 'V137_native_endpoint_stream_exact')
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    values = {k: result[k] for k in frozen.original.frozen.prior.METRICS}
    values.update({k: rule.terminal[k] for k in ('inclusive_terminal_delay_s', 'qualification_shift_s')})
    previous = read(args.v135_root/'paths'/(name+'.json'))
    pilot.joint.assert_same_state(previous['values'], values, 'V137_all_V135_values_exact')
    pilot.joint.assert_same_state(previous['contract'], rule.contract, 'V137_same_frozen_physics')
    terminal = dict(plans=rule.plans, dispatches=rule.dispatches,
        returns=list(rule.returns.values()), reference_clocks=rule.reference_clocks)
    pilot.joint.assert_same_state(read(args.v135_root/'terminal'/(name+'.json')), terminal,
        'V137_all_terminal_plans_launches_returns_exact')
    record_path = args.out_root/'records'/(name+'.npz')
    rule.save(record_path)
    passive.assert_arrays_and_actions(args.v135_root/'records'/(name+'.npz'), record_path, rule.decisions)
    rows = []
    for r in rule.decisions:
        key = passive.key(r)
        row = dict(r, **rule.queries[key])
        row.update(holding_person_s=holding_exposure(row, rule.departure_exposure[key]),
            holding_load=rule.departure_exposure[key]['holding_load'])
        if not row['prediction_time_s'] == row['evidence']['ready_time_s'] <= row['policy_time_s']:
            raise AssertionError('ready/policy forecast clocks differ')
        rows.append(row)
    close(sum(r['holding_person_s'] for r in rows), values['holding_person_s'], 'all stop exposures recover holding ledger')
    np.testing.assert_allclose(sum(credit.stream.bands.values()), credit.stream.total, rtol=0, atol=1e-7)
    sparse = [r for r in rows if r['evidence']['follower_source'] in frozen.frozen.prior.SOURCES]
    old_sparse = identity.read_trace(args.v135_root/'launch_traces'/(name+'.jsonl.gz'))
    identity.assert_decisions(old_sparse, [{k: r[k] for k in old} for old, r in zip(old_sparse, sparse)])
    if len(sparse) != len(old_sparse):
        raise AssertionError('passive replay changed sparse decision support')
    identity.write_trace(args.out_root/'traces'/(name+'.jsonl.gz'), rows)
    identity.write_trace(args.out_root/'ready_labels'/(name+'.jsonl.gz'),
        [dict(trip_id=k[0], station_id=k[1], direction=k[2], **v) for k, v in rule.ready.items()])
    pilot.write_json(args.out_root/'exposure'/(name+'.json'),
        {str(k): v.tolist() for k, v in credit.stream.bands.items()})
    receipt = dict(seed=seed, day=day, mode=mode, values=values, decisions=len(rows),
        ready_labels=len(rule.ready), endpoints_exact=True, all_timing_and_actions_exact=True,
        sparse_evidence_and_forecasts_exact=True, terminal_path_exact=True, holding_exposure_exact=True,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    pilot.write_json(receipt_path, receipt)
    print('V137_EXACT_PASSIVE_PATH_DONE', seed, day, mode, flush=True)
    return receipt


def label_path(root, seed, day, mode):
    name = f'{seed}_{day}_{mode}'
    trace = identity.read_trace(root/'traces'/(name+'.jsonl.gz'))
    ready = {passive.key(r): r for r in identity.read_trace(root/'ready_labels'/(name+'.jsonl.gz'))}
    with np.load(root/'records'/(name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        departures = [dict(zip(fields, r.tolist())) for r in record['departures']]
    moment = identity.audit.series([dict(r, before=r['time_s'], current_action=0.) for r in departures],
        ('before', 'current_action'))
    next_dep = identity.successors(departures)
    groups, locations = defaultdict(list), {}
    for r in departures:
        key = passive.key(r)
        group = groups[key[1:]]
        locations[key] = len(group)
        group.append(r)
    labels = []
    for row in trace:
        if row['online']['status'] != 'launch_handoff_component' or not row['component_used']:
            continue
        key, e = passive.key(row), row['evidence']
        resolved = ready_label(row, ready)
        end = groups[key[1:]][locations[key]]
        elapsed = end['time_s']-e['ready_time_s']
        lag = elapsed-row['executed_s']
        restore = row['raw_shadow_s']-row['chosen_s']
        follower = e['predicted_follower_trip_id'], key[1], key[2]
        label = dict(trip_id=key[0], station_id=key[1], direction=int(key[2]),
            policy_time_s=row['policy_time_s'], original_s=row['raw_shadow_s'], chosen_s=row['chosen_s'],
            direct_delta_s=-restore, query_load=row['query_load'], holding_person_s=row['holding_person_s'],
            direct_same_state_exposure_delta_person_s=-restore*row['query_load'],
            forward_gap_s=e['forward_gap_s'], original_gap_s=row['original_prediction']['gap_s'],
            handoff_gap_s=row['online']['gap_s'], execution_lag_s=lag,
            predicted_identity_is_next_departure=follower == next_dep.get(key), **resolved)
        if resolved['resolved']:
            for name, gap in (('same_ready', resolved['same_ready_gap_s']),
                    ('original_prediction', row['original_prediction']['gap_s']), ('handoff_prediction', row['online']['gap_s'])):
                label[name+'_candidate_minus_shadow_pair_loss_s2'] = pair_loss_change(
                    e['forward_gap_s'], gap, row['raw_shadow_s']+lag, elapsed)
        times = np.asarray([r['time_s'] for r in groups[key[1:]]])
        shifted = times.copy()
        shifted[locations[key]] += restore
        geometry = identity.audit.geometry_change(times, shifted, moment)
        label['frozen_departure_geometry_candidate_minus_shadow_cv2'] = -geometry['cv_squared_change']
        label['shadow_departure_order_changes'] = not np.array_equal(np.argsort(shifted, kind='stable'), np.arange(len(times)))
        labels.append(label)
    pilot.write_json(root/'labels'/(name+'.json'), labels)
    return trace, labels


def label_summary(rows):
    valid = [r for r in rows if r['resolved']]
    action = [r for r in rows if r['direct_delta_s'] != 0]
    names = ('original_error_s', 'handoff_error_s', 'original_abs_error_s', 'handoff_abs_error_s')
    return dict(handoff_decisions=len(rows), resolved=len(valid), missing_reasons=dict(Counter(
        r['reason'] for r in rows if not r['resolved'])),
        prediction_means={k: fmean(r[k] for r in valid) for k in names} if valid else None,
        changed_actions=len(action), direct_delta_s=sum(r['direct_delta_s'] for r in rows),
        direct_same_state_exposure_delta_person_s=sum(r['direct_same_state_exposure_delta_person_s'] for r in rows),
        next_departure_identity_matches=sum(r['predicted_identity_is_next_departure'] for r in rows),
        forecast_error_improved=sum(r['handoff_abs_error_s'] < r['original_abs_error_s'] for r in valid),
        ready_pair_candidate_harm=sum(r['resolved'] and r['same_ready_candidate_minus_shadow_pair_loss_s2'] > 1e-8 for r in action),
        ready_pair_candidate_benefit=sum(r['resolved'] and r['same_ready_candidate_minus_shadow_pair_loss_s2'] < -1e-8 for r in action),
        departure_geometry_candidate_harm=sum(r['frozen_departure_geometry_candidate_minus_shadow_cv2'] > 1e-12 for r in action),
        departure_geometry_candidate_benefit=sum(r['frozen_departure_geometry_candidate_minus_shadow_cv2'] < -1e-12 for r in action))


def aggregate(args):
    cases, all_labels = [], []
    for seed, day in ITEMS:
        paths = {m: read(args.out_root/'paths'/f'{seed}_{day}_{m}.json') for m in frozen.MODES}
        traces, labelled = {}, {}
        for m in frozen.MODES:
            traces[m], labelled[m] = label_path(args.out_root, seed, day, m)
        b, c = [{passive.key(r): r for r in traces[m]} for m in frozen.MODES]
        if b.keys() != c.keys():
            raise AssertionError('paired observed lower decision support differs')
        direct_keys = {passive.key(r) for r in labelled['handoff']}
        exposure = dict(on_handoff_keys_person_s=0., other_keys_person_s=0.,
            holding_change_at_candidate_load_person_s=0., load_change_at_baseline_holding_person_s=0.)
        for k in b:
            delta = c[k]['holding_person_s']-b[k]['holding_person_s']
            exposure['on_handoff_keys_person_s' if k in direct_keys else 'other_keys_person_s'] += delta
            exposure['holding_change_at_candidate_load_person_s'] += (c[k]['executed_s']-b[k]['executed_s'])*c[k]['query_load']
            exposure['load_change_at_baseline_holding_person_s'] += b[k]['executed_s']*(c[k]['query_load']-b[k]['query_load'])
        effect = {k: paths['handoff']['values'][k]-paths['old']['values'][k] for k in frozen.METRICS}
        close(exposure['on_handoff_keys_person_s']+exposure['other_keys_person_s'], effect['holding_person_s'], 'paired exposed holding')
        close(exposure['holding_change_at_candidate_load_person_s']+exposure['load_change_at_baseline_holding_person_s'],
            effect['holding_person_s'], 'holding/load exposure allocation')
        bands = {m: read(args.out_root/'exposure'/f'{seed}_{day}_{m}.json') for m in frozen.MODES}
        timeline = {k: (np.asarray(bands['handoff'].get(k, [0.]*5))-bands['old'].get(k, [0.]*5)).tolist()
            for k in bands['old'].keys() | bands['handoff'].keys()}
        sums = np.sum(list(timeline.values()), axis=0)
        for i, name in ((0, 'waiting_person_s'), (1, 'onboard_person_s'), (4, 'backlog_trip_s')):
            close(sums[i], effect[name], 'paired exposure timeline')
        pilot.write_json(args.out_root/'timelines'/f'{seed}_{day}.json', timeline)
        largest = sorted(timeline, key=lambda k: abs(timeline[k][0]+timeline[k][1]), reverse=True)[:4]
        cases.append(dict(seed=seed, day=day, effect=effect, exposure_allocation=exposure,
            labels=label_summary(labelled['handoff']), handoff_decisions=labelled['handoff'],
            largest_passenger_change_bands=[dict(start_s=int(k)*BIN_S, waiting_person_s=timeline[k][0],
                onboard_person_s=timeline[k][1], backlog_trip_s=timeline[k][4]) for k in largest]))
        all_labels.extend(labelled['handoff'])
    pilot.write_json(args.out_root/'summary/status.json', dict(status='v137_exact_passive_ready_exposure_complete',
        cases=cases, labels=label_summary(all_labels), replayed_consumed_paths=len(CASES), new_demand_days=0,
        all_original_endpoints_actions_timing_and_terminal_paths_exact=True, exposure_ledgers_exact=True,
        actor_updates=0, critic_updates=0, dual_updates=0, parameters_changed=False, main_policy_changed=False,
        limitations='Six outcome-selected consumed pairs from V136, not independent confirmation or representative '
            'MAE. Labels are post-rollout outcomes on each frozen path, never controller inputs. Same-trip/bus '
            'ready truth and next-departure identity are separate. Pair losses use fixed observed follower ready '
            'time; departure geometry shifts one event with other events frozen, including reordering. Neither '
            'is a realized per-action intervention or daily causal value. Same-state exposure and holding/load '
            'allocations are arithmetic, not passenger mediation.15-minute bands conserve exact observed '
            'exposures, but do not assign delayed changes to individual actions. No fitting or gate changes.'))
    print('V137_PASSIVE_READY_EXPOSURE_COMPLETE', label_summary(all_labels), flush=True)


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v137_*.py', '-v'], check=True)
    for stage, cases in (('preflight', CASES[:2]), ('remaining', CASES[2:])):
        with ProcessPoolExecutor(max_workers=min(args.workers, len(cases)), mp_context=get_context('spawn')) as pool:
            tasks = [pool.submit(replay, args, c) for c in cases]
            for task in as_completed(tasks):
                task.result()
        pilot.write_json(args.out_root/'stages'/stage/'done.json', dict(status='exact', paths=len(cases)))
        print('V137_STAGE_EXACT', stage, flush=True)
    aggregate(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v135-root', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=10)
    run(p.parse_args())
