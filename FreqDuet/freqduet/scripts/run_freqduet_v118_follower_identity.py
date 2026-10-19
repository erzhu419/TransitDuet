#!/usr/bin/env python3
"""Passively reproduce V115 before labeling the exact predicted follower."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v117_decision_evidence as prior


audit, frozen = prior.prior, prior.prior.frozen
ITEMS = list(itertools.product(frozen.SEEDS, frozen.DAYS))
CI_FIELDS = ('identity_matches_next_arrival', 'identity_matches_next_departure',
    'same_ready_resolved', 'same_ready_temporal_valid', 'next_service_error_s',
    'same_ready_error_s', 'identity_gap_component_s', 'same_eta_error_s', 'same_dwell_error_s',
    'predicted_vs_same_sign_disagrees', 'same_vs_next_sign_disagrees',
    'next_ready_vs_departure_sign_disagrees')


def capture_prediction(bus, now, trip_by_bus):
    result = dict(bus_id=int(bus.bus_id), ready_time_s=float(now),
        follower_source=str(bus.pre_action_follower_source),
        service_dwell_proxy_s=float(bus.last_service_dwell_s))
    follower = bus.pre_action_follower_bus_id
    follower = None if follower is None else int(follower)
    result.update(predicted_follower_bus_id=follower, predicted_follower_trip_id=trip_by_bus.get(follower))
    for name, attribute in (
            ('eta_prediction_s', 'pre_action_follower_eta'),
            ('spatial_gap_m', 'pre_action_follower_spatial_gap'),
            ('speed_mps', 'pre_action_follower_speed')):
        value = getattr(bus, attribute)
        result[name] = None if value is None else float(value)
    return result


class IdentityAudit(frozen.FundedAudit):
    def __init__(self, seed):
        super().__init__(seed)
        self.ready, self.evidence = {}, {}

    def attach(self, runner):
        super().attach(runner)
        from env.bus import Bus
        self.bus_type, self.original_prepare = Bus, Bus._prepare_for_action

        def prepare(bus, now, buses, debug):
            identities = {int(b.bus_id): int(b.trip_id) for b in buses}
            result = self.original_prepare(bus, now, buses, debug)
            key = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
            if key in self.ready:
                raise AssertionError('a trip/stop has multiple action-ready captures')
            self.ready[key] = capture_prediction(bus, now, identities)
            return result

        Bus._prepare_for_action = prepare

    def detach(self):
        self.bus_type._prepare_for_action = self.original_prepare
        super().detach()

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        identity = audit.key(self.decisions[-1])
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        self.evidence[identity] = dict(self.ready[identity],
            forward_gap_s=context.forward_headway_s, target_headway_s=context.target_headway_s,
            follower_gap_prediction_s=context.follower_departure_gap_s)
        return law


def read_trace(path):
    with gzip.open(path, 'rt', encoding='ascii') as stream:
        return [json.loads(line) for line in stream]


def write_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='ascii') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')


def assert_decisions(original, current):
    if len(original) != len(current):
        raise AssertionError('passive replay changed the number of original decisions')
    for old, new in zip(original, current):
        frozen.pilot.joint.assert_same_state(old, new, 'V118_exact_original_decision_path')


def run_item(root, old, index):
    receipt = root/'pairs'/f'p{index}.json'
    if receipt.exists():
        return
    seed, day = ITEMS[index]
    rule, credit = IdentityAudit(seed), frozen.pilot.CreditObserver()
    result, _, _, _ = frozen.pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v115_{seed}_{day}_{frozen.PRIMARY}_0', probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    actual = dict(seed=seed, day=day, method=frozen.PRIMARY, rep=0, results=result,
        contract=rule.contract, coverage=rule.coverage, feedforward=rule.feedforward, accounting=rule.accounting)
    cached = json.loads((old/'pairs'/f'p{frozen.ITEMS.index((seed, day, frozen.PRIMARY, 0))}.json').read_text())
    frozen.pilot.joint.assert_same_state(cached, actual, 'V118_all_original_endpoints_and_accounts')
    name = f'{seed}_{day}_{frozen.PRIMARY}_0'
    assert_decisions(read_trace(old/'traces'/(name+'.jsonl.gz')), rule.decisions)
    record = root/'records'/(name+'.npz')
    rule.save(record)
    with np.load(old/'records'/(name+'.npz')) as before, np.load(record) as after:
        if before.files != after.files:
            raise AssertionError('passive replay changed timing record fields')
        for field in before.files:
            np.testing.assert_array_equal(before[field], after[field])
    departures = {audit.key(r): r for r in rule.departures}
    arrivals = {audit.key(r): r for r in rule.arrivals}
    for row in rule.decisions:
        key = audit.key(row)
        predecessor = departures.get((row['predecessor_trip_id'], key[1], key[2]))
        # Rebuild V117's coordinates independently of the newly captured context.
        departure = dict(departures[key], arrival_time=arrivals[key]['time_s'])
        recovered = prior.recover(row, departure, predecessor)
        evidence = rule.evidence[key]
        np.testing.assert_allclose(recovered['ready_time_s'], evidence['ready_time_s'], atol=1e-9, rtol=0)
        np.testing.assert_allclose(recovered['forward_gap_s'], evidence['forward_gap_s'], atol=1e-9, rtol=0)
        if row['follower_valid']:
            np.testing.assert_allclose(recovered['predicted_follower_gap_s'],
                evidence['follower_gap_prediction_s'], atol=1e-9, rtol=0)
    write_trace(root/'traces'/(name+'.jsonl.gz'),
        [dict(row, evidence=rule.evidence[audit.key(row)]) for row in rule.decisions])
    write_trace(root/'ready'/(name+'.jsonl.gz'), [dict(trip_id=k[0], station_id=k[1], direction=k[2], **v)
        for k, v in rule.ready.items()])
    frozen.pilot.write_json(receipt, dict(seed=seed, day=day, index=index, original_endpoints_exact=True,
        original_decisions_exact=True, full_timing_arrays_exact=True, v117_coordinates_exact=True,
        decisions=len(rule.decisions), ready_events=len(rule.ready), actor_updates=0, dual_updates=0))
    print('V118_EXACT_PATH_DONE', index, seed, day, flush=True)


def successors(events):
    previous, following = {}, {}
    for row in events:
        identity = audit.key(row)
        if identity[1:] in previous:
            following[previous[identity[1:]]] = identity
        previous[identity[1:]] = identity
    return following


def resolved_label(evidence, own_ready, identity, ready, arrivals, departures):
    exists = identity is not None and identity in ready
    matches = exists and ready[identity]['bus_id'] == evidence['predicted_follower_bus_id']
    resolved = matches and identity in arrivals and identity in departures
    result = dict(forecast_trip_ready_exists=exists, same_ready_resolved=resolved)
    if exists:
        result['same_trip_bus_matches'] = matches
    if not resolved:
        return result
    gap = ready[identity]['ready_time_s']-own_ready
    result['same_ready_temporal_valid'] = gap >= 0
    if gap < 0:
        return result
    arrival_eta = arrivals[identity]['time_s']-own_ready
    dwell = ready[identity]['ready_time_s']-arrivals[identity]['time_s']
    predicted = evidence['follower_gap_prediction_s']
    eta_error = evidence['eta_prediction_s']-arrival_eta
    dwell_error = evidence['service_dwell_proxy_s']-dwell
    adjustment = predicted-evidence['eta_prediction_s']-evidence['service_dwell_proxy_s']
    np.testing.assert_allclose(eta_error+dwell_error+adjustment, predicted-gap, atol=1e-8, rtol=0)
    return dict(result, same_ready_gap_s=gap, same_ready_error_s=predicted-gap,
        same_ready_abs_error_s=abs(predicted-gap), same_eta_error_s=eta_error,
        same_eta_abs_error_s=abs(eta_error), same_dwell_error_s=dwell_error,
        gap_adjustment_s=adjustment,
        same_arrival_before_own_ready=arrival_eta < 0,
        same_after_ready_s=departures[identity]['time_s']-ready[identity]['ready_time_s'])


def error_parts(predicted, same_gap, next_gap):
    return dict(next_service_error_s=predicted-next_gap, same_ready_error_s=predicted-same_gap,
        identity_gap_component_s=same_gap-next_gap)


def summarize(rows):
    names = sorted({k for r in rows for k, v in r.items() if isinstance(v, (bool, int, float))})
    metrics = {}
    for name in names:
        values = np.asarray([float(r[name]) for r in rows if r.get(name) is not None])
        metrics[name] = dict(n=len(values), mean=float(values.mean()), p10=float(np.percentile(values, 10)),
            median=float(np.median(values)), p90=float(np.percentile(values, 90)))
    return dict(decisions=len(rows), cut_s=sum(r.get('cut_s', 0.) for r in rows), metrics=metrics,
        follower_sources=dict(Counter(r['follower_source'] for r in rows if r['follower_eligible'])))


def analyse(root, index):
    seed, day = ITEMS[index]
    name = f'{seed}_{day}_{frozen.PRIMARY}_0'
    trace = {audit.key(r): r for r in read_trace(root/'traces'/(name+'.jsonl.gz'))}
    ready = {audit.key(r): r for r in read_trace(root/'ready'/(name+'.jsonl.gz'))}
    with np.load(root/'records'/(name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        events = {kind: [dict(zip(fields, row.tolist())) for row in record[kind]]
                  for kind in ('arrivals', 'departures')}
    arrivals = {audit.key(r): r for r in events['arrivals']}
    departures = {audit.key(r): r for r in events['departures']}
    observed_departures = [r for r in events['departures'] if audit.key(r) in arrivals]
    next_arr, next_dep = successors(events['arrivals']), successors(observed_departures)
    moment = audit.series([dict(r, current_action=0., before=r['time_s']) for r in observed_departures], ('before', 'current_action'))
    groups, rows = defaultdict(list), {}
    for identity, row in trace.items():
        e, f = row['evidence'], row['follower_valid']
        record = dict(follower_eligible=f, follower_source=e['follower_source'])
        if f:
            predicted_id = (e['predicted_follower_trip_id'], identity[1], identity[2])
            record.update(resolved_label(e, e['ready_time_s'], predicted_id, ready, arrivals, departures))
            record['identity_matches_next_arrival'] = predicted_id == next_arr.get(identity)
            record['identity_matches_next_departure'] = predicted_id == next_dep.get(identity)
            successor = next_dep.get(identity)
            if successor in ready:
                next_gap = ready[successor]['ready_time_s']-e['ready_time_s']
                record['next_ready_temporal_valid'] = next_gap >= 0
                if next_gap >= 0:
                    record['next_ready_gap_s'] = next_gap
                    record['next_service_error_s'] = e['follower_gap_prediction_s']-next_gap
                    if 'same_ready_gap_s' in record:
                        record.update(error_parts(e['follower_gap_prediction_s'], record['same_ready_gap_s'], next_gap))
        rows[identity] = record
        groups['all_decisions'].append(record)
        if f and identity in next_dep:
            groups['forecast_identity_'+('match' if record['identity_matches_next_departure'] else 'mismatch')].append(record)
    stops = defaultdict(list)
    for end in observed_departures:
        stops[audit.key(end)[1:]].append(end)
    for stop in stops.values():
        times = np.asarray([r['time_s'] for r in stop])
        for i, end in enumerate(stop):
            identity = audit.key(end)
            if identity not in trace:
                continue
            row, e = trace[identity], trace[identity]['evidence']
            parts = audit.action_parts(row)
            for cause, cut in (('negative', parts['negative_cut_s']), ('quota', -parts['quota'])):
                if cut <= 0:
                    continue
                shifted = times.copy()
                shifted[i] += cut
                delta = audit.geometry_change(times, shifted, moment)['cv_squared_change']
                label = 'useful' if delta < -1e-12 else 'harmful' if delta > 1e-12 else 'neutral'
                record = dict(rows[identity], cut_s=cut)
                executed = end['time_s']-e['ready_time_s']
                forward = e['forward_gap_s']
                if row['follower_valid']:
                    predicted_sign = prior.pair_restore_change(forward, e['follower_gap_prediction_s'], executed, cut) < 0
                    record['predicted_pair_beneficial'] = predicted_sign
                    if 'same_ready_gap_s' in record:
                        same_sign = prior.pair_restore_change(forward, record['same_ready_gap_s'], executed, cut) < 0
                        record['same_target_pair_beneficial'] = same_sign
                        record['predicted_vs_same_sign_disagrees'] = predicted_sign != same_sign
                if 0 < i < len(stop)-1 and int(stop[i-1]['trip_id']) == row['predecessor_trip_id']:
                    next_identity = audit.key(stop[i+1])
                    next_gap = ready[next_identity]['ready_time_s']-e['ready_time_s']
                    if next_gap >= 0:
                        next_sign = prior.pair_restore_change(forward, next_gap, executed, cut) < 0
                        record['next_ready_pair_beneficial'] = next_sign
                        if 'same_target_pair_beneficial' in record:
                            record['same_vs_next_sign_disagrees'] = record['same_target_pair_beneficial'] != next_sign
                        departure_sign = prior.pair_restore_change(forward,
                            stop[i+1]['time_s']-e['ready_time_s'], executed, cut) < 0
                        record['next_ready_vs_departure_sign_disagrees'] = next_sign != departure_sign
                groups[cause].append(record)
                groups[cause+'_'+label].append(record)
                if row['follower_valid'] and identity in next_dep:
                    groups[cause+'_identity_'+('match' if record['identity_matches_next_departure'] else 'mismatch')].append(record)
    return groups


def aggregate(root, v116):
    receipts = [json.loads((root/'pairs'/f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [(r['seed'], r['day']) for r in receipts] != ITEMS or not all(all(r[k] for k in
            ('original_endpoints_exact', 'original_decisions_exact', 'full_timing_arrays_exact', 'v117_coordinates_exact'))
            for r in receipts):
        raise AssertionError('all original paths must match before identity/ETA analysis')
    combined, cells = defaultdict(list), []
    for index, (seed, day) in enumerate(ITEMS):
        groups = analyse(root, index)
        means = {}
        for name, rows in groups.items():
            combined[name].extend(rows)
            summary = summarize(rows)
            means[name] = {m: summary['metrics'][m]['mean'] for m in CI_FIELDS if m in summary['metrics']}
        cells.append(dict(seed=seed, day=day, means=means))
    baseline = json.loads((v116/'summary/status.json').read_text())
    for cause in ('negative', 'quota'):
        for label in ('useful', 'harmful'):
            rows = combined[cause+'_'+label]
            expected = baseline['cohorts'][cause]
            if len(rows) != expected[label+'_restores'] or sum(r['cut_s'] for r in rows) != expected[label+'_cut_s']:
                raise AssertionError('passive identity replay changed V116 cut geometry')
    common = {}
    for name in combined:
        if all(name in c['means'] for c in cells):
            metrics = set.intersection(*(set(c['means'][name]) for c in cells))
            if metrics:
                common[name] = sorted(metrics)
    inference_rows = [dict(seed=c['seed'], day=c['day'], values={g: {m: c['means'][g][m] for m in fields}
        for g, fields in common.items()}) for c in cells]
    frozen.pilot.write_json(root/'summary/status.json', dict(status='v115_passive_identity_replay_complete',
        receipts=receipts, cohorts={g: summarize(rows) for g, rows in combined.items()}, actor_day=cells,
        cell_mean_intervals=audit.inference.describe(inference_rows, 'values'),
        replayed_consumed_paths=len(ITEMS), new_demand_days=0, altered_action_rollouts=0,
        actor_updates=0, critic_updates=0, dual_updates=0, quota_changed=False, main_policy_changed=False,
        limitations='Identity matching compares the frozen predicted trip/bus with the next actual service. '
            'Same-target pair signs are not actual neighbouring-headway CV2 when identities differ. '
            'All restoration signs hold other completed times fixed; no changed-action closed-loop return '
            'was evaluated. Ready-to-departure timing includes holding and execution ticks. '
            'Post-hoc restoration cohorts select on realized geometry, not independent predictive validation. '
            'Two archived actors/four consumed days; missing labels have observed counts, never zero errors.'))
    print('V118_PASSIVE_IDENTITY_REPLAY_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v115-root', type=Path, required=True)
    p.add_argument('--v116-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v118_*.py', '-v'], check=True)
    run_item(a.out_root, a.v115_root, 0)
    print('V118_FIRST_EXACT_PATH_PREFLIGHT_PASSED', flush=True)
    with ProcessPoolExecutor(max_workers=min(a.workers, len(ITEMS)-1), mp_context=get_context('spawn')) as pool:
        list(pool.map(run_item, itertools.repeat(a.out_root), itertools.repeat(a.v115_root), range(1, len(ITEMS))))
    aggregate(a.out_root, a.v116_root)
