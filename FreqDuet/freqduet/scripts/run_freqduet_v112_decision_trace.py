#!/usr/bin/env python3
"""Passive decision traces on consumed V110 paths, gated by exact reproduction."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v110_observed_disturbance as prior


pilot = prior.pilot
METHODS = ('bilateral025', prior.CANDIDATE)
ITEMS = list(itertools.product(prior.SEEDS, prior.DAYS, METHODS))
GROUPS = ('all', 'valid_forward', 'bilateral', 'fallback', 'missing_forward',
    'adjacent_same_predecessor', 'repeated_negative_after_holding',
    'repeated_negative_feedback_nonpositive', 'base_zero_override',
    'low_request', 'high_request', 'innovation_shadow_differs')


def command(desired, feasible):
    return float(pilot.step.ACTIONS[prior.quantized_action(desired, feasible)])


def decision_math(context, coordinates, feasible, candidate, previous=None):
    valid = bool(context.evidence_valid)
    bilateral = valid and bool(context.follower_evidence_valid)
    error = ((context.follower_departure_gap_s - context.forward_headway_s) / 2 if bilateral
             else context.target_headway_s - context.forward_headway_s) if valid else 0.
    residual = float(coordinates[2]) if valid else None
    base = .25 * error
    ff = -.75 * residual if valid else 0.
    desired = base + ff if candidate else base
    selected, shadow = command(desired, feasible), command(base, feasible)
    clipped = float(np.clip(desired, 0., 45.) - np.clip(base, 0., 45.))
    raw_delta = ff if candidate else 0.
    result = dict(valid_forward=valid, valid_bilateral=bilateral,
        forward_gap_s=float(context.forward_headway_s) if valid else None,
        follower_gap_prediction_s=float(context.follower_departure_gap_s) if bilateral else None,
        target_headway_s=float(context.target_headway_s), feedback_error_s=float(error),
        release_gap_s=float(coordinates[0]) if valid else None,
        executed_holding_gap_s=float(coordinates[1]) if valid else None,
        residual_s=residual, base_desired_s=float(base), desired_s=float(desired),
        proposed_feedforward_s=float(ff), actual_feedforward_s=float(raw_delta),
        chosen_s=selected, base_shadow_s=shadow,
        ff_shadow_s=command(base + ff, feasible),
        clipping_delta_s=clipped-raw_delta,
        quantization_delta_s=selected-shadow-clipped,
        previous_residual_s=previous, innovation_s=None, innovation_shadow_s=None,
        feasible_bins_s=pilot.step.ACTIONS[feasible].tolist())
    if valid and previous is not None:
        result['innovation_s'] = residual - previous
        result['innovation_shadow_s'] = command(base - .75 * (residual - previous), feasible)
    np.testing.assert_allclose(raw_delta + result['clipping_delta_s'] + result['quantization_delta_s'],
        selected-shadow, atol=1e-10, rtol=0)
    return result


def completed_follower(row, ready, departures, arrivals, next_arrival):
    key = row['follower_trip_id'], row['station_id'], row['direction']
    if not row['valid_bilateral'] or key not in ready:
        return None
    gap = ready[key] - row['ready_time_s']
    departure, arrival = departures.get(key), arrivals.get(key)
    return dict(ready_gap_s=gap, ready_label_temporal_valid=gap >= 0.,
        prediction_error_s=row['follower_gap_prediction_s']-gap,
        next_arrival_identity_matches=next_arrival.get((row['trip_id'], row['station_id'], row['direction'])) == key[0],
        follower_departure_gap_from_own_ready_s=departure['time_s']-row['ready_time_s'] if departure else None,
        follower_after_ready_elapsed_s=departure['time_s']-ready[key] if departure else None,
        follower_executed_holding_s=departure['realized_holding']-arrival['realized_holding']
            if departure and arrival else None)


class DecisionTrace(prior.DisturbanceAudit):
    def __init__(self, seed, method):
        super().__init__(seed, method)
        self.decisions, self.ready, self.previous = [], {}, {}
        self.ledger = defaultdict(lambda: dict(actual=0., shadow=0., positive_excess=0.))

    def attach(self, runner):
        super().attach(runner)
        from env.bus import Bus
        self.bus_type, self.original_prepare = Bus, Bus._prepare_for_action

        def prepare(bus, now, buses, debug):
            result = self.original_prepare(bus, now, buses, debug)
            key = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
            self.ready[key] = float(now)
            return result

        Bus._prepare_for_action = prepare

    def detach(self):
        self.bus_type._prepare_for_action = self.original_prepare
        super().detach()

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        key = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
        predecessor = int(bus.forward_predecessor_trip_id) if context.evidence_valid else None
        last = self.previous.get(key[0])
        previous = (last['residual_s'] if last is not None and last['valid_forward']
            and context.evidence_valid and last['predecessor_trip_id'] == predecessor
            and abs(last['station_id']-key[1]) == 1 else None)
        coordinates = self.observable_coordinates(bus, context) if context.evidence_valid else None
        row = decision_math(context, coordinates, policy.feasible_action_mask(states)[0].numpy(),
            self.variant == prior.CANDIDATE, previous)
        np.testing.assert_array_equal(law[0].numpy(), np.eye(7)[np.flatnonzero(pilot.step.ACTIONS == row['chosen_s'])[0]])
        launch = self.launches[key[0]]
        held = float(bus.episode_hold_vehicle_seconds) - launch['holding_counter_at_launch']
        ledger = self.ledger[key[0]]
        np.testing.assert_allclose(held, ledger['actual'], atol=1e-9, rtol=0)
        row.update(trip_id=key[0], station_id=key[1], direction=key[2],
            bus_id=int(bus.bus_id), ready_time_s=self.ready[key], policy_time_s=float(self.env.current_time),
            predecessor_trip_id=predecessor,
            follower_trip_id=int(bus.pre_action_follower_trip_id) if bus.pre_action_follower_trip_id is not None else None,
            follower_source=str(bus.pre_action_follower_source),
            own_executed_holding_s=held, prior_shadow_baseline_sum_s=ledger['shadow'],
            prior_signed_excess_s=held-ledger['shadow'], prior_positive_excess_s=ledger['positive_excess'])
        if coordinates is not None:
            before = self.completed[predecessor, key[1], key[2]]
            np.testing.assert_allclose(before['time_s'] + context.forward_headway_s,
                row['ready_time_s'], atol=1e-9, rtol=0)
        self.decisions.append(row)
        self.previous[key[0]] = row
        ledger['actual'] += row['chosen_s']
        ledger['shadow'] += row['base_shadow_s']
        ledger['positive_excess'] += max(row['chosen_s']-row['base_shadow_s'], 0.)
        return law

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        self.policy_records = records

    def complete_trace(self):
        if len(self.decisions) != len(self.policy_records):
            raise AssertionError('passive trace is not aligned with all sampled decisions')
        fields = ('trip_id', 'station_id', 'direction')
        departures = {tuple(r[k] for k in fields): r for r in self.departures}
        arrivals = {tuple(r[k] for k in fields): r for r in self.arrivals}
        next_arrival, previous = {}, {}
        for arrival in self.arrivals:
            key = tuple(arrival[k] for k in fields)
            if key[1:] in previous:
                next_arrival[previous[key[1:]]] = key[0]
            previous[key[1:]] = key
        for row, sampled in zip(self.decisions, self.policy_records):
            key = tuple(row[k] for k in fields)
            if key != tuple(sampled[k] for k in fields) or row['chosen_s'] != pilot.step.ACTIONS[sampled['action']]:
                raise AssertionError('trace changed a sampled command or decision identity')
            np.testing.assert_allclose(departures[key]['realized_holding']-arrivals[key]['realized_holding'],
                row['chosen_s'], atol=1e-9, rtol=0)
            row['completed_follower'] = completed_follower(row, self.ready, departures, arrivals, next_arrival)
            label = row['completed_follower']
            if label is not None and label['ready_label_temporal_valid']:
                error = (label['ready_gap_s']-row['forward_gap_s']) / 2
                desired = .25*error + row['actual_feedforward_s']
                mask = np.isin(pilot.step.ACTIONS, row['feasible_bins_s'])
                label.update(completed_feedback_error_s=error, completed_desired_s=desired,
                    completed_shadow_command_s=command(desired, mask))
                np.testing.assert_allclose(row['desired_s']-desired,
                    .125*label['prediction_error_s'], atol=1e-9, rtol=0)


def summarize(rows):
    groups = {g: dict(count=0, chosen_sum_s=0., base_shadow_sum_s=0., command_delta_sum_s=0.,
        raw_feedforward_sum_s=0., clipping_delta_sum_s=0., quantization_delta_sum_s=0.,
        proposed_feedforward_abs_sum_s=0., innovation_abs_sum_s=0., innovation_observed=0) for g in GROUPS}
    follower = dict(eligible=0, resolved=0, temporal_invalid=0, prediction_error_sum_s=0.,
        absolute_error_sum_s=0., squared_error_sum_s2=0., next_identity_mismatch=0,
        feedback_sign_reversal=0, completed_shadow_differs=0, completed_shadow_delta_sum_s=0.,
        future_holding_sum_s=0., future_holding_observed=0)
    for row in rows:
        valid, bilateral = row['valid_forward'], row['valid_bilateral']
        repeated = (row['previous_residual_s'] is not None and row['previous_residual_s'] < 0
            and row['residual_s'] < 0 and row['own_executed_holding_s'] > 0)
        selected = ['all', 'valid_forward' if valid else 'missing_forward']
        if valid:
            selected.append('bilateral' if bilateral else 'fallback')
            if row['previous_residual_s'] is not None:
                selected.append('adjacent_same_predecessor')
                if row['innovation_shadow_s'] != row['ff_shadow_s']:
                    selected.append('innovation_shadow_differs')
            if repeated:
                selected.append('repeated_negative_after_holding')
                if row['feedback_error_s'] <= 0 and row['chosen_s'] > 0:
                    selected.append('repeated_negative_feedback_nonpositive')
            if row['base_shadow_s'] == 0 and row['chosen_s'] > 0:
                selected.append('base_zero_override')
            if row['desired_s'] <= 0:
                selected.append('low_request')
            if row['desired_s'] >= 45:
                selected.append('high_request')
        for name in selected:
            g = groups[name]
            g['count'] += 1
            for out, field in (('chosen_sum_s', 'chosen_s'), ('base_shadow_sum_s', 'base_shadow_s'),
                ('raw_feedforward_sum_s', 'actual_feedforward_s'), ('clipping_delta_sum_s', 'clipping_delta_s'),
                ('quantization_delta_sum_s', 'quantization_delta_s')):
                g[out] += row[field]
            g['command_delta_sum_s'] += row['chosen_s']-row['base_shadow_s']
            g['proposed_feedforward_abs_sum_s'] += abs(row['proposed_feedforward_s'])
            if row['innovation_s'] is not None:
                g['innovation_observed'] += 1
                g['innovation_abs_sum_s'] += abs(row['innovation_s'])
        follower['eligible'] += int(bilateral)
        label = row['completed_follower']
        if label is None:
            continue
        if not label['ready_label_temporal_valid']:
            follower['temporal_invalid'] += 1
            continue
        follower['resolved'] += 1
        err = label['prediction_error_s']
        follower['prediction_error_sum_s'] += err
        follower['absolute_error_sum_s'] += abs(err)
        follower['squared_error_sum_s2'] += err**2
        follower['next_identity_mismatch'] += int(not label['next_arrival_identity_matches'])
        follower['feedback_sign_reversal'] += int(row['feedback_error_s']*label['completed_feedback_error_s'] < 0)
        follower['completed_shadow_differs'] += int(row['chosen_s'] != label['completed_shadow_command_s'])
        follower['completed_shadow_delta_sum_s'] += row['chosen_s']-label['completed_shadow_command_s']
        if label['follower_executed_holding_s'] is not None:
            follower['future_holding_observed'] += 1
            follower['future_holding_sum_s'] += label['follower_executed_holding_s']
    return dict(groups=groups, follower=follower)


def assert_timing(rule, old_path):
    with np.load(old_path) as old:
        for kind, rows in (('arrivals', rule.arrivals), ('departures', rule.departures)):
            fields = old['event_fields'].tolist()
            np.testing.assert_array_equal([[r[k] for k in fields] for r in rows], old[kind])
        np.testing.assert_array_equal([[trip, r['bus_id'], r['time_s'], r['readiness_s']]
            for trip, r in rule.launches.items()], old['launches'])


def run_item(root, old, index):
    path = root / 'pairs' / f'p{index}.json'
    if path.exists():
        return
    seed, day, method = ITEMS[index]
    cached = json.loads((old / 'pairs' / f'p{prior.ITEMS.index((seed, day, method, 0))}.json').read_text())
    rule, credit = DecisionTrace(seed, method), pilot.CreditObserver()
    row, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v112_{seed}_{day}_{method}', probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    current = dict(seed=seed, day=day, method=method, rep=0, results=row,
        contract=rule.contract, coverage=rule.coverage, feedforward=rule.feedforward)
    pilot.joint.assert_same_state(current, cached, 'V112_exact_V110_endpoints_and_audits')
    assert_timing(rule, old / 'records' / f'{seed}_{day}_{method}_0.npz')
    # Future labels and interpreted summaries are only formed after reproduction passes.
    rule.complete_trace()
    trace = root / 'traces' / f'p{index}.jsonl.gz'
    trace.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(trace, 'wt', encoding='ascii') as output:
        for decision in rule.decisions:
            output.write(json.dumps(decision, allow_nan=False)+'\n')
    pilot.write_json(path, dict(index=index, seed=seed, day=day, method=method, rep=0,
        exact_v110_endpoints_and_timing=True, decisions=len(rule.decisions), diagnostics=summarize(rule.decisions)))
    print('V112_TRACE_PATH_EXACT', index, flush=True)


def aggregate(root):
    rows = [json.loads((root / 'pairs' / f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [(r['seed'], r['day'], r['method']) for r in rows] != ITEMS or not all(r['exact_v110_endpoints_and_timing'] for r in rows):
        raise AssertionError('V112 lacks an exact replay of every selected consumed path')
    def pool(chosen):
        return dict(groups={g: {k: sum(r['diagnostics']['groups'][g][k] for r in chosen)
            for k in chosen[0]['diagnostics']['groups'][g]} for g in GROUPS},
            follower={k: sum(r['diagnostics']['follower'][k] for r in chosen)
                for k in chosen[0]['diagnostics']['follower']})
    pilot.write_json(root / 'summary/status.json', dict(status='consumed_v110_decision_trace_complete',
        methods={m: pool([r for r in rows if r['method'] == m]) for m in METHODS},
        actor_day=[{k: r[k] for k in ('seed', 'day', 'method', 'diagnostics')} for r in rows],
        consumed_path_replays=len(rows), fresh_demand_days=0, sampler_rep=0,
        all_cached_endpoints_audits_and_timing_exact=True, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, v110_candidate_closed=True, controller_admitted=False,
        scope='two actors and four consumed development days; descriptive state/action accounting, not causal mediation',
        limitations='Post-completion follower labels and innovation shadows never drive actions. '
            'Shadow excess is relative to the original rule on the same observed states, not a causal compensation ledger. '
            'Forecast error, repeated level response and clipping overlap and are not additive causes of service loss.'))
    print('V112_DECISION_TRACE_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v110-root', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(ITEMS))
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.command == 'aggregate':
        aggregate(a.out_root)
    else:
        if a.command == 'smoke':
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                str(Path(__file__).resolve().parents[1] / 'tests'), '-p', 'test_v112_*.py', '-v'], check=True)
        indices = range(2) if a.command == 'smoke' else range(a.start, a.end)
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(run_item, itertools.repeat(a.out_root), itertools.repeat(a.v110_root), indices))
        marker = ('smoke/done.json' if a.command == 'smoke' else f'shards/{a.start}-{a.end}/done.json')
        pilot.write_json(a.out_root / marker, dict(status='exact_cached_replays_complete',
            indices=list(indices), main_policy_changed=False))
