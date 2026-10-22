#!/usr/bin/env python3
"""Persist consumed V211 selection paths without changing their decisions."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v211_joint_resources as prior


EPS = 1e-8
BASE = prior.JointAudit
SMOKE = ((0, 0, 0), (0, 0, 1), (2, 0, 9), (3, 0, 14))


def key(row):
    return int(row['trip_id']), int(row['station_id']), bool(row['direction'])


def save_trace(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(data, stream, separators=(',', ':'), allow_nan=False)


def load_trace(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def public_plan(env):
    return {int(t.launch_turn): dict(trip_id=int(t.launch_turn), direction=bool(t.direction),
        launch_s=float(t.launch_time), planned_s=float(getattr(t, '_freqduet_scheduled_launch', t.launch_time)),
        target_s=float(t.target_headway)) for t in env.timetables if not t.launched}


def plan_changes(before, after):
    prior.require(before.keys() == after.keys(), 'upper callback unexpectedly changed launched-trip set')
    return [dict(trip_id=k, before=before[k], after=after[k]) for k in before if before[k] != after[k]]


def mechanical_gain(forward, follower, target, action):
    if target <= 0 or min(forward, follower) < 0 or action > follower:
        return None
    return float(((forward-target)**2+(follower-target)**2
        -(forward+action-target)**2-(follower-action-target)**2)/(2*target**2))


def unique_decisions(rows):
    result = {key(r): r for r in rows}
    prior.require(len(result) == len(rows), 'repeated lower decision identity needs explicit occurrence accounting')
    return result


def completed_labels(decisions, departures):
    """Offline ready-neighbour labels; none of these labels feed a decision."""
    unique_decisions(decisions)
    departed = {key(r): r for r in departures}
    groups = defaultdict(list)
    for r in decisions:
        groups[r['station_id'], r['direction']].append(r)
    labels = {}
    for rows in groups.values():
        rows = sorted(rows, key=lambda r: (r['time_s'], r['ordinal']))
        for index, r in enumerate(rows):
            k, context = key(r), r['context']
            following = rows[index+1] if index+1 < len(rows) else None
            observed = (following is not None and following['bus_id'] != r['bus_id']
                and k in departed and context['evidence_valid'])
            gap = following['time_s']-r['time_s'] if observed else None
            gain = mechanical_gain(context['forward_headway_s'], gap,
                context['target_headway_s'], r['command_s']) if observed else None
            labels[k] = dict(follower_ready_gap_s=gap,
                follower_trip_id=following['trip_id'] if observed else None,
                follower_label_available_s=following['time_s'] if observed else None,
                own_departure_s=departed[k]['time_s'] if k in departed else None,
                ready_stage_gain=gain)
    return labels


class PassiveAudit(BASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries, self.upper_calls, self.upper_commands = [], [], []
        self.completed_departures, self.departure_cursor = {}, 0

    def attach(self, runner):
        super().attach(runner)
        self.passive_upper = runner._upper_callback_v2
        self.passive_action = runner.upper_trainer.policy_net.get_action

        def action(*args, **kwargs):
            value = self.passive_action(*args, **kwargs)
            trip = self.current_upper_trip
            self.upper_commands.append(dict(time_s=float(self.env.current_time),
                trip_id=int(trip.launch_turn) if trip is not None else None,
                coefficient=np.asarray(value).reshape(-1).tolist()))
            return value

        def upper(state, trip):
            before, count = public_plan(self.env), len(self.upper_commands)
            value = self.passive_upper(state, trip)
            plans = runner._active_timetable_plans
            active = [dict(key=str(k), origin_s=float(v['origin']),
                plan_id=int(v.get('plan_id', -1)), action=np.asarray(v['action']).reshape(-1).tolist())
                for k, v in plans.items()]
            self.upper_calls.append(dict(time_s=float(self.env.current_time), trip_id=int(trip.launch_turn),
                direction=bool(trip.direction), returned_target_s=float(value),
                fresh_commands=self.upper_commands[count:], active_plans=active,
                changes=plan_changes(before, public_plan(self.env))))
            return value

        runner._upper_callback_v2 = upper
        runner.upper_trainer.policy_net.get_action = action

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        for row in self.departures[self.departure_cursor:]:
            self.completed_departures[key(row)] = row
        self.departure_cursor = len(self.departures)
        predecessor = int(bus.forward_predecessor_trip_id) if bus.forward_predecessor_trip_id is not None else None
        previous = self.completed_departures.get((predecessor, int(bus.last_station.station_id), bool(bus.direction)))
        feasible = policy.feasible_action_mask(states)[0].numpy()
        base_bin = prior.prior.rule_action(context, 'bilateral', .25, feasible)
        command = float(prior.pilot.step.ACTIONS[int(law.argmax(dim=1).item())])
        launch = self.launches[int(bus.trip_id)]
        self.queries.append(dict(ordinal=len(self.queries), time_s=float(self.env.current_time),
            trip_id=int(bus.trip_id), station_id=int(bus.last_station.station_id), direction=bool(bus.direction),
            bus_id=int(bus.bus_id), command_s=command, base_bin=base_bin,
            context=asdict(context), onboard=len(bus.passengers),
            predecessor_trip_id=predecessor, predecessor_departure=previous,
            launch_s=launch['time_s'],
            executed_holding_before_s=float(bus.episode_hold_vehicle_seconds)-launch['holding_counter_at_launch']))
        return law

    def detach(self):
        self.runner._upper_callback_v2 = self.passive_upper
        self.runner.upper_trainer.policy_net.get_action = self.passive_action
        super().detach()

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        prior.require(len(records) == len(self.queries), 'passive lower query count differs')
        for query, record in zip(self.queries, records):
            prior.require(key(query) == key(record) and query['time_s'] == record['time_s']
                and query['command_s'] == prior.pilot.step.ACTIONS[record['action']], 'passive action differs from actual sample')
        prior.require(abs(sum(q['command_s'] for q in self.queries)-sum(h['action_s'] for h in holds)) <= EPS,
            'sampled and issued holding totals differ')
        self.full_trace = dict(lower_queries=self.queries, issued_holding=holds, upper_callbacks=self.upper_calls,
            actual_launches=self.dispatch_ledger, vehicle_returns=self.return_ledger,
            arrivals=self.arrivals, departures=self.departures)


def replay(root, ci, future, candidate, reference, label):
    # Copy V211's replay assembly to inject only the passive observer, not a policy.
    seed, day = prior.CONTEXTS[ci]
    sampler = 187000000+seed+100*day
    uniforms, credit = prior.pilot.event.EventUniforms(sampler), prior.pilot.CreditObserver()
    rule = PassiveAudit(seed, candidate, reference=reference, fork=prior.future_seed(ci, future), uniforms=uniforms)
    row, events, _, _ = prior.pilot.rollout(root, seed, day, 'stochastic', sampler, [0.]*4, label,
        probability_rule=rule, trace_observer=rule, event_uniforms=uniforms, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s,
        inclusive_launch_delay_s=rule.ledger['inclusive_launch_delay_s'], **rule.dispatch)
    result = dict(context=ci, seed=seed, day=day, candidate=candidate, continuation='bilateral025',
        future_seed=prior.future_seed(ci, future), results={k: row[k] for k in (*prior.METRICS, 'action_counts')},
        anchor=rule.anchor, upper=rule.applied_upper, anchor_command=rule.anchor_command, lower=rule.lower_changes,
        ledger=rule.ledger, authority=rule.authority, physical_contract=rule.contract)
    if hasattr(rule, 'execution'):
        result['execution'] = rule.execution
    return result, rule, events


def check_cached(actual, cached):
    prior.require(set(cached) <= set(actual), 'replay omitted an archived field')
    prior.pilot.joint.assert_same_state({k: actual[k] for k in cached}, cached, 'V213_archived_selection_path')


def trace_path(root, ci, future, candidate):
    return root/'records'/f'c{ci}_f{future}_a{candidate}.json.gz'


def collect(root, old, ci, future, candidates):
    group = prior.read(old/'select'/f'c{ci}_f{future}.json')
    reference = prior.read(old/'references'/f'c{ci}.json')
    checked = []
    for candidate in candidates:
        path = trace_path(root, ci, future, candidate)
        if not path.exists():
            actual, rule, events = replay(root, ci, future, candidate, reference, f'v213_c{ci}_f{future}_a{candidate}')
            check_cached(actual, group['rows'][candidate])
            save_trace(path, dict(result={k: v for k, v in actual.items() if k != 'anchor'}, trace=rule.full_trace,
                archived_endpoints_and_saved_actions_exact=True))
        checked.append(candidate)
    return checked


def compare_launches(base, candidate):
    b = {r['trip']: r for r in base}
    c = {r['trip']: r for r in candidate}
    common = b.keys() & c.keys()
    return dict(matched=len(common), unmatched_base=len(b.keys()-c.keys()), unmatched_candidate=len(c.keys()-b.keys()),
        changed_time=sum(abs(c[k]['actual_s']-b[k]['actual_s']) > EPS for k in common),
        changed_vehicle=sum(c[k]['bus'] != b[k]['bus'] for k in common),
        absolute_time_delta_s=sum(abs(c[k]['actual_s']-b[k]['actual_s']) for k in common),
        lateness_delta_s=sum(max(0., c[k]['actual_s']-c[k]['public_baseline_s'])
            -max(0., b[k]['actual_s']-b[k]['public_baseline_s']) for k in common))


def paired_timeline(base, candidate):
    b, c = base['trace'], candidate['trace']
    bm, cm = unique_decisions(b['lower_queries']), unique_decisions(c['lower_queries'])
    labels = completed_labels(c['lower_queries'], c['departures'])
    overrides = {tuple(r['identity'][:3]) for r in candidate['result']['lower']}
    cutoff = max(r['time_s'] for r in (*base['result']['lower'], *candidate['result']['lower']))
    counts, seconds = Counter(), Counter()
    counts['matched'] = len(bm.keys() & cm.keys())
    counts['unmatched_base'] = len(bm.keys()-cm.keys())
    counts['unmatched_candidate'] = len(cm.keys()-bm.keys())
    for k in bm.keys() | cm.keys():
        bv, cv = bm.get(k), cm.get(k)
        delta = (cv['command_s'] if cv else 0.)-(bv['command_s'] if bv else 0.)
        seconds['all_command_delta_s'] += delta
        if bv is None or cv is None:
            seconds['unmatched_command_delta_s'] += delta
            continue
        registered = k in overrides
        both_after = min(bv['time_s'], cv['time_s']) > cutoff
        period = 'registered' if registered else 'post_second' if both_after else 'other'
        seconds[period+'_command_delta_s'] += delta
        if abs(delta) <= EPS:
            continue
        counts[period+'_changed'] += 1
        if period != 'post_second' or delta <= EPS:
            continue
        counts['post_second_increases'] += 1
        seconds['post_second_positive_command_delta_s'] += delta
        seconds['post_second_onboard_exposure_proxy_delta_person_s'] += cv['onboard']*cv['command_s']-bv['onboard']*bv['command_s']
        label = labels[k]
        gain = label['ready_stage_gain']
        marginal = None
        if gain is not None:
            context = cv['context']
            baseline_gain = mechanical_gain(context['forward_headway_s'], label['follower_ready_gap_s'],
                context['target_headway_s'], bv['command_s'])
            marginal = gain-baseline_gain if baseline_gain is not None else None
        category = 'unresolved' if marginal is None else 'compensatory' if marginal > EPS else (
            'locally_counterproductive' if marginal < -EPS else 'neutral')
        counts['post_second_increase_'+category] += 1
        seconds['post_second_increase_'+category+'_s'] += delta
    accounted = sum(seconds[k] for k in ('registered_command_delta_s', 'post_second_command_delta_s',
        'other_command_delta_s', 'unmatched_command_delta_s'))
    prior.require(abs(accounted-seconds['all_command_delta_s']) <= EPS, 'timeline holding partition does not close')
    expected = candidate['result']['ledger']['commanded_holding_s']-base['result']['ledger']['commanded_holding_s']
    prior.require(abs(expected-seconds['all_command_delta_s']) <= EPS, 'timeline holding does not reproduce V211')
    plan_b = sum(bool(r['changes']) for r in b['upper_callbacks'])
    plan_c = sum(bool(r['changes']) for r in c['upper_callbacks'])
    return dict(counts=dict(counts), seconds=dict(seconds), common_cutoff_s=cutoff,
        launches=compare_launches(b['actual_launches'], c['actual_launches']),
        plan_update_calls_base=plan_b, plan_update_calls_candidate=plan_c,
        effects={k: candidate['result']['results'][k]-base['result']['results'][k]
            for k in prior.METRICS})


def group_summary(root, ci, future):
    paths = [load_trace(trace_path(root, ci, future, a)) for a in range(15)]
    contrasts = {}
    for u, l in prior.CANDIDATES:
        if l:
            contrasts[f'lower{l}_at_upper{u}'] = paired_timeline(paths[5*u], paths[5*u+l])
        elif u:
            contrasts[f'upper{u}_only'] = paired_timeline(paths[0], paths[5*u])
    prior.pilot.write_json(root/'groups'/f'c{ci}_f{future}.json', dict(context=ci, future=future,
        paths=15, cached_exact=True, contrasts=contrasts,
        trace_bytes=sum(trace_path(root, ci, future, a).stat().st_size for a in range(15))))


def run_group(root, old, index):
    prior.require(prior.read(root/'smoke/done.json')['status'] == 'v213_passive_equivalence_passed', 'smoke not passed')
    ci, future = divmod(index, 16)
    collect(root, old, ci, future, range(15))
    group_summary(root, ci, future)
    print('V213_GROUP_DONE', index, flush=True)


def smoke(root, old):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v213_*.py', '-v'], check=True)
    prior.require(not (root/'protocol.json').exists(), 'do not overwrite V213 registration')
    prior.pilot.write_json(root/'protocol.json', dict(stage='v213_passive_consumed_selection_timeline',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], contexts=[list(c) for c in prior.CONTEXTS],
        futures=list(range(16)), candidates=list(range(15)), paths=960, smoke=[list(v) for v in SMOKE],
        old_root=str(old), evaluation_opened=False, training_updates=0, main_policy_changed=False,
        protocol='Reuse exact V211 reference prefixes and selection future namespaces. No selector update, '
            'new future, action override, authority change or field supplied to the controller. '
            'Test passive versus original full event/action/timing paths, then match every archived scalar/ledger/command. '
            'Split matched command differences into registered,after both second interventions,and other;unmatched separately. '
            'Completed ready-neighbour marginal squared-gap gain is a mechanical diagnostic,not a causal action value.'))
    for ci, future, candidate in SMOKE:
        reference = prior.read(old/'references'/f'c{ci}.json')
        native, native_rule, native_events = prior.replay(root, ci, candidate, reference=reference,
            future=prior.future_seed(ci, future), label=f'passive_off_c{ci}_f{future}_a{candidate}')
        passive, rule, events = replay(root, ci, future, candidate, reference, f'passive_on_c{ci}_f{future}_a{candidate}')
        check_cached(passive, prior.read(old/'select'/f'c{ci}_f{future}.json')['rows'][candidate])
        prior.pilot.joint.assert_same_state(passive, native, 'V213_passive_full_result')
        prior.pilot.joint.assert_same_state(events, native_events, 'V213_passive_full_action_event_path')
        for name in ('arrivals', 'departures', 'dispatch_ledger', 'return_ledger'):
            prior.pilot.joint.assert_same_state(getattr(rule, name), getattr(native_rule, name), 'V213_passive_'+name)
        save_trace(trace_path(root, ci, future, candidate), dict(result={k: v for k, v in passive.items() if k != 'anchor'},
            trace=rule.full_trace, archived_endpoints_and_saved_actions_exact=True))
    prior.pilot.write_json(root/'smoke/done.json', dict(status='v213_passive_equivalence_passed',
        smoke_paths=len(SMOKE), rollouts=2*len(SMOKE), full_actions_and_timing_exact=True,
        archived_endpoints_exact=True, training_updates=0, main_policy_changed=False))
    print('V213_SMOKE_PASSED', flush=True)


def aggregate(root):
    rows = [prior.read(root/'groups'/f'c{c}_f{f}.json') for c, f in itertools.product(range(4), range(16))]
    summary = {}
    for contrast in rows[0]['contrasts']:
        chosen = [r['contrasts'][contrast] for r in rows]
        counts, launches = Counter(), Counter()
        for r in chosen:
            counts.update(r['counts'])
            launches.update(r['launches'])
        names = set().union(*(r['seconds'] for r in chosen))
        metrics = [r['effects'] for r in chosen]
        summary[contrast] = dict(counts=dict(counts), launches=dict(launches),
            mean_seconds={k: float(np.mean([r['seconds'].get(k, 0.) for r in chosen])) for k in names},
            mean_effects={k: float(np.mean([r[k] for r in metrics])) for k in prior.METRICS},
            context_mean_effects=[dict(context=c, **{k: float(np.mean([r['contrasts'][contrast]['effects'][k]
                for r in rows if r['context'] == c])) for k in prior.METRICS}) for c in range(4)],
            paths_with_changed_actual_launches=sum(r['launches']['changed_time'] > 0 for r in chosen),
            paths_with_changed_vehicle_assignment=sum(r['launches']['changed_vehicle'] > 0 for r in chosen))
    prior.pilot.write_json(root/'summary/status.json', dict(status='v213_passive_selection_timelines_complete',
        groups=64, paths=960, all_cached_endpoints_and_saved_commands_exact=all(r['cached_exact'] for r in rows),
        smoke=prior.read(root/'smoke/done.json'), contrasts=summary, trace_bytes=sum(r['trace_bytes'] for r in rows),
        consumed_selection_futures=list(range(16)), evaluation_files_opened=0, training_updates=0,
        main_policy_changed=False, fleet=12, lower_bins_s=prior.pilot.step.ACTIONS.tolist(),
        upper_authority='unchanged [-60,60],shared antisymmetric coefficient,EMA and rolling zero-sum plan',
        limitations='No individual rebound suppression replay:mechanical compensation labels and full-day '
            'outcomes establish associations,not whether each later action is causally necessary or harmful. '
            'Old full trajectories were not archived;all old stored fields match,and full passive/off paths match in smoke.'))
    print('V213_PASSIVE_AGGREGATE_DONE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v211-root', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=64)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.command == 'smoke':
        smoke(a.out_root, a.v211_root)
    elif a.command == 'aggregate':
        aggregate(a.out_root)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(run_group, itertools.repeat(a.out_root), itertools.repeat(a.v211_root), range(a.start, a.end)))
        prior.pilot.write_json(a.out_root/'shards'/f'{a.start}-{a.end}.json', dict(status='complete', groups=a.end-a.start))


if __name__ == '__main__':
    main()
