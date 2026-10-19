#!/usr/bin/env python3
"""Recover decision evidence on two exact, consumed V130 candidate paths."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v118_follower_identity as identity
import run_freqduet_v130_plan_terminal as original

pilot = original.frozen.pilot
ITEMS = ((12011, 2231004), (933013, 2231004))


def key(row):
    return int(row['trip_id']), int(row['station_id']), bool(row['direction'])


def snapshot_avl(bus):
    snapshots = getattr(bus, '_follower_avl_snapshots', ())
    behind = sorted((s for s in snapshots if s.on_route and s.direction == bus.direction
        and s.bus_id != bus.bus_id and s.progress_m < bus.travel_distance-1e-6),
        key=lambda s: -s.progress_m)
    follower = next((s for s in snapshots if s.bus_id == bus.pre_action_follower_bus_id), None)
    return dict(controlled_progress_m=float(bus.travel_distance),
        estimator_follower_trip_id=bus.pre_action_follower_trip_id,
        follower_snapshot=None if follower is None else asdict(follower),
        nearest_active_followers=[asdict(s) for s in behind[:3]])


class PassiveTrace(original.PlanAudit):
    def __init__(self, seed, geometry):
        super().__init__(seed, geometry, original.frozen.PRIMARY, True)
        self.ready_avl, self.queries = {}, {}

    def attach(self, runner):
        super().attach(runner)
        self.trace_prepare = self.bus_type._prepare_for_action

        def prepare(bus, now, buses, debug):
            result = self.trace_prepare(bus, now, buses, debug)
            stop = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
            self.ready_avl[stop] = snapshot_avl(bus)
            return result

        self.bus_type._prepare_for_action = prepare

    def detach(self):
        self.bus_type._prepare_for_action = self.trace_prepare
        super().detach()

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        stop = key(self.decisions[-1])
        if stop in self.queries:
            raise AssertionError('duplicate lower query at a consumed trip/stop')
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        # Copy already-produced evidence only; never query the predictor or actor again.
        self.queries[stop] = deepcopy(dict(evidence=self.evidence[stop],
            online=self.online[stop], context=asdict(context), ready_avl=self.ready_avl[stop]))
        return law


def assert_arrays_and_actions(before_path, after_path, decisions):
    with np.load(before_path) as before, np.load(after_path) as after:
        if before.files != after.files:
            raise AssertionError('passive observation changed timing fields')
        for name in before.files:
            np.testing.assert_array_equal(before[name], after[name])
        fields = before['event_fields'].tolist()
        arrivals = {key(dict(zip(fields, r))): r[fields.index('realized_holding')]
            for r in before['arrivals']}
        actions = {key(dict(zip(fields, r))): r[fields.index('realized_holding')]-arrivals[key(dict(zip(fields, r)))]
            for r in before['departures'] if key(dict(zip(fields, r))) in arrivals}
    for row in decisions:
        if row['chosen_s'] != actions[key(row)] or row['executed_s'] != actions[key(row)]:
            raise AssertionError('passive query changed an original executed lower action')


def replay(args):
    seed, day = ITEMS[args.index]
    label = f'{seed}_{day}_split_plan'
    rule = PassiveTrace(seed, original.frozen.prior.model.load_geometry(args.runtime_root))
    credit = pilot.CreditObserver()
    result, _, _, _ = pilot.rollout(args.out_root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, 'v130_'+label, probability_rule=rule, trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    values = {k: result[k] for k in original.frozen.prior.METRICS}
    values.update({k: rule.terminal[k] for k in ('inclusive_terminal_delay_s', 'qualification_shift_s')})
    actual = dict(seed=seed, day=day, method='split_plan', values=values,
        terminal=rule.terminal, contract=rule.contract, neutral_exact=False)
    # Consumed future labels are opened only after the frozen live rollout finishes.
    cached = json.loads((args.v130_root/'pairs'/f'p{original.ITEMS.index((seed, day))}.json').read_text())
    old = next(r for r in cached['paths'] if r['method'] == 'split_plan')
    pilot.joint.assert_same_state(old, actual, 'V132_original_V130_endpoints')
    terminal = dict(plans=rule.plans, dispatches=rule.dispatches,
        returns=list(rule.returns.values()), reference_clocks=rule.reference_clocks)
    terminal_path = args.out_root/'terminal'/(label+'.json')
    pilot.write_json(terminal_path, terminal)
    pilot.joint.assert_same_state(json.loads((args.v130_root/'terminal'/(label+'.json')).read_text()),
        json.loads(terminal_path.read_text()), 'V132_original_terminal_path')
    record = args.out_root/'records'/(label+'.npz')
    rule.save(record)
    assert_arrays_and_actions(args.v130_root/'records'/(label+'.npz'), record, rule.decisions)
    rows = [dict(r, **rule.queries[key(r)]) for r in rule.decisions]
    for row in rows:
        if not row['prediction_time_s'] == row['evidence']['ready_time_s'] <= row['policy_time_s']:
            raise AssertionError('passive trace confused ready and query clocks')
    identity.write_trace(args.out_root/'traces'/(label+'.jsonl.gz'), rows)
    receipt = dict(seed=seed, day=day, endpoints_exact=True, full_timing_arrays_exact=True,
        terminal_commands_dispatches_returns_exact=True, all_realized_lower_actions_exact=True,
        decisions=len(rows), values=values, actor_updates=0, critic_updates=0, dual_updates=0)
    pilot.write_json(args.out_root/'shards'/str(args.index)/'done.json', receipt)
    print('V132_EXACT_PASSIVE_PATH_DONE', json.dumps(receipt), flush=True)


def brief(row):
    names = ('trip_id', 'station_id', 'direction', 'policy_time_s', 'chosen_s', 'executed_s',
        'raw_shadow_s', 'feedback_error_s', 'raw_feedback_error_s', 'prediction_time_s',
        'component_used', 'request_gap_s', 'evidence', 'online', 'context', 'ready_avl')
    result = {k: row[k] for k in names if k in row}
    result['unclipped_request_s'] = .25*row['feedback_error_s']
    return result


def compare_paths(baseline, treated):
    old = {key(r): r for r in baseline}
    new = {key(r): r for r in treated}
    if len(old) != len(baseline) or len(new) != len(treated):
        raise AssertionError('duplicate trip/stop decision in path comparison')
    ordered = sorted(old.keys() & new.keys(), key=lambda k: (new[k]['policy_time_s'], k))
    changed = [k for k in ordered if old[k]['chosen_s'] != new[k]['chosen_s']]
    evidence_changed = [k for k in ordered if old[k]['evidence'] != new[k]['evidence']]
    def pair(k):
        return dict(baseline=brief(old[k]), candidate=brief(new[k]))
    transitions = Counter((old[k]['evidence']['follower_source'], new[k]['evidence']['follower_source'])
        for k in changed)
    return dict(common_decisions=len(ordered), baseline_only=len(old.keys()-new.keys()),
        candidate_only=len(new.keys()-old.keys()), changed_actions=len(changed),
        first_changed_queries=[pair(k) for k in changed[:6]],
        first_evidence_change=None if not evidence_changed else pair(evidence_changed[0]),
        changed_source_pairs=[dict(baseline=a, candidate=b, count=n)
            for (a, b), n in transitions.most_common()],
        changed_follower_ids=sum(old[k]['evidence']['predicted_follower_bus_id']
            != new[k]['evidence']['predicted_follower_bus_id'] for k in changed),
        total_chosen_delta_s=sum(new[k]['chosen_s'] for k in new)-sum(old[k]['chosen_s'] for k in old))


def aggregate(args):
    paths = []
    for index, (seed, day) in enumerate(ITEMS):
        receipt = json.loads((args.out_root/'shards'/str(index)/'done.json').read_text())
        if (receipt['seed'], receipt['day']) != (seed, day) or not all(receipt[k] for k in
                ('endpoints_exact', 'full_timing_arrays_exact', 'terminal_commands_dispatches_returns_exact',
                 'all_realized_lower_actions_exact')):
            raise AssertionError('passive path admission failed')
        baseline = identity.read_trace(args.v124_root/'traces'/f'{seed}_{day}_split_bilateral025_0.jsonl.gz')
        treated = identity.read_trace(args.out_root/'traces'/f'{seed}_{day}_split_plan.jsonl.gz')
        terminal = json.loads((args.out_root/'terminal'/f'{seed}_{day}_split_plan.json').read_text())
        commands = [r for r in terminal['plans'] if r['command_s'] > r['published_release_s']]
        paths.append(dict(receipt, comparison=compare_paths(baseline, treated),
            first_terminal_commands=commands[:4]))
    pilot.write_json(args.out_root/'summary/status.json', dict(status='v132_exact_passive_trace_complete',
        paths=paths, new_candidate_replays=2, reused_baseline_traces=2, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Two consumed paths only. Exact replay verifies passive instrumentation, not '
            'a new control effect. Ready-time AVL/prediction and later query clocks are distinct. '
            'Whole-path contrasts do not identify the daily value of one changed action.'))
    print('V132_EXACT_PASSIVE_TRACE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    parser.add_argument('--out-root', type=Path, required=True)
    for name in ('v124-root', 'v130-root', 'runtime-root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--index', type=int, choices=range(len(ITEMS)))
    args = parser.parse_args()
    if args.command == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v132_*.py', '-v'], check=True)
        pilot.write_json(args.out_root/'smoke/done.json', dict(status='passive_trace_tests_passed'))
    elif args.command == 'run':
        if not (args.out_root/'smoke/done.json').is_file():
            raise AssertionError('passive replay requires passed trace tests')
        replay(args)
    else:
        aggregate(args)
