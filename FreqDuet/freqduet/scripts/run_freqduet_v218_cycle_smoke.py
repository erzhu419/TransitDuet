#!/usr/bin/env python3
"""Bounded four-arm execution smoke for one persistent resource-cycle law."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import freqduet_v218_cycle_policy as law
import run_freqduet_v216_passive_query as passive


pilot = passive.prior.pilot
METRICS = (*passive.prior.prior.METRICS, 'public_launch_delay_total_s', 'readiness_delay_total_s')


class CycleAudit(passive.QueryAudit):
    def __init__(self, seed, arm):
        super().__init__(seed, True)
        self.arm, self.cycles, self.responses, self.launch_plans = arm, [], [], {}

    def attach(self, runner):
        super().attach(runner)
        assert runner.timetable_replan_interval_s == law.WINDOW_S
        assert runner.timetable_action_ema_alpha == .35
        self.captured_action, self.captured_upper = runner.upper_trainer.policy_net.get_action, runner._upper_callback_v2
        self.captured_launch = runner.env.launch_bus

        def action(state, *args, **kwargs):
            native = self.captured_action(state, *args, **kwargs)
            if self.env.current_time < law.START_S:
                return native
            q = self.fresh_queries[-1]
            cycle = law.cycle_context(q)
            shift = cycle['shift_s'] if self.arm in ('upper', 'joint') else 0.
            chosen = passive.prior.upper_candidate(native, runner.upper_action_low, runner.upper_action_high, shift)
            cycle.update(native=np.asarray(native).reshape(-1).tolist(), command=chosen.tolist(),
                         applied_shift_s=float(chosen[0]-np.asarray(native).reshape(-1)[0]))
            self.cycles.append(cycle)
            self.commands[-1]['coefficient'] = chosen.tolist()
            return native if shift == 0 else chosen

        def upper(state, trip):
            count = len(self.cycles)
            value = self.captured_upper(state, trip)
            if len(self.cycles) > count:
                self.cycles[-1]['plan_changes'] = self.callbacks[-1]['changes']
                key = '__all__' if runner.timetable_plan_all_directions else bool(trip.direction)
                self.cycles[-1]['effective_action'] = np.asarray(
                    runner._active_timetable_plans[key]['action']).reshape(-1).tolist()
            return value

        def launch(trip, actual_launch=None):
            planned = float(getattr(trip, '_freqduet_scheduled_launch', trip.launch_time))
            bus = self.captured_launch(trip, actual_launch=actual_launch)
            if bus is not None:
                self.launch_plans[int(trip.launch_turn)] = planned
            return bus
        runner.upper_trainer.policy_net.get_action, runner._upper_callback_v2 = action, upper
        runner.env.launch_bus = launch

    def release_gap(self, query, bus):
        for row in self.departures[self.cursor:]:
            self.completed[row['trip_id'], row['station_id'], row['direction']] = row
        self.cursor = len(self.departures)
        predecessor = bus.forward_predecessor_trip_id
        prior = self.completed.get((predecessor, query['station_id'], query['direction']))
        if prior is None or not prior['time_s'] <= prior['time_s']+query['context']['forward_headway_s'] <= query['time_s']:
            return None
        return (self.launches[query['trip_id']]['time_s']-prior['release']
                +query['executed_holding_before_s']-prior['realized_holding'])

    def __call__(self, policy, states, features, theta):
        result = super().__call__(policy, states, features, theta)
        if not self.cycles:
            return result
        q = self.queries[-1]
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        gap = self.release_gap(q, bus) if q['forward_valid'] else None
        selected, reason, link = law.lower_request(q, self.cycles[-1], gap)
        base = q['base_bin']
        if self.arm not in ('lower', 'joint'):
            selected = base
        self.responses.append(dict(time_s=q['time_s'], event_seq=q['event_seq'], cycle_seq=self.cycles[-1]['event_seq'],
            bus_id=q['bus_id'], trip_id=q['trip_id'], station_id=q['station_id'], direction=q['direction'],
            base_bin=base, selected_bin=selected, reason=reason, link=link,
            release_holding_gap_s=gap, executed_before_s=q['executed_holding_before_s']))
        if selected != base:
            result = torch.zeros_like(result)
            result[0, selected] = 1.
            q['command_s'] = law.ACTIONS[selected]
        return result

    def detach(self):
        self.runner.upper_trainer.policy_net.get_action = self.captured_action
        self.runner._upper_callback_v2, self.env.launch_bus = self.captured_upper, self.captured_launch
        super().detach()


def common(rule, events, holds):
    return dict(lower_actions=rule.actions, lower_policy_inputs=rule.policy_inputs,
        upper_commands=rule.commands, upper_callbacks=rule.callbacks, issued_holding=holds,
        actual_launches=rule.dispatches, vehicle_returns=rule.returns, arrivals=rule.arrivals,
        departures=rule.departures, action_uniforms=[[list(k), v] for k, v in events.items()])


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def run_item(root, archived, item):
    ci, arm = item
    output = root/'records'/f'c{ci}_{arm}.json.gz'
    if output.exists():
        return
    progress = root/'progress'/f'c{ci}_{arm}.json'
    assert not progress.exists(), 'a started path failed; inspect it rather than automatically extend the budget'
    passive.write(progress, dict(context=ci, arm=arm, started=True))
    seed, day = law.CONTEXTS[ci]
    rule, credit = CycleAudit(seed, arm), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v218_c{ci}_{arm}', probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s,
        public_launch_delay_total_s=sum(max(0., r['actual_s']-r['public_baseline_s']) for r in rule.dispatches),
        readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in rule.dispatches), **rule.dispatch_metrics)
    data = dict(context=ci, arm=arm, results=row, ledger=rule.ledger, physical_contract=rule.contract,
        trace=common(rule, events, holds), cycles=rule.cycles, responses=rule.responses,
        launch_plans=rule.launch_plans, events=rule.clock.events, main_policy_changed=False)
    if arm == 'v107':
        old = read(archived/'records'/f'v216_c{ci}_capture1.json.gz')
        for key, value in old['result']['results'].items():
            passive.exact(row[key], value, 'V218_archived_'+key)
        passive.exact(data['physical_contract'], old['result']['physical_contract'], 'V218_archived_physics')
        passive.exact(data['trace'], old['trace'], 'V218_full_archived_actions_inputs_uniforms_and_timing')
    passive.trace(output, data)
    passive.write(progress, dict(context=ci, arm=arm, started=True, completed=True, archived_exact=arm == 'v107'))
    print('V218_CYCLE_PATH_DONE', ci, arm, flush=True)


def next_launch(data, bus_id, trip_id):
    returned = next((e for e in data['events'] if e['kind'] == 'return'
                     and (e['bus_id'], e['trip_id']) == (bus_id, trip_id)), None)
    if returned is None:
        return None
    launched = next((e for e in data['events'] if e['kind'] == 'launch' and e['bus_id'] == bus_id
                     and e['event_seq'] > returned['event_seq']), None)
    return None if launched is None else launched['trip_id']


def execution(data, lower_control):
    trace = data['trace']
    departures = {(r['trip_id'], r['station_id'], r['direction']): r for r in trace['departures']}
    controls = {r['trip']: r for r in lower_control['trace']['actual_launches']}
    executed, linked, coupled, missing = 0, 0, 0, 0
    for q in data['responses']:
        if q['selected_bin'] == q['base_bin']:
            continue
        dep = departures.get((q['trip_id'], q['station_id'], q['direction']))
        if dep is None:
            missing += 1
            continue
        held = dep['realized_holding']-q['executed_before_s']
        assert held == law.ACTIONS[q['selected_bin']], 'issued resource response differs from executed holding'
        executed += 1
        anchor = dict(bus_id=q['bus_id'], trip_id=q['trip_id'], station_id=q['station_id'],
            inbound_direction=q['direction'], target_direction=q['link']['target_direction'],
            due_trip_ids=q['link']['due_trip_ids'], base_bin=q['selected_bin'])
        label = passive.obs.supply_label(anchor, trace['vehicle_returns'], trace['actual_launches'],
            data['events'], trace['departures'], dict(executed_holding_before_s=q['executed_before_s']))
        if not label.get('same_vehicle_due_supply'):
            continue
        linked += 1
        launched = label['next_same_vehicle_launch']
        control = controls.get(launched['trip'])
        control_plan = lower_control['launch_plans'].get(str(launched['trip']))
        planned = data['launch_plans'][str(launched['trip'])]
        if (control is not None and control_plan is not None and control['bus'] == launched['bus']
                and control['direction'] == launched['direction']
                and next_launch(lower_control, q['bus_id'], q['trip_id']) == launched['trip']
                and planned != control_plan and launched['actual_s'] != control['actual_s']):
            coupled += 1
    return dict(changed_requests=sum(r['selected_bin'] != r['base_bin'] for r in data['responses']),
        executed_responses=executed, missing_departures=missing, same_vehicle_due_supply=linked,
        same_chain_planned_and_actual_launch_changed_vs_lower_only=coupled)


def aggregate(root):
    rows = []
    for ci in range(4):
        arms = {a: read(root/'records'/f'c{ci}_{a}.json.gz') for a in law.ARMS}
        for data in arms.values():
            passive.exact(data['physical_contract'], arms['v107']['physical_contract'], 'V218_four_arm_physics')
        joint = arms['joint']
        e = execution(joint, arms['lower'])
        upper_changes = sum(c['applied_shift_s'] != 0 for c in joint['cycles'])
        plans_changed = sum(x['before']['planned_s'] != x['after']['planned_s']
                            for c in joint['cycles'] for x in c['plan_changes'])
        cleared = all(r['results']['trips_unlaunched'] == r['results']['trips_incomplete'] == 0 for r in arms.values())
        passed = (cleared and upper_changes > 0 and plans_changed > 0 and e['missing_departures'] == 0
                  and e['same_chain_planned_and_actual_launch_changed_vs_lower_only'] > 0)
        rows.append(dict(context=ci, seed=law.CONTEXTS[ci][0], day=law.CONTEXTS[ci][1],
            outcomes={a: {m: r['results'][m] for m in METRICS} for a, r in arms.items()},
            joint_minus={a: {m: joint['results'][m]-arms[a]['results'][m] for m in METRICS}
                         for a in ('v107', 'upper', 'lower')},
            upper_requests=upper_changes, changed_plan_instances=plans_changed, execution=e,
            joint_reasons=dict(Counter(r['reason'] for r in joint['responses'])), execution_gate_passed=bool(passed)))
    passive.write(root/'summary/status.json', dict(status='v218_fixed_candidate_execution_smoke_complete',
        contexts=rows, paths=16, all_baseline_trajectories_archived_exact=True,
        execution_gate_passed=all(r['execution_gate_passed'] for r in rows),
        selection_performed=False, independent_futures_opened=False, main_policy_changed=False,
        interpretation='engineering execution and descriptive outcomes only; not independent efficacy or frequency evidence'))
    print('V218_EXECUTION_SMOKE_COMPLETE', [r['execution_gate_passed'] for r in rows], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v218_*.py', '-v'], check=True)
    passive.write(a.out_root/'protocol.json', law.registration(os.environ['FREQDUET_DIAGNOSTIC_COMMIT']))
    run_item(a.out_root, a.archive_root, (0, 'v107'))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
        list(pool.map(run_item, itertools.repeat(a.out_root), itertools.repeat(a.archive_root),
                      [(ci, 'v107') for ci in range(1, 4)]))
        list(pool.map(run_item, itertools.repeat(a.out_root), itertools.repeat(a.archive_root),
                      list(itertools.product(range(4), law.ARMS[1:]))))
    aggregate(a.out_root)


if __name__ == '__main__':
    main()
