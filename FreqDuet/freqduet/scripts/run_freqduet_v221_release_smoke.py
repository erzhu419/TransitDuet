#!/usr/bin/env python3
"""Budgeted V220 actual-release gate, then conditional four-arm execution smoke."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import freqduet_v219_projection as preview
import freqduet_v219_service_task as law
import freqduet_v220_supply_projection as supply
import freqduet_v221_release_gate as gate
import run_freqduet_v218_cycle_smoke as prior


passive, pilot = prior.passive, prior.pilot
METRICS = (*prior.METRICS, 'waiting_person_s', 'onboard_person_s',
           'public_early_launch_total_s', 'public_early_launch_count')


class ReleaseAudit(passive.QueryAudit):
    def __init__(self, seed, arm):
        super().__init__(seed, True)
        self.arm, self.cycles, self.tasks, self.responses, self.launch_plans = arm, [], {}, [], {}

    release_gap = prior.CycleAudit.release_gap

    def attach(self, runner):
        super().attach(runner)
        planner = runner.timetable_planner
        assert not planner.plan_all_directions and runner.timetable_replan_interval_s == law.WINDOW_S
        assert runner.timetable_action_ema_alpha == .35
        self.planner, self.preview_planner = planner, copy.copy(planner)
        self.original_apply = planner.apply
        self.captured_upper, self.captured_launch = runner._upper_callback_v2, self.env.launch_bus

        def apply(timetables, trip, action, **kwargs):
            q = self.fresh_queries[-1]
            assert q['trip_id'] == trip.launch_turn and q['time_s'] == self.env.current_time
            raw = np.asarray(q['native_coefficient'], np.float32)
            effective = preview.ema_action(raw, self.previous_action, .35)
            np.testing.assert_array_equal(action, effective)
            trips = preview.query_timetable(q, self.dispatches)
            native = preview.preview(self.preview_planner, trips, q, effective)
            summary = self.original_apply(timetables, trip, action, **kwargs)
            actual = {t: {k: r[k] for k in ('planned_s', 'target_s')}
                      for t, r in passive.obs.plans(self.env).items()}
            preview.verify_native(actual, native, self.env.time_step)
            if q['time_s'] < law.START_S:
                return summary
            request = np.clip(raw+30., planner.action_low, planner.action_high).astype(np.float32)
            desired = preview.preview(self.preview_planner, trips, q,
                preview.ema_action(request, self.previous_action, .35))
            solution, diagnostics = supply.project(self.preview_planner, q, trips, native, desired)
            evidence = preview.inspect_proposal(q, native, solution, self.env.time_step)
            assert not evidence['continuous_later_trip_ids'] and not evidence['clock_later_trip_ids']
            assert not evidence['unfunded_within_cycle_trip_ids'] and not evidence['outside_cycle_advanced']
            assert diagnostics['max_headway_error_s'] <= 1e-6
            resource = law.ready_prefix(q)
            tasks = [dict(trip_id=t, reserved_bus_id=resource['ready_bus_by_trip'][t],
                native_s=native[t]['planned_s'], proposed_s=solution[t]['planned_s'],
                query_event_seq=q['event_seq']) for t in evidence['law_admission']['task_ids']]
            for task in tasks:
                assert task['trip_id'] not in self.tasks, 'same departure funded twice'
                self.tasks[task['trip_id']] = dict(task)
            applied = evidence['supported'] and self.arm in ('upper', 'joint')
            if applied:
                summary = gate.overwrite_curve(planner, q, timetables, native, solution, summary)
            expected = solution if applied else native
            after = {t: {k: r[k] for k in ('planned_s', 'target_s')}
                     for t, r in passive.obs.plans(self.env).items()}
            preview.verify_native(after, expected, self.env.time_step)
            # The constrained curve persists in trips, not a fictitious scalar cache.
            np.testing.assert_array_equal(runner._active_timetable_plans[bool(trip.direction)]['action'], action)
            self.cycles.append(dict(query=q, tasks=tasks, applied=applied,
                request_raw=request.tolist(), effective_request=preview.ema_action(request, self.previous_action, .35).tolist(),
                projection=diagnostics, **evidence))
            return summary

        def upper(state, trip):
            active = runner._active_timetable_plans.get(bool(trip.direction))
            self.previous_action = None if active is None else np.asarray(active['action'], np.float32).copy()
            return self.captured_upper(state, trip)

        def launch(trip, actual_launch=None):
            plan = float(getattr(trip, '_freqduet_scheduled_launch', trip.launch_time))
            bus = self.captured_launch(trip, actual_launch=actual_launch)
            if bus is not None:
                self.launch_plans[int(trip.launch_turn)] = plan
                task = self.tasks.get(int(trip.launch_turn))
                if task is not None:
                    event = self.clock.events[-1]
                    assert event['kind'] == 'launch' and event['trip_id'] == trip.launch_turn
                    task.update(actual_bus_id=int(bus.bus_id), actual_launch_s=float(self.env.current_time),
                        launch_event_seq=event['event_seq'], bus_id=task['reserved_bus_id'])
            return bus

        planner.apply, runner._upper_callback_v2, self.env.launch_bus = apply, upper, launch

    def __call__(self, policy, states, features, theta):
        result = super().__call__(policy, states, features, theta)
        q = self.queries[-1]
        task = self.tasks.get(q['trip_id'])
        if task is None or task.get('launch_event_seq') is None or task['actual_bus_id'] != task['reserved_bus_id']:
            return result
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        gap = self.release_gap(q, bus) if q['forward_valid'] else None
        chosen = law.lower_request(q, task, gap)
        selected = chosen if self.arm in ('lower', 'joint') else q['base_bin']
        self.responses.append(dict(time_s=q['time_s'], event_seq=q['event_seq'], bus_id=q['bus_id'],
            trip_id=q['trip_id'], station_id=q['station_id'], direction=q['direction'],
            base_bin=q['base_bin'], compensation_bin=chosen, selected_bin=selected,
            release_holding_gap_s=gap, executed_before_s=q['executed_holding_before_s']))
        if selected != q['base_bin']:
            result = torch.zeros_like(result)
            result[0, selected] = 1.
            q['command_s'] = law.ACTIONS[selected]
        return result

    def detach(self):
        self.planner.apply = self.original_apply
        self.runner._upper_callback_v2, self.env.launch_bus = self.captured_upper, self.captured_launch
        super().detach()


def run_item(root, archived, item):
    ci, arm = item
    output, progress = root/'records'/f'c{ci}_{arm}.json.gz', root/'progress'/f'c{ci}_{arm}.json'
    if output.exists():
        assert prior.read(output)['arm'] == arm
        return
    assert not progress.exists(), 'started path consumes its slot; inspect failure, no automatic rerun'
    passive.write(progress, dict(context=ci, arm=arm, started=True))
    print('V221_PATH_STARTED', ci, arm, flush=True)
    seed, day = law.CONTEXTS[ci]
    rule, credit = ReleaseAudit(seed, arm), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v221_c{ci}_{arm}', probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s,
        public_launch_delay_total_s=sum(max(0., r['actual_s']-r['public_baseline_s']) for r in rule.dispatches),
        public_early_launch_total_s=sum(max(0., r['public_baseline_s']-r['actual_s']) for r in rule.dispatches),
        public_early_launch_count=sum(r['actual_s'] < r['public_baseline_s'] for r in rule.dispatches),
        readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in rule.dispatches), **rule.dispatch_metrics)
    data = dict(context=ci, arm=arm, results=row, ledger=rule.ledger, physical_contract=rule.contract,
        trace=prior.common(rule, events, holds), cycles=rule.cycles, tasks=list(rule.tasks.values()),
        responses=rule.responses, fresh_upper=rule.fresh_queries, launch_plans=rule.launch_plans,
        events=rule.clock.events, main_policy_changed=False)
    if arm == 'v107':
        old = prior.read(archived/'records'/f'v216_c{ci}_capture1.json.gz')
        for key, value in old['result']['results'].items():
            passive.exact(row[key], value, 'V221_archived_'+key)
        passive.exact(data['physical_contract'], old['result']['physical_contract'], 'V221_archived_physics')
        passive.exact(data['trace'], old['trace'], 'V221_full_archived_actions_inputs_uniforms_and_timing')
    passive.trace(output, data)
    passive.write(progress, dict(context=ci, arm=arm, started=True, completed=True, archived_exact=arm == 'v107'))
    print('V221_PATH_DONE', ci, arm, flush=True)


def authority(candidate, reference):
    passive.exact(candidate['physical_contract'], reference['physical_contract'], 'V221_authority_physics')
    first = next((r for r in candidate['cycles'] if r['supported']), None)
    prefix_exact, checks = False, []
    if first is not None:
        q = first['query']
        matched = next((r for r in reference['fresh_upper'] if
            (r['time_s'], r['trip_id']) == (q['time_s'], q['trip_id'])), None)
        passive.exact(q, matched, 'V221_FIRST_admitted_query_and_ready_prefix')
        prefix_exact = True
        old = {r['trip']: r for r in reference['trace']['actual_launches']}
        new = {r['trip']: r for r in candidate['trace']['actual_launches']}
        for task in first['tasks']:
            trip = task['trip_id']
            checks.append(gate.task_execution(task, old.get(trip), new.get(trip),
                reference['launch_plans'].get(str(trip)), candidate['launch_plans'].get(str(trip)),
                candidate['physical_contract']['dt_s']))
    cleared = all(d['results']['trips_unlaunched'] == d['results']['trips_incomplete'] == 0
                  for d in (candidate, reference))
    return dict(context=candidate['context'], first_query_exact=prefix_exact, cleared=cleared,
        first_admitted_tasks=checks, supported_queries=sum(r['supported'] for r in candidate['cycles']),
        passed=bool(prefix_exact and checks and all(r['passed'] for r in checks) and cleared),
        upper_minus_reference={m: candidate['results'][m]-reference['results'][m] for m in METRICS})


def execution(data):
    departures = {(r['trip_id'], r['station_id'], r['direction']): r for r in data['trace']['departures']}
    changed = [r for r in data['responses'] if r['selected_bin'] != r['base_bin']]
    missing = 0
    for r in changed:
        departure = departures.get((r['trip_id'], r['station_id'], r['direction']))
        if departure is None:
            missing += 1
        else:
            assert departure['realized_holding']-r['executed_before_s'] == law.ACTIONS[r['selected_bin']]
    return dict(changed_requests=len(changed), missing_departures=missing,
        executed_changed_requests=len(changed)-missing, tasks=len(data['tasks']),
        launched_on_reserved_bus=sum(t.get('actual_bus_id') == t['reserved_bus_id'] for t in data['tasks']))


def aggregate(root, authority_rows):
    contexts = []
    for ci in range(4):
        arms = {a: prior.read(root/'records'/f'c{ci}_{a}.json.gz') for a in gate.ARMS}
        for data in arms.values():
            passive.exact(data['physical_contract'], arms['v107']['physical_contract'], 'V221_four_arm_physics')
        effects = {a: {m: arms['joint']['results'][m]-arms[a]['results'][m] for m in METRICS}
                   for a in gate.ARMS[:-1]}
        ex = {a: execution(d) for a, d in arms.items()}
        joint_launch = {r['trip']: r for r in arms['joint']['trace']['actual_launches']}
        lower_launch = {r['trip']: r for r in arms['lower']['trace']['actual_launches']}
        same_bus_advanced = sum(t in lower_launch and r['bus'] == lower_launch[t]['bus']
            and r['actual_s'] < lower_launch[t]['actual_s'] for t, r in joint_launch.items())
        cleared = all(d['results']['trips_unlaunched'] == d['results']['trips_incomplete'] == 0 for d in arms.values())
        constraints = ('cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s',
                       'trips_unlaunched', 'trips_incomplete')
        joint_ok = all(e['passenger_person_s'] < 0 and all(e[k] <= 0 for k in constraints) for e in effects.values())
        contexts.append(dict(context=ci, actor=law.CONTEXTS[ci][0], day=law.CONTEXTS[ci][1],
            outcomes={a: {m: d['results'][m] for m in METRICS} for a, d in arms.items()},
            joint_minus=effects, execution=ex, joint_same_bus_earlier_than_lower=same_bus_advanced,
            descriptive_joint_gate=bool(joint_ok and cleared and ex['joint']['executed_changed_requests'] > 0
                and ex['joint']['missing_departures'] == 0 and same_bus_advanced > 0)))
    passive.write(root/'summary/status.json', dict(status='v221_four_arm_execution_complete',
        authority=authority_rows, contexts=contexts, started_paths=16,
        authority_gate_passed=True, descriptive_joint_gate=all(r['descriptive_joint_gate'] for r in contexts),
        training_updates=0, independent_futures_opened=False, main_policy_changed=False,
        scope='Consumed development contexts;execution/descriptive outcomes,not independent efficacy or frequency evidence.'))
    print('V221_FOUR_ARM_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    assert 1 <= a.workers <= 4
    spec = gate.registration(os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])
    frozen = a.out_root/'protocol.json'
    if frozen.exists():
        passive.exact(json.loads(frozen.read_text()), json.loads(json.dumps(spec)), 'V221_frozen_budget')
    else:
        passive.write(frozen, spec)
    for pattern in ('test_v219_*.py', 'test_v220_*.py', 'test_v221_*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
        for arm in ('v107', 'upper'):
            list(pool.map(run_item, itertools.repeat(a.out_root), itertools.repeat(a.archive_root),
                          [(ci, arm) for ci in range(4)]))
        rows = [authority(prior.read(a.out_root/'records'/f'c{ci}_upper.json.gz'),
                          prior.read(a.out_root/'records'/f'c{ci}_v107.json.gz')) for ci in range(4)]
        passed = all(r['passed'] for r in rows)
        result = dict(status='v221_authority_passed' if passed else 'v221_stopped_at_authority_gate',
            contexts=rows, started_paths=8, authority_gate_passed=passed, lower_joint_started=False,
            training_updates=0, main_policy_changed=False, independent_futures_opened=False)
        passive.write(a.out_root/'summary/authority.json', result)
        passive.write(a.out_root/'summary/status.json', result)
        print('V221_AUTHORITY_GATE', [r['passed'] for r in rows], flush=True)
        if not passed:
            return
        list(pool.map(run_item, itertools.repeat(a.out_root), itertools.repeat(a.archive_root),
                      list(itertools.product(range(4), ('lower', 'joint')))))
    aggregate(a.out_root, rows)


if __name__ == '__main__':
    main()
