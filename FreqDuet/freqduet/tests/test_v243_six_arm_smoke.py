import ast
from collections import defaultdict
import copy
import os
from pathlib import Path
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from env.scenario import ScenarioTape
from upper.timetable_planner import TimetableCurvePlanner
import freqduet_v219_projection as preview
import freqduet_v219_service_task as law
import freqduet_v220_supply_projection as supply
import freqduet_v221_release_gate as gate
from test_v219_projection import query
from test_v241_passive_requests import packet_inputs
from freqduet_v241_passive_requests import request_packet
from freqduet_v243_six_arm_smoke import (
    ARM_RULES, SMOKE_SEED, smoke_audit, selected_request, execution_chain, check_smoke)


def definitions(path, names, namespace):
    # Execute the actual frozen definitions without importing their rollout CLI/dependency graph.
    tree = ast.parse(Path(path).read_text())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in body} == set(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return NS(**{name: namespace[name] for name in names})


SCRIPTS = Path(os.environ['FREQDUET_V243_PRIMITIVES'])
common = dict(copy=copy, np=np, torch=torch, random=random)
same = definitions(SCRIPTS/'run_freqduet_protocol_v6_checkpoint_trajectory.py',
                   ('assert_same_state',), dict(common)).assert_same_state
rng = definitions(SCRIPTS/'run_freqduet_v36_lower_credit.py', ('rng_state',), dict(common)).rng_state
repeated = definitions(SCRIPTS/'run_freqduet_v68_repeated_continuations.py',
    ('physical_prefix', 'fork_future'), dict(common, joint=NS(assert_same_state=same, rng_state=rng)))
event = definitions(SCRIPTS/'run_freqduet_v69_event_coupling.py',
    ('categorical_index', 'EventUniforms', 'alignment'), dict(common, defaultdict=defaultdict))
resource_snapshot = definitions(SCRIPTS/'run_freqduet_v211_joint_resources.py', ('resource_snapshot',),
    dict(common, require=lambda condition, message: condition or (_ for _ in ()).throw(AssertionError(message)))).resource_snapshot
joint = NS(resource_snapshot=resource_snapshot, pilot=NS(joint=NS(assert_same_state=same, rng_state=rng)))


class ToyPolicy:
    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)
        self.calls, self.last_return = 0, None
        self.weights = torch.tensor([3., 7.])

    def sampling_state(self):
        return self.generator.get_state()

    def set_sampling_state(self, state):
        self.generator.set_state(state)

    def get_action(self, state):
        self.calls += 1
        torch.rand((), generator=self.generator)
        self.last_return = np.array([0.], np.float32)
        return self.last_return


class NativeCallbacks:
    """Synthetic native producer for the actual V221 callback class and V243 adapter."""
    def __init__(self, seed, arm):
        self.clock = NS(events=[])
        self.commands, self.callbacks, self.queries, self.policy_inputs = [], [], [], []
        self.dispatches, self.returns, self.arrivals, self.departures = [], [], [], []
        self.fresh_queries, self.launches = [], {}
        self.handoff = NS(data=lambda: dict(prepares=[dict(context='cached')]))
        self.resource = NS(data=lambda complete: dict(complete=complete, versions=[0], H=70.))
        self.cycle_capture = NS(data=lambda complete: dict(complete=complete, cycle=[2, 1, 0], H=70.))
        self.request_capture = NS(queries=[])
        self.lower_calls, self.current_trip = 0, None

    def attach(self, runner):
        self.runner, self.env = runner, runner.env
        self.current_upper = runner._upper_callback_v2
        self.current_sample = runner.upper_trainer.policy_net.get_action
        for b in self.env.bus_all:
            if b.on_route:
                self.launches[b.trip_id] = dict(time_s=0., holding_counter_at_launch=0.)

        def sample(state):
            value = self.current_sample(state)
            q = query((2, 2))
            q.update(time_s=self.env.current_time, event_seq=len(self.clock.events),
                direction=True, fresh_upper=True, native_coefficient=value.tolist())
            self.clock.events.append(dict(kind='fresh_upper', time_s=q['time_s'], event_seq=q['event_seq']))
            self.fresh_queries.append(q)
            self.commands.append(dict(coefficient=value.tolist()))
            return value

        def upper(state, trip):
            self.current_trip = trip
            try:
                raw = runner.upper_trainer.policy_net.get_action(state)
                self.returned_native_object = raw
                effective = preview.ema_action(raw, runner._active_timetable_plans[True]['action'], .35)
                runner._active_timetable_plans[True]['action'] = effective
                result = runner.timetable_planner.apply(self.env.timetables, trip, effective,
                    origin_launch_s=7200., write_scheduled_launch=True, plan_id=40)
                self.callbacks.append(dict(native=raw.tolist(), summary=result))
                return result
            finally:
                self.current_trip = None
        runner._upper_callback_v2, runner.upper_trainer.policy_net.get_action = upper, sample

    def __call__(self, policy, states, features, theta):
        self.lower_calls += 1
        packet = copy.deepcopy(self.packet)
        self.request_capture.queries.append(packet)
        p = packet['identity']
        self.queries.append(dict(p, base_bin=packet['native_bin'], command_s=law.ACTIONS[packet['native_bin']],
            feasible_bins=list(range(7)), time_s=packet['query_s'], event_seq=packet['native_event_seq'],
            forward_valid=True, executed_holding_before_s=packet['executed_holding_before_s']))
        value = torch.zeros((1, 7))
        value[0, packet['native_bin']] = 1.
        return value

    def detach(self):
        self.runner._upper_callback_v2 = self.current_upper
        self.runner.upper_trainer.policy_net.get_action = self.current_sample

    def finish(self, runner, records, holds):
        self.actions = records


legacy = definitions(Path(os.environ['FREQDUET_V243_CONTROLLER']), ('ReleaseAudit',),
    dict(common, passive=NS(QueryAudit=NativeCallbacks, obs=NS(plans=lambda env: {
        t.launch_turn: dict(planned_s=float(getattr(t, '_freqduet_scheduled_launch', t.launch_time)),
                            target_s=t.target_headway) for t in env.timetables})),
        prior=NS(CycleAudit=NS(release_gap=lambda *args: 0.)), preview=preview, supply=supply, gate=gate, law=law))
Audit = smoke_audit(legacy.ReleaseAudit, repeated, event, joint)


def packet(unassociated=False, holding=70., seq=10, unsupported=False):
    row, cyc, prepare = packet_inputs(unassociated)
    row['executed_holding_before_s'] = cyc['view']['executed_holding_s'] = holding
    if unsupported:
        row['shadow_frontier']['status'] = 'missing_follower'
    p = request_packet(0, row, cyc, prepare)
    p.update(native_event_seq=seq, query_s=7201.)
    return p


def setup(arm='v107', reference=None):
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    trips = preview.query_timetable(dict(query((2, 2)), direction=True), [])
    planner = TimetableCurvePlanner(basis_per_direction=2, shared_directions=True, plan_all_directions=False,
        coefficient_parameterization='antisymmetric_linear_v5', terminal_schedule_mode='exact_headway_curve',
        headway_budget_mode='rolling_zero_sum_delta_v6', delta_min_s=-60, delta_max_s=60)
    station = NS(station_id=3, _pending_passengers=[], waiting_passengers=[])
    buses = [NS(bus_id=i, trip_id=1 if i == 2 else 100+i, direction=i >= 6,
        on_route=i not in (0, 1, 10, 11), state=NS(name='Running'), current_speed=7.,
        last_station=station, next_station=station, travel_distance=10., absolute_distance=10.,
        last_station_dis=10., next_station_dis=90., holding_time=0., episode_hold_vehicle_seconds=70.,
        passengers=[]) for i in range(12)]
    pending = NS(appear_time=7300., boarding_time=None, arrive_time=None, appear_station=station,
                 destination_station=station, boarded=False, arrived=False)
    station._pending_passengers.append(pending)
    env = NS(current_time=7200., time_step=1., bus_all=buses, timetables=trips,
        bus_in_terminal=[b for b in buses if not b.on_route], stations=[station], routes=[NS(speed_limit=7.)],
        _demand_multipliers={6: 1.2}, _od_multipliers={(1, True): .8}, _peak_shift=1.,
        scenario_tape=ScenarioTape(123), launch_bus=lambda *a, **k: None,
        _measurement_details=dict(trips_unlaunched=0, trips_incomplete=0))
    env.scenario_tape.normal_stream(0, 1, 'road', 1)
    upper, lower = ToyPolicy(41), ToyPolicy(42)
    streams = dict(fleet=np.random.RandomState(1), selector=np.random.RandomState(2))
    runner = NS(env=env, timetable_planner=planner, timetable_replan_interval_s=900.,
        timetable_action_ema_alpha=.35, upper_trainer=NS(policy_net=upper), _upper_callback_v2=lambda *a: None,
        _active_timetable_plans={True: dict(action=np.array([0.], np.float32))},
        _runtime_numpy_streams=lambda: streams, _bus_for_agent=lambda _: buses[2], _pilot_agent=0)
    uniforms = event.EventUniforms(402112137)
    k, u = uniforms.draw(1, 3, False)
    uniforms.records[k] = dict(uniform=u, order=0, time_s=7199., action_s=20.)
    a = Audit(12037, arm, uniforms, reference)
    a.attach(runner)
    a.lower_policy = lower
    return a, runner, lower, pending


def anchor(a, runner):
    trip = next(t for t in runner.env.timetables if t.launch_turn == 40)
    return runner._upper_callback_v2(np.zeros(3, np.float32), trip)


def ref_of(a):
    return dict(anchor=copy.deepcopy(a.fork_anchor), future=copy.deepcopy(a.future_receipt))


def actual_rows(p):
    ident = p['identity']
    actions = [dict(time_s=p['query_s'], trip_id=ident['trip_id'], station_id=ident['station_id'],
                    direction=ident['direction'], action=p['selected_bin'],
                    probabilities=np.eye(7)[p['selected_bin']])]
    holds = [dict(time_s=p['query_s'], bus_id=ident['bus_id'], trip_id=ident['trip_id'],
                  station_id=ident['station_id'], action_s=p['issued_s'])]
    deps = [dict(time_s=p['query_s']+p['issued_s'], trip_id=ident['trip_id'], station_id=ident['station_id'],
                 direction=ident['direction'], realized_holding=p['executed_holding_before_s']+p['issued_s'])]
    return actions, holds, deps


def synthetic_branches():
    reference, r, _, _ = setup()
    anchor(reference, r)
    branches = {}
    for arm in ARM_RULES:
        a, runner, policy, _ = setup(arm, ref_of(reference) if arm != 'v107' else None)
        anchor(a, runner)
        actions, holds, deps = [], [], []
        H = 70.
        for i, unassociated in enumerate((False, True)):
            a.packet = packet(unassociated, holding=H, seq=10+i)
            a.packet.update(query_index=i, query_s=7201.+60*i)
            a.packet['identity']['station_id'] += i
            a(policy, None, None, None)
            p = a.issued_requests[-1]
            x, h, d = actual_rows(p)
            actions.extend(x)
            holds.extend(h)
            deps.extend(d)
            H += p['issued_s']
            k, u = a.uniforms.draw(p['identity']['trip_id'], p['identity']['station_id'], p['identity']['direction'])
            a.uniforms.records[k] = dict(uniform=u, order=len(a.uniforms.records),
                time_s=p['query_s'], action_s=p['issued_s'])
        advanced = next(t for t in runner.env.timetables if t.launch_turn == 42)
        plan = advanced._freqduet_scheduled_launch
        execution = execution_chain(a.issued_requests, actions, holds, deps)
        branches[arm] = dict(arm=arm, anchor=a.fork_anchor, future=a.future_receipt,
            requests=a.issued_requests, execution=execution, uniforms=a.uniforms.records,
            physical_contract=dict(fleet=12, bins=list(law.ACTIONS)),
            ledger=dict(physical_vehicles=12, incomplete_trips=0,
                commanded_holding_s=H, executed_holding_s=H),
            completion=dict(trips_unlaunched=0, trips_incomplete=0),
            tasks=copy.deepcopy(list(a.tasks.values())),
            dispatches=[dict(trip=42, bus=11, actual_s=float(np.ceil(plan)))],
            launch_plans={42: plan}, main_policy_changed=False)
    return branches


class SixArmSmokeTest(unittest.TestCase):
    def setUp(self):
        self.saved_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()

    def tearDown(self):
        random.setstate(self.saved_rng[0])
        np.random.set_state(self.saved_rng[1])
        torch.set_rng_state(self.saved_rng[2])

    def test_six_arms_match_before_fork_and_keep_one_native_sample(self):
        native, runner, _, _ = setup()
        anchor(native, runner)
        reference = ref_of(native)
        for arm in ARM_RULES:
            a, r, policy, pending = setup(arm, reference if arm != 'v107' else None)
            old_weights = policy.weights.clone()
            anchor(a, r)
            same(a.fork_anchor, reference['anchor'])
            same(a.future_receipt, reference['future'])
            self.assertIs(a.returned_native_object, r.upper_trainer.policy_net.last_return)
            self.assertEqual(r.upper_trainer.policy_net.calls, 1)
            self.assertIs(r.env.stations[0]._pending_passengers[0], pending)
            self.assertEqual(r.env.routes[0].speed_limit, 7.)
            self.assertTrue(torch.equal(policy.weights, old_weights))
            self.assertEqual(a.uniforms.counts[(1, 3, 0)], 1)
            self.assertEqual(r.env.scenario_tape.seed, SMOKE_SEED)

    def test_actual_frozen_v220_materializes_only_upper_arms(self):
        ref, r, _, _ = setup()
        anchor(ref, r)
        for arm in ARM_RULES:
            a, r, _, _ = setup(arm, ref_of(ref) if arm != 'v107' else None)
            anchor(a, r)
            advanced = next(t for t in r.env.timetables if t.launch_turn == 42)
            upper = ARM_RULES[arm][0] == 'v220'
            self.assertEqual(advanced._freqduet_scheduled_launch < 7560., upper)
            for t in r.env.timetables:
                if not t.direction:
                    self.assertEqual(getattr(t, '_freqduet_scheduled_launch', t.launch_time), t.launch_time)
            self.assertTrue(all(c['applied'] == upper for c in a.cycles))
            np.testing.assert_array_equal(r._active_timetable_plans[True]['action'], [0.])

    def test_prefix_mismatch_stops_before_fork_or_plan_write(self):
        a, r, _, _ = setup()
        anchor(a, r)
        reference = ref_of(a)
        for mutate in ('H', 'plan', 'rng', 'prepare', 'native', 'pending'):
            bad = copy.deepcopy(reference)
            if mutate == 'H':
                bad['anchor']['v235']['H'] += 1
            elif mutate == 'plan':
                bad['anchor']['resources']['plans'][0][3] += 1
            elif mutate == 'rng':
                bad['anchor']['rng']['torch'][0] ^= 1
            elif mutate == 'prepare':
                bad['anchor']['v224']['prepares'] = []
            elif mutate == 'native':
                bad['anchor']['native_action'][0] += 1
            else:
                bad['anchor']['physical']['pending'][0] = []
            current, runner, _, _ = setup('joint_supply', bad)
            with self.subTest(mutate=mutate), self.assertRaises(AssertionError):
                anchor(current, runner)
            self.assertIsNone(current.fork_anchor)
            self.assertEqual(runner.env.scenario_tape.seed, 123)
            self.assertFalse(current.cycles)

    def test_anchor_is_first_fresh_after_threshold_and_is_not_repeated(self):
        a, r, _, _ = setup()
        r.env.current_time = 7199.
        anchor(a, r)
        self.assertIsNone(a.fork_anchor)
        r.env.current_time = 7200.
        anchor(a, r)
        receipt = a.future_receipt
        r.env.current_time = 8100.
        anchor(a, r)
        self.assertIs(a.future_receipt, receipt)
        self.assertEqual(r.upper_trainer.policy_net.calls, 3)

    def test_future_coupling_uses_keys_not_global_order_and_keeps_branch_only_events(self):
        a, r, _, _ = setup()
        anchor(a, r)
        b, s, _, _ = setup('joint_blind', ref_of(a))
        anchor(b, s)
        keys = [(1, 4, False), (2, 3, True)]
        def draw(audit, order):
            for key in order:
                k, u = audit.uniforms.draw(*key)
                audit.uniforms.records[k] = dict(uniform=u, order=len(audit.uniforms.records), time_s=7201., action_s=10.)
        draw(a, keys)
        draw(b, [(99, 1, False), *reversed(keys)])
        report = event.alignment(a.uniforms.records, b.uniforms.records)
        self.assertEqual((report['uniform_mismatches'], report['alternative_only']), (0, 1))
        self.assertGreater(report['order_shifted'], 0)
        self.assertEqual(r.env.scenario_tape.normal_stream(0, 1, 'road', 9),
                         s.env.scenario_tape.normal_stream(0, 1, 'road', 9))
        self.assertNotEqual(event.EventUniforms(SMOKE_SEED+10000001).draw(*keys[0])[1],
                            a.uniforms.records[(*map(int, keys[0]), 0)]['uniform'])

    def test_every_arm_uses_its_current_packet_and_keeps_old_lower_inactive(self):
        ref, r, _, _ = setup()
        anchor(ref, r)
        for arm in ARM_RULES:
            a, r, policy, _ = setup(arm, ref_of(ref) if arm != 'v107' else None)
            anchor(a, r)
            for unassociated, H in ((False, 70.), (True, 70.), (False, 0.)):
                a.packet = packet(unassociated, H)
                value = a(policy, None, None, None)
                expected, _ = selected_request(arm, a.packet, a.fork_anchor['fresh_query']['event_seq'])
                self.assertEqual(value.argmax().item(), expected)
                self.assertEqual(a.queries[-1]['command_s'], law.ACTIONS[expected])
                self.assertEqual(a.issued_requests[-1]['executed_holding_before_s'], H)
            self.assertEqual(a.lower_calls, 3)
            self.assertIn(a.arm, ('v107', 'upper'))

    def test_prefork_lower_is_native_and_zero_holding_aliases_native(self):
        a, _, policy, _ = setup()
        a.branch = 'joint_supply'
        a.packet = packet()
        self.assertEqual(a(policy, None, None, None).argmax().item(), a.packet['native_bin'])
        for arm in ARM_RULES:
            p = packet(holding=0.)
            self.assertEqual(selected_request(arm, p, 0)[0], p['native_bin'])
            self.assertEqual(selected_request(arm, packet(), 10)[0], p['native_bin'])

    def test_old_lower_request_is_never_stacked_and_missing_support_keeps_native(self):
        ref, r, _, _ = setup()
        anchor(ref, r)
        a, r, policy, _ = setup('joint_supply', ref_of(ref))
        anchor(a, r)
        a.tasks[1] = dict(actual_bus_id=2, reserved_bus_id=2, launch_event_seq=1)
        a.packet = packet()
        with patch.object(law, 'lower_request', return_value=0) as old_lower:
            value = a(policy, None, None, None)
        old_lower.assert_called_once()
        self.assertEqual(value.argmax().item(), 2)
        self.assertNotEqual(value.argmax().item(), 0)
        for arm in ('lower_supply', 'joint_supply', 'lower_blind', 'joint_blind'):
            a, r, policy, _ = setup(arm, ref_of(ref))
            anchor(a, r)
            a.packet = packet(unsupported=True)
            self.assertEqual(a(policy, None, None, None).argmax().item(), a.packet['native_bin'])

    def test_sampler_issue_and_physical_holding_must_all_agree(self):
        ref, r, _, _ = setup()
        anchor(ref, r)
        a, r, policy, _ = setup('lower_supply', ref_of(ref))
        anchor(a, r)
        a.packet = packet()
        law_out = a(policy, None, None, None).numpy()[0]
        selected = event.categorical_index(law_out, .999)
        self.assertEqual(selected, 2)
        p = a.issued_requests[-1]
        actions, holds, deps = actual_rows(p)
        report = execution_chain([p], actions, holds, deps)
        self.assertEqual(report['executed_changed_requests'], 1)
        for what in ('sample', 'probability', 'issue', 'execution'):
            x, h, d = copy.deepcopy((actions, holds, deps))
            if what == 'sample':
                x[0]['action'] = 4
            elif what == 'probability':
                x[0]['probabilities'] = np.eye(7)[4]
            elif what == 'issue':
                h[0]['action_s'] = 20.
            else:
                d[0]['realized_holding'] += 1.
            with self.subTest(what=what), self.assertRaises(AssertionError):
                execution_chain([p], x, h, d)
        pending = execution_chain([p], actions, holds, [])
        self.assertFalse(pending['complete'])
        self.assertIsNone(pending['missing_departures'][0]['executed_s'])

    def test_missing_anchor_and_missing_reference_stop(self):
        with self.assertRaises(AssertionError):
            setup('joint_supply')
        a, r, _, _ = setup()
        with self.assertRaises(AssertionError):
            a.finish(r, [], [])

    def test_hooks_restore_after_detach(self):
        a, r, _, _ = setup()
        a.detach()
        self.assertEqual(r.upper_trainer.policy_net.get_action, a.current_sample)

    def test_six_arm_gate_requires_every_actual_execution_pair(self):
        branches = synthetic_branches()
        result = check_smoke(branches, same, event.alignment, gate.task_execution, 1.)
        self.assertTrue(result['passed'])
        self.assertGreater(result['same_state_dual_request_differences'], 0)
        for candidate in ('upper_only', 'joint_supply', 'joint_blind'):
            bad = copy.deepcopy(branches)
            bad[candidate]['dispatches'][0]['actual_s'] = 7560.
            self.assertFalse(check_smoke(bad, same, event.alignment, gate.task_execution, 1.)['passed'])
        for arm in ('lower_supply', 'joint_supply', 'lower_blind', 'joint_blind'):
            bad = copy.deepcopy(branches)
            bad[arm]['execution']['executed_changed_requests'] = 0
            self.assertFalse(check_smoke(bad, same, event.alignment, gate.task_execution, 1.)['passed'])

    def test_gate_rejects_missing_arm_prefix_coupling_or_uncleared_service(self):
        branches = synthetic_branches()
        for what in ('arm', 'prefix', 'future', 'uniform', 'fleet', 'unlaunched', 'incomplete'):
            bad = copy.deepcopy(branches)
            if what == 'arm':
                bad.pop('joint_blind')
            elif what == 'prefix':
                bad['joint_supply']['anchor']['v235']['H'] += 1
            elif what == 'future':
                bad['joint_supply']['future']['scenario_seed'] += 1
            elif what == 'uniform':
                row = next(iter(bad['joint_supply']['uniforms'].values()))
                row['uniform'] += .01
            elif what == 'fleet':
                bad['joint_supply']['ledger']['physical_vehicles'] = 13
            else:
                bad['joint_supply']['completion']['trips_'+what] = 1
            with self.subTest(what=what), self.assertRaises(AssertionError):
                check_smoke(bad, same, event.alignment, gate.task_execution, 1.)


if __name__ == '__main__':
    unittest.main()
