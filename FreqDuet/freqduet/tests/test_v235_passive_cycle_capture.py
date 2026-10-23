from collections import Counter
import copy
from types import SimpleNamespace
import random
import unittest

import numpy as np
import torch

from freqduet_v230_resource_capture import ResourceCapture
from freqduet_v235_passive_cycle_capture import recording_audit, validate_capture


class Clock:
    def __init__(self):
        self.events = []

    def stamp(self, kind, now, **identity):
        row = dict(kind=kind, event_seq=len(self.events), time_s=now, **identity)
        self.events.append(row)
        return row


class BaseAudit:
    """Synthetic callback producer; never imports a transit runner or model."""
    def __init__(self, seed, arm):
        self.clock = Clock()
        self.native_stamp = self.clock.stamp
        self.callbacks, self.dispatches, self.returns, self.queries = [], [], [], []
        self.handoff = SimpleNamespace(calls=Counter(), checked=Counter(), queries=[],
                                       prepares=[], by_prepare_seq={})

    def attach(self, runner):
        self.env = runner.env

    def __call__(self, policy, states, features, theta):
        self.handoff.calls['lower_rule'] += 1
        self.queries.append(states)
        return policy

    def detach(self):
        self.detached = True


class ResourceOnlyAudit(BaseAudit):
    # Construction/call order from the frozen V230 collector, without simulation.
    def __init__(self, seed, arm):
        super().__init__(seed, arm)
        self.resource = ResourceCapture(self)

    def attach(self, runner):
        super().attach(runner)
        self.resource.attach()

    def __call__(self, policy, states, features, theta):
        value = super().__call__(policy, states, features, theta)
        self.resource.query(self.queries[-1])
        return value

    def detach(self):
        self.resource.detach()
        super().detach()


class Fixture:
    def __init__(self, capture=True):
        self.audit = (recording_audit(BaseAudit) if capture else ResourceOnlyAudit)(0, 'v107')
        plans = [(1, False, 0.), (2, True, 0.), (3, True, 100.), (4, False, 200.), (5, True, 300.)]
        self.env = SimpleNamespace(time_step=1., current_time=0.,
            bus_all=[SimpleNamespace(bus_id=b) for b in range(12)],
            timetables=[SimpleNamespace(launch_turn=t, direction=d, launched=False,
                                       launch_time=p, target_headway=100.) for t, d, p in plans])
        self.audit.attach(SimpleNamespace(env=self.env))
        self.running, self.ready_directions, self.return_times, self.draws = {}, dict.fromkeys(range(12), False), {}, []
        self.ready_directions[1] = True
        self.audit.clock.stamp('fresh_upper', 0., trip_id=1, direction=False)
        self.replan(0., 1, False)

    def replan(self, now, trip, direction, changes=None):
        self.env.current_time = now
        self.audit.callbacks.append(dict(trip_id=trip, time_s=now, direction=direction, changes=changes or []))
        self.audit.clock.stamp('plan_update', now, trip_id=trip)

    def launch(self, bus, trip, direction, now):
        self.env.current_time = now
        self.running[bus] = dict(bus_id=bus, trip_id=trip, direction=direction)
        self.audit.dispatches.append(dict(trip=trip, bus=bus, direction=direction, actual_s=now))
        self.audit.clock.stamp('launch', now, bus_id=bus, trip_id=trip)

    def returned(self, bus, now, holding):
        self.env.current_time = now
        active = self.running.pop(bus)
        launch = next(r for r in self.audit.dispatches if r['trip'] == active['trip_id'])
        self.audit.returns.append(dict(trip_id=active['trip_id'], bus_id=bus, launch_s=launch['actual_s'],
            return_s=now, holding_s=holding, other_elapsed_s=now-launch['actual_s']-holding,
            inbound_direction=active['direction'], target_direction=not active['direction']))
        self.ready_directions[bus], self.return_times[bus] = not active['direction'], now
        self.audit.clock.stamp('return', now, bus_id=bus, trip_id=active['trip_id'])

    def query(self, bus, now, holding, masked=False):
        self.env.current_time = now
        a, identity = self.audit, dict(self.running[bus], station_id=4)
        prepared = a.clock.stamp('prepare', now, **identity)
        seq = a.clock.stamp('lower_query', now, **identity)['event_seq']
        order = 100+10*len(a.handoff.queries)
        context = dict(evidence_valid=not masked, follower_evidence_valid=not masked,
                       forward_headway_s=100., follower_departure_gap_s=120.)
        packet = dict(clock=dict(handoff_order=order, prepare_s=now), current=identity, context=context)
        a.handoff.prepares.append(packet)
        a.handoff.by_prepare_seq[prepared['event_seq']] = packet
        a.handoff.queries.append(dict(legacy_query_event_seq=seq, query_order=order+1,
            prepare_order=order, query_occurrence=1, identity=identity, query_s=now))
        ready = [dict(bus_id=b, direction=self.ready_directions[b], position_sample_s=now,
                      return_observed=b in self.return_times, last_observed_return_s=self.return_times.get(b))
                 for b in range(12) if b not in self.running]
        active = [dict(self.running[b], position_sample_s=now, remaining_distance_m=200.+b*100.,
                       state='on_route', last_station_id=4, next_station_id=5, progress_m=50.)
                  for b in range(12) if b in self.running]
        q = dict(identity, time_s=now, event_seq=seq, snapshot_read_s=now, snapshot_event_seq=seq,
            state_captured_s=now, prepare_event_seq=prepared['event_seq'], context=context,
            ready=ready, active=active, plans=copy.deepcopy(list(a.resource.plans.values())),
            onboard=0 if masked else 3, terminal_masked=masked, feasible_bins=[0] if masked else [0, 1, 3],
            base_bin=0 if masked else 1, executed_holding_before_s=holding)
        result = object()
        assert a(result, q, None, None) is result, 'observer changed returned action distribution'
        self.draws.append([random.random(), float(np.random.random()), float(torch.rand(()))])

    def complete(self):
        self.launch(0, 1, False, 0.)
        self.launch(1, 2, True, 0.)
        self.query(0, 10., 0.)
        before = copy.deepcopy(self.audit.resource.plans[3])
        self.replan(11., 3, True, [dict(trip_id=3, before=before, after=dict(before, planned_s=90.))])
        self.query(0, 12., 5.)
        self.returned(1, 13., 7.)
        self.returned(0, 14., 10.)
        self.launch(0, 3, True, 15.)
        self.query(0, 16., 0., masked=True)
        self.returned(0, 17., 2.)

    def originals(self):
        a = self.audit
        return copy.deepcopy(dict(legacy=dict(events=a.clock.events,
            trace=dict(lower_actions=a.queries, upper_callbacks=a.callbacks, vehicle_returns=a.returns,
                       actual_launches=a.dispatches, action_uniforms=self.draws),
            ledger=dict(executed_holding_s=sum(r['holding_s'] for r in a.returns)), results=dict(journey=18.)),
            v224=dict(prepares=a.handoff.prepares, queries=a.handoff.queries,
                      observer_checks=dict(a.handoff.checked), native_calls=dict(a.handoff.calls)),
            v230=a.resource.data(True)))


class PassiveCycleCaptureTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.getstate(), np.random.get_state(), torch.get_rng_state()

    def tearDown(self):
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])

    def paired(self):
        pairs = []
        for capture in (False, True):
            random.seed(23)
            np.random.seed(23)
            torch.manual_seed(23)
            fixture = Fixture(capture)
            fixture.complete()
            pairs.append(fixture)
        return pairs

    def test_actual_frozen_resource_hooks_preserve_all_three_original_payloads(self):
        reference, current = self.paired()
        self.assertEqual(reference.originals(), current.originals())
        verdict = validate_capture(reference.originals(), current.originals(), current.audit.cycle_capture.data(True))
        self.assertEqual(verdict['executed_holding_s'], 19.)
        self.assertEqual((verdict['queries'], verdict['closed_cycles'], verdict['observed_links']), (3, 3, 1))

    def test_plan_write_keeps_holding_and_links_do_not_backfill_queries(self):
        f = Fixture()
        f.complete()
        data = f.audit.cycle_capture.data(True)
        first, second, new_cycle = [q['view'] for q in data['queries']]
        self.assertEqual((first['executed_holding_s'], second['executed_holding_s']), (0., 5.))
        self.assertEqual(second['delta_executed_since_last_sample_s'], 5.)
        self.assertNotEqual(first['pending_plan_version_refs']['3'], second['pending_plan_version_refs']['3'])
        self.assertEqual(first['own_association']['obligation']['planned_s'], 100.)
        self.assertEqual(second['own_association']['obligation']['planned_s'], 90.)
        self.assertEqual(new_cycle['executed_holding_s'], 0.)
        self.assertIsNone(new_cycle['previous_query_s'])
        self.assertIsNone(new_cycle['own_association']['obligation'])
        self.assertEqual(data['observed_links'][0]['previous']['executed_holding_s'], 10.)
        self.assertNotIn('observed_previous_cycle', first)

    def test_zero_query_cycles_and_return_remainders_are_complete(self):
        f = Fixture()
        f.complete()
        cycles = f.audit.cycle_capture.data(True)['cycles']
        self.assertEqual([c['queries'] for c in cycles], [2, 0, 1])
        self.assertEqual([c['after_last_query_s'] for c in cycles], [5., 7., 2.])
        self.assertEqual(len(f.audit.cycle_capture.data(True)['terminal']['returned_without_next_launch']), 2)

    def test_open_cycle_keeps_observed_prefix_and_unknown_return(self):
        f = Fixture()
        f.launch(0, 1, False, 0.)
        f.query(0, 10., 5.)
        data = f.audit.cycle_capture.data(False)
        self.assertFalse(data['complete'])
        self.assertEqual(data['terminal']['open_cycles'][0]['executed_holding_s'], 5.)
        self.assertIsNone(data['cycles'][0]['executed_holding_s'])
        self.assertIsNone(data['cycles'][0]['after_last_query_s'])

    def test_current_ready_return_changes_association_without_resetting_holding(self):
        f = Fixture()
        f.launch(0, 1, False, 0.)
        f.launch(1, 2, True, 0.)
        f.query(0, 10., 5.)
        f.launch(2, 4, False, 11.)
        f.returned(2, 12., 0.)
        f.query(0, 13., 10.)
        first, second = [q['view'] for q in f.audit.cycle_capture.data(False)['queries']]
        self.assertEqual([q['own_association']['obligation']['trip_id'] for q in (first, second)], [3, 5])
        self.assertEqual(second['delta_executed_since_last_sample_s'], 5.)

    def test_missing_rows_changed_clocks_and_filled_terminal_states_fail_acceptance(self):
        reference, current = self.paired()
        originals, data = current.originals(), current.audit.cycle_capture.data(True)
        for corruption in ('query', 'clock', 'holding', 'terminal', 'future', 'transition'):
            bad = copy.deepcopy(data)
            if corruption == 'query':
                bad['queries'].pop()
            elif corruption == 'clock':
                bad['queries'][0]['view']['query_s'] += 1.
            elif corruption == 'holding':
                bad['cycles'][0]['after_last_query_s'] = 0.
            elif corruption == 'terminal':
                bad['terminal']['returned_without_next_launch'].pop()
            elif corruption == 'transition':
                bad['transitions'].pop()
            else:
                bad['queries'][0]['actual_next_trip'] = 3
            with self.subTest(corruption=corruption), self.assertRaises(AssertionError):
                validate_capture(reference.originals(), originals, bad)

    def test_old_counter_changes_fail_acceptance(self):
        reference, current = self.paired()
        originals = current.originals()
        originals['v230']['observer_checks']['query'] += 1
        with self.assertRaises(AssertionError):
            validate_capture(reference.originals(), originals, current.audit.cycle_capture.data(True))

    def test_observer_guards_all_rngs_and_native_calls(self):
        f = Fixture()
        c = f.audit.cycle_capture
        for mutate in (random.random, np.random.random, lambda: torch.rand(()),
                       lambda: f.audit.handoff.calls.update(['estimator'])):
            with self.assertRaises(AssertionError):
                with c.read('injected_side_effect'):
                    mutate()

    def test_detach_restores_hooks_and_serialization_does_not_touch_old_counters(self):
        f = Fixture()
        f.complete()
        before = f.originals()
        self.assertEqual(f.audit.cycle_capture.data(True), f.audit.cycle_capture.data(True))
        self.assertEqual(before, f.originals())
        f.audit.detach()
        self.assertEqual(f.audit.clock.stamp, f.audit.native_stamp)
        self.assertEqual(f.audit.resource.query.__func__, ResourceCapture.query)
        self.assertTrue(f.audit.detached)


if __name__ == '__main__':
    unittest.main()
