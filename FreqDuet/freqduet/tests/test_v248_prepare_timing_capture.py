from collections import Counter
import copy
import json
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from freqduet_v224_prepare_capture import PrepareCapture, shadow
from freqduet_v235_passive_cycle_capture import recording_audit as cycle_audit
from freqduet_v241_passive_requests import recording_audit as request_audit
from freqduet_v248_prepare_timing_capture import recording_audit, validate_capture
from preregister_freqduet_v246_compensation_supply import persistent_change
from preregister_freqduet_v247_prepare_timing import map_prepare
from test_v235_passive_cycle_capture import BaseAudit, Fixture as ResourceFixture


class NativeAudit(BaseAudit):
    """Synthetic producer using the actual V224/V230/V235/V241 observer chain."""
    def __init__(self, seed, arm):
        super().__init__(seed, arm)
        self.handoff = PrepareCapture(self)
        self.launches, self.buses, self.departures, self.observed_ready = {}, {}, [], {}
        audit = self

        class Bus:
            def _prepare_for_action(bus, now):
                audit.handoff.calls['prepare'] += 1
                identity = dict(bus_id=bus.bus_id, trip_id=bus.trip_id,
                    station_id=bus.last_station.station_id, direction=bus.direction)
                stamp = audit.clock.stamp('prepare', now, **identity)
                audit.observed_ready[tuple(identity[k] for k in
                    ('bus_id','trip_id','station_id','direction'))] = stamp
                with audit.handoff.read('prepare'):
                    audit.handoff.prepare(bus.result, (bus, now), {})
                return bus.result

        class Recorder:
            def record_departure(recorder, station_id, direction, departure_time_s, trip_id):
                audit.handoff.calls['departure'] += 1
                bus, launch = audit.buses[trip_id], audit.launches[trip_id]
                held = bus.episode_hold_vehicle_seconds-launch['holding_counter_at_launch']
                audit.departures.append(dict(trip_id=trip_id, station_id=station_id,
                    direction=direction, time_s=departure_time_s, release=launch['time_s'],
                    realized_holding=held, other_elapsed=departure_time_s-launch['time_s']-held))
                with audit.handoff.read('departure'):
                    audit.handoff.departure(recorder.result,
                        (recorder, station_id, direction, departure_time_s, trip_id), {})
                return recorder.result

        self.bus_type, self.recorder_type = Bus, Recorder

    def attach(self, runner):
        super().attach(runner)
        self.handoff.env = self.env

    def __call__(self, policy, states, features, theta):
        value = super().__call__(policy, states, features, theta)
        self.handoff.query(self.queries[-1])
        return value


class Fixture(ResourceFixture):
    def __init__(self, capture=True, arm='v107'):
        base = request_audit(cycle_audit(NativeAudit))
        self.audit = (recording_audit(base) if capture else base)(0, arm)
        self.base_action = base.__call__
        plans = [(1, False, 0.), (2, True, 0.), (3, True, 100.), (4, False, 200.), (5, True, 300.)]
        recorder = self.audit.recorder_type()
        recorder.result = object()
        self.env = SimpleNamespace(time_step=1., current_time=0., headway_events=recorder,
            bus_all=[SimpleNamespace(bus_id=b) for b in range(12)],
            timetables=[SimpleNamespace(launch_turn=t, direction=d, launched=False,
                launch_time=p, target_headway=100.) for t,d,p in plans])
        self.audit.attach(SimpleNamespace(env=self.env))
        self.physical, self.running, self.return_times, self.draws = {}, {}, {}, []
        self.ready_directions = dict.fromkeys(range(12), False)
        self.ready_directions[1] = True
        for _ in range(20):
            self.audit.handoff.mark('tick')  # Deliberately distinct clock domains.
        self.audit.clock.stamp('fresh_upper', 0., trip_id=1, direction=False)
        self.replan(0., 1, False)

    def launch(self, bus, trip, direction, now):
        a, self.env.current_time = self.audit, now
        vehicle = self.physical.setdefault(bus, a.bus_type())
        vehicle.bus_id, vehicle.trip_id, vehicle.direction = bus, trip, direction
        vehicle.episode_hold_vehicle_seconds = getattr(vehicle, 'episode_hold_vehicle_seconds', 100.)
        vehicle.result = object()
        a.buses[trip] = vehicle
        a.launches[trip] = dict(time_s=now, bus_id=bus,
            holding_counter_at_launch=vehicle.episode_hold_vehicle_seconds)
        self.running[bus] = dict(bus_id=bus, trip_id=trip, direction=direction)
        a.dispatches.append(dict(trip=trip, bus=bus, direction=direction, actual_s=now))
        a.clock.stamp('launch', now, bus_id=bus, trip_id=trip)
        plan = next(t for t in self.env.timetables if t.launch_turn == trip)
        with a.handoff.read('launch'):
            a.handoff.launch(vehicle, (plan,), {})

    def holding(self, bus, seconds):
        vehicle = self.physical[bus]
        vehicle.episode_hold_vehicle_seconds = self.audit.launches[vehicle.trip_id]['holding_counter_at_launch']+seconds

    def departure(self, bus, station, now, held):
        self.env.current_time = now
        self.holding(bus, held)
        vehicle = self.physical[bus]
        assert self.env.headway_events.record_departure(station, vehicle.direction, now,
            vehicle.trip_id) is self.env.headway_events.result

    def prepare(self, bus, station, now, held, front=None):
        self.env.current_time = now
        self.holding(bus, held)
        vehicle = self.physical[bus]
        vehicle.last_station = SimpleNamespace(station_id=station)
        vehicle.forward_predecessor_trip_id = front
        vehicle.pre_action_forward_headway = None if front is None else now-self.audit.handoff.departures[
            front, station, vehicle.direction]['departure_s']
        vehicle.pre_action_forward_headway_source = 'unavailable' if front is None else 'matched_departure_event'
        vehicle.pre_action_follower_departure_gap = None
        vehicle.pre_action_follower_source = 'unavailable'
        vehicle._target_headway = 100.
        assert vehicle._prepare_for_action(now) is vehicle.result

    def query(self, bus, now, held, masked=False):
        self.env.current_time = now
        self.holding(bus, held)
        a, vehicle = self.audit, self.physical[bus]
        identity = dict(self.running[bus], station_id=vehicle.last_station.station_id)
        prepare = a.handoff.prepares[-1]
        seq = a.clock.stamp('lower_query', now, **identity)['event_seq']
        feasible = [0] if masked else list(range(7))
        command = shadow.command(SimpleNamespace(**prepare['context']), feasible)
        ready = [dict(bus_id=b, direction=self.ready_directions[b], position_sample_s=now,
            return_observed=b in self.return_times, last_observed_return_s=self.return_times.get(b))
            for b in range(12) if b not in self.running]
        active = [dict(self.running[b], position_sample_s=now, remaining_distance_m=200.+b*100.,
            state='on_route', last_station_id=4, next_station_id=5, progress_m=50.)
            for b in range(12) if b in self.running]
        q = dict(identity, time_s=now, event_seq=seq, snapshot_read_s=now, snapshot_event_seq=seq,
            state_captured_s=prepare['clock']['prepare_s'],
            prepare_event_seq=prepare['clock']['legacy_prepare_event_seq'], context=prepare['context'],
            ready=ready, active=active, plans=copy.deepcopy(list(a.resource.plans.values())),
            onboard=0 if masked else 3, terminal_masked=masked, feasible_bins=feasible,
            base_bin=command['bin'], executed_holding_before_s=held)
        policy = torch.eye(7)[[command['bin']]]
        assert a(policy, q, None, None) is policy
        self.draws.append([random.random(), float(np.random.random()), float(torch.rand(()))])

    def complete(self):
        self.launch(0, 1, False, 0.)
        self.launch(2, 4, False, 200.)
        self.launch(1, 2, True, 201.)
        self.departure(0, 7, 500., 40.)
        self.prepare(2, 7, 600., 20., front=1)
        self.query(2, 610., 25.)
        self.query(2, 615., 25.)
        before = copy.deepcopy(self.audit.resource.plans[3])
        self.replan(620., 3, True, [dict(trip_id=3, before=before, after=dict(before, planned_s=90.))])
        self.departure(0, 8, 650., 50.)
        self.prepare(2, 8, 700., 30., front=1)
        self.query(2, 705., 35.)
        self.prepare(2, 9, 710., 35.)  # Unused PREPARE must remain present.
        self.returned(0, 715., 50.)
        self.returned(2, 720., 35.)
        self.launch(0, 3, True, 730.)  # Reuses the physical front bus.
        self.prepare(0, 1, 740., 0.)
        self.query(0, 740., 0., masked=True)
        self.returned(0, 745., 0.)
        self.returned(1, 746., 0.)  # Zero-query cycle.
        self.launch(2, 5, True, 750.)
        self.prepare(2, 1, 755., 0.)  # Open zero-query cycle.

    def originals(self):
        data = super().originals()
        data['legacy']['trace']['departures'] = copy.deepcopy(self.audit.departures)
        data['v224'] = copy.deepcopy(dict(self.audit.handoff.data(), complete=True))
        data['v235'] = self.audit.cycle_capture.data(True)
        data['v241'] = self.audit.request_capture.data(True)
        return data


class PrepareCaptureTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.fixtures = []

    def tearDown(self):
        for f in reversed(self.fixtures):
            f.audit.detach()
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])

    def make(self, capture=True, arm='v107'):
        random.seed(248)
        np.random.seed(248)
        torch.manual_seed(248)
        f = Fixture(capture, arm)
        self.fixtures.append(f)
        f.complete()
        return f

    def pair(self, arm='v107'):
        ref, current = self.make(False, arm), self.make(True, arm)
        return ref.originals(), current.originals(), current.audit.prepare_timing_capture.data(True)

    def test_all_five_payloads_exact_in_both_native_arms(self):
        for arm in ('v107', 'upper'):
            with self.subTest(arm=arm):
                result = validate_capture(*self.pair(arm))
                self.assertTrue(result['all_five_old_payloads_exact'])
                self.assertEqual((result['prepare_occurrences'], result['query_occurrences']), (5,4))
                self.assertEqual((result['unused_prepares'], result['zero_query_cycles'], result['open_cycles']), (2,3,1))

    def test_prepare_h_is_not_later_query_h_or_plan_reset(self):
        f = self.make()
        data = f.audit.prepare_timing_capture.data(True)
        self.assertEqual([r['mapped']['own']['holding_s'] for r in data['prepares']], [20.,30.,35.,0.,0.])
        self.assertEqual([(r['join']['prepare_holding_s'],r['join']['query_holding_s'])
                         for r in data['queries']], [(20.,25.),(20.,25.),(30.,35.),(0.,0.)])
        self.assertEqual([r['prepare_index'] for r in data['queries']], [0,0,1,3])
        self.assertEqual(data['prepares'][1]['change']['status'], 'same_physical_cycle_and_front')
        self.assertEqual(data['prepares'][0]['mapped']['front_identity'], data['prepares'][1]['mapped']['front_identity'])
        self.assertNotEqual(data['prepares'][4]['mapped']['cycle_key'], data['prepares'][0]['mapped']['cycle_key'])

    def test_completed_front_is_frozen_before_bus_reuse(self):
        f = self.make()
        data = f.audit.prepare_timing_capture.data(True)
        self.assertEqual(f.audit.buses[1].trip_id, 3)
        self.assertEqual(data['samples'][0]['sample']['completed_front']['timing']['realized_holding'], 40.)
        self.assertEqual(data['samples'][1]['sample']['completed_front']['timing']['realized_holding'], 50.)
        self.assertEqual([s['completed_departure_count'] for s in data['samples']], [1,2,2,2,2])
        self.assertIsNone(data['prepares'][2]['mapped']['front']['holding_s'])

    def test_native_callback_and_all_four_hook_returns_unchanged(self):
        f = self.make()
        self.assertIs(type(f.audit).__call__, f.base_action)
        self.assertEqual(sum(f.audit.prepare_timing_capture.checks.values()), 5+2+5+5+4)
        self.assertGreater(f.audit.prepare_timing_capture.prepares[0]['mapped']['prepare_order'],
                           f.audit.prepare_timing_capture.prepares[0]['mapped']['prepare_seq'])

    def test_wrong_fields_missing_rows_and_future_prefix_are_rejected(self):
        ref, old, data = self.pair()
        for kind in ('missing_prepare','query_join','holding','clock','front','future_prefix','cross_station'):
            bad = copy.deepcopy(data)
            if kind == 'missing_prepare':
                bad['prepares'].pop()
            elif kind == 'query_join':
                bad['queries'][0]['prepare_index'] = 1
            elif kind == 'holding':
                bad['samples'][0]['sample']['holding_s'] = 25.
            elif kind == 'clock':
                bad['samples'][0]['sample']['native']['event_seq'] = bad['prepares'][0]['mapped']['prepare_order']
            elif kind == 'front':
                bad['samples'][0]['sample']['completed_front']['timing']['realized_holding'] += 1
            elif kind == 'future_prefix':
                bad['samples'][0]['front_departure_index'] = 1
            else:
                bad['prepares'][1]['mapped']['front_identity']['station_id'] = 8
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                validate_capture(ref, old, bad)

    def test_each_old_payload_difference_stops_acceptance(self):
        ref, old, data = self.pair()
        for name in old:
            bad = copy.deepcopy(old)
            bad[name]['injected_change'] = 1
            with self.subTest(payload=name), self.assertRaises(AssertionError):
                validate_capture(ref, bad, data)

    def test_unavailable_valid_front_cannot_pass_as_abstention(self):
        ref, old, data = self.pair()
        saved = data['samples'][0]
        saved['front_departure_index'] = saved['sample']['completed_front'] = None
        row = data['prepares'][0]
        row['mapped'] = map_prepare(old['v224']['prepares'][0], saved['sample'])
        row['change'] = persistent_change(None, row['mapped'])
        with self.assertRaises(AssertionError):
            validate_capture(ref, old, data)

    def test_observer_checks_all_rngs_and_native_calls(self):
        observer = self.make().audit.prepare_timing_capture
        for mutate in (random.random, np.random.random, lambda: torch.rand(()),
                       lambda: observer.audit.handoff.calls.update(['estimator'])):
            with self.assertRaises(AssertionError):
                with observer.read('injected'):
                    mutate()

    def test_serialization_and_detach_leave_old_chain_intact(self):
        f = self.make()
        before = f.originals()
        observer = f.audit.prepare_timing_capture
        data = observer.data(True)
        self.assertEqual(json.loads(json.dumps(data)), data)
        observer.detach()
        self.assertEqual(f.audit.clock.stamp, observer.original_stamp)
        self.assertIs(f.audit.bus_type._prepare_for_action, observer.original_prepare)
        self.assertIs(type(f.env.headway_events).record_departure, observer.original_departure)
        self.assertEqual(f.originals(), before)

    def test_incomplete_capture_and_active_arm_are_rejected(self):
        ref, old, data = self.pair()
        data['complete'] = False
        with self.assertRaises(AssertionError):
            validate_capture(ref, old, data)
        with self.assertRaises(AssertionError):
            recording_audit(request_audit(cycle_audit(NativeAudit)))(0, 'joint')


if __name__ == '__main__':
    unittest.main()
