import copy
from dataclasses import fields
import inspect
import random
from types import SimpleNamespace
import unittest

import numpy as np

from freqduet_v224_prepare_capture import (
    PrepareCapture, cached_context, check_packet, empty_phase, seal_launch_phase, validate_saved_capture,
)


def fixture():
    audit = SimpleNamespace(observed_ready={(3, 7, 2, True): dict(event_seq=77)})
    capture = PrepareCapture(audit)
    capture.env = SimpleNamespace(current_time=100.)
    for trip, bus, now in ((5, 2, 100.), (7, 3, 150.), (9, 4, 190.)):
        capture.env.current_time = now
        order = capture.mark('launch')
        capture.launches[trip] = seal_launch_phase(None, trip, bus, True, now, now, order, 1.)
    capture.env.current_time = 200.
    capture.departures[5, 2, True] = dict(status='supported', trip_id=5, bus_id=2, direction=True,
        station_id=2, departure_s=200., departure_order=capture.mark('departure'), launch_order=0)
    capture.env.current_time = 500.
    identities = dict(time_s=500., order=capture.mark('avl_identities'),
                      vehicles=[dict(bus_id=4, trip_id=9, direction=True, on_route=True)])
    sample = dict(time_s=500., order=capture.mark('avl_sample'), origin='native_inline_snapshot',
                  vehicles=[dict(bus_id=4, direction=True, on_route=True)], identity_sample=identities)
    capture.estimates[3] = dict(time_s=500., order=capture.mark('estimate'), snapshot=sample,
                              result=dict(follower_bus_id=4))
    bus = SimpleNamespace(trip_id=7, bus_id=3, direction=True, last_station=SimpleNamespace(station_id=2),
        pre_action_forward_headway=300., _target_headway=360.,
        pre_action_forward_headway_source='matched_departure_event', forward_predecessor_trip_id=5,
        pre_action_follower_departure_gap=420., pre_action_follower_source='same_time_avl_journey_speed_eta',
        pre_action_follower_bus_id=4)
    with capture.read('prepare'):
        capture.prepare(None, (bus, 500.), {})
    q = dict(trip_id=7, bus_id=3, station_id=2, direction=True, prepare_event_seq=77,
        state_captured_s=500., context=cached_context(bus), feasible_bins=list(range(7)),
        base_bin=3, time_s=500., event_seq=88)
    return capture, bus, q


def basis(**kw):
    return dict(native_plan_s=100.2, proposed_plan_s=99.2, applied_plan_s=99.2,
                admission_s=80., admission_order=0, reserved_bus_id=3) | kw


class PrepareCaptureTest(unittest.TestCase):
    def test_frozen_runtime_uses_inline_snapshot_and_cached_bus_identity(self):
        from env.bus import Bus
        from env.scenario import ScenarioTape
        from lower.causal_follower_eta import AVLVehicleSnapshot, FollowerDepartureEstimate
        self.assertEqual(tuple(inspect.signature(Bus._update_pre_action_follower_eta).parameters),
                         ('self', 'current_time', 'bus_all'))
        self.assertEqual([f.name for f in fields(AVLVehicleSnapshot)],
            ['bus_id', 'direction', 'on_route', 'progress_m', 'launch_time_s',
             'current_speed_mps', 'route_speed_mps'])
        self.assertIn('follower_bus_id', {f.name for f in fields(FollowerDepartureEstimate)})
        self.assertTrue(callable(ScenarioTape._rng) and callable(ScenarioTape._stream_rng))

    def test_delivered_phase_uses_ticks_not_continuous_plan(self):
        r = seal_launch_phase(basis(), 7, 3, True, 99.2, 100., 1, 1.)
        self.assertEqual((r['status'], r['phase_s']), ('supported', -1.))
        self.assertEqual(r['launch_event_seq'], r['launch_order'])

    def test_shadow_proposal_does_not_issue_phase_in_v107(self):
        r = seal_launch_phase(basis(applied_plan_s=100.2), 7, 3, True, 100.2, 140., 1, 1.)
        self.assertEqual((r['status'], r['phase_s']), ('known_zero', 0.))

    def test_undelivered_or_overwritten_nonzero_is_not_zero(self):
        for b, bus, plan, actual in ((basis(), 4, 99.2, 100.), (basis(), 3, 99.2, 101.),
                (basis(), 3, 100.2, 101.), (basis(invalidated=True), 3, 99.2, 100.)):
            r = seal_launch_phase(b, 7, bus, True, plan, actual, 1, 1.)
            self.assertEqual((r['status'], r['phase_s']), ('undelivered_plan', None))

    def test_initial_unmodified_trip_is_explicit_zero(self):
        r = seal_launch_phase(None, 7, 3, True, 100., 120., 1, 1.)
        self.assertEqual((r['status'], r['phase_s']), ('known_zero', 0.))

    def test_reused_vehicle_does_not_resolve_a_different_trip(self):
        capture = PrepareCapture(None)
        capture.launches[7] = seal_launch_phase(None, 7, 3, True, 100., 100., 1, 1.)
        who = dict(trip_id=7, bus_id=4, direction=True)
        self.assertEqual(capture.phase(who, 'missing_launch')['status'], 'identity_unresolved')
        self.assertIsNone(empty_phase('no_native_follower')['phase_s'])

    def test_copied_phase_is_not_later_backfilled(self):
        capture = PrepareCapture(None)
        capture.launches[7] = seal_launch_phase(None, 7, 3, True, 100., 100., 1, 1.)
        r = capture.phase(dict(trip_id=7, bus_id=3, direction=True), 'missing_launch')
        capture.launches[7]['phase_s'] = -10.
        self.assertEqual(r['phase_s'], 0.)

    def test_cached_context_preserves_distinct_missing_evidence(self):
        bus = SimpleNamespace(pre_action_forward_headway=300., _target_headway=360.,
            pre_action_forward_headway_source='matched_departure_event',
            pre_action_follower_departure_gap=None, pre_action_follower_source='no_same_direction_avl_follower')
        c = cached_context(bus)
        self.assertTrue(c['evidence_valid'])
        self.assertFalse(c['follower_evidence_valid'])
        self.assertIsNone(c['follower_departure_gap_s'])

    def test_observer_guard_accepts_only_reading(self):
        capture = PrepareCapture(None)
        with capture.read('read'):
            copy.deepcopy(dict(a=[1, 2]))
        self.assertEqual(capture.checked['read'], 1)

    def test_observer_guard_rejects_random_draw_or_native_call(self):
        capture = PrepareCapture(None)
        state = random.getstate()
        try:
            with self.assertRaises(AssertionError):
                with capture.read('bad_rng'):
                    random.random()
        finally:
            random.setstate(state)
        with self.assertRaises(AssertionError):
            with capture.read('bad_call'):
                capture.calls['estimator'] += 1

    def test_hooks_return_original_value_and_detach_exactly(self):
        owner = SimpleNamespace(f=lambda x: x+1)
        original = owner.f
        capture = PrepareCapture(None)
        seen = []
        capture.watch(owner, 'f', 'test', lambda v, a, k: seen.append(v))
        self.assertEqual(owner.f(3), 4)
        self.assertEqual(seen, [4])
        self.assertEqual(capture.calls['test'], 1)
        capture.detach()
        self.assertIs(owner.f, original)

    def test_complete_prepare_packet_joins_without_using_later_identity(self):
        capture, bus, q = fixture()
        bus.pre_action_follower_trip_id = 99
        bus.pre_action_follower_bus_id = 10
        capture.env.current_time = q['time_s']
        capture.query(q)
        self.assertEqual(capture.prepares[0]['follower']['trip_id'], 9)
        self.assertEqual(capture.queries[0]['shadow_bin'], 3)
        self.assertFalse(capture.queries[0]['executed_shadow'])

    def test_repeated_prepares_and_queries_are_not_overwritten(self):
        capture, bus, q = fixture()
        capture.query(q)
        capture.audit.observed_ready[3, 7, 2, True]['event_seq'] = 78
        with capture.read('prepare'):
            capture.prepare(None, (bus, 500.), {})
        capture.query(dict(q, prepare_event_seq=78, event_seq=89))
        self.assertEqual([p['clock']['prepare_occurrence'] for p in capture.prepares], [0, 1])
        self.assertEqual([r['query_occurrence'] for r in capture.queries], [0, 1])
        self.assertNotEqual(capture.queries[0]['prepare_order'], capture.queries[1]['prepare_order'])

    def test_query_context_or_identity_drift_fails(self):
        capture, _, q = fixture()
        for changed in (dict(q, bus_id=9), dict(q, context=dict(q['context'], forward_headway_s=301.))):
            with self.assertRaises(AssertionError):
                capture.query(changed)

    def test_future_phase_or_wrong_avl_sample_clock_fails(self):
        capture, _, _ = fixture()
        packet = capture.prepares[0]
        bad = copy.deepcopy(packet)
        bad['phases']['follower']['launch_order'] = packet['clock']['handoff_order']+1
        with self.assertRaises(AssertionError):
            check_packet(bad)
        bad = copy.deepcopy(packet)
        bad['follower']['avl_sample_s'] += 1.
        with self.assertRaises(AssertionError):
            check_packet(bad)

    def test_saved_complete_sidecar_can_be_checked_without_replay(self):
        capture, _, q = fixture()
        capture.query(q)
        data = dict(capture.data(), complete=True)
        result = validate_saved_capture(data, dict(trace=dict(lower_actions=[{}])))
        self.assertTrue(result['recording_complete'])
        self.assertEqual(result['queries'], 1)
        self.assertEqual(result['shadow_actions_executed'], 0)


if __name__ == '__main__':
    unittest.main()
