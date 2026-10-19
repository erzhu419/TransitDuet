import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v118_follower_identity as run


class FollowerIdentityTests(unittest.TestCase):
    def test_capture_uses_frozen_forecast_trip(self):
        bus = SimpleNamespace(bus_id=1, last_service_dwell_s=20., pre_action_follower_bus_id=2,
            pre_action_follower_source='same_time_avl_journey_speed_eta', pre_action_follower_eta=100.,
            pre_action_follower_spatial_gap=500., pre_action_follower_speed=5.)
        identities = {2: 10}
        record = run.capture_prediction(bus, 1000., identities)
        identities[2] = 12
        self.assertEqual(record['predicted_follower_trip_id'], 10)
        self.assertEqual(record['ready_time_s'], 1000.)

    def test_identity_gap_and_same_target_error_add_exactly(self):
        p = run.error_parts(400., 380., 280.)
        self.assertEqual(p['next_service_error_s'], p['same_ready_error_s']+p['identity_gap_component_s'])
        self.assertEqual(p['same_ready_error_s'], 20.)

    def test_resolved_same_trip_checks_physical_bus_and_error_components(self):
        key = 2, 3, True
        evidence = dict(predicted_follower_bus_id=9, follower_gap_prediction_s=140.,
            eta_prediction_s=120., service_dwell_proxy_s=20.)
        ready = {key: dict(bus_id=9, ready_time_s=230.)}
        label = run.resolved_label(evidence, 100., key, ready,
            {key: dict(time_s=200.)}, {key: dict(time_s=240.)})
        self.assertTrue(label['same_ready_resolved'])
        self.assertEqual(label['same_eta_error_s'], 20.)
        self.assertEqual(label['same_dwell_error_s'], -10.)
        self.assertEqual(label['same_ready_error_s'], 10.)
        bad = run.resolved_label(dict(evidence, predicted_follower_bus_id=8), 100., key, ready,
            {key: dict(time_s=200.)}, {key: dict(time_s=240.)})
        self.assertFalse(bad['same_ready_resolved'])
        self.assertNotIn('same_ready_error_s', bad)

    def test_missing_or_past_ready_is_not_zero_error(self):
        e = dict(predicted_follower_bus_id=9)
        missing = run.resolved_label(e, 100., (2, 3, True), {}, {}, {})
        self.assertFalse(missing['same_ready_resolved'])
        key = 2, 3, True
        past = run.resolved_label(e, 100., key, {key: dict(bus_id=9, ready_time_s=99.)},
            {key: dict(time_s=80.)}, {key: dict(time_s=110.)})
        self.assertFalse(past['same_ready_temporal_valid'])
        self.assertNotIn('same_ready_error_s', past)

    def test_original_action_or_trace_length_change_rejected(self):
        old = [dict(trip_id=2, station_id=3, chosen_s=5.)]
        run.assert_decisions(old, old)
        with self.assertRaises(AssertionError):
            run.assert_decisions(old, [])
        with self.assertRaises((AssertionError, ValueError)):
            run.assert_decisions(old, [dict(old[0], chosen_s=10.)])

    def test_non_neighbour_target_can_reverse_local_pair_sign(self):
        self.assertLess(run.prior.pair_restore_change(300., 450., 5., 10.), 0.)
        self.assertGreater(run.prior.pair_restore_change(300., 250., 5., 10.), 0.)


if __name__ == '__main__':
    unittest.main()
