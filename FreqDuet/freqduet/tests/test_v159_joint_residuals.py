import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import extract_freqduet_v159_joint_residuals as extract


class ResidualTests(unittest.TestCase):
    def test_day_fields_include_nested_splits_not_actor_ids(self):
        self.assertEqual(extract.day_ids(dict(registration=dict(days=dict(
            calibration=[1001], confirmation=[2001]), archived_actors=[12011],
            sampler_seed=999), cases=[dict(day=3001, seed=12037)])), {1001, 2001, 3001})

    def test_query_inputs_exclude_later_policy_load_and_labels(self):
        row = dict(prediction_time_s=10., policy_time_s=11., query_load=42,
            evidence=dict(bus_id=2, ready_time_s=10., speed_mps=3., future_holding_s=100.),
            chosen_s=60., future_arrival_s=50.)
        state = extract.observable(row, None, 'opaque', 9.)
        self.assertNotIn('query_load', state)
        self.assertNotIn('future_holding_s', state['evidence'])
        self.assertNotIn('chosen_s', state)
        self.assertTrue(state['clearance'])

    def test_unresolved_label_is_missing_not_zero(self):
        row = dict(prediction_time_s=10., context=dict(follower_evidence_valid=False))
        label = extract.label_record(dict(ready={}), row)
        self.assertNotEqual(label['status'], 'resolved')
        self.assertIsNone(label['full_eta_s'])
        self.assertIsNone(label['available_time_s'])

    def test_label_matches_same_physical_vehicle_and_ready_clock(self):
        row = dict(prediction_time_s=10., trip_id=6, station_id=3, direction=True,
            evidence=dict(predicted_follower_trip_id=7, predicted_follower_bus_id=8),
            context=dict(follower_evidence_valid=True))
        path = dict(ready={(7, 3, True): dict(bus_id=8, ready_time_s=25.)})
        label = extract.label_record(path, row)
        self.assertEqual((label['full_eta_s'], label['available_time_s']), (15., 25.))
        self.assertFalse(label['available_time_s'] < 25.)
        self.assertTrue(label['available_time_s'] < 26.)

    def test_coarse_joint_sum_and_inseparable_holding(self):
        packet = dict(current_service=1., downstream_service=2., running=10.,
            holding=3., execution_wait=1., current_ready_joint=2.)
        actual = dict(current_service=2., downstream_service=2., running=12.,
            holding=1., execution_wait=2., current_ready_joint=0.)
        value = extract.components(packet, actual, (1, 2, True))
        self.assertEqual(sum(value['residual_s'].values()), 0.)
        self.assertFalse(value['holding_separable'])
        self.assertIsNone(value['holding_residual_s'])

    def test_changed_moments_stop_extraction(self):
        expected = dict(x=dict(queries=2, episodes=1, metrics=dict(e_running=3.)))
        extract.reconcile(expected, expected)
        with self.assertRaises(AssertionError):
            extract.reconcile(dict(x=dict(queries=2, episodes=1,
                metrics=dict(e_running=4.))), expected)


if __name__ == '__main__':
    unittest.main()
