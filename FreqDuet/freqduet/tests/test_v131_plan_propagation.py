import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v131_plan_propagation import launch_state, clipped_parts, allocate_supply, stop_actions


class PlanPropagationTests(unittest.TestCase):
    def test_control_wait_can_be_masked_by_vehicle_unavailability(self):
        row = dict(actual_s=200., eligible_s=130., first_dispatch_visible_s=200.)
        state = launch_state(row, 100.)
        self.assertEqual(state['control_s'], 0.)
        self.assertEqual(state['supply_s'], 100.)
        self.assertEqual(state['residual_s'], 0.)
        row.update(actual_s=130., first_dispatch_visible_s=100.)
        self.assertEqual(launch_state(row, 100.)['control_s'], 30.)

    def test_clipping_allocates_threshold_crossing_without_order_bias(self):
        self.assertEqual(clipped_parts(-10., 10., dict(launch=30., held=-10.)), dict(launch=15., held=-5.))
        self.assertEqual(clipped_parts(10., -10., dict(launch=-30., held=10.)), dict(launch=-15., held=5.))

    def test_zero_net_change_preserves_cancellation_when_supply_bound(self):
        self.assertEqual(clipped_parts(10., 10., dict(launch=5., held=-5.)), dict(launch=5., held=-5.))

    def test_unknown_vehicle_mapping_is_not_encoded_as_no_effect(self):
        result = allocate_supply(dict(supply_s=10.), dict(supply_s=20.), None)
        self.assertEqual(result, dict(unmatched_supply_s=10.))

    def test_positive_residual_is_retained_and_future_vehicle_rejected(self):
        row = dict(actual_s=140., eligible_s=130., first_dispatch_visible_s=100.)
        self.assertEqual(launch_state(row, 100.)['residual_s'], 10.)
        row['first_dispatch_visible_s'] = 150.
        with self.assertRaises(AssertionError):
            launch_state(row, 100.)

    def test_stop_action_uses_increment_not_cumulative_holding(self):
        arrival = dict(trip_id=1, station_id=2, direction=True, time_s=100., release=0.,
            realized_holding=30., other_elapsed=70.)
        departure = dict(arrival, time_s=130., realized_holding=45., other_elapsed=85.)
        action = stop_actions([arrival], [departure])[1, 2, 1]
        self.assertEqual(action['action_s'], 15.)
        self.assertEqual(action['incoming_holding_s'], 30.)


if __name__ == '__main__':
    unittest.main()
