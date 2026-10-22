import unittest

import numpy as np
from upper.timetable_planner import TimetableCurvePlanner

import freqduet_v219_projection as native_preview
import freqduet_v220_supply_projection as supply
from test_v219_projection import query


class SupplyProjectionTest(unittest.TestCase):
    def setUp(self):
        self.planner = TimetableCurvePlanner(basis_per_direction=2, shared_directions=True,
            plan_all_directions=False, coefficient_parameterization='antisymmetric_linear_v5',
            terminal_schedule_mode='exact_headway_curve', headway_budget_mode='rolling_zero_sum_delta_v6',
            delta_min_s=-60, delta_max_s=60)

    def run_projection(self, q, action=10.5):
        q['direction'] = True
        trips = native_preview.query_timetable(q, [])
        unchanged = [vars(t).copy() for t in trips]
        native = native_preview.preview(self.planner, trips, q, [0.])
        desired = native_preview.preview(self.planner, trips, q, [action])
        result, evidence = supply.project(self.planner, q, trips, native, desired)
        self.assertEqual(unchanged, [vars(t) for t in trips])
        return native, desired, result, evidence

    def test_joint_solution_not_post_masked_times(self):
        q = query((1, 2))
        native, desired, result, evidence = self.run_projection(q)
        self.assertFalse(native_preview.inspect_proposal(q, native, desired, 1.)['supported'])
        self.assertTrue(native_preview.inspect_proposal(q, native, result, 1.)['supported'])
        self.assertLess(result[42]['planned_s'], native[42]['planned_s'])
        self.assertEqual(result[44]['planned_s'], native[44]['planned_s'])
        self.assertGreater(result[44]['target_s'], native[44]['target_s'])
        self.assertEqual(result[46]['planned_s'], native[46]['planned_s'])
        self.assertEqual(evidence['variable_trip_ids'], [42])
        self.assertLess(evidence['objective_after'], evidence['objective_before'])
        self.assertLess(evidence['max_headway_error_s'], 1e-6)

    def test_no_stock_or_only_anchor_stock_means_native(self):
        for count in (0, 1):
            native, _, result, evidence = self.run_projection(query((1, count)))
            self.assertEqual(result, native)
            self.assertEqual(evidence['variable_trip_ids'], [])

    def test_zero_request_reproduces_native_ticks(self):
        q = query()
        native, _, result, _ = self.run_projection(q, 0.)
        native_preview.verify_native(result, native, 1.)

    def test_older_due_trip_reserves_vehicle_before_mutable_slot(self):
        q = query((1, 2))
        q['plans'].append(dict(trip_id=38, direction=True, public_s=6840., planned_s=6840., target_s=360.))
        native, _, result, evidence = self.run_projection(q)
        self.assertEqual(native, result)
        self.assertEqual(evidence['variable_trip_ids'], [])

    def test_due_clock_never_pushed_back_to_query_time(self):
        q = query((1, 2))
        q['time_s'] = 7600.
        native, _, result, _ = self.run_projection(q)
        self.assertEqual(native, result)

    def test_bounds_and_block_closures_hold_at_large_request(self):
        q = query((4, 4))
        native, _, result, evidence = self.run_projection(q, 60.)
        self.assertEqual(result[40], native[40])
        self.assertEqual(result[46]['planned_s'], native[46]['planned_s'])
        for t in (42, 44, 46):
            self.assertGreaterEqual(result[t]['target_s'], 300.-1e-6)
            self.assertLessEqual(result[t]['target_s'], 420.+1e-6)
        np.testing.assert_allclose(evidence['rolling_budget_residual_s'], 0., atol=1e-6)
        self.assertEqual(supply.block_ends([360, 720, 1080, 1440, 1800, 2160, 2520], 900), [2, 5, 6])

    def test_next_cycle_supported_stock_does_not_authorize_tail_clock_change(self):
        q = query((6, 6))
        q['plans'].extend(dict(trip_id=40+2*i, direction=True, public_s=7200.+360*i,
                              planned_s=7200.+360*i, target_s=360.) for i in (4, 5, 6, 7))
        native, _, result, _ = self.run_projection(q)
        for t in (46, 48, 50, 52, 54):
            self.assertEqual(result[t]['planned_s'], native[t]['planned_s'])


if __name__ == '__main__':
    unittest.main()
