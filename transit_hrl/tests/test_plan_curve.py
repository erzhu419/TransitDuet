import unittest

import numpy as np

from freq_hrl.policies import BernsteinPlanCurve, CausalPlanCurveState
from freq_transitduet.upper.timetable_planner import TimetableCurvePlanner


class PlanCurveTest(unittest.TestCase):
    def test_curve_evaluates_inside_bounds(self):
        curve = BernsteinPlanCurve(
            horizon_s=100.0,
            basis_dim=4,
            min_value=0.0,
            max_value=10.0,
            delta_min=-2.0,
            delta_max=2.0,
        )
        action = np.array([2.0, 2.0, 2.0, 2.0])
        self.assertAlmostEqual(curve.value_at(5.0, action, 50.0), 7.0)
        clipped = curve.value_at(9.0, action, 50.0)
        self.assertEqual(clipped, 10.0)

    def test_smoothness_penalty_detects_curvature(self):
        curve = BernsteinPlanCurve(basis_dim=4, delta_min=-2.0, delta_max=2.0)
        smooth = np.array([0.0, 0.0, 0.0, 0.0])
        curved = np.array([0.0, 2.0, -2.0, 0.0])
        self.assertGreater(curve.smoothness_penalty(curved), curve.smoothness_penalty(smooth))

    def test_causal_plan_state_reuses_active_curve(self):
        curve = BernsteinPlanCurve(
            horizon_s=10.0,
            basis_dim=2,
            min_value=-1.0,
            max_value=1.0,
            delta_min=-1.0,
            delta_max=1.0,
            n_entities=2,
        )
        state = CausalPlanCurveState(
            curve=curve,
            replan_interval_s=10.0,
            desired_change_threshold=2.0,
            gross_cap=None,
        )
        first = state.target_toward(0.0, [0.0, 0.0], [0.8, -0.4])
        mid = state.target_toward(5.0, [0.0, 0.0], [0.8, -0.4])
        self.assertTrue(first["replan"])
        self.assertFalse(mid["replan"])
        np.testing.assert_allclose(mid["target"], [0.4, -0.2], atol=1e-8)
        self.assertEqual(state.decisions, 1)
        self.assertEqual(state.reuses, 1)

    def test_transit_timetable_planner_uses_shared_curve_semantics(self):
        class Trip:
            def __init__(self, launch_time, launch_turn, direction):
                self.launch_time = launch_time
                self.launch_turn = launch_turn
                self.direction = direction
                self.target_headway = 360.0
                self.launched = False

        planner = TimetableCurvePlanner(
            horizon_s=600.0,
            basis_per_direction=2,
            min_headway_s=180.0,
            max_headway_s=720.0,
            delta_min_s=-120.0,
            delta_max_s=120.0,
        )
        trips = [Trip(0.0, 0, True), Trip(300.0, 1, True), Trip(300.0, 2, False)]
        summary = planner.apply(trips, trips[0], [0.0, 60.0, 0.0, -30.0])
        self.assertAlmostEqual(summary["target_headway"], 360.0)
        self.assertAlmostEqual(trips[1].target_headway, 390.0)
        self.assertAlmostEqual(trips[2].target_headway, 360.0)


if __name__ == "__main__":
    unittest.main()
