import unittest

import numpy as np

from freq_hrl.policies import BernsteinPlanCurve


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


if __name__ == "__main__":
    unittest.main()
