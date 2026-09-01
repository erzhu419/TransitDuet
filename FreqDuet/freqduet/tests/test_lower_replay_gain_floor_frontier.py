import unittest

import numpy as np

from scripts.audit_lower_replay_gain_floor_frontier import (
    _gain_floor_cost_arrays,
    _minimum_primary_at_constraint,
)


class GainFloorFrontierTest(unittest.TestCase):
    def test_frontier_mixes_adjacent_deterministic_policies(self):
        primary = np.asarray([[0.0, 1.0], [0.0, 1.0]])
        constraint = np.asarray([[1.0, 0.0], [1.0, 0.0]])
        feasible = np.ones_like(primary, dtype=bool)

        result = _minimum_primary_at_constraint(
            primary, constraint, feasible, 0.5)

        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["minimum_primary_mean"], 0.5)
        self.assertAlmostEqual(result["achieved_constraint_mean"], 0.5)

    def test_frontier_reports_an_infeasible_constraint(self):
        primary = np.asarray([[0.0, 1.0]])
        constraint = np.asarray([[1.0, 0.0]])
        feasible = np.asarray([[True, False]])

        result = _minimum_primary_at_constraint(
            primary, constraint, feasible, 0.5)

        self.assertFalse(result["feasible"])
        self.assertEqual(result["minimum_constraint_mean"], 1.0)
        self.assertIsNone(result["minimum_primary_mean"])

    def test_gain_floor_distinguishes_relative_and_absolute_shortfall(self):
        zero_hold = np.asarray([0.04, 0.0])
        absolute = np.asarray([
            [0.04, 0.01, 0.0],
            [0.0, 0.01, 0.02],
        ])
        required = np.asarray([0.5, 0.5])

        result = _gain_floor_cost_arrays(zero_hold, absolute, required)

        np.testing.assert_allclose(
            result["relative_shortfall"][0], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(
            result["absolute_shortfall"][0], [0.02, 0.0, 0.0])
        np.testing.assert_allclose(result["relative_shortfall"][1], 0.0)
        self.assertEqual(result["eligible"].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
