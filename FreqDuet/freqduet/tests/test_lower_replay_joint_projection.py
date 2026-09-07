import unittest

import numpy as np

from scripts.audit_lower_replay_joint_projection import _joint_kl_projection


class JointReplayProjectionTest(unittest.TestCase):
    def test_joint_projection_meets_two_active_constraints(self):
        base = np.asarray([[0.45, 0.10, 0.45]])
        regularity = np.asarray([[0.0, 0.4, 1.0]])
        passenger = np.asarray([[1.0, 0.4, 0.0]])
        feasible = np.ones_like(base, dtype=bool)

        result = _joint_kl_projection(
            base,
            regularity,
            passenger,
            feasible,
            regularity_limit=0.45,
            passenger_limit=0.45,
        )

        self.assertTrue(result["joint_feasible"])
        self.assertTrue(result["converged"])
        self.assertLessEqual(result["expected_costs"][0], 0.45 + 1e-8)
        self.assertLessEqual(result["expected_costs"][1], 0.45 + 1e-8)
        np.testing.assert_allclose(
            np.asarray(result["probabilities"]).sum(axis=1), 1.0)
        self.assertGreater(result["multipliers"][0], 0.0)
        self.assertGreater(result["multipliers"][1], 0.0)

    def test_feasible_base_distribution_is_unchanged(self):
        base = np.asarray([[0.8, 0.2], [0.6, 0.4]])
        regularity = np.asarray([[0.0, 0.2], [0.0, 0.2]])
        passenger = np.asarray([[0.0, 0.3], [0.0, 0.3]])
        feasible = np.ones_like(base, dtype=bool)

        result = _joint_kl_projection(
            base,
            regularity,
            passenger,
            feasible,
            regularity_limit=0.2,
            passenger_limit=0.3,
        )

        self.assertTrue(result["converged"])
        np.testing.assert_allclose(result["multipliers"], [0.0, 0.0])
        np.testing.assert_allclose(result["probabilities"], base)

    def test_reports_empty_joint_frontier(self):
        base = np.asarray([[0.5, 0.5]])
        regularity = np.asarray([[0.0, 1.0]])
        passenger = np.asarray([[1.0, 0.0]])
        feasible = np.ones_like(base, dtype=bool)

        result = _joint_kl_projection(
            base,
            regularity,
            passenger,
            feasible,
            regularity_limit=0.1,
            passenger_limit=0.1,
        )

        self.assertFalse(result["joint_feasible"])
        self.assertFalse(result["converged"])
        self.assertIsNone(result["probabilities"])


if __name__ == "__main__":
    unittest.main()
