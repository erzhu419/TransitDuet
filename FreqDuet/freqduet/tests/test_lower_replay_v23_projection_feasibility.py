import unittest

import numpy as np

from scripts.audit_lower_replay_v23_projection_feasibility import (
    _audit_minibatches,
    _project_indices,
)


def _table(*, jointly_feasible=True):
    states = 8
    if jointly_feasible:
        regularity = np.tile([1.0, 0.04, 0.0], (states, 1))
        passenger = np.tile([0.0, 0.04, 1.0], (states, 1))
    else:
        regularity = np.tile([0.0, 1.0], (states, 1))
        passenger = np.tile([1.0, 0.0], (states, 1))
    actions = regularity.shape[1]
    return {
        "valid": np.ones(states, dtype=bool),
        "required_gain": np.ones(states),
        "absolute_shortfall": regularity,
        "passenger_costs": passenger,
        "probabilities": np.full((states, actions), 1.0 / actions),
        "feasible": np.ones((states, actions), dtype=bool),
        "action_bins_s": np.arange(actions, dtype=float),
    }


class V23ProjectionFeasibilityTest(unittest.TestCase):
    def test_full_subset_meets_both_locked_targets(self):
        result = _project_indices(
            _table(), np.arange(8),
            regularity_target=0.05, passenger_target=0.05)

        self.assertTrue(result["passes"])
        self.assertLessEqual(result["projected_regularity_cost"], 0.05 + 1e-8)
        self.assertLessEqual(result["projected_passenger_cost"], 0.05 + 1e-8)

    def test_deterministic_minibatches_all_pass(self):
        result = _audit_minibatches(
            _table(), regularity_target=0.05, passenger_target=0.05,
            batch_size=4, batch_count=5, sample_seed=17)

        self.assertTrue(result["passes"])
        self.assertEqual(result["pass_count"], 5)
        self.assertEqual(result["failed_batches"], [])

    def test_reports_infeasible_minibatches(self):
        result = _audit_minibatches(
            _table(jointly_feasible=False),
            regularity_target=0.1, passenger_target=0.1,
            batch_size=4, batch_count=3, sample_seed=17)

        self.assertFalse(result["passes"])
        self.assertEqual(result["joint_feasible_count"], 0)
        self.assertEqual(len(result["failed_batches"]), 3)


if __name__ == "__main__":
    unittest.main()
