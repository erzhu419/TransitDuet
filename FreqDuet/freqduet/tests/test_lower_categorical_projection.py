import unittest

import torch

from lower.categorical_projection import joint_weighted_kl_projection


class JointWeightedCategoricalProjectionTest(unittest.TestCase):
    def test_joint_projection_meets_both_limits(self):
        base_logits = torch.zeros((8, 3), dtype=torch.float32)
        regularity = torch.tensor(
            [[1.0, 0.04, 0.0]] * 8, dtype=torch.float32)
        passenger = torch.tensor(
            [[0.0, 0.04, 1.0]] * 8, dtype=torch.float32)

        result = joint_weighted_kl_projection(
            base_logits,
            torch.stack((regularity, passenger), dim=0),
            torch.ones_like(base_logits, dtype=torch.bool),
            torch.ones(8),
            torch.tensor([0.05, 0.05]),
        )

        self.assertTrue(result["converged"])
        self.assertLessEqual(result["expected_costs"][0], 0.05 + 1e-8)
        self.assertLessEqual(result["expected_costs"][1], 0.05 + 1e-8)
        torch.testing.assert_close(
            result["probabilities"].sum(dim=-1),
            torch.ones(8, dtype=torch.float64),
        )
        self.assertGreater(result["weighted_kl"], 0.0)

    def test_projection_uses_weights_and_excludes_masked_actions(self):
        logits = torch.tensor([
            [0.0, 1.0, 5.0],
            [0.0, 1.0, 5.0],
        ])
        costs = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]).unsqueeze(0)
        feasible = torch.tensor([
            [True, True, False],
            [True, True, False],
        ])
        weights = torch.tensor([9.0, 1.0])

        result = joint_weighted_kl_projection(
            logits, costs, feasible, weights, torch.tensor([0.2]))

        self.assertTrue(result["converged"])
        self.assertTrue(torch.equal(
            result["probabilities"][:, 2], torch.zeros(2, dtype=torch.float64)))
        manual = (
            0.9 * result["probabilities"][0, 1]
            + 0.1 * result["probabilities"][1, 0])
        torch.testing.assert_close(result["expected_costs"][0], manual)
        self.assertLessEqual(manual, 0.2 + 1e-8)

    def test_inactive_constraints_return_the_supported_soft_policy(self):
        logits = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
        costs = torch.zeros((2, 2, 2))

        result = joint_weighted_kl_projection(
            logits,
            costs,
            torch.ones_like(logits, dtype=torch.bool),
            torch.tensor([1.0, 3.0]),
            torch.tensor([0.1, 0.2]),
        )

        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 1)
        torch.testing.assert_close(
            result["probabilities"], result["base_probabilities"])
        torch.testing.assert_close(
            result["multipliers"], torch.zeros(2, dtype=torch.float64))

    def test_rejects_a_row_without_an_executable_action(self):
        with self.assertRaisesRegex(ValueError, "feasible action"):
            joint_weighted_kl_projection(
                torch.zeros((1, 2)),
                torch.zeros((1, 1, 2)),
                torch.zeros((1, 2), dtype=torch.bool),
                torch.ones(1),
                torch.ones(1),
            )


if __name__ == "__main__":
    unittest.main()
