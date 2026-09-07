import unittest

import numpy as np
import torch

from lower.cost_replay_buffer import CostReplayBuffer
from lower.resac_lagrangian import RESACLagrangianTrainer


def _projection_objective():
    return {
        "enable": True,
        "mode": "analytic_two_sided_hf_aggregate_gain_projection_v10",
        "target_feature_index": 1,
        "valid_feature_index": 2,
        "target_headway_feature_index": 0,
        "action_target_scale_s": 45.0,
        "target_headway_scale_s": 600.0,
        "cost_limit": 0.05,
        "cost_cap": 1.0,
        "constraint_scale_mode": "cost_limit_ratio_v1",
        "lambda_lr": 1e-3,
        "lambda_min": 1e-4,
        "lambda_max": 2.0,
        "initial_lambda": 0.01,
        "dual_update_mode": "exact_projection_v1",
        "augmented_lagrangian_rho": 0.0,
        "regularity_gain_floor": {
            "enable": True,
            "mode": "causal_hf_aggregate_gain_floor_v2",
            "hf_energy_feature_index": 3,
            "base_fraction": 0.30,
            "hf_increment": 0.30,
            "hf_energy_scale": 0.04,
            "hf_energy_exponent": 1.0,
        },
        "passenger_holding_constraint": {
            "enable": True,
            "mode": "causal_apc_person_delay_dual_v1",
            "load_feature_index": 4,
            "action_norm_s": 45.0,
            "load_clip": 1.0,
            "cost_limit": 0.08,
            "constraint_scale_mode": "cost_limit_ratio_v1",
            "lambda_lr": 1e-3,
            "lambda_min": 1e-4,
            "lambda_max": 2.0,
            "initial_lambda": 0.01,
            "dual_update_mode": "exact_projection_v1",
            "augmented_lagrangian_rho": 0.0,
        },
        "categorical_projection": {
            "enable": True,
            "mode": "joint_kl_soft_policy_target_v1",
            "regularity_target": 0.036,
            "passenger_target": 0.075,
            "tolerance": 1e-8,
            "max_iterations": 200,
            "support_floor": 1e-12,
        },
    }


def _trainer():
    return RESACLagrangianTrainer(
        state_dim=5,
        action_range=45.0,
        action_bins=[0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
        discrete_critic="zero_hold_advantage",
        ensemble_size=2,
        hidden_dim=8,
        auto_entropy=False,
        regularity_policy_objective=_projection_objective(),
    )


class V23ProjectedPolicyTrainerTest(unittest.TestCase):
    def test_exact_target_meets_locked_costs(self):
        trainer = _trainer()
        state = torch.tensor([
            [0.5, 0.5, 1.0, 0.04, 0.1],
            [0.6, 0.4, 1.0, 0.02, 0.2],
        ] * 4)
        q_lcb = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 8)
        safety_cost = torch.zeros_like(q_lcb)

        result = trainer._regularity_projected_soft_policy_target(
            state, q_lcb, safety_cost, torch.ones(8), torch.tensor(1.0))

        self.assertTrue(result["applied"])
        self.assertTrue(result["converged"])
        self.assertLessEqual(result["expected_costs"][0], 0.036 + 1e-8)
        self.assertLessEqual(result["expected_costs"][1], 0.075 + 1e-8)
        self.assertGreater(result["weighted_kl"], 0.0)

    def test_update_distills_projection_without_soft_duals_and_round_trips(self):
        torch.manual_seed(2309)
        trainer = _trainer()
        replay = CostReplayBuffer(64, seed=2311)
        state = np.array([0.5, 0.5, 1.0, 0.04, 0.1], dtype=np.float32)
        for index in range(32):
            replay.push(state, 0.0, 0.0, 0.0, state, True, index)

        metrics = trainer.update(replay, 16, reward_scale=1.0)

        self.assertEqual(metrics["regularity_projection_enabled"], 1.0)
        self.assertEqual(metrics["regularity_projection_applied"], 1.0)
        self.assertEqual(metrics["regularity_projection_converged"], 1.0)
        self.assertLessEqual(
            metrics["regularity_projection_target_regularity_cost"],
            0.036 + 1e-8,
        )
        self.assertLessEqual(
            metrics["regularity_projection_target_passenger_cost"],
            0.075 + 1e-8,
        )
        self.assertEqual(metrics["regularity_policy_penalty"], 0.0)
        self.assertEqual(
            metrics["regularity_passenger_holding_penalty"], 0.0)
        self.assertFalse(trainer.regularity_soft_dual_enabled)
        self.assertFalse(trainer.regularity_passenger_soft_dual_enabled)
        self.assertIsNone(trainer.log_regularity_lambda)
        self.assertIsNone(trainer.log_regularity_passenger_lambda)

        state_dict = trainer.training_state_dict()
        self.assertIsNone(state_dict["log_regularity_lambda"])
        self.assertIsNone(state_dict["log_regularity_passenger_lambda"])
        restored = _trainer()
        restored.load_training_state_dict(state_dict)
        self.assertEqual(
            restored.regularity_policy_contract,
            trainer.regularity_policy_contract,
        )
        for expected, observed in zip(
                trainer.policy_net.parameters(),
                restored.policy_net.parameters()):
            torch.testing.assert_close(observed, expected)

    def test_projection_target_is_fail_closed(self):
        objective = _projection_objective()
        objective["categorical_projection"]["regularity_target"] = 0.037

        with self.assertRaisesRegex(ValueError, "locked at 0.036"):
            RESACLagrangianTrainer(
                state_dim=5,
                action_range=45.0,
                action_bins=[0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
                ensemble_size=2,
                hidden_dim=8,
                regularity_policy_objective=objective,
            )


if __name__ == "__main__":
    unittest.main()
