import unittest

import numpy as np
import torch

from lower.cost_replay_buffer import CostReplayBuffer
from lower.resac_lagrangian import RESACLagrangianTrainer


def _projection_objective(distillation="forward_kl_v2", steps=4):
    return {
        "enable": True,
        "mode": "analytic_two_sided_hf_aggregate_gain_projection_v11",
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
            "mode": "joint_kl_soft_policy_distillation_v2",
            "regularity_target": 0.036,
            "passenger_target": 0.075,
            "tolerance": 1e-8,
            "max_iterations": 200,
            "support_floor": 1e-12,
            "distillation": distillation,
            "distillation_steps": steps,
        },
    }


def _trainer(distillation="forward_kl_v2", steps=4):
    return RESACLagrangianTrainer(
        state_dim=5,
        action_range=45.0,
        action_bins=[0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
        discrete_critic="zero_hold_advantage",
        ensemble_size=2,
        hidden_dim=8,
        auto_entropy=False,
        policy_sample_seed=2401,
        regularity_policy_objective=(
            _projection_objective(distillation, steps)),
    )


def _replay():
    replay = CostReplayBuffer(64, seed=2411)
    states = (
        np.array([0.5, 0.5, 1.0, 0.04, 0.1], dtype=np.float32),
        np.array([0.6, 0.4, 1.0, 0.02, 0.2], dtype=np.float32),
    )
    for index in range(32):
        state = states[index % len(states)]
        replay.push(state, 0.0, 0.0, 0.0, state, True, index)
    return replay


class V24ProjectedPolicyTrainerTest(unittest.TestCase):
    def test_four_forward_kl_steps_track_teacher_better_than_one(self):
        torch.manual_seed(2421)
        one_step = _trainer(steps=1)
        torch.manual_seed(2421)
        four_steps = _trainer(steps=4)

        torch.manual_seed(2431)
        one_metrics = one_step.update(_replay(), 16, reward_scale=1.0)
        torch.manual_seed(2431)
        four_metrics = four_steps.update(_replay(), 16, reward_scale=1.0)

        self.assertEqual(
            one_metrics["regularity_projection_actor_distillation_steps"],
            1.0,
        )
        self.assertEqual(
            four_metrics["regularity_projection_actor_distillation_steps"],
            4.0,
        )
        self.assertLess(
            four_metrics["regularity_projection_actor_post_forward_kl"],
            one_metrics["regularity_projection_actor_post_forward_kl"],
        )
        self.assertLess(
            abs(four_metrics[
                "regularity_projection_actor_post_target_action_change_mean_s"]),
            abs(one_metrics[
                "regularity_projection_actor_post_target_action_change_mean_s"]),
        )
        self.assertLessEqual(
            four_metrics["regularity_projection_target_regularity_cost"],
            0.036 + 1e-8,
        )
        self.assertLessEqual(
            four_metrics["regularity_projection_target_passenger_cost"],
            0.075 + 1e-8,
        )

    def test_reverse_kl_multistep_is_an_explicit_control(self):
        torch.manual_seed(2441)
        trainer = _trainer(distillation="reverse_kl_v1", steps=4)
        metrics = trainer.update(_replay(), 16, reward_scale=1.0)

        projection = trainer.regularity_policy_contract[
            "categorical_projection"]
        self.assertEqual(projection["distillation"], "reverse_kl_v1")
        self.assertEqual(projection["distillation_steps"], 4)
        self.assertEqual(
            metrics["regularity_projection_actor_distillation_steps"], 4.0)
        self.assertTrue(np.isfinite(
            metrics["regularity_projection_actor_post_reverse_kl"]))

    def test_unregistered_step_count_is_rejected(self):
        objective = _projection_objective(steps=2)
        with self.assertRaisesRegex(ValueError, "one of 1, 4, or 8"):
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
