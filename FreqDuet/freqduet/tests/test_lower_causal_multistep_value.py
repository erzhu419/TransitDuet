import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lower.causal_multistep_value import (
    CausalMultiStepRegularityObjective,
    CausalMultiStepValueReplay,
)
from lower.resac_lagrangian import RESACLagrangianTrainer


def objective_config(**overrides):
    config = {
        "enable": True,
        "mode": "discounted_future_arrival_cost_change_v1",
        "horizon_steps": 3,
        "discount": 1.0,
        "ucb_beta": 0.0,
        "min_replay_size": 1,
        "min_critic_updates": 1,
        "replay_capacity": 100,
        "hidden_dim": 8,
        "ensemble_size": 2,
        "n_layers": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    config.update(overrides)
    return config


class _MainReplay:
    def sample(self, batch_size):
        state = np.asarray([
            [0.2, 1.0, 1.0],
            [0.4, 1.0, 1.0],
        ], dtype=np.float32)
        action = np.asarray([[0.0], [5.0]], dtype=np.float32)
        reward = np.asarray([[-0.1], [-0.2]], dtype=np.float32)
        cost = np.asarray([[0.01], [0.02]], dtype=np.float32)
        next_state = state + 0.01
        done = np.zeros((2, 1), dtype=np.float32)
        trip_ids = np.asarray([100, 101], dtype=np.int64)
        return state, action, reward, cost, next_state, done, trip_ids


class CausalMultiStepValueTest(unittest.TestCase):
    def test_exact_horizon_target_is_future_mean_minus_decision_cost(self):
        replay = CausalMultiStepValueReplay(
            capacity=20, horizon_steps=3, discount=1.0, seed=3)
        state = np.asarray([1.0, 2.0], dtype=np.float32)
        self.assertFalse(replay.append(state, 0.0, 0.2, 0.1, 7))
        self.assertFalse(replay.append(state, 5.0, 0.4, 0.2, 7))
        self.assertTrue(replay.append(state, 10.0, 0.6, 0.3, 7))
        self.assertEqual(len(replay), 1)
        _, action, target, stream_id = replay.sample(1)
        self.assertEqual(float(action[0, 0]), 0.0)
        self.assertAlmostEqual(float(target[0, 0]), 0.3, places=6)
        self.assertEqual(int(stream_id[0]), 7)

    def test_trip_and_episode_boundaries_never_complete_short_targets(self):
        replay = CausalMultiStepValueReplay(
            capacity=20, horizon_steps=3, discount=1.0, seed=3)
        state = np.asarray([1.0], dtype=np.float32)
        replay.append(state, 0.0, 0.1, 0.2, 1)
        replay.append(state, 0.0, 0.1, 0.2, 2)
        replay.append(state, 0.0, 0.1, 0.2, 1, done=True)
        self.assertEqual(len(replay), 0)
        self.assertEqual(replay.terminal_tails_discarded, 2)
        self.assertEqual(replay.end_episode(), 1)
        self.assertEqual(replay.episode_tails_discarded, 1)
        self.assertEqual(replay.pending, {})

    def _objective(self, **overrides):
        return CausalMultiStepRegularityObjective(
            state_dim=3,
            action_candidates=[0.0, 5.0, 10.0],
            config=objective_config(**overrides),
            replay_seed=9,
        )

    def _populate_one_target(self, objective):
        state = np.asarray([0.2, 1.0, 1.0], dtype=np.float32)
        for index, cost in enumerate((0.2, 0.3, 0.4)):
            objective.observe(
                state=state + index * 0.01,
                action=[0.0, 5.0, 10.0][index],
                outcome_cost=cost,
                baseline_cost=0.1,
                stream_id=12,
                done=index == 2,
            )

    def test_policy_cost_is_positive_value_regret_against_zero_hold(self):
        objective = self._objective()
        self._populate_one_target(objective)
        objective.critic_updates = 1
        with torch.no_grad():
            for weight, bias in zip(
                    objective.critic.weights, objective.critic.biases):
                weight.zero_()
                bias.zero_()
            objective.critic.biases[-1][:, 0, :] = torch.tensor(
                [0.0, 0.1, -0.1])
        state = torch.zeros((1, 3), dtype=torch.float32)
        probs = torch.full((1, 3), 1.0 / 3.0)
        expected, valid, costs = objective.policy_cost(
            state, probs, torch.ones(1), cost_cap=0.25)
        self.assertTrue(torch.allclose(
            costs, torch.tensor([[0.0, 0.1, 0.0]])))
        self.assertAlmostEqual(float(expected.item()), 1.0 / 30.0, places=6)
        self.assertEqual(float(valid.item()), 1.0)

    def test_training_checkpoint_restores_replay_optimizer_and_rng(self):
        objective = self._objective()
        self._populate_one_target(objective)
        objective.update(batch_size=1)
        state = objective.training_state_dict()
        restored = self._objective()
        restored.load_training_state_dict(state)
        self.assertEqual(restored.contract, objective.contract)
        self.assertEqual(restored.critic_updates, objective.critic_updates)
        self.assertEqual(len(restored.replay), len(objective.replay))
        for left, right in zip(
                restored.critic.parameters(), objective.critic.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_trainer_updates_multistep_critic_and_actor_objective(self):
        objective = self._objective(horizon_steps=2)
        state = np.asarray([0.2, 1.0, 1.0], dtype=np.float32)
        objective.observe(state, 0.0, 0.2, 0.1, 1)
        objective.observe(state, 5.0, 0.3, 0.1, 1)
        objective.observe(state, 10.0, 0.4, 0.1, 1, done=True)
        trainer = RESACLagrangianTrainer(
            state_dim=3,
            action_bins=[0.0, 5.0, 10.0],
            action_range=10.0,
            hidden_dim=8,
            ensemble_size=2,
            weight_reg_mode="mean",
            regularity_policy_objective={
                "enable": True,
                "mode": "causal_multistep_arrival_delta_regret_dual_v12",
                "target_feature_index": 0,
                "valid_feature_index": 1,
                "target_headway_feature_index": 2,
                "action_target_scale_s": 10.0,
                "target_headway_scale_s": 360.0,
                "cost_limit": 0.001,
                "cost_cap": 0.25,
                "constraint_scale_mode": "cost_limit_ratio_v1",
                "lambda_lr": 1e-3,
                "lambda_min": 1e-4,
                "lambda_max": 2.0,
                "initial_lambda": 0.01,
                "multi_step_value": objective.contract,
            },
            multistep_regularity_objective=objective,
        )
        metrics = trainer.update(
            _MainReplay(), batch_size=2, reward_scale=1.0)
        self.assertEqual(metrics["multistep_value_enabled"], 1.0)
        self.assertEqual(metrics["multistep_value_ready"], 1.0)
        self.assertEqual(metrics["multistep_value_critic_updates"], 1.0)
        self.assertTrue(np.isfinite(
            metrics["multistep_value_critic_loss"]))
        self.assertTrue(np.isfinite(metrics["policy_loss"]))

    def test_deployment_checkpoint_rejects_different_horizon(self):
        objective = self._objective(horizon_steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save(objective.deployment_state_dict(), path)
            state = torch.load(path, weights_only=True)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                self._objective(horizon_steps=4).load_deployment_state_dict(
                    state)

    def test_deployment_checkpoint_preserves_ready_value_policy(self):
        objective = self._objective()
        self._populate_one_target(objective)
        objective.update(batch_size=1)
        state = torch.zeros((1, 3), dtype=torch.float32)
        probabilities = torch.full((1, 3), 1.0 / 3.0)
        expected_before, _, costs_before = objective.policy_cost(
            state, probabilities, torch.ones(1), cost_cap=0.25)

        restored = self._objective()
        restored.load_deployment_state_dict(objective.deployment_state_dict())
        expected_after, valid_after, costs_after = restored.policy_cost(
            state, probabilities, torch.ones(1), cost_cap=0.25)
        telemetry = restored.telemetry()

        self.assertTrue(restored.ready)
        self.assertTrue(torch.equal(expected_before, expected_after))
        self.assertTrue(torch.equal(costs_before, costs_after))
        self.assertEqual(float(valid_after.item()), 1.0)
        self.assertEqual(telemetry["multistep_value_ready"], 1.0)
        self.assertEqual(telemetry["multistep_value_replay_size"], 1.0)
        self.assertEqual(telemetry["multistep_value_targets_emitted"], 1.0)
        reloaded = self._objective()
        reloaded.load_deployment_state_dict(restored.deployment_state_dict())
        self.assertTrue(reloaded.ready)

    def test_deployment_checkpoint_rejects_inconsistent_readiness(self):
        objective = self._objective()
        checkpoint = objective.deployment_state_dict()
        checkpoint["training_evidence"]["ready"] = True
        with self.assertRaisesRegex(ValueError, "readiness is inconsistent"):
            self._objective().load_deployment_state_dict(checkpoint)


if __name__ == "__main__":
    unittest.main()
