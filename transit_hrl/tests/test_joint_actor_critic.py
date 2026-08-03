import copy
import unittest

import numpy as np
import torch

from freq_hrl.experiments.trading.ppo_actor_critic import (
    capacity_matched_joint_hidden_dim,
    causal_raw_lag_stack,
    joint_parameter_count,
    smdp_rollout,
    smdp_parameter_count,
    train_ppo_actor_critic,
)
from freq_hrl.rl import (
    JointActorCriticPPO,
    JointPPOConfig,
    JointTrajectoryBatch,
    FrequencySeparatedActorCriticPPO,
    SMDPPPOConfig,
)


class JointActorCriticPPOTest(unittest.TestCase):
    @staticmethod
    def _batch(seed: int = 0) -> JointTrajectoryBatch:
        rng = np.random.default_rng(seed)
        return JointTrajectoryBatch(
            state=rng.normal(size=(12, 5)).astype(np.float32),
            action=rng.normal(size=(12, 4)).astype(np.float32),
            reward=rng.normal(scale=0.1, size=12).astype(np.float32),
            done=np.asarray([0.0] * 11 + [1.0], dtype=np.float32),
            old_logp=np.zeros(12, dtype=np.float32),
            old_value=np.zeros(12, dtype=np.float32),
        )

    def test_standard_joint_update_uses_one_actor_and_one_value(self):
        model = JointActorCriticPPO(JointPPOConfig(
            state_dim=5,
            action_dim=4,
            hidden_dim=16,
            epochs=2,
            minibatch_size=6,
        ))
        metrics = model.update(self._batch())
        self.assertEqual(metrics["actor_optimizer_steps"], 4.0)
        self.assertEqual(metrics["value_optimizer_steps"], 4.0)
        self.assertIn("approx_kl", metrics)
        self.assertIn("clip_fraction", metrics)
        self.assertFalse(hasattr(model, "upper_actor"))
        self.assertFalse(hasattr(model, "lower_actor"))

    def test_reward_scale_does_not_change_normalized_actor_direction(self):
        config = JointPPOConfig(
            state_dim=5,
            action_dim=4,
            hidden_dim=16,
            epochs=1,
            minibatch_size=12,
        )
        torch.manual_seed(17)
        small = JointActorCriticPPO(config)
        large = JointActorCriticPPO(config)
        large.load_state_dict(copy.deepcopy(small.state_dict()))
        small_batch = self._batch(4)
        large_batch = copy.deepcopy(small_batch)
        large_batch.reward *= 100.0
        np.random.seed(23)
        small.update(small_batch)
        np.random.seed(23)
        large.update(large_batch)
        small_actor = torch.cat([
            parameter.detach().reshape(-1) for parameter in small.actor.parameters()
        ])
        large_actor = torch.cat([
            parameter.detach().reshape(-1) for parameter in large.actor.parameters()
        ])
        torch.testing.assert_close(small_actor, large_actor, rtol=1e-5, atol=1e-6)

    def test_capacity_match_uses_active_parameters_without_padding(self):
        smdp = SMDPPPOConfig(
            upper_state_dim=23,
            lower_state_dim=25,
            upper_action_dim=3,
            lower_action_dim=3,
            hidden_dim=64,
        )
        target = smdp_parameter_count(smdp)
        hidden, actual, ratio = capacity_matched_joint_hidden_dim(
            target_parameter_count=target,
            state_dim=25,
            action_dim=6,
            requested_hidden_dim=64,
        )
        joint = JointPPOConfig(state_dim=25, action_dim=6, hidden_dim=hidden)
        self.assertEqual(actual, joint_parameter_count(joint))
        self.assertLessEqual(abs(ratio - 1.0), 0.05)

    def test_flat_runner_reports_canonical_joint_contract(self):
        payload, rows, model = train_ppo_actor_critic(
            train_seeds=[42],
            validation_seeds=[57721],
            eval_seeds=[123],
            steps=24,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            seed=7,
            hidden_dim=16,
            ppo_epochs=1,
            minibatch_size=32,
            policy_mode="flat_ppo",
        )
        self.assertIsInstance(model, JointActorCriticPPO)
        self.assertEqual(payload["trainer"], "canonical_joint_flat_ppo_v1")
        self.assertLessEqual(abs(float(payload["capacity_ratio"]) - 1.0), 0.05)
        self.assertEqual(payload["trajectory_contract"]["critic"], "one state-value function")
        self.assertEqual(rows[0]["upper_decision_count"], 24)
        self.assertEqual(rows[0]["routing_contract"], "causal_raw_lag_history")

    def test_legacy_smdp_flat_path_is_forbidden(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=17,
            lower_state_dim=17,
            upper_action_dim=2,
            lower_action_dim=2,
            hidden_dim=8,
        ))
        with self.assertRaisesRegex(ValueError, "joint_flat_rollout"):
            smdp_rollout(
                model,
                seed=1,
                steps=4,
                assets=2,
                scenario="persistent_shift",
                sample=False,
                policy_mode="flat_ppo",
            )


class CausalRawHistoryTest(unittest.TestCase):
    def test_lag_stack_contains_only_current_and_past_samples(self):
        history = np.arange(12, dtype=np.float64).reshape(6, 2)
        observed = causal_raw_lag_stack(
            history[:4], assets=2, lags=(0, 1, 3, 8)
        )
        expected = np.concatenate([history[3], history[2], history[0], history[0]])
        np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
