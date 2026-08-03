import copy
import unittest

import numpy as np
import torch

from freq_hrl.experiments.trading.ppo_actor_critic import (
    capacity_matched_joint_hidden_dim,
    capacity_matched_smdp_hidden_dim,
    causal_raw_history_window,
    joint_parameter_count,
    raw_lower_state_dim,
    raw_upper_state_dim,
    smdp_rollout,
    smdp_parameter_count,
    train_ppo_actor_critic,
)
from freq_hrl.rl import (
    CausalGRUStateEncoder,
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

    def test_smdp_capacity_count_includes_learned_promotion_actor_and_critic(self):
        config = SMDPPPOConfig(
            upper_state_dim=17,
            lower_state_dim=19,
            upper_action_dim=4,
            lower_action_dim=2,
            promotion_state_dim=28,
            hidden_dim=16,
        )
        model = FrequencySeparatedActorCriticPPO(config)
        modules = [
            model.upper_actor,
            model.lower_actor,
            model.upper_value,
            model.lower_value,
            model.lower_cost_value,
            model.promotion_actor,
            model.promotion_value,
        ]
        actual = sum(
            parameter.numel()
            for module in modules
            if module is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(actual, smdp_parameter_count(config))

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
            state_dim=raw_lower_state_dim(3),
            action_dim=6,
            requested_hidden_dim=64,
        )
        joint = JointPPOConfig(
            state_dim=raw_lower_state_dim(3), action_dim=6, hidden_dim=hidden
        )
        self.assertEqual(actual, joint_parameter_count(joint))
        self.assertLessEqual(abs(ratio - 1.0), 0.05)

    def test_generic_hrl_matches_capacity_after_full_window_expansion(self):
        reference = SMDPPPOConfig(
            upper_state_dim=23,
            lower_state_dim=25,
            upper_action_dim=3,
            lower_action_dim=3,
            hidden_dim=64,
        )
        target = smdp_parameter_count(reference)
        hidden, actual, ratio = capacity_matched_smdp_hidden_dim(
            target_parameter_count=target,
            upper_state_dim=raw_upper_state_dim(3),
            lower_state_dim=raw_lower_state_dim(3),
            upper_action_dim=3,
            lower_action_dim=3,
            requested_hidden_dim=64,
        )
        generic = SMDPPPOConfig(
            upper_state_dim=raw_upper_state_dim(3),
            lower_state_dim=raw_lower_state_dim(3),
            upper_action_dim=3,
            lower_action_dim=3,
            hidden_dim=hidden,
        )
        self.assertEqual(actual, smdp_parameter_count(generic))
        self.assertLess(hidden, reference.hidden_dim)
        self.assertLessEqual(abs(ratio - 1.0), 0.05)

    def test_causal_gru_analytic_counts_match_trainable_parameters(self):
        joint_config = JointPPOConfig(
            state_dim=raw_lower_state_dim(3, 20),
            action_dim=6,
            hidden_dim=12,
            state_encoder="causal_gru",
            raw_history_window=20,
            raw_feature_dim=3,
        )
        joint_model = JointActorCriticPPO(joint_config)
        self.assertEqual(
            joint_parameter_count(joint_config),
            sum(
                parameter.numel()
                for parameter in joint_model.parameters()
                if parameter.requires_grad
            ),
        )

        smdp_config = SMDPPPOConfig(
            upper_state_dim=raw_upper_state_dim(3, 20),
            lower_state_dim=raw_lower_state_dim(3, 20),
            upper_action_dim=3,
            lower_action_dim=3,
            hidden_dim=12,
            state_encoder="causal_gru",
            raw_history_window=20,
            raw_feature_dim=3,
        )
        smdp_model = FrequencySeparatedActorCriticPPO(smdp_config)
        modules = (
            smdp_model.upper_actor,
            smdp_model.lower_actor,
            smdp_model.upper_value,
            smdp_model.lower_value,
            smdp_model.lower_cost_value,
        )
        self.assertEqual(
            smdp_parameter_count(smdp_config),
            sum(
                parameter.numel()
                for module in modules
                for parameter in module.parameters()
                if parameter.requires_grad
            ),
        )

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
            hidden_dim=64,
            ppo_epochs=1,
            minibatch_size=32,
            policy_mode="flat_ppo",
        )
        self.assertIsInstance(model, JointActorCriticPPO)
        self.assertEqual(payload["trainer"], "canonical_joint_flat_ppo_v1")
        self.assertLessEqual(abs(float(payload["capacity_ratio"]) - 1.0), 0.05)
        self.assertEqual(model.config.state_dim, raw_lower_state_dim(2))
        self.assertEqual(payload["raw_history_window"], 120)
        self.assertEqual(payload["trajectory_contract"]["critic"], "one state-value function")
        self.assertEqual(rows[0]["upper_decision_count"], 24)
        self.assertEqual(
            rows[0]["routing_contract"], "causal_raw_contiguous_window"
        )

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

    def test_causal_gru_flat_and_hierarchical_runners(self):
        for mode, model_type in (
            ("flat_gru_ppo", JointActorCriticPPO),
            ("generic_hrl_gru_ppo", FrequencySeparatedActorCriticPPO),
        ):
            with self.subTest(mode=mode):
                payload, rows, model = train_ppo_actor_critic(
                    train_seeds=[42],
                    validation_seeds=[57721],
                    eval_seeds=[123],
                    steps=16,
                    assets=2,
                    scenario="persistent_shift",
                    iterations=1,
                    seed=7,
                    hidden_dim=64,
                    ppo_epochs=1,
                    minibatch_size=32,
                    policy_mode=mode,
                )
                self.assertIsInstance(model, model_type)
                self.assertEqual(model.config.state_encoder, "causal_gru")
                self.assertEqual(model.config.raw_history_window, 16)
                self.assertLessEqual(
                    abs(float(payload["capacity_ratio"]) - 1.0), 0.05
                )
                self.assertEqual(rows[0]["baseline"], mode)
                self.assertEqual(
                    rows[0]["routing_contract"], "causal_raw_full_episode_gru"
                )
                self.assertEqual(payload["raw_history_sampling"],
                                 "complete_contiguous_oldest_to_newest")


class CausalRawHistoryTest(unittest.TestCase):
    def test_causal_gru_is_invariant_to_left_padding_values(self):
        torch.manual_seed(5)
        encoder = CausalGRUStateEncoder(
            state_dim=4 * 2 + 1,
            history_window=4,
            raw_feature_dim=2,
            hidden_dim=8,
        )
        observed = np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
        first = np.concatenate([
            np.full((2, 2), -9.0, dtype=np.float32).reshape(-1),
            observed.reshape(-1),
            np.asarray([0.5], dtype=np.float32),
        ])
        second = np.concatenate([
            np.full((2, 2), 17.0, dtype=np.float32).reshape(-1),
            observed.reshape(-1),
            np.asarray([0.5], dtype=np.float32),
        ])
        with torch.no_grad():
            first_encoded = encoder(torch.as_tensor(first).view(1, -1))
            second_encoded = encoder(torch.as_tensor(second).view(1, -1))
        torch.testing.assert_close(first_encoded, second_encoded)

    def test_full_window_contains_every_causal_sample_and_left_padding(self):
        history = np.arange(12, dtype=np.float64).reshape(6, 2)
        observed = causal_raw_history_window(
            history[:4], assets=2, window=6
        ).reshape(6, 2)
        expected = np.vstack([history[0], history[0], history[:4]])
        np.testing.assert_array_equal(observed, expected)
        self.assertFalse(np.any(np.isin(observed, history[4:])))

    def test_full_window_truncates_oldest_samples_without_subsampling(self):
        history = np.arange(16, dtype=np.float64).reshape(8, 2)
        observed = causal_raw_history_window(
            history, assets=2, window=5
        ).reshape(5, 2)
        np.testing.assert_array_equal(observed, history[-5:])

if __name__ == "__main__":
    unittest.main()
