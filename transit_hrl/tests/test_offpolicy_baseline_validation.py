import unittest

import numpy as np

from freq_hrl.experiments.trading.offpolicy_baseline_validation import (
    capacity_matched_offpolicy_hidden_dim,
    decode_flat_action,
    flat_state,
    offpolicy_trainable_parameter_count,
    train_flat_offpolicy_baseline,
)
from freq_hrl.experiments.trading.ppo_actor_critic import (
    flat_latent_speed,
    raw_lower_state_dim,
)
from freq_hrl.rl import FlatOffPolicyActorCritic, OffPolicyConfig


class OffPolicyBaselineValidationTest(unittest.TestCase):
    def test_offpolicy_capacity_count_and_matching_use_trainable_online_modules(self):
        for algorithm in ("sac", "td3"):
            with self.subTest(algorithm=algorithm):
                hidden, count, ratio = capacity_matched_offpolicy_hidden_dim(
                    target_parameter_count=39_445,
                    state_dim=248,
                    action_dim=4,
                    algorithm=algorithm,
                    requested_hidden_dim=64,
                )
                config = OffPolicyConfig(
                    state_dim=248,
                    action_dim=4,
                    algorithm=algorithm,
                    hidden_dim=hidden,
                )
                model = FlatOffPolicyActorCritic(config)
                actual = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                self.assertEqual(count, offpolicy_trainable_parameter_count(config))
                self.assertEqual(actual, count)
                self.assertLessEqual(abs(ratio - 1.0), 0.05)

    def test_flat_state_and_action_contract(self):
        state = flat_state([0.001, -0.001], [0.1, -0.1], [0.2, 0.0], progress=0.5)
        self.assertEqual(state.shape, (raw_lower_state_dim(2),))
        target, speed = decode_flat_action([1.0, -1.0, -1.0, 1.0], assets=2)
        self.assertLessEqual(abs(target).sum(), 1.0)
        self.assertGreaterEqual(speed.min(), 0.05)
        self.assertLessEqual(speed.max(), 1.0)

    def test_flat_algorithms_share_execution_action_semantics(self):
        latent = np.asarray([-0.8, 0.4], dtype=np.float64)
        bounded = np.tanh(latent)
        _, offpolicy_speed = decode_flat_action(
            np.concatenate([np.zeros(2), bounded]),
            assets=2,
        )
        np.testing.assert_allclose(offpolicy_speed, flat_latent_speed(latent))

    def test_sac_and_td3_end_to_end_smoke(self):
        for mode in ("flat_sac", "flat_td3"):
            with self.subTest(mode=mode):
                payload, rows, agent = train_flat_offpolicy_baseline(
                    policy_mode=mode,
                    train_seeds=[42],
                    eval_seeds=[123],
                    steps=40,
                    assets=2,
                    scenario="persistent_shift",
                    iterations=1,
                    seed=7,
                    hidden_dim=16,
                    warmup_steps=8,
                    batch_size=8,
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["baseline"], mode)
                self.assertEqual(rows[0]["metric_contract_version"], "trading_metrics_v2")
                self.assertLess(rows[0]["equity_reconstruction_max_abs_error"], 1e-10)
                self.assertGreater(payload["gradient_updates_train"], 0)
                self.assertIn("selected_checkpoint_iteration", payload)
                self.assertIn("validation_learning_gain", payload)
                self.assertEqual(payload["training_reward_scale"], 100.0)
                self.assertEqual(payload["raw_history_window"], 120)
                self.assertEqual(
                    payload["raw_history_sampling"],
                    "complete_contiguous_oldest_to_newest",
                )
                self.assertEqual(
                    rows[0]["routing_contract"], "causal_raw_contiguous_window"
                )
                self.assertEqual(
                    payload["gradient_updates_train"],
                    payload["actor_optimizer_steps_train"]
                    + payload["critic_optimizer_steps_train"]
                    + payload["temperature_optimizer_steps_train"],
                )
                self.assertGreater(sum(p.numel() for p in agent.parameters() if p.requires_grad), 0)

    def test_offpolicy_baseline_can_share_causal_execution_environment(self):
        payload, rows, _ = train_flat_offpolicy_baseline(
            policy_mode="flat_sac",
            train_seeds=[42],
            validation_seeds=[84],
            eval_seeds=[123],
            steps=24,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            seed=7,
            hidden_dim=16,
            warmup_steps=4,
            batch_size=4,
            execution_timeline_contract="causal_post_trade_v3",
            volume_impact_bps=10.0,
        )
        self.assertEqual(
            payload["execution_timeline_contract"], "causal_post_trade_v3"
        )
        self.assertEqual(payload["mark_to_market_timing"], "post_trade")
        self.assertEqual(rows[0]["mark_to_market_timing"], "post_trade")
        self.assertEqual(rows[0]["volume_impact_bps"], 10.0)


if __name__ == "__main__":
    unittest.main()
