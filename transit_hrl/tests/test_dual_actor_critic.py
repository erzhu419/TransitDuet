import unittest

import numpy as np

from freq_hrl.experiments.transit.ppo_surrogate import train_transit_surrogate_ppo
from freq_hrl.experiments.trading.ppo_actor_critic import train_ppo_actor_critic
from freq_hrl.policies import BernsteinPlanCurve
from freq_hrl.rl import (
    DualActorCriticPPO,
    DualPPOConfig,
    LearnedPlanActionMapper,
    TrajectoryBatch,
    apply_replay_updates,
)


class DualActorCriticTest(unittest.TestCase):
    def test_dual_ppo_update_runs(self):
        cfg = DualPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=0,
            epochs=1,
            minibatch_size=4,
        )
        model = DualActorCriticPPO(cfg)
        batch = TrajectoryBatch(
            upper_state=np.zeros((4, 3), dtype=np.float32),
            lower_state=np.zeros((4, 2), dtype=np.float32),
            upper_action=np.zeros((4, 1), dtype=np.float32),
            lower_action=np.zeros((4, 1), dtype=np.float32),
            reward=np.ones(4, dtype=np.float32) * 0.01,
            done=np.array([0, 0, 0, 1], dtype=np.float32),
            old_upper_logp=np.zeros(4, dtype=np.float32),
            old_lower_logp=np.zeros(4, dtype=np.float32),
            old_upper_value=np.zeros(4, dtype=np.float32),
            old_lower_value=np.zeros(4, dtype=np.float32),
            constraint=np.ones(4, dtype=np.float32) * 0.2,
        )
        metrics = model.update(batch)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("constraint_mean", metrics)

    def test_apply_replay_updates_records_shared_kernel_rows(self):
        cfg = DualPPOConfig(
            upper_state_dim=3,
            lower_state_dim=2,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=0,
            epochs=1,
            minibatch_size=4,
        )
        model = DualActorCriticPPO(cfg)
        batch = TrajectoryBatch(
            upper_state=np.zeros((4, 3), dtype=np.float32),
            lower_state=np.zeros((4, 2), dtype=np.float32),
            upper_action=np.zeros((4, 1), dtype=np.float32),
            lower_action=np.zeros((4, 1), dtype=np.float32),
            reward=np.ones(4, dtype=np.float32) * 0.01,
            done=np.array([0, 0, 0, 1], dtype=np.float32),
            old_upper_logp=np.zeros(4, dtype=np.float32),
            old_lower_logp=np.zeros(4, dtype=np.float32),
            old_upper_value=np.zeros(4, dtype=np.float32),
            old_lower_value=np.zeros(4, dtype=np.float32),
            constraint=np.ones(4, dtype=np.float32) * 0.2,
        )
        rows = []
        metrics = apply_replay_updates(
            model,
            batch,
            rows,
            episode=7,
            replay_updates=2,
            metadata={"domain": "unit"},
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode"], 7)
        self.assertEqual(rows[1]["replay_update"], 1)
        self.assertEqual(rows[0]["domain"], "unit")
        self.assertIn("policy_loss", metrics)

    def test_learned_plan_action_mapper(self):
        mapper = LearnedPlanActionMapper(
            curve=BernsteinPlanCurve(horizon_s=600.0, basis_dim=3, n_entities=2, delta_min=-0.5, delta_max=0.5),
            coefficient_scale=0.5,
            eval_offset_s=300.0,
        )
        out = mapper.target(np.zeros(2, dtype=np.float64), np.ones(mapper.action_dim, dtype=np.float64) * 0.25)
        self.assertEqual(out.target.shape, (2,))
        self.assertEqual(out.coefficients.shape, (6,))
        self.assertGreaterEqual(out.smoothness_penalty, 0.0)

    def test_trading_ppo_actor_critic_smoke(self):
        payload, rows, _ = train_ppo_actor_critic(
            train_seeds=[42],
            eval_seeds=[123],
            steps=40,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            seed=7,
        )
        self.assertEqual(payload["trainer"], "frequency_separated_smdp_ppo_v2")
        self.assertEqual(payload["trajectory_contract"]["policy_ratios"], "independent upper and lower PPO ratios")
        self.assertEqual(len(rows), 1)
        self.assertIn("sharpe_mean", payload["summary"])
        self.assertLess(rows[0]["upper_decision_count"], rows[0]["lower_decision_count"])
        self.assertEqual(rows[0]["protocol_valid"], 1.0)

    def test_trading_ppo_learned_baseline_modes_emit_main_metrics(self):
        for mode in ("flat_ppo", "generic_hrl_ppo"):
            with self.subTest(mode=mode):
                payload, rows, _ = train_ppo_actor_critic(
                    train_seeds=[42],
                    eval_seeds=[123],
                    steps=32,
                    assets=2,
                    scenario="persistent_shift",
                    iterations=1,
                    seed=7,
                    policy_mode=mode,
                )
                self.assertEqual(payload["policy_mode"], mode)
                self.assertEqual(payload["trainer"], "frequency_separated_smdp_ppo_v2")
                self.assertFalse(payload["frequency_routing_enabled"])
                self.assertEqual(rows[0]["baseline"], mode)
                if mode == "flat_ppo":
                    self.assertEqual(rows[0]["upper_decision_count"], 32)
                    self.assertEqual(rows[0]["temporal_contract"], "primitive_joint_action")
                else:
                    self.assertLess(rows[0]["upper_decision_count"], 32)
                    self.assertEqual(rows[0]["temporal_contract"], "asynchronous_hierarchy")
                self.assertIn("FocusScore", rows[0])
                self.assertIn("FocusScore_mean", payload["summary"])

    def test_transit_surrogate_ppo_smoke(self):
        payload, rows, _ = train_transit_surrogate_ppo(
            train_seeds=[11],
            eval_seeds=[101],
            steps=30,
            corridors=2,
            scenario="persistent_shift",
            iterations=1,
            seed=7,
        )
        self.assertEqual(payload["trainer"], "frequency_separated_smdp_ppo_v2")
        self.assertEqual(
            payload["trajectory_contract"]["policy_ratios"],
            "independent upper and lower PPO ratios",
        )
        self.assertEqual(payload["domain"], "transit_surrogate")
        self.assertEqual(len(rows), 1)
        self.assertIn("wait_proxy_mean", payload["summary"])
        self.assertLess(rows[0]["upper_transition_count"], rows[0]["lower_transition_count"])
        self.assertEqual(rows[0]["protocol_valid"], 1.0)


if __name__ == "__main__":
    unittest.main()
