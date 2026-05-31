import unittest

import numpy as np

from freq_hrl.experiments.transit.ppo_surrogate import train_transit_surrogate_ppo
from freq_hrl.experiments.trading.ppo_actor_critic import train_ppo_actor_critic
from freq_hrl.policies import BernsteinPlanCurve
from freq_hrl.rl import DualActorCriticPPO, DualPPOConfig, LearnedPlanActionMapper, TrajectoryBatch


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
        self.assertEqual(payload["trainer"], "shared_dual_level_ppo")
        self.assertEqual(len(rows), 1)
        self.assertIn("sharpe_mean", payload["summary"])

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
        self.assertEqual(payload["trainer"], "shared_dual_level_ppo")
        self.assertEqual(payload["domain"], "transit_surrogate")
        self.assertEqual(len(rows), 1)
        self.assertIn("wait_proxy_mean", payload["summary"])


if __name__ == "__main__":
    unittest.main()
