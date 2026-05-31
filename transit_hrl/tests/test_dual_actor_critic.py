import unittest

import numpy as np

from freq_hrl.experiments.trading.ppo_actor_critic import train_ppo_actor_critic
from freq_hrl.rl import DualActorCriticPPO, DualPPOConfig, TrajectoryBatch


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
        )
        metrics = model.update(batch)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)

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


if __name__ == "__main__":
    unittest.main()
