import unittest

import numpy as np
import torch

from freq_hrl.rl import FlatOffPolicyActorCritic, OffPolicyConfig, ReplayBuffer


def _batch(state_dim: int, action_dim: int, size: int = 16):
    rng = np.random.default_rng(7)
    return {
        "state": torch.as_tensor(rng.normal(size=(size, state_dim)), dtype=torch.float32),
        "action": torch.as_tensor(rng.uniform(-1, 1, size=(size, action_dim)), dtype=torch.float32),
        "reward": torch.as_tensor(rng.normal(size=(size, 1)), dtype=torch.float32),
        "next_state": torch.as_tensor(rng.normal(size=(size, state_dim)), dtype=torch.float32),
        "done": torch.zeros((size, 1), dtype=torch.float32),
    }


class OffPolicyActorCriticTest(unittest.TestCase):
    def test_replay_buffer_shapes(self):
        replay = ReplayBuffer(capacity=8, state_dim=3, action_dim=2)
        for index in range(10):
            replay.add(
                np.ones(3) * index,
                np.ones(2),
                reward=float(index),
                next_state=np.ones(3) * (index + 1),
                done=index == 9,
            )
        self.assertEqual(replay.size, 8)
        batch = replay.sample(4, np.random.default_rng(1), torch.device("cpu"))
        self.assertEqual(batch["state"].shape, (4, 3))
        self.assertEqual(batch["action"].shape, (4, 2))

    def test_sac_update_is_finite(self):
        agent = FlatOffPolicyActorCritic(OffPolicyConfig(
            state_dim=4,
            action_dim=2,
            algorithm="sac",
            hidden_dim=16,
        ))
        metrics = agent.update(_batch(4, 2))
        self.assertTrue(np.isfinite(metrics["critic_loss"]))
        self.assertTrue(np.isfinite(metrics["actor_loss"]))
        self.assertGreater(metrics["alpha"], 0.0)
        self.assertIn("log_alpha", agent.state_dict())
        action = agent.act(np.zeros(4, dtype=np.float32), sample=False)
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_td3_delays_actor_update(self):
        agent = FlatOffPolicyActorCritic(OffPolicyConfig(
            state_dim=4,
            action_dim=2,
            algorithm="td3",
            hidden_dim=16,
            policy_delay=2,
        ))
        first = agent.update(_batch(4, 2))
        second = agent.update(_batch(4, 2))
        self.assertEqual(first["actor_updated"], 0.0)
        self.assertEqual(second["actor_updated"], 1.0)
        self.assertTrue(np.isfinite(second["critic_loss"]))


if __name__ == "__main__":
    unittest.main()
