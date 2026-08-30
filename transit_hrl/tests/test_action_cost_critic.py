import unittest

import numpy as np
import torch

from freq_hrl.rl.action_cost_critic import (
    ActionCostCritic,
    discounted_smdp_cost_returns,
    transform_latent_action,
)


class ActionCostCriticTest(unittest.TestCase):
    def test_smdp_returns_respect_duration_and_episode_boundaries(self):
        returns = discounted_smdp_cost_returns(
            np.array([1.0, 2.0, 4.0, 8.0]),
            np.array([2, 1, 3, 1]),
            np.array([0, 1, 0, 1]),
            gamma=0.5,
        )
        np.testing.assert_allclose(returns, [1.5, 2.0, 5.0, 8.0])

    def test_action_input_supports_actor_gradients(self):
        critic = ActionCostCritic(3, 2, 8, zero_init_output=False)
        state = torch.ones((4, 3))
        action = torch.randn((4, 2), requires_grad=True)
        critic(state, action).mean().backward()
        self.assertIsNotNone(action.grad)
        self.assertGreater(float(torch.linalg.vector_norm(action.grad)), 0.0)

    def test_latent_action_transform_matches_deployment_coordinates(self):
        action = torch.tensor([[0.0, 1.0]])
        transformed = transform_latent_action(
            action, transform="tanh", scale=0.5
        )
        torch.testing.assert_close(
            transformed, 0.5 * torch.tanh(action)
        )
        with self.assertRaises(ValueError):
            transform_latent_action(action, transform="bad", scale=1.0)


if __name__ == "__main__":
    unittest.main()
