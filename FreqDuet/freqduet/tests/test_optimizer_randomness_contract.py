import math
import random
import unittest

import numpy as np
import torch

from lower.cost_replay_buffer import CostReplayBuffer
from lower.resac_lagrangian import GaussianPolicy, RESACLagrangianTrainer
from randomness import RandomnessContract
from upper.resac_upper import RESACUpperTrainer, UpperReplayBuffer


class RandomnessContractTest(unittest.TestCase):
    def test_named_seeds_are_stable_and_distinct(self):
        first = RandomnessContract(42, "isolated_streams_v4")
        second = RandomnessContract(42, "isolated_streams_v4")
        self.assertEqual(first.seed("lower_policy"), second.seed("lower_policy"))
        self.assertNotEqual(
            first.seed("lower_policy"), first.seed("upper_policy"))
        self.assertEqual(
            first.manifest(["upper_policy", "lower_policy"]),
            second.manifest(["lower_policy", "upper_policy"]),
        )

    def test_initialization_scope_restores_global_torch_stream(self):
        contract = RandomnessContract(7, "isolated_streams_v4")
        torch.manual_seed(123)
        expected = torch.rand(4)
        torch.manual_seed(123)
        with contract.torch_initialization("lower_init"):
            _ = torch.rand(100)
        observed = torch.rand(4)
        torch.testing.assert_close(observed, expected)

    def test_replay_sampling_is_independent_of_global_random(self):
        lower_a = CostReplayBuffer(32, seed=11)
        lower_b = CostReplayBuffer(32, seed=11)
        upper_a = UpperReplayBuffer(32, seed=13)
        upper_b = UpperReplayBuffer(32, seed=13)
        for idx in range(16):
            state = np.array([idx], dtype=np.float32)
            lower_a.push(state, idx, idx, 0.0, state, False, idx)
            lower_b.push(state, idx, idx, 0.0, state, False, idx)
            upper_a.push(state, [idx], idx, state, False)
            upper_b.push(state, [idx], idx, state, False)
        random.seed(999)
        _ = [random.random() for _ in range(100)]
        np.testing.assert_array_equal(
            lower_a.sample(8)[-1], lower_b.sample(8)[-1])
        np.testing.assert_array_equal(
            upper_a.sample(8)[1], upper_b.sample(8)[1])


class OptimizerContractTest(unittest.TestCase):
    def test_lower_entropy_uses_dimensionless_action_coordinates(self):
        physical = GaussianPolicy(
            3, action_range=60.0,
            entropy_action_coordinates="physical_legacy", sample_seed=17)
        normalized = GaussianPolicy(
            3, action_range=60.0,
            entropy_action_coordinates="normalized_unit_interval",
            sample_seed=17)
        normalized.load_state_dict(physical.state_dict())
        state = torch.zeros((5, 3))
        action_p, log_p, *_ = physical.evaluate(state)
        action_n, log_n, *_ = normalized.evaluate(state)
        torch.testing.assert_close(action_p, action_n)
        torch.testing.assert_close(
            log_n - log_p,
            torch.full_like(log_p, math.log(60.0)),
            atol=2e-5,
            rtol=0.0,
        )

    def test_policy_sampling_stream_ignores_global_torch_consumption(self):
        first = GaussianPolicy(2, sample_seed=23)
        second = GaussianPolicy(2, sample_seed=23)
        second.load_state_dict(first.state_dict())
        state = torch.zeros((4, 2))
        action_first, *_ = first.evaluate(state)
        _ = torch.randn(1000)
        action_second, *_ = second.evaluate(state)
        torch.testing.assert_close(action_first, action_second)

    def test_v4_temperature_starts_inside_and_respects_bounds(self):
        lower = RESACLagrangianTrainer(
            state_dim=2,
            ensemble_size=2,
            hidden_dim=8,
            maximum_alpha=0.05,
            minimum_alpha=0.01,
            initial_alpha=0.2,
            temperature_contract="bounded_log_parameter_v4",
            entropy_action_coordinates="normalized_unit_interval",
        )
        upper = RESACUpperTrainer(
            state_dim=2,
            action_dim=1,
            action_low=[-1.0],
            action_high=[1.0],
            ensemble_size=2,
            hidden_dim=8,
            maximum_alpha=0.04,
            minimum_alpha=0.01,
            initial_alpha=0.2,
            temperature_contract="bounded_log_parameter_v4",
        )
        self.assertAlmostEqual(lower.alpha, 0.05, places=6)
        self.assertAlmostEqual(upper.alpha, 0.04, places=6)
        self.assertAlmostEqual(lower.log_alpha.exp().item(), lower.alpha, places=6)
        self.assertAlmostEqual(upper.log_alpha.exp().item(), upper.alpha, places=6)

    def test_lagrange_dual_uses_per_decision_cost_limit(self):
        def updated_lambda(cost):
            torch.manual_seed(31)
            trainer = RESACLagrangianTrainer(
                state_dim=2,
                ensemble_size=2,
                hidden_dim=8,
                cost_limit=0.5,
                cost_limit_semantics="per_decision_rate",
                lambda_lr=1e-2,
                auto_entropy=False,
            )
            replay = CostReplayBuffer(64, seed=37)
            for idx in range(16):
                state = np.array([idx / 16.0, 0.0], dtype=np.float32)
                replay.push(state, 0.0, 0.0, cost, state, True, idx)
            trainer.update(replay, 16, reward_scale=1.0)
            return trainer.lambda_param

        self.assertGreater(updated_lambda(1.0), 1.0)
        self.assertLess(updated_lambda(0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
