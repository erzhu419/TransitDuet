import copy
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
    def test_standard_sac_ablation_uses_twin_min_backup_and_actor_value(self):
        trainer = RESACLagrangianTrainer(
            state_dim=2,
            ensemble_size=2,
            hidden_dim=8,
            critic_aggregation="twin_min",
            auto_entropy=False,
        )
        q_all = torch.tensor([[1.0, 3.0], [2.0, -1.0]])

        torch.testing.assert_close(
            trainer._aggregate_target_q(q_all),
            torch.tensor([1.0, -1.0]),
        )
        actor_q, q_mean, _ = trainer._policy_q_value(q_all)
        torch.testing.assert_close(actor_q, torch.tensor([1.0, -1.0]))
        torch.testing.assert_close(q_mean, torch.tensor([1.5, 1.0]))

    def test_twin_min_rejects_more_than_two_critics(self):
        with self.assertRaisesRegex(ValueError, "requires ensemble_size=2"):
            RESACUpperTrainer(
                state_dim=2,
                action_dim=1,
                action_low=[-1.0],
                action_high=[1.0],
                ensemble_size=3,
                critic_aggregation="twin_min",
            )

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

    def test_lagrange_dual_uses_tpc_weighted_cost_occupancy(self):
        torch.manual_seed(41)
        trainer = RESACLagrangianTrainer(
            state_dim=2,
            ensemble_size=2,
            hidden_dim=8,
            cost_limit=0.5,
            cost_limit_semantics="per_decision_rate",
            lambda_lr=1e-2,
            auto_entropy=False,
        )
        replay = CostReplayBuffer(64, seed=43)
        for idx in range(16):
            state = np.array([idx / 16.0, 0.0], dtype=np.float32)
            cost = 0.0 if idx < 8 else 1.0
            replay.push(state, 0.0, 0.0, cost, state, True, idx)

        def weight_fn(trip_ids):
            return np.where(trip_ids < 8, 1.8, 0.2).astype(np.float32)

        metrics = trainer.update(
            replay, 16, reward_scale=1.0, weight_fn=weight_fn)

        self.assertAlmostEqual(metrics["batch_cost_mean"], 0.1, places=6)
        self.assertLess(trainer.lambda_param, 1.0)

    @staticmethod
    def _regularity_trainer(
            cost_limit=0.001, conditional_entropy=False,
            constraint_scale_mode="raw_cost_v1", initial_lambda=1.0,
            policy_mode="analytic_two_sided_target_dual_v1",
            capacity_gain_weight=0.02, capacity_exponent=1.0,
            action_efficiency_penalty=0.0,
            fleet_pressure_start=0.75,
            opportunity_cost_penalty=0.0,
            target_pressure_exponent=0.0,
            hf_opportunity_cost_penalty=0.0,
            hf_energy_scale=0.04,
            hf_energy_exponent=1.0,
            action_bins=None):
        conditional = (
            {
                "enable": True,
                "mode": "evidence_split_temperature_v1",
                "target_fraction": 0.25,
                "lr": 1e-2,
                "minimum_alpha": 1e-4,
                "maximum_alpha": 0.1,
                "initial_alpha": 0.05,
            }
            if conditional_entropy else {"enable": False}
        )
        capacity_mode = policy_mode in {
            "analytic_two_sided_capacity_gain_regret_dual_v3",
            "analytic_two_sided_efficiency_gain_regret_dual_v4",
            "analytic_two_sided_fleet_efficiency_gain_regret_dual_v5",
            "analytic_two_sided_target_preserving_gain_regret_dual_v6",
            "analytic_two_sided_hf_opportunity_gain_regret_dual_v7",
        }
        regularity = {
            "enable": True,
            "mode": policy_mode,
            "target_feature_index": 1,
            "valid_feature_index": 2,
            "target_headway_feature_index": 0,
            "action_target_scale_s": 45.0,
            "target_headway_scale_s": 600.0,
            "cost_limit": cost_limit,
            "cost_cap": 0.25,
            "constraint_scale_mode": constraint_scale_mode,
            "lambda_lr": 1e-2,
            "lambda_min": 1e-3,
            "lambda_max": 20.0,
            "initial_lambda": initial_lambda,
            "conditional_entropy": conditional,
        }
        if capacity_mode:
            regularity["capacity_gated_gain"] = {
                "enable": True,
                "mode": (
                    "positive_zero_hold_target_preserving_gain_v4"
                    if policy_mode
                    == "analytic_two_sided_target_preserving_gain_regret_dual_v6"
                    else
                    "positive_zero_hold_hf_opportunity_gain_v5"
                    if policy_mode
                    == "analytic_two_sided_hf_opportunity_gain_regret_dual_v7"
                    else
                    "positive_zero_hold_fleet_efficiency_gain_v3"
                    if policy_mode
                    == "analytic_two_sided_fleet_efficiency_gain_regret_dual_v5"
                    else
                    "positive_zero_hold_efficiency_gain_v2"
                    if action_efficiency_penalty > 0.0
                    else "positive_zero_hold_gain_v1"),
                "capacity_feature_index": 3,
                "weight": capacity_gain_weight,
                "gain_scale": 0.002,
                "capacity_exponent": capacity_exponent,
                "action_efficiency_penalty": action_efficiency_penalty,
            }
            if policy_mode in {
                    "analytic_two_sided_fleet_efficiency_gain_regret_dual_v5",
                    "analytic_two_sided_target_preserving_gain_regret_dual_v6",
            }:
                regularity["capacity_gated_gain"].update({
                    "fleet_utilization_feature_index": 4,
                    "fleet_pressure_start": fleet_pressure_start,
                    "fleet_pressure_full": 1.0,
                    "fleet_pressure_exponent": 1.0,
                })
            if (policy_mode
                    == "analytic_two_sided_target_preserving_gain_regret_dual_v6"):
                regularity["capacity_gated_gain"].update({
                    "opportunity_cost_penalty": opportunity_cost_penalty,
                    "target_pressure_exponent": target_pressure_exponent,
                })
            if (policy_mode
                    == "analytic_two_sided_hf_opportunity_gain_regret_dual_v7"):
                regularity["capacity_gated_gain"].update({
                    "hf_energy_feature_index": 4,
                    "hf_energy_scale": hf_energy_scale,
                    "hf_energy_exponent": hf_energy_exponent,
                    "hf_opportunity_cost_penalty": (
                        hf_opportunity_cost_penalty),
                })
        return RESACLagrangianTrainer(
            state_dim=(
                5 if policy_mode
                in {
                    "analytic_two_sided_fleet_efficiency_gain_regret_dual_v5",
                    "analytic_two_sided_target_preserving_gain_regret_dual_v6",
                    "analytic_two_sided_hf_opportunity_gain_regret_dual_v7",
                }
                else 4 if capacity_mode else 3),
            action_range=45.0,
            action_bins=(action_bins or [0.0, 45.0]),
            ensemble_size=2,
            hidden_dim=8,
            auto_entropy=conditional_entropy,
            regularity_policy_objective=regularity,
        )

    def test_causal_regularity_cost_matches_two_sided_action_term(self):
        trainer = self._regularity_trainer()
        state = torch.tensor([[0.6, 1.0, 1.0]], dtype=torch.float32)
        probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

        expected, valid, costs = trainer._regularity_policy_cost(state, probs)

        self.assertAlmostEqual(costs[0, 0].item(), (45.0 / 360.0) ** 2)
        self.assertEqual(costs[0, 1].item(), 0.0)
        self.assertAlmostEqual(expected.item(), 0.5 * (45.0 / 360.0) ** 2)
        self.assertEqual(valid.item(), 1.0)

    def test_causal_regularity_regret_is_relative_to_zero_holding(self):
        trainer = self._regularity_trainer(
            policy_mode="analytic_two_sided_zero_hold_regret_dual_v2")
        state = torch.tensor([
            [0.6, 1.0 / 3.0, 1.0],
        ], dtype=torch.float32)
        probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

        expected, valid, regrets = trainer._regularity_policy_cost(
            state, probs)

        expected_action_regret = (
            (30.0 / 360.0) ** 2 - (15.0 / 360.0) ** 2)
        self.assertEqual(regrets[0, 0].item(), 0.0)
        self.assertAlmostEqual(
            regrets[0, 1].item(), expected_action_regret, places=7)
        self.assertAlmostEqual(
            expected.item(), 0.5 * expected_action_regret, places=7)
        self.assertEqual(valid.item(), 1.0)
        self.assertEqual(
            trainer.regularity_policy_contract["mode"],
            "analytic_two_sided_zero_hold_regret_dual_v2",
        )

    def test_capacity_gain_rewards_only_positive_low_load_improvement(self):
        trainer = self._regularity_trainer(
            policy_mode="analytic_two_sided_capacity_gain_regret_dual_v3")
        state = torch.tensor([
            [0.6, 1.0, 1.0, 1.0],
            [0.6, 1.0, 1.0, 0.0],
        ], dtype=torch.float32)
        probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)

        expected, valid, gains = trainer._regularity_policy_capacity_gain(
            state, probs)

        zero_hold_cost = (45.0 / 360.0) ** 2
        self.assertAlmostEqual(
            gains[0, 1].item(), zero_hold_cost, places=7)
        self.assertAlmostEqual(
            expected[0].item(), 0.5 * zero_hold_cost, places=7)
        self.assertEqual(expected[1].item(), 0.0)
        self.assertTrue(torch.equal(valid, torch.ones_like(valid)))

    def test_capacity_gain_bonus_uses_only_valid_evidence_and_round_trips(self):
        torch.manual_seed(109)
        trainer = self._regularity_trainer(
            cost_limit=0.00025,
            constraint_scale_mode="cost_limit_ratio_v1",
            initial_lambda=0.01,
            policy_mode="analytic_two_sided_capacity_gain_regret_dual_v3",
            capacity_gain_weight=0.02,
            capacity_exponent=2.0,
        )
        replay = CostReplayBuffer(64, seed=113)
        for idx in range(16):
            state = np.array([0.6, 1.0, 1.0, 1.0], dtype=np.float32)
            replay.push(state, 0.0, 0.0, 0.0, state, True, idx)

        metrics = trainer.update(replay, 16, reward_scale=1.0)

        self.assertGreater(
            metrics["regularity_policy_capacity_gain_mean"], 0.0)
        self.assertAlmostEqual(
            metrics["regularity_policy_capacity_gain_bonus"],
            0.02 * metrics[
                "regularity_policy_scaled_capacity_gain_mean"],
            places=6,
        )
        self.assertAlmostEqual(
            metrics["regularity_policy_capacity_gate_mean"], 1.0)
        restored = self._regularity_trainer(
            cost_limit=0.00025,
            constraint_scale_mode="cost_limit_ratio_v1",
            initial_lambda=0.01,
            policy_mode="analytic_two_sided_capacity_gain_regret_dual_v3",
            capacity_gain_weight=0.02,
            capacity_exponent=2.0,
        )
        restored.load_training_state_dict(trainer.training_state_dict())
        self.assertEqual(
            restored.regularity_policy_contract,
            trainer.regularity_policy_contract)

        invalid_replay = CostReplayBuffer(64, seed=127)
        for idx in range(16):
            state = np.array([0.6, 1.0, 0.0, 1.0], dtype=np.float32)
            invalid_replay.push(state, 0.0, 0.0, 0.0, state, True, idx)
        invalid_metrics = restored.update(
            invalid_replay, 16, reward_scale=1.0)
        self.assertEqual(
            invalid_metrics["regularity_policy_capacity_gain_bonus"], 0.0)

    def test_efficiency_gain_discounts_large_holding_in_actor_objective(self):
        trainer = self._regularity_trainer(
            policy_mode="analytic_two_sided_efficiency_gain_regret_dual_v4",
            action_efficiency_penalty=1.0,
        )
        state = torch.tensor([[0.6, 1.0, 1.0, 1.0]], dtype=torch.float32)
        probs = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        expected, valid, gains = trainer._regularity_policy_capacity_gain(
            state, probs)

        zero_hold_cost = (45.0 / 360.0) ** 2
        self.assertAlmostEqual(gains[0, 0].item(), 0.0, places=7)
        self.assertAlmostEqual(
            gains[0, 1].item(), 0.5 * zero_hold_cost, places=7)
        self.assertAlmostEqual(expected.item(), 0.5 * zero_hold_cost, places=7)
        self.assertEqual(valid.item(), 1.0)
        self.assertEqual(
            trainer.regularity_policy_contract["capacity_gated_gain"]
            ["action_efficiency_penalty"],
            1.0,
        )

    def test_fleet_efficiency_gain_only_discounts_under_fleet_pressure(self):
        trainer = self._regularity_trainer(
            policy_mode=(
                "analytic_two_sided_fleet_efficiency_gain_regret_dual_v5"),
            action_efficiency_penalty=1.0,
            fleet_pressure_start=0.75,
        )
        state = torch.tensor([
            [0.6, 1.0, 1.0, 1.0, 0.75],
            [0.6, 1.0, 1.0, 1.0, 1.0],
        ], dtype=torch.float32)
        probs = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

        expected, valid, gains = trainer._regularity_policy_capacity_gain(
            state, probs)

        zero_hold_cost = (45.0 / 360.0) ** 2
        self.assertAlmostEqual(gains[0, 1].item(), zero_hold_cost, places=7)
        self.assertAlmostEqual(
            gains[1, 1].item(), 0.5 * zero_hold_cost, places=7)
        self.assertAlmostEqual(expected[0].item(), zero_hold_cost, places=7)
        self.assertAlmostEqual(expected[1].item(), 0.5 * zero_hold_cost, places=7)
        self.assertTrue(torch.equal(valid, torch.ones_like(valid)))
        contract = trainer.regularity_policy_contract["capacity_gated_gain"]
        self.assertEqual(contract["fleet_utilization_feature_index"], 4)
        self.assertEqual(contract["fleet_pressure_start"], 0.75)

    def test_target_preserving_gate_keeps_v14_action_ordering(self):
        bins = [0.0, 15.0, 30.0, 45.0]
        v14 = self._regularity_trainer(
            policy_mode="analytic_two_sided_capacity_gain_regret_dual_v3",
            action_bins=bins,
        )
        v17 = self._regularity_trainer(
            policy_mode=(
                "analytic_two_sided_target_preserving_gain_regret_dual_v6"),
            opportunity_cost_penalty=1.0,
            target_pressure_exponent=1.0,
            action_bins=bins,
        )
        state_v14 = torch.tensor(
            [[0.6, 2.0 / 3.0, 1.0, 1.0]], dtype=torch.float32)
        state_v17 = torch.tensor(
            [[0.6, 2.0 / 3.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        probs = torch.full((1, len(bins)), 1.0 / len(bins))

        _, _, base_gains = v14._regularity_policy_capacity_gain(
            state_v14, probs)
        _, _, gated_gains = v17._regularity_policy_capacity_gain(
            state_v17, probs)
        gate = v17._regularity_policy_action_efficiency_gate(state_v17)

        self.assertTrue(torch.allclose(gate, gate[:, :1].expand_as(gate)))
        torch.testing.assert_close(gated_gains, base_gains * gate)
        self.assertEqual(
            int(gated_gains.argmax(dim=-1).item()),
            int(base_gains.argmax(dim=-1).item()),
        )
        self.assertEqual(int(gated_gains.argmax(dim=-1).item()), 2)
        contract = v17.regularity_policy_contract["capacity_gated_gain"]
        self.assertEqual(contract["opportunity_cost_penalty"], 1.0)
        self.assertEqual(contract["target_pressure_exponent"], 1.0)

    def test_hf_opportunity_gate_is_v14_at_zero_and_preserves_ordering(self):
        bins = [0.0, 15.0, 30.0, 45.0]
        v14 = self._regularity_trainer(
            policy_mode="analytic_two_sided_capacity_gain_regret_dual_v3",
            action_bins=bins,
        )
        v18 = self._regularity_trainer(
            policy_mode=(
                "analytic_two_sided_hf_opportunity_gain_regret_dual_v7"),
            hf_opportunity_cost_penalty=1.0,
            hf_energy_scale=0.04,
            action_bins=bins,
        )
        state_v14 = torch.tensor(
            [[0.6, 2.0 / 3.0, 1.0, 1.0]], dtype=torch.float32)
        state_v18 = torch.tensor([
            [0.6, 2.0 / 3.0, 1.0, 1.0, 0.0],
            [0.6, 2.0 / 3.0, 1.0, 1.0, 0.04],
        ], dtype=torch.float32)
        probs_v14 = torch.full((1, len(bins)), 1.0 / len(bins))
        probs_v18 = torch.full((2, len(bins)), 1.0 / len(bins))

        _, _, base_gains = v14._regularity_policy_capacity_gain(
            state_v14, probs_v14)
        _, _, gated_gains = v18._regularity_policy_capacity_gain(
            state_v18, probs_v18)
        gates = v18._regularity_policy_action_efficiency_gate(state_v18)

        torch.testing.assert_close(gated_gains[0], base_gains[0])
        torch.testing.assert_close(
            gated_gains[1], base_gains[0] * (2.0 / 3.0))
        self.assertTrue(torch.allclose(gates, gates[:, :1].expand_as(gates)))
        self.assertTrue(torch.equal(
            gated_gains.argmax(dim=-1),
            torch.tensor([2, 2]),
        ))
        contract = v18.regularity_policy_contract["capacity_gated_gain"]
        self.assertEqual(contract["hf_energy_feature_index"], 4)
        self.assertEqual(contract["hf_energy_scale"], 0.04)
        self.assertEqual(contract["hf_opportunity_cost_penalty"], 1.0)

    def test_causal_regularity_dual_uses_only_valid_evidence(self):
        torch.manual_seed(53)
        trainer = self._regularity_trainer(cost_limit=0.0)
        replay = CostReplayBuffer(64, seed=59)
        for idx in range(16):
            state = np.array([0.6, 1.0, 1.0], dtype=np.float32)
            replay.push(state, 0.0, 0.0, 0.0, state, True, idx)

        metrics = trainer.update(replay, 16, reward_scale=1.0)

        self.assertEqual(metrics["regularity_policy_valid_fraction"], 1.0)
        self.assertGreater(metrics["regularity_policy_cost_mean"], 0.0)
        self.assertGreater(trainer.regularity_lambda_param, 1.0)

        torch.manual_seed(61)
        invalid_trainer = self._regularity_trainer(cost_limit=0.0)
        invalid_replay = CostReplayBuffer(64, seed=67)
        for idx in range(16):
            state = np.array([0.6, 1.0, 0.0], dtype=np.float32)
            invalid_replay.push(state, 0.0, 0.0, 0.0, state, True, idx)

        invalid_metrics = invalid_trainer.update(
            invalid_replay, 16, reward_scale=1.0)

        self.assertEqual(
            invalid_metrics["regularity_policy_valid_fraction"], 0.0)
        self.assertEqual(invalid_metrics["regularity_policy_cost_mean"], 0.0)
        self.assertEqual(invalid_trainer.regularity_lambda_param, 1.0)

    def test_causal_regularity_limit_ratio_is_dimensionless(self):
        trainer = self._regularity_trainer(
            cost_limit=0.002,
            constraint_scale_mode="cost_limit_ratio_v1",
            initial_lambda=0.1,
        )
        raw_cost = torch.tensor(0.003, dtype=torch.float32)

        scaled_cost = trainer._scale_regularity_constraint_cost(raw_cost)

        self.assertAlmostEqual(scaled_cost.item(), 1.5, places=6)
        self.assertAlmostEqual(trainer.regularity_scaled_cost_limit, 1.0)
        self.assertEqual(
            trainer.regularity_policy_contract["constraint_scale_mode"],
            "cost_limit_ratio_v1",
        )

    def test_causal_regularity_limit_ratio_rejects_zero_limit(self):
        with self.assertRaisesRegex(ValueError, "positive limit"):
            self._regularity_trainer(
                cost_limit=0.0,
                constraint_scale_mode="cost_limit_ratio_v1",
            )

    def test_causal_regularity_limit_ratio_drives_actor_and_dual(self):
        torch.manual_seed(101)
        trainer = self._regularity_trainer(
            cost_limit=0.002,
            constraint_scale_mode="cost_limit_ratio_v1",
            initial_lambda=0.1,
        )
        replay = CostReplayBuffer(64, seed=103)
        for idx in range(16):
            state = np.array([0.6, 0.5, 1.0], dtype=np.float32)
            replay.push(state, 0.0, 0.0, 0.0, state, True, idx)

        metrics = trainer.update(replay, 16, reward_scale=1.0)

        self.assertAlmostEqual(
            metrics["regularity_policy_scaled_cost_mean"],
            metrics["regularity_policy_cost_mean"] / 0.002,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["regularity_policy_penalty"],
            0.1 * metrics["regularity_policy_scaled_cost_mean"],
            places=5,
        )
        self.assertGreater(trainer.regularity_lambda_param, 0.1)

    def test_causal_regularity_dual_round_trips_exact_training_state(self):
        torch.manual_seed(71)
        trainer = self._regularity_trainer(cost_limit=0.0)
        replay = CostReplayBuffer(64, seed=73)
        for idx in range(16):
            state = np.array([0.6, 0.5, 1.0], dtype=np.float32)
            replay.push(state, 0.0, 0.0, 0.0, state, True, idx)
        trainer.update(replay, 16, reward_scale=1.0)

        state = trainer.training_state_dict()
        restored = self._regularity_trainer(cost_limit=0.0)
        restored.load_training_state_dict(state)

        self.assertAlmostEqual(
            restored.regularity_lambda_param,
            trainer.regularity_lambda_param,
            places=7,
        )
        for expected, observed in zip(
                trainer.policy_net.parameters(),
                restored.policy_net.parameters()):
            torch.testing.assert_close(observed, expected)

    def test_legacy_raw_regularity_checkpoint_defaults_scale_contract(self):
        trainer = self._regularity_trainer()
        state = copy.deepcopy(trainer.training_state_dict())
        state["format"] = "freqduet-lower-training-v6"
        state["regularity_policy_contract"].pop("constraint_scale_mode")
        restored = self._regularity_trainer()

        restored.load_training_state_dict(state)

        self.assertEqual(
            restored.regularity_constraint_scale_mode,
            "raw_cost_v1",
        )

    def test_causal_entropy_split_uses_independent_valid_temperature(self):
        trainer = self._regularity_trainer(conditional_entropy=True)
        trainer.alpha = 0.08
        state = torch.tensor([
            [0.6, 0.5, 0.0],
            [0.6, 0.5, 1.0],
        ], dtype=torch.float32)

        temperatures = trainer._entropy_alpha_for_state(state)

        self.assertAlmostEqual(temperatures[0].item(), 0.08, places=6)
        self.assertAlmostEqual(temperatures[1].item(), 0.05, places=6)
        self.assertEqual(
            trainer.regularity_policy_contract["conditional_entropy"]["mode"],
            "evidence_split_temperature_v1",
        )

    def test_causal_entropy_split_updates_only_valid_temperature(self):
        torch.manual_seed(79)
        trainer = self._regularity_trainer(conditional_entropy=True)
        replay = CostReplayBuffer(64, seed=83)
        for idx in range(16):
            state = np.array([0.6, 0.5, 1.0], dtype=np.float32)
            replay.push(state, 0.0, 0.0, 0.0, state, True, idx)
        base_before = trainer.log_alpha.detach().clone()
        valid_before = trainer.regularity_alpha_param

        metrics = trainer.update(replay, 16, reward_scale=1.0)

        torch.testing.assert_close(trainer.log_alpha.detach(), base_before)
        self.assertLess(trainer.regularity_alpha_param, valid_before)
        self.assertEqual(metrics["regularity_entropy_split_enabled"], 1.0)
        self.assertGreater(metrics["regularity_entropy_valid_mean"], 0.0)

    def test_causal_entropy_split_round_trips_exact_training_state(self):
        torch.manual_seed(89)
        trainer = self._regularity_trainer(conditional_entropy=True)
        replay = CostReplayBuffer(64, seed=97)
        for idx in range(16):
            state = np.array([0.6, 0.5, 1.0], dtype=np.float32)
            replay.push(state, 0.0, 0.0, 0.0, state, True, idx)
        trainer.update(replay, 16, reward_scale=1.0)

        state = trainer.training_state_dict()
        restored = self._regularity_trainer(conditional_entropy=True)
        restored.load_training_state_dict(state)

        self.assertEqual(state["format"], "freqduet-lower-training-v7")
        self.assertAlmostEqual(
            restored.regularity_alpha_param,
            trainer.regularity_alpha_param,
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
