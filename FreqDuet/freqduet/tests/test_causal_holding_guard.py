import unittest
from types import SimpleNamespace

import numpy as np
import torch

from lower.causal_holding_guard import CausalHoldingActionGuard
from lower.resac_lagrangian import CategoricalPolicy
from runner_v3 import TransitDuetV2Runner


class CausalHoldingActionGuardTest(unittest.TestCase):
    def test_caps_action_at_observed_forward_headway_deficit(self):
        guard = CausalHoldingActionGuard(
            enabled=True, max_deficit_fraction=0.5)
        result = guard.evaluate(
            30.0,
            forward_headway_s=320.0,
            target_headway_s=360.0,
            evidence_valid=True,
        )

        self.assertEqual(result.limit_s, 20.0)
        self.assertEqual(result.allowed_s, 20.0)
        self.assertEqual(result.adjustment_s, 10.0)
        self.assertTrue(result.active)

    def test_masks_action_without_deployable_arrival_event(self):
        guard = CausalHoldingActionGuard(enabled=True)
        result = guard.evaluate(
            15.0,
            forward_headway_s=300.0,
            target_headway_s=360.0,
            evidence_valid=False,
        )

        self.assertEqual(result.allowed_s, 0.0)
        self.assertFalse(result.evidence_valid)

    def test_runner_rounds_guarded_action_down_to_discrete_bin(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.lower_causal_holding_guard = CausalHoldingActionGuard(
            enabled=True)
        runner.lower_action_bins = np.asarray(
            [0.0, 5.0, 10.0, 15.0], dtype=np.float32)
        runner.lower_action_bins_gate_enabled = False
        runner._ep_lower_causal_guard_active = []
        runner._ep_lower_causal_guard_limits = []
        runner._ep_lower_causal_guard_adjustments = []
        bus = SimpleNamespace(
            forward_headway_source="arrival_event",
            forward_headway=353.0,
            _target_headway=360.0,
        )

        adjusted = runner._apply_causal_holding_guard(
            np.asarray([15.0], dtype=np.float32), bus)

        self.assertEqual(float(adjusted[0]), 5.0)
        self.assertEqual(runner._ep_lower_causal_guard_limits, [7.0])
        self.assertEqual(runner._ep_lower_causal_guard_adjustments, [10.0])

    def test_v6_guard_uses_matched_pre_action_departure_gap(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.lower_causal_holding_guard = CausalHoldingActionGuard(
            enabled=True, evidence_mode="pre_action_departure_v6")
        runner.lower_action_bins = np.asarray(
            [0.0, 5.0, 10.0, 15.0], dtype=np.float32)
        runner.lower_action_bins_gate_enabled = False
        runner._ep_lower_causal_guard_active = []
        runner._ep_lower_causal_guard_limits = []
        runner._ep_lower_causal_guard_adjustments = []
        bus = SimpleNamespace(
            forward_headway_source="arrival_event",
            forward_headway=300.0,
            pre_action_forward_headway_source="matched_departure_event",
            pre_action_forward_headway=353.0,
            _target_headway=360.0,
        )

        adjusted = runner._apply_causal_holding_guard(
            np.asarray([15.0], dtype=np.float32), bus)

        self.assertEqual(float(adjusted[0]), 5.0)
        self.assertEqual(runner._ep_lower_causal_guard_limits, [7.0])

    def test_v6_guard_masks_when_immediate_predecessor_has_not_departed(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.lower_causal_holding_guard = CausalHoldingActionGuard(
            enabled=True, evidence_mode="pre_action_departure_v6")
        runner.lower_action_bins = np.asarray(
            [0.0, 5.0, 10.0], dtype=np.float32)
        runner.lower_action_bins_gate_enabled = False
        runner._ep_lower_causal_guard_active = []
        runner._ep_lower_causal_guard_limits = []
        runner._ep_lower_causal_guard_adjustments = []
        bus = SimpleNamespace(
            pre_action_forward_headway_source="predecessor_not_departed",
            pre_action_forward_headway=None,
            _target_headway=360.0,
        )

        adjusted = runner._apply_causal_holding_guard(
            np.asarray([10.0], dtype=np.float32), bus)

        self.assertEqual(float(adjusted[0]), 0.0)

    def test_categorical_policy_masks_infeasible_actions_in_distribution(self):
        policy = CategoricalPolicy(
            num_inputs=3,
            action_bins=[0.0, 5.0, 10.0, 15.0, 30.0, 45.0],
            action_limit_feature_index=1,
        )
        state = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 10.5 / 45.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        probs, log_probs, _ = policy.dist_info(state)
        feasible = policy.feasible_action_mask(state)

        self.assertEqual(feasible[0].tolist(), [True, False, False, False, False, False])
        self.assertEqual(feasible[1].tolist(), [True, True, True, False, False, False])
        self.assertTrue(bool(feasible[2].all()))
        self.assertTrue(bool(torch.all(probs[~feasible] == 0.0)))
        self.assertTrue(bool(torch.isfinite(log_probs).all()))
        self.assertTrue(bool(torch.allclose(
            probs.sum(dim=-1), torch.ones(3))))


if __name__ == "__main__":
    unittest.main()
