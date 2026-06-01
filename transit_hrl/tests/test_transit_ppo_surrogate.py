import unittest

import numpy as np
import torch

from freq_hrl.experiments.transit.ppo_surrogate import (
    initialize_transit_prior,
    make_plan_mapper,
    make_tracker,
    rollout,
    split_upper_action,
)
from freq_hrl.rl import DualActorCriticPPO, DualPPOConfig


class TransitPPOSurrogateTest(unittest.TestCase):
    def test_rollout_uses_wait_attribution_and_promotion_replans(self):
        corridors = 2
        tracker = make_tracker(method="poisson_harmonic")
        upper_dim = int(tracker.upper_features("low_mid").size + corridors + 1)
        lower_dim = int(
            corridors * tracker.lower_features(0, True, "high_mid").size
            + 2 * corridors
            + 1
            + 3 * corridors
        )
        mapper = make_plan_mapper(
            corridors=corridors,
            plan_basis_dim=2,
            plan_horizon_s=900.0,
            plan_eval_offset_s=120.0,
            plan_coefficient_scale_s=12.0,
        )
        model = DualActorCriticPPO(DualPPOConfig(
            upper_state_dim=upper_dim,
            lower_state_dim=lower_dim,
            upper_action_dim=mapper.action_dim,
            lower_action_dim=corridors,
            hidden_dim=0,
            init_log_std=-3.0,
        ))
        torch.manual_seed(7)
        np.random.seed(7)
        initialize_transit_prior(model, corridors, plan_basis_dim=2)
        batch, row = rollout(
            model,
            seed=7,
            steps=48,
            corridors=corridors,
            scenario="localized_burst",
            sample=True,
            plan_mapper=mapper,
            tracker_method="poisson_harmonic",
            include_native_lower_context=True,
            wait_upper_weight=0.05,
            wait_lower_weight=0.10,
            wait_lower_board_credit_weight=0.05,
            upper_decision_interval=8,
            promotion_forced_replan=True,
            promotion_replan_strength_min=0.0,
        )
        self.assertIsNotNone(batch)
        self.assertEqual(batch.lower_state.shape[1], lower_dim)
        self.assertGreater(row["wait_attr_penalty"], 0.0)
        self.assertGreaterEqual(row["wait_high_share"], 0.0)
        self.assertGreater(row["upper_decision_count"], 1)
        self.assertGreaterEqual(row["promotion_replan_count"], 0)

    def test_learned_promotion_gate_adds_upper_action_and_replans(self):
        corridors = 2
        tracker = make_tracker(method="dynamic_harmonic_nb")
        upper_dim = int(tracker.upper_features("low_mid").size + corridors + 1)
        lower_dim = int(
            corridors * tracker.lower_features(0, True, "high_mid").size
            + 2 * corridors
            + 1
            + 3 * corridors
        )
        mapper = make_plan_mapper(
            corridors=corridors,
            plan_basis_dim=2,
            plan_horizon_s=900.0,
            plan_eval_offset_s=120.0,
            plan_coefficient_scale_s=1.0,
        )
        model = DualActorCriticPPO(DualPPOConfig(
            upper_state_dim=upper_dim,
            lower_state_dim=lower_dim,
            upper_action_dim=mapper.action_dim + 1,
            lower_action_dim=corridors,
            hidden_dim=0,
            init_log_std=-3.0,
        ))
        torch.manual_seed(13)
        np.random.seed(13)
        initialize_transit_prior(
            model,
            corridors,
            plan_basis_dim=2,
            include_native_lower_context=True,
            promotion_learned_replan=True,
        )
        plan_action, gate = split_upper_action(
            np.zeros(mapper.action_dim + 1, dtype=np.float32),
            mapper.action_dim,
            promotion_learned_replan=True,
        )
        self.assertEqual(plan_action.size, mapper.action_dim)
        self.assertAlmostEqual(gate, 0.5)
        batch, row = rollout(
            model,
            seed=13,
            steps=48,
            corridors=corridors,
            scenario="persistent_shift",
            sample=True,
            plan_mapper=mapper,
            tracker_method="dynamic_harmonic_nb",
            include_native_lower_context=True,
            upper_decision_interval=8,
            promotion_learned_replan=True,
            promotion_learned_gate_threshold=0.55,
            promotion_replan_strength_min=0.0,
            promotion_replan_recovery_gain=0.06,
            promotion_residual_threshold=0.55,
            promotion_persistence_ratio=0.20,
            wait_credit_control_gain=2.0,
            lower_lf_effect_filter_window=12,
            lower_lf_effect_filter_gain=1.0,
            lower_lf_raw_recenter_gain=1.0,
        )
        self.assertIsNotNone(batch)
        self.assertEqual(batch.upper_action.shape[1], mapper.action_dim + 1)
        self.assertGreater(row["promotion_gate_value"], 0.0)
        self.assertGreater(row["promotion_replan_count"], 0)
        self.assertGreater(row["promotion_recovery_relief"], 0.0)


if __name__ == "__main__":
    unittest.main()
