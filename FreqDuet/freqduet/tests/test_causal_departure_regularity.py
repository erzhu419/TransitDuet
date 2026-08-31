import unittest
from types import SimpleNamespace

import numpy as np

from lower.causal_departure_regularity import (
    CausalDepartureRegularityCost,
    causal_two_sided_action_excess_cost,
    causal_two_sided_holding_target_s,
    causal_two_sided_zero_hold_regret_cost,
    fleet_utilization_pressure,
    holding_action_efficiency_gate,
    holding_fleet_efficiency_gate,
    holding_target_preserving_fleet_efficiency_gate,
    holding_target_pressure,
)
from runner_v3 import TransitDuetV2Runner


class CausalDepartureRegularityCostTest(unittest.TestCase):
    def test_action_excess_cost_is_zero_at_the_analytic_target(self):
        target = causal_two_sided_holding_target_s(
            forward_headway_s=300.0,
            follower_departure_gap_s=420.0,
            action_cap_s=90.0,
        )

        exact = causal_two_sided_action_excess_cost(
            action_s=target,
            target_action_s=target,
            target_headway_s=360.0,
            cost_cap=0.25,
        )
        offset = causal_two_sided_action_excess_cost(
            action_s=30.0,
            target_action_s=target,
            target_headway_s=360.0,
            cost_cap=0.25,
        )

        self.assertEqual(exact, 0.0)
        self.assertAlmostEqual(offset, (30.0 / 360.0) ** 2)

    def test_zero_hold_regret_only_penalizes_worse_actions(self):
        kwargs = {
            'target_action_s': 30.0,
            'target_headway_s': 600.0,
            'cost_cap': 0.25,
        }

        self.assertEqual(causal_two_sided_zero_hold_regret_cost(
            action_s=0.0, **kwargs), 0.0)
        self.assertEqual(causal_two_sided_zero_hold_regret_cost(
            action_s=30.0, **kwargs), 0.0)
        self.assertEqual(causal_two_sided_zero_hold_regret_cost(
            action_s=60.0, **kwargs), 0.0)
        self.assertAlmostEqual(
            causal_two_sided_zero_hold_regret_cost(
                action_s=75.0, **kwargs),
            ((75.0 - 30.0) / 600.0) ** 2 - (30.0 / 600.0) ** 2,
        )

    def test_compact_target_is_the_clipped_two_sided_balance_action(self):
        self.assertEqual(causal_two_sided_holding_target_s(
            forward_headway_s=300.0,
            follower_departure_gap_s=360.0,
            action_cap_s=45.0,
        ), 30.0)
        self.assertEqual(causal_two_sided_holding_target_s(
            forward_headway_s=300.0,
            follower_departure_gap_s=500.0,
            action_cap_s=45.0,
        ), 45.0)
        self.assertEqual(causal_two_sided_holding_target_s(
            forward_headway_s=420.0,
            follower_departure_gap_s=360.0,
            action_cap_s=45.0,
        ), 0.0)

    def test_compact_target_fails_closed_for_invalid_evidence(self):
        self.assertIsNone(causal_two_sided_holding_target_s(
            forward_headway_s=None,
            follower_departure_gap_s=360.0,
            action_cap_s=45.0,
        ))
        self.assertIsNone(causal_two_sided_holding_target_s(
            forward_headway_s=300.0,
            follower_departure_gap_s=float("nan"),
            action_cap_s=45.0,
        ))
        with self.assertRaisesRegex(ValueError, "action_cap_s"):
            causal_two_sided_holding_target_s(
                forward_headway_s=300.0,
                follower_departure_gap_s=360.0,
                action_cap_s=0.0,
            )

    def test_cost_is_minimized_by_action_that_closes_observed_deficit(self):
        regularity = CausalDepartureRegularityCost(
            enabled=True, cost_weight=2.0)
        context = regularity.capture(
            forward_headway_s=330.0,
            target_headway_s=360.0,
            evidence_source="matched_departure_event",
        )

        exact = regularity.evaluate(context, 30.0)
        short = regularity.evaluate(context, 10.0)
        long = regularity.evaluate(context, 50.0)

        self.assertEqual(exact.cost, 0.0)
        self.assertAlmostEqual(short.cost, long.cost)
        self.assertGreater(short.cost, exact.cost)
        self.assertEqual(exact.predicted_headway_s, 360.0)

    def test_tolerance_and_cap_bound_the_soft_cost(self):
        regularity = CausalDepartureRegularityCost(
            enabled=True,
            cost_weight=3.0,
            tolerance_fraction=0.05,
            cost_cap=0.04,
        )
        context = regularity.capture(
            forward_headway_s=360.0,
            target_headway_s=360.0,
            evidence_source="matched_departure_event",
        )

        inside = regularity.evaluate(context, 18.0)
        outside = regularity.evaluate(context, 360.0)

        self.assertEqual(inside.cost, 0.0)
        self.assertAlmostEqual(outside.cost, 0.12)

    def test_missing_matched_departure_evidence_has_zero_cost(self):
        regularity = CausalDepartureRegularityCost(
            enabled=True, cost_weight=1.0)
        context = regularity.capture(
            forward_headway_s=None,
            target_headway_s=360.0,
            evidence_source="predecessor_not_departed",
        )

        result = regularity.evaluate(context, 45.0)

        self.assertEqual(result.cost, 0.0)
        self.assertFalse(result.evidence_valid)

    def test_forward_incremental_reward_credits_only_action_improvement(self):
        regularity = CausalDepartureRegularityCost(
            enabled=True,
            objective_mode="forward_incremental_reward",
            reward_weight=2.0,
            cost_cap=1.0,
        )
        context = regularity.capture(
            forward_headway_s=300.0,
            target_headway_s=360.0,
            evidence_source="matched_departure_event",
        )

        improving = regularity.evaluate(context, 30.0)
        exact = regularity.evaluate(context, 60.0)
        worsening = regularity.evaluate(context, 180.0)

        self.assertGreater(exact.reward_adjustment, improving.reward_adjustment)
        self.assertGreater(improving.reward_adjustment, 0.0)
        self.assertLess(worsening.reward_adjustment, 0.0)
        self.assertEqual(exact.cost, 0.0)

    def test_two_sided_reward_favors_balancing_forward_and_follower_gaps(self):
        regularity = CausalDepartureRegularityCost(
            enabled=True,
            objective_mode="avl_two_sided_incremental_reward",
            reward_weight=1.0,
        )
        context = regularity.capture(
            forward_headway_s=300.0,
            target_headway_s=360.0,
            evidence_source="matched_departure_event",
            follower_departure_gap_s=420.0,
            follower_evidence_source="same_time_avl_journey_speed_eta",
        )

        no_hold = regularity.evaluate(context, 0.0)
        balanced = regularity.evaluate(context, 60.0)
        too_long = regularity.evaluate(context, 120.0)

        self.assertEqual(no_hold.reward_adjustment, 0.0)
        self.assertGreater(balanced.reward_adjustment, 0.0)
        self.assertLess(too_long.reward_adjustment, balanced.reward_adjustment)
        self.assertEqual(balanced.predicted_headway_s, 360.0)
        self.assertEqual(balanced.predicted_follower_gap_s, 360.0)

    def test_two_sided_reward_fails_closed_without_same_time_avl(self):
        regularity = CausalDepartureRegularityCost(
            enabled=True,
            objective_mode="avl_two_sided_incremental_reward",
            reward_weight=1.0,
        )
        context = regularity.capture(
            forward_headway_s=300.0,
            target_headway_s=360.0,
            evidence_source="matched_departure_event",
            follower_departure_gap_s=420.0,
            follower_evidence_source="legacy_backward_headway",
        )

        result = regularity.evaluate(context, 60.0)

        self.assertEqual(result.reward_adjustment, 0.0)
        self.assertFalse(result.follower_evidence_valid)

    def test_runner_consumes_frozen_action_time_context(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.lower_departure_regularity = CausalDepartureRegularityCost(
            enabled=True, cost_weight=1.0)
        runner._pending_lower_action_context = {}
        bus = SimpleNamespace(
            pre_action_forward_headway=320.0,
            pre_action_forward_headway_source="matched_departure_event",
            _target_headway=360.0,
        )
        runner._capture_lower_action_context(7, bus)

        bus.pre_action_forward_headway = 100.0
        bus._target_headway = 200.0
        context = runner._consume_lower_action_context(7)
        result = runner.lower_departure_regularity.evaluate(context, 40.0)

        self.assertEqual(result.predicted_headway_s, 360.0)
        self.assertEqual(result.cost, 0.0)
        self.assertNotIn(7, runner._pending_lower_action_context)

    def test_runner_records_realized_capacity_gated_gain(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.env = SimpleNamespace(
            lower_context_features={
                "regularity_hold_target_norm",
                "regularity_hold_target_valid",
                "capacity",
            },
            lower_causal_holding_action_scale_s=90.0,
        )
        runner.lower_trainer = SimpleNamespace(
            regularity_policy_contract={
                "action_target_scale_s": 90.0,
                "cost_cap": 0.25,
            },
            regularity_capacity_gain_enabled=True,
            regularity_capacity_feature_index=2,
            regularity_capacity_exponent=2.0,
            regularity_capacity_action_efficiency_penalty=0.0,
            discrete_actions=None,
        )
        runner._ep_lower_regularity_policy_evidence_valid = []
        runner._ep_lower_regularity_policy_action_costs = []
        runner._ep_lower_regularity_policy_zero_hold_action_costs = []
        runner._ep_lower_regularity_policy_action_regrets = []
        runner._ep_lower_regularity_policy_capacity_gates = []
        runner._ep_lower_regularity_policy_capacity_gains = []
        runner._ep_lower_regularity_policy_action_efficiency_gates = []
        runner._ep_lower_regularity_policy_oracle_action_costs = []
        runner._ep_lower_regularity_policy_excess_action_costs = []
        runner._ep_lower_regularity_policy_target_actions = []
        runner._ep_lower_regularity_policy_abs_errors = []
        regularity = CausalDepartureRegularityCost(enabled=False)
        context = regularity.capture(
            forward_headway_s=300.0,
            target_headway_s=360.0,
            evidence_source="matched_departure_event",
            follower_departure_gap_s=420.0,
            follower_evidence_source="same_time_avl_journey_speed_eta",
        )

        runner._record_causal_regularity_policy_execution(
            context,
            action_s=60.0,
            state=np.asarray([0.0, 0.0, 0.5], dtype=np.float32),
        )

        expected_zero_hold_cost = (60.0 / 360.0) ** 2
        self.assertEqual(runner._ep_lower_regularity_policy_evidence_valid, [1.0])
        self.assertEqual(runner._ep_lower_regularity_policy_capacity_gates, [0.25])
        self.assertAlmostEqual(
            runner._ep_lower_regularity_policy_capacity_gains[0],
            0.25 * expected_zero_hold_cost,
        )
        self.assertEqual(
            runner._ep_lower_regularity_policy_action_efficiency_gates,
            [1.0],
        )

    def test_holding_efficiency_gate_is_smooth_and_dimensionless(self):
        self.assertEqual(holding_action_efficiency_gate(
            action_s=0.0, action_scale_s=45.0, penalty=2.0), 1.0)
        self.assertAlmostEqual(holding_action_efficiency_gate(
            action_s=22.5, action_scale_s=45.0, penalty=2.0), 0.5)
        self.assertAlmostEqual(holding_action_efficiency_gate(
            action_s=45.0, action_scale_s=45.0, penalty=2.0), 1.0 / 3.0)

    def test_fleet_efficiency_gate_recovers_v14_below_pressure_start(self):
        self.assertEqual(fleet_utilization_pressure(
            utilization=0.75,
            pressure_start=0.75,
            pressure_full=1.0,
        ), 0.0)
        self.assertAlmostEqual(fleet_utilization_pressure(
            utilization=0.875,
            pressure_start=0.75,
            pressure_full=1.0,
        ), 0.5)
        self.assertEqual(fleet_utilization_pressure(
            utilization=1.0,
            pressure_start=0.75,
            pressure_full=1.0,
        ), 1.0)
        self.assertEqual(holding_fleet_efficiency_gate(
            action_s=45.0,
            action_scale_s=45.0,
            penalty=1.0,
            fleet_pressure=0.0,
        ), 1.0)
        self.assertAlmostEqual(holding_fleet_efficiency_gate(
            action_s=45.0,
            action_scale_s=45.0,
            penalty=1.0,
            fleet_pressure=1.0,
        ), 0.5)

    def test_target_preserving_gate_is_state_scalar(self):
        self.assertEqual(holding_target_pressure(
            target_action_s=0.0,
            action_scale_s=45.0,
            exponent=0.0,
        ), 1.0)
        self.assertAlmostEqual(holding_target_pressure(
            target_action_s=22.5,
            action_scale_s=45.0,
            exponent=1.0,
        ), 0.5)
        self.assertAlmostEqual(
            holding_target_preserving_fleet_efficiency_gate(
                target_action_s=22.5,
                action_scale_s=45.0,
                penalty=1.0,
                fleet_pressure=0.5,
                target_pressure_exponent=1.0,
            ),
            0.8,
        )
        self.assertEqual(
            holding_target_preserving_fleet_efficiency_gate(
                target_action_s=45.0,
                action_scale_s=45.0,
                penalty=2.0,
                fleet_pressure=0.0,
                target_pressure_exponent=1.0,
            ),
            1.0,
        )

    def test_enabled_cost_requires_positive_weight(self):
        with self.assertRaisesRegex(ValueError, "cost_weight > 0"):
            CausalDepartureRegularityCost(enabled=True, cost_weight=0.0)

    def test_incremental_reward_requires_positive_reward_weight(self):
        with self.assertRaisesRegex(ValueError, "reward_weight > 0"):
            CausalDepartureRegularityCost(
                enabled=True,
                objective_mode="forward_incremental_reward",
            )


if __name__ == "__main__":
    unittest.main()
