import unittest
from types import SimpleNamespace

from lower.causal_departure_regularity import (
    CausalDepartureRegularityCost,
    causal_two_sided_action_excess_cost,
    causal_two_sided_holding_target_s,
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
