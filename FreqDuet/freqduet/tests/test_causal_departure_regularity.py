import unittest
from types import SimpleNamespace

from lower.causal_departure_regularity import CausalDepartureRegularityCost
from runner_v3 import TransitDuetV2Runner


class CausalDepartureRegularityCostTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
