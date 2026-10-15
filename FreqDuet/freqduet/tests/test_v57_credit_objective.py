from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_freqduet_v57_credit_objective import audit_context, choose
from run_freqduet_v36_lower_credit import OUTCOMES


def example():
    scores = pd.DataFrame([dict(action_s=a, model_ep=199, lambda_value=1, energy=e)
                           for a, e in [(0, 0), (15, -1), (45, -2)]])
    rows = []
    for action, reward, cost, journey in [(0, 0, 0, 20), (15, 3, 2, 19), (45, 2, 0, 21)]:
        row = dict.fromkeys(OUTCOMES, 0.0)
        row.update(action_s=action, actor_action_s=0, trip_done=True,
                   trip_reward_return=reward, trip_cost_return=cost,
                   restricted_total_journey_horizon_min=journey)
        rows.append(row)
    return scores, pd.DataFrame(rows)


class CreditObjectiveTest(unittest.TestCase):
    def test_realized_local_preference_can_harm_full_day_journey(self):
        selected, counts = audit_context(*example(), 199)
        local = next(r for r in selected if r["selector"] == "realized_trip_objective")
        self.assertEqual(local["selected_action_s"], 45)
        self.assertEqual(local["delta_restricted_total_journey_horizon_min"], 1)
        self.assertTrue(counts["trip_best_day_harm"])
        self.assertEqual(counts["trip_gain_day_harm_actions"], 1)

    def test_cost_uses_the_anchor_models_recorded_lambda(self):
        values, rollouts = example()
        values.lambda_value = 0
        selected, _ = audit_context(values, rollouts, 199)
        local = next(r for r in selected if r["selector"] == "realized_trip_objective")
        self.assertEqual(local["selected_action_s"], 15)

    def test_exact_ties_keep_incumbent_not_arbitrary_larger_action(self):
        frame = pd.DataFrame([dict(action_s=0, value=1), dict(action_s=15, value=1)])
        self.assertEqual(choose(frame, "value", True, 15).action_s, 15)

    def test_incomplete_trip_credit_cannot_enter_audit(self):
        values, rollouts = example()
        rollouts.loc[0, "trip_done"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit_context(values, rollouts, 199)

    def test_early_score_must_not_be_used_for_late_continuation(self):
        values, rollouts = example()
        values.model_ep = 39
        with self.assertRaisesRegex(ValueError, "contracts differ"):
            audit_context(values, rollouts, 199)


if __name__ == "__main__":
    unittest.main()
