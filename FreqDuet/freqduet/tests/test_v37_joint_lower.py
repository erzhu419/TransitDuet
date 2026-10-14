from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v37_joint_lower import OUTCOMES, PRIMARY, context_spaces, pilot_items, second_matches


class JointLowerTest(unittest.TestCase):
    def test_flattened_batch_covers_every_prespecified_context_and_first_action(self):
        items = pilot_items()
        self.assertEqual(len(items), 84)
        self.assertEqual(len(set(items)), 84)
        self.assertEqual(len({item[:3] for item in items}), 12)

    def grid(self):
        rows = []
        outcomes = {(0, 0): (10, .2, 100), (5, 0): (9.8, .2, 100),
                    (0, 5): (9.5, .19, 100), (5, 5): (9, .18, 90)}
        for (first, second), triple in outcomes.items():
            row = dict(first_action_s=first, second_action_s=second, first_actor_s=0)
            for metric in OUTCOMES:
                row[metric] = row[f"reference_{metric}"] = row[f"first_only_{metric}"] = 0.0
            for i, metric in enumerate(PRIMARY):
                row[metric] = triple[i]
                row[f"reference_{metric}"] = outcomes[(0, 0)][i]
                row[f"first_only_{metric}"] = outcomes[(first, 0)][i]
            rows.append(row)
        return pd.DataFrame(rows)

    def test_second_decision_requires_same_stop_direction_and_another_vehicle(self):
        first = dict(station_id=3, direction=True, bus_id=1)
        self.assertTrue(second_matches(first, dict(first, bus_id=2)))
        for change in (dict(station_id=4), dict(direction=False), dict(bus_id=1)):
            self.assertFalse(second_matches(first, {**first, "bus_id": 2, **change}))

    def test_joint_compares_against_single_intervention_at_either_decision(self):
        result = {row["contrast"]: row for row in context_spaces(self.grid())}
        metric = "selected_delta_restricted_total_journey_horizon_min"
        self.assertAlmostEqual(result["first_only_vs_actor"][metric], -.2)
        self.assertAlmostEqual(result["single_union_vs_actor"][metric], -.5)
        self.assertAlmostEqual(result["joint_vs_actor"][metric], -1)
        self.assertAlmostEqual(result["joint_vs_best_single"][metric], -.5)
        self.assertEqual(result["joint_vs_best_single"]["has_all_three_strict_improvement"], 1)

    def test_no_joint_gain_is_manufactured_when_second_only_is_already_best(self):
        frame = self.grid()
        frame.loc[(frame.first_action_s == 5) & (frame.second_action_s == 5), PRIMARY[0]] = 11
        result = {row["contrast"]: row for row in context_spaces(frame)}
        self.assertEqual(result["joint_vs_best_single"]["selected_delta_restricted_total_journey_horizon_min"], 0)

    def test_changed_reference_cannot_be_aggregated(self):
        frame = self.grid()
        frame.loc[0, "reference_headway_cv"] += .01
        with self.assertRaisesRegex(ValueError, "reference differs"):
            context_spaces(frame)


if __name__ == "__main__":
    unittest.main()
