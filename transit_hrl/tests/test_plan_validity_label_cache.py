from copy import deepcopy
import unittest

from freq_hrl.experiments.plan_validity_label_continuation import HEADS
from scripts.read_plan_validity_label_cache import margin_readout
from test_plan_validity_label_continuation import banks
from test_plan_validity_phase_support import clock_rule


class LabelCacheTest(unittest.TestCase):
    def rows(self):
        rows = banks()["candidate"]
        for i, row in enumerate(rows):
            row["offset_fraction"] = 0.2 * (i % 5)
        return rows

    def test_common_score_and_cut_shift_does_not_change_actions(self):
        rows = self.rows()
        rules = {name: clock_rule() for name in HEADS}
        rules[HEADS[0]].threshold = 0.37
        rules[HEADS[1]] = deepcopy(rules[HEADS[0]])
        rules[HEADS[1]].model["model"]["weights"][0] += 0.7
        rules[HEADS[1]].threshold += 0.7
        result = margin_readout(rows, rules)
        self.assertEqual(result["action_disagreements"], 0)
        self.assertAlmostEqual(result["mean_score_difference"], 0.7)
        self.assertLess(result["state_dependent_score_difference_SD"], 1e-14)
        self.assertLess(result["margin_difference_RMS"], 1e-14)

    def test_cut_only_difference_reports_the_actual_causal_states(self):
        rows = self.rows()
        rules = {name: clock_rule() for name in HEADS}
        rules[HEADS[0]].threshold, rules[HEADS[1]].threshold = 0.37, 0.57
        result = margin_readout(rows, rules)
        expected = [{"seed": r["seed"], "step": r["opportunity_step"]} for r in rows if r["offset_fraction"] == 0.4]
        self.assertEqual(result["changed_states"], expected)
        self.assertEqual(result["action_disagreements"], len(expected))
        self.assertAlmostEqual(result["margin_difference_RMS"], 0.2)
        self.assertEqual(result["mean_score_difference"], 0)


if __name__ == "__main__":
    unittest.main()
