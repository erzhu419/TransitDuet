from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze_plan_validity_window_coverage_cache as analysis


class WindowCoverageCacheTest(unittest.TestCase):
    def test_contrast_sign_and_window_tail_mismatch_are_preserved(self):
        rows = [{"seed": seed, "factual_stop": False,
            "A_replicas": [{"adaptive_window_advantage": 1, "full_advantage": -2}] * 4,
            "B_replicas": [{"adaptive_window_advantage": 1, "full_advantage": -2}] * 4} for seed in (1, 2)]
        rules = {k: SimpleNamespace(threshold=0.5) for pair in analysis.CONTRASTS.values() for k in pair}
        mapping = {id(rule): np.ones(2) if k.startswith("wide") else np.zeros(2) for k, rule in rules.items()}
        with patch.object(analysis, "predict", side_effect=lambda rule, _: mapping[id(rule)]):
            result = analysis.panel(rows, rules)
        c = result["factorial_action_contrasts"]["coverage_df2"]
        self.assertEqual(c["changed_states"], 2)
        self.assertEqual(c["changed_paths"], 2)
        self.assertEqual(c["blocks"]["A"]["window_gain"], 1)
        self.assertEqual(c["blocks"]["A"]["full_gain"], -2)
        self.assertEqual(c["blocks"]["A"]["changed_state_window_full_sign_disagreement"], 1)
        unchanged = result["factorial_action_contrasts"]["complexity_wide"]
        self.assertEqual(unchanged["changed_states"], 0)
        self.assertIsNone(unchanged["blocks"]["B"]["changed_state_window_full_sign_disagreement"])


if __name__ == "__main__":
    unittest.main()
