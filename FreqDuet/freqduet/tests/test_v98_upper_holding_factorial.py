import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v98_upper_holding_factorial as experiment


class FactorialTests(unittest.TestCase):
    def test_budget_counts_shared_zero_paths_once(self):
        self.assertEqual(len(experiment.ITEMS), 1280)
        self.assertEqual(len(set(experiment.ITEMS)), 1280)
        self.assertEqual(sum(r[3] == "learned_zero" for r in experiment.ITEMS), 320)
        self.assertEqual(sum(r[3] == "fixed_preserve" for r in experiment.ITEMS), 960)

    def test_factorial_interaction_and_gap_identity(self):
        values = dict(learned_preserve=12., learned_zero=7., fixed_preserve=9., fixed_zero=3.)
        rows = {k: dict.fromkeys(experiment.frozen.METRICS, v) for k, v in values.items()}
        result = experiment.contrasts(rows)
        for m in experiment.frozen.METRICS:
            self.assertEqual(result["full_gap"][m], 9.)
            self.assertEqual(result["holding_at_learned"][m], 5.)
            self.assertEqual(result["upper_without_holding"][m], 4.)
            self.assertEqual(result["interaction"][m], -1.)

    def test_zero_is_native_feasible_action_not_mask_replacement(self):
        class Policy:
            def feasible_action_mask(self, states):
                return torch.tensor([[True, True, False, False, False, False, False]])
        np.testing.assert_array_equal(experiment.zero_law(Policy(), torch.zeros(1, 2)), [[1, 0, 0, 0, 0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
