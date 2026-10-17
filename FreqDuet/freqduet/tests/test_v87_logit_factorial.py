from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v87_logit_factorial import CAL_DAYS, EVAL_DAYS, ITEMS, centered, distribution, factorial, mean_shift


class LogitFactorialTest(unittest.TestCase):
    def test_calibration_and_evaluation_are_separate_complete_days(self):
        self.assertFalse(set(CAL_DAYS) & set(EVAL_DAYS))
        self.assertEqual(len(ITEMS), 2 * 16 * 3)

    def test_day_weight_is_not_decision_count_weight_and_ignores_outcomes(self):
        days = [dict(logit_delta_mean=[1., -1.], decisions=100, journey_min=2.),
                dict(logit_delta_mean=[3., -3.], decisions=1, journey_min=999.)]
        np.testing.assert_array_equal(mean_shift(days), [2., -2.])
        days[0]["journey_min"] = -1000.
        np.testing.assert_array_equal(mean_shift(days), [2., -2.])

    def test_centering_removes_softmax_gauge_not_state_response(self):
        x = np.array([[1., 2., 3.], [4., 2., 6.]])
        np.testing.assert_array_equal(centered(x), centered(x + np.array([[10.], [20.]])))
        shift = np.array([.3, -.1, -.2])
        np.testing.assert_allclose(centered(x + shift)[1] - centered(x + shift)[0], centered(x)[1] - centered(x)[0])

    def test_four_corners_and_feasible_support(self):
        late, early = torch.tensor([[1., 2., 3.]]), torch.tensor([[2., 3., 1.]])
        mask = torch.tensor([[True, True, False]])
        shift = np.array([1., -1., 0.])
        for label, logits in (("late", late), ("early", early),
                              ("late_earlymean", late + torch.tensor(shift, dtype=torch.float32)),
                              ("early_latemean", early - torch.tensor(shift, dtype=torch.float32))):
            expected = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), -1)
            torch.testing.assert_close(distribution(late, early, mask, shift, label), expected, atol=0, rtol=0)
            self.assertEqual(float(expected[0, 2]), 0.)
        for label in ("late_earlymean", "early_latemean"):
            reference = "late" if label == "late_earlymean" else "early"
            torch.testing.assert_close(distribution(late, early, mask, np.zeros(3), label),
                distribution(late, early, mask, np.zeros(3), reference), atol=0, rtol=0)

    def test_factorial_interaction_and_total_are_not_double_counted(self):
        result = factorial(dict(late=10., early=5., late_earlymean=8., early_latemean=6.))
        self.assertEqual(result["interaction"], -1.)
        self.assertEqual(result["early_vs_late"], result["early_mean_at_late_feedback"] -
                         result["late_feedback_at_early_mean"])


if __name__ == "__main__":
    unittest.main()
