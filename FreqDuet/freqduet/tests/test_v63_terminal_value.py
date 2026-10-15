from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v63_terminal_value as terminal


class TerminalValueTest(unittest.TestCase):
    def test_validation_is_disjoint_from_all_v62_days(self):
        old = {seed for seed, _ in terminal.warmup.DAYS}
        self.assertTrue(set(terminal.VALIDATION_SEEDS).isdisjoint(old))
        self.assertEqual(len(terminal.ITEMS), 16)
        self.assertEqual(len({terminal.specification(i)["label"] for i in range(16)}), 16)

    def test_training_target_is_final_value_not_remaining_or_clipped_cost(self):
        day = dict(returns=np.array([.02, -.08]), prefix=np.array([.08, .18]), final_cv_squared=.10)
        targets = terminal.terminal_targets([day])
        np.testing.assert_array_equal(targets[0]["returns"], [.10, .10])
        np.testing.assert_array_equal(day["returns"], [.02, -.08])
        np.testing.assert_allclose(targets[0]["returns"] - day["prefix"], day["returns"])

    def test_prefix_subtraction_preserves_action_preferences_and_policy_gradient(self):
        values = torch.tensor([.08, .07, .09, .10, .06, .12, .11], dtype=torch.float64)
        logits = torch.zeros(7, dtype=torch.float64, requires_grad=True)
        probabilities = logits.softmax(0)
        original = torch.autograd.grad((probabilities * values).sum(), logits, retain_graph=True)[0]
        factored = torch.autograd.grad((probabilities * (values - .15)).sum(), logits)[0]
        torch.testing.assert_close(original, factored, atol=1e-12, rtol=0)
        self.assertEqual(int(values.argmin()), int((values - .15).argmin()))

    def test_prediction_uses_no_final_label_and_keeps_negative_remaining_cost(self):
        class ConstantFinal(torch.nn.Module):
            def forward(self, states, actions):
                return torch.full((len(states), 1), .10)

        day = dict(states=np.zeros((2, 3), dtype=np.float32), actions=np.zeros((2, 1), dtype=np.float32),
                   returns=np.array([.02, -.08]), prefix=np.array([.08, .18]), final_cv_squared=.10)
        prediction = terminal.predict_remaining(ConstantFinal(), day)
        np.testing.assert_allclose(prediction, [.02, -.08], atol=1e-8)
        changed_labels = dict(day, final_cv_squared=.90, returns=np.array([.82, .72]))
        np.testing.assert_array_equal(prediction, terminal.predict_remaining(ConstantFinal(), changed_labels))

    def test_gate_cannot_pass_by_only_beating_the_direct_v62_model(self):
        metrics = dict(mae=.02, rmse=.03)
        day = dict(factored=metrics, direct_v62=dict(mae=.1, rmse=.2), frozen=dict(mae=.2, rmse=.3),
                   zero=dict(mae=.025, rmse=.035), causal_final_mean=dict(mae=.004, rmse=.004))
        self.assertFalse(terminal.calibration_gate([day]))
        better = dict(day, factored=dict(mae=.003, rmse=.0035))
        self.assertTrue(terminal.calibration_gate([better]))
        self.assertFalse(terminal.calibration_gate([better, day]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
