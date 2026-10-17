from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v95_compensation_specificity as study


class CompensationSpecificityTest(unittest.TestCase):
    def test_reference_probability_uses_only_observable_reduction_budgets(self):
        contexts = dict(nonloss=dict(expected_holding_reduction_s=3., count=99),
                        compensation_loss=dict(expected_holding_reduction_s=7., count=1))
        self.assertEqual(study.reference_probability(contexts), .3)
        contexts["nonloss"]["count"] = 0
        self.assertEqual(study.reference_probability(contexts), .3)

    def test_uniform_law_ignores_which_compensation_stratum_was_logged(self):
        late = torch.tensor([[.1, .2, .7, 0., 0., 0., 0.]])
        early = torch.tensor([[.6, .3, .1, 0., 0., 0., 0.]])
        policy = SimpleNamespace(forward=lambda states: late.log(), feasible_action_mask=lambda states: late > 0)
        outputs = []
        for group in study.frozen.GROUPS:
            rule = study.UniformMixture(12037, .3)
            def parent(self, *args):
                self.early_decisions += 1
                self.selected_counts[group] += 1
                return early
            with patch.object(study.frozen.ReleaseHolding, "__call__", parent):
                outputs.append(rule(policy, None, None, None))
        torch.testing.assert_close(outputs[0], .7 * late + .3 * early)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        self.assertEqual(float(outputs[0][:, 3:].sum()), 0.)

    def test_mixture_endpoints_and_ineligible_native_return_are_exact(self):
        late, early = torch.tensor([.8, .2]), torch.tensor([.9, .1])
        self.assertIs(study.mixture(late, early, 0.), late)
        self.assertIs(study.mixture(late, early, 1.), early)
        with patch.object(study.frozen.ReleaseHolding, "__call__", return_value=late):
            self.assertIs(study.UniformMixture(12037, .3)(None, None, None, None), late)

    def test_reference_budget_equality_is_not_claimed_for_shifted_contexts(self):
        contexts = {g: dict(expected_holding_reduction_s=v, count=10) for g, v in zip(study.frozen.GROUPS, (3., 7.))}
        self.assertAlmostEqual(study.dose(contexts, "uniform", .3)["expected_reduction_minus_shadow_preserve_s"], 0.)
        contexts["nonloss"]["expected_holding_reduction_s"] = 6.
        self.assertAlmostEqual(study.dose(contexts, "uniform", .3)["expected_reduction_minus_shadow_preserve_s"], -2.1)

    def test_fixed_grid_reuses_all_ten_actor_day_repeat_cells(self):
        self.assertEqual(len(study.ITEMS), 480)
        self.assertEqual({r[1] for r in study.ITEMS}, set(study.confirmation.VALIDATION_SEEDS))
        for old_index, *coordinates in study.ITEMS:
            self.assertEqual(tuple(coordinates), study.confirmation.ITEMS[old_index])

    def test_saved_timing_reconstructs_native_coordinates(self):
        events = np.array([[i, 2, 1, t, 100.*i, h, t-100.*i-h]
            for i, (t, h) in enumerate(zip((100., 220., 410.), (0., 10., 20.)))])
        data = dict(event_fields=np.array(study.frozen.timing.EVENT_FIELDS), arrivals=events)
        with patch.object(study.np, "load") as load:
            load.return_value.__enter__.return_value = data
            saved = study.saved_timing(Path("native.npz"))
        effects = study.confirmation.timing_effect(saved, saved)
        self.assertTrue(all(v == 0 for v in effects.values()))


if __name__ == "__main__":
    unittest.main()
