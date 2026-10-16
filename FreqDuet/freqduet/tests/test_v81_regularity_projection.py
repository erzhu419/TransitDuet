from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freqduet_regularity_projection as projection


class RegularityProjectionTest(unittest.TestCase):
    def test_binding_linear_gain_recovers_the_feasible_base(self):
        base = np.full(7, 1. / 7.)
        gain = np.arange(7.)
        raw = np.exp(-gain) / np.exp(-gain).sum()
        actual, active = projection.project_distribution(base, raw, gain)
        self.assertTrue(active)
        np.testing.assert_allclose(actual, base, rtol=1e-10, atol=1e-12)

    def test_physical_quadratic_gain_is_feasible_with_minimum_kl(self):
        context = SimpleNamespace(target_headway_s=300., forward_headway_s=150.,
            follower_departure_gap_s=450., evidence_valid=True, follower_evidence_valid=True)
        gains = projection.step.action_features(context, .5, 0, 2, 60.)[:, 0]
        base = np.full(7, 1. / 7.)
        raw = np.exp(-projection.step.ACTIONS / 45.)
        raw /= raw.sum()
        actual, active = projection.project_distribution(base, raw, gains)
        self.assertTrue(active)
        self.assertGreaterEqual(actual @ gains, base @ gains - 2e-12)
        self.assertLessEqual(actual @ np.log(actual / raw), base @ np.log(base / raw) + 1e-10)
        self.assertAlmostEqual(actual.sum(), 1.)

    def test_zero_gain_information_preserves_the_raw_law(self):
        base = np.full(7, 1. / 7.)
        raw = np.arange(1., 8.) / 28.
        actual, active = projection.project_distribution(base, raw, np.zeros(7))
        self.assertFalse(active)
        np.testing.assert_allclose(actual, raw, rtol=0, atol=1e-16)

    def test_zero_tilt_preserves_the_exact_native_tensor_law(self):
        from lower.resac_lagrangian import CategoricalPolicy

        torch.set_num_threads(1)
        policy = CategoricalPolicy(3, projection.step.ACTIONS, hidden_dim=8)
        states = torch.tensor([[.2, .4, .6]])
        features = np.arange(28.).reshape(7, 4) / 10.
        rule = projection.RegularityProjection()
        actual = rule(policy, states, features, np.zeros(4))
        expected = projection.step.probabilities(policy, states, features, np.zeros(4))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(rule.active, 0)


if __name__ == "__main__":
    unittest.main()
