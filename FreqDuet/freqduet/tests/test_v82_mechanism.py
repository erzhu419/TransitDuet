from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_freqduet_v82_mechanism import variance_effect


class VarianceEffectTest(unittest.TestCase):
    def test_constant_mean_is_pure_variance_effect(self):
        result = variance_effect(dict(mean_headway_s=300., cv_squared=.04),
                                 dict(mean_headway_s=300., cv_squared=.05))
        self.assertAlmostEqual(result["variance_at_base_mean"], .01)
        self.assertEqual(result["mean_normalization"], 0.)

    def test_fixed_variance_is_pure_mean_normalization(self):
        result = variance_effect(dict(mean_headway_s=300., cv_squared=.04),
                                 dict(mean_headway_s=150., cv_squared=.16))
        self.assertEqual(result["variance_at_base_mean"], 0.)
        self.assertAlmostEqual(result["mean_normalization"], .12)

    def test_both_effects_can_have_opposite_signs(self):
        result = variance_effect(dict(mean_headway_s=300., cv_squared=.04),
                                 dict(mean_headway_s=150., cv_squared=.08))
        self.assertLess(result["variance_at_base_mean"], 0.)
        self.assertGreater(result["mean_normalization"], 0.)
        self.assertAlmostEqual(sum(result.values()), .04)


if __name__ == "__main__":
    unittest.main()
