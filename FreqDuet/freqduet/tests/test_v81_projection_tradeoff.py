from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_freqduet_v81_projection_tradeoff import repeat_variance


class ProjectionTradeoffTest(unittest.TestCase):
    def test_identical_repeats_keep_day_variation(self):
        x = np.arange(1., 6.)[:, None]
        result = repeat_variance(np.stack((x, x, x), axis=1))
        np.testing.assert_allclose(result["conditional_sampling_variance"], [0.])
        np.testing.assert_allclose(result["between_day_variance_untruncated"], [x.var(ddof=1)])
        self.assertEqual(result["all_repeats_positive"], [5])

    def test_unresolved_day_signal_is_not_clipped(self):
        values = np.tile(np.array([-1., 0., 1.])[None, :, None], (6, 1, 1))
        result = repeat_variance(values)
        np.testing.assert_allclose(result["conditional_sampling_variance"], [1.])
        np.testing.assert_allclose(result["between_day_variance_untruncated"], [-1. / 3.])
        self.assertEqual(result["mixed_repeat_sign"], [6])

    def test_scaling_preserves_units(self):
        values = np.arange(45.).reshape(5, 3, 3) - 20.
        base, scaled = repeat_variance(values), repeat_variance(values * 60.)
        np.testing.assert_allclose(scaled["conditional_sampling_variance"],
                                   np.asarray(base["conditional_sampling_variance"]) * 3600.)
        np.testing.assert_allclose(scaled["mean_standard_error"], np.asarray(base["mean_standard_error"]) * 60.)


if __name__ == "__main__":
    unittest.main()
