from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_freqduet_v80_response_noise import repeat_moments


class ResponseNoiseTest(unittest.TestCase):
    def test_identical_repeats_preserve_day_signal_without_sampling_noise(self):
        x = np.arange(1., 5.)[:, None]
        result = repeat_moments(np.stack((x, x), axis=1))
        np.testing.assert_allclose(result["conditional_sampling_variance"], [0.])
        np.testing.assert_allclose(result["between_day_covariance"], [np.var(x, ddof=1)])
        self.assertEqual(result["both_positive"], [4])

    def test_negative_covariance_is_not_clipped_into_a_signal(self):
        x = np.array([-2., -1., 1., 2.])[:, None]
        result = repeat_moments(np.stack((x, -x), axis=1))
        self.assertLess(result["between_day_covariance"][0], 0.)
        self.assertLessEqual(result["between_day_covariance_ci95"][0][1], 0.)
        self.assertEqual(result["opposite_sign"], [4])

    def test_unit_scaling_changes_variance_quadratically(self):
        values = np.arange(24.).reshape(4, 2, 3) - 10.
        base, scaled = repeat_moments(values), repeat_moments(values * 10.)
        np.testing.assert_allclose(scaled["conditional_sampling_variance"],
                                   np.asarray(base["conditional_sampling_variance"]) * 100.)
        np.testing.assert_allclose(scaled["current_mean_standard_error"],
                                   np.asarray(base["current_mean_standard_error"]) * 10.)


if __name__ == "__main__":
    unittest.main()
