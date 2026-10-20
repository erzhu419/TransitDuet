from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import summarize_freqduet_v158_variance as summary


class VarianceSummaryTests(unittest.TestCase):
    def fixture(self):
        mu = np.array([[[1., -1.], [2., -4.], [4., -2.]],
            [[3., -2.], [7., -3.], [5., -5.]]])
        within = np.array([[4., -1.], [-1., 2.]])
        return mu, np.einsum('sdi,sdj->sdij', mu, mu)+within

    def test_uniform_weights_reproduce_original_decomposition(self):
        mu, second = self.fixture()
        result = summary.decomposition(mu, second, np.full((1, 2), .5), np.full((1, 3), 1./3))
        old = summary.frozen.covariance_report(mu, second)
        for name, field in (('total', 'covariance_s2'), ('within', 'within_cell_covariance_s2'),
                ('actor', 'actor_mean_covariance_s2'), ('day', 'day_mean_covariance_s2'),
                ('interaction', 'actor_day_interaction_covariance_s2')):
            np.testing.assert_allclose(result[name][0], old[field], atol=1e-12)

    def test_shared_crossed_weights_match_explicit_resampling(self):
        mu, second = self.fixture()
        a, d = np.array([[0, 0], [0, 1]]), np.array([[2, 0, 2], [1, 1, 2]])
        result = summary.decomposition(mu, second, summary.weights(2, a), summary.weights(3, d))
        for b in range(2):
            old = summary.frozen.covariance_report(mu[a[b]][:, d[b]], second[a[b]][:, d[b]])
            np.testing.assert_allclose(result['total'][b], old['covariance_s2'], atol=1e-12)
            np.testing.assert_allclose(result['day'][b], old['day_mean_covariance_s2'], atol=1e-12)
            np.testing.assert_allclose(result['within'][b], old['within_cell_covariance_s2'], atol=1e-12)

    def test_opposite_constant_bias_is_not_noise_correlation(self):
        mu = np.tile([10., -10.], (2, 3, 1))
        second = np.einsum('sdi,sdj->sdij', mu, mu)
        result = summary.decomposition(mu, second, np.full((1, 2), .5), np.full((1, 3), 1./3))
        np.testing.assert_allclose(result['total'], 0., atol=1e-12)
        self.assertIsNone(summary.estimate(np.nan, np.array([np.nan]))['ci95'])


if __name__ == '__main__':
    unittest.main()
