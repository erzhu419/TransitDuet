from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v180_service_increment as audit


class ServiceIncrementTests(unittest.TestCase):
    def test_strata_remove_composition_only_association(self):
        groups = []
        for offset in (0., 20.):
            groups.append(np.mean([audit.moment(x+offset, np.full(len(audit.TARGETS), y+offset))
                for x, y in ((0., 0.), (1., 0.), (0., 1.), (1., 1.))], axis=0))
        conditional = audit.covariance(np.array(groups))[0]
        marginal = audit.covariance(np.array(groups).sum(axis=0, keepdims=True))[0]
        np.testing.assert_allclose(conditional, 0.)
        self.assertTrue((marginal > .9).all())

    def test_within_stratum_signal_survives(self):
        value = np.array([np.mean([audit.moment(x, np.full(len(audit.TARGETS), -x)) for x in range(4)], axis=0)])
        np.testing.assert_allclose(audit.covariance(value)[0], -1.)

    def test_missing_variation_is_not_zero_correlation(self):
        value = np.array([audit.moment(1., np.arange(len(audit.TARGETS)))])
        self.assertTrue(np.isnan(audit.covariance(value)[0]).all())
        self.assertIsNone(audit.interval(np.nan, np.array([np.nan]))['mean'])

    def test_batches_match_individual_crossed_moments(self):
        values = np.array([[audit.moment(x, np.full(len(audit.TARGETS), x*x)) for x in (i, i+1)] for i in (1., 3.)])
        batch = audit.covariance(values)[0]
        for i in range(2):
            np.testing.assert_allclose(batch[i], audit.covariance(values[i])[0])

    def test_score_covariance_is_width_minus_saved_penalty(self):
        rows = []
        for x in range(5):
            y = np.zeros(len(audit.TARGETS))
            y[5:8] = [2*x, 3*x, -x]
            rows.append(audit.moment(x, y))
        cov = audit.covariance(np.array([np.mean(rows, axis=0)]))[1]
        self.assertAlmostEqual(cov[7], cov[5]-cov[6])


if __name__ == '__main__':
    unittest.main()
