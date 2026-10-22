from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import preregister_freqduet_v208_query_covariance as v


class QueryCovarianceTests(unittest.TestCase):
    def test_second_moments_do_not_determine_state_cross_covariance(self):
        a = np.array([(s, x, s*x) for s, x in product((-1., 1.), repeat=2)])
        b = a.copy()
        b[:, 2] *= -1.
        np.testing.assert_array_equal(a.mean(axis=0), b.mean(axis=0))
        np.testing.assert_array_equal(a.T@a, b.T@b)
        self.assertEqual(np.prod(a, axis=1).mean(), 1.)
        self.assertEqual(np.prod(b, axis=1).mean(), -1.)

    def test_aligned_sums_and_relative_mean_decomposition(self):
        s, x, y = np.array([[1., 2., 4., 7.], [2., 5., -1., 6.], [8., -3., 4., 2.]])
        w = np.array([.02, .09, .04, .01])
        m = w.sum()
        mean = lambda z: w@z/m
        calc = lambda p, q: v.centered_cross(m, w@s, w@x, w@y, w@(x*y), w@(s*x), w@(s*y), w@(s*x*y), p, q)
        p, q = 5., -2.
        self.assertAlmostEqual(calc(p, q), w@((s-mean(s))*(x-p)*(y-q)))
        rel = (mean(x)-p)*(w@((s-mean(s))*y))+(mean(y)-q)*(w@((s-mean(s))*x))
        self.assertAlmostEqual(calc(p, q), calc(mean(x), mean(y))+rel)

    def test_missing_exposure_is_not_zero_effect(self):
        with self.assertRaisesRegex(AssertionError, 'zero exposure'):
            v.centered_cross(0., *([0.]*9))

    def test_header_inventory_does_not_read_large_payload(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'moments.npz'
            np.savez_compressed(path, moments=np.zeros((64, 2, 9, 820)), days=np.arange(64))
            with patch.object(np, 'load', side_effect=AssertionError('payload access')):
                got = v.npz_headers(path)
        self.assertEqual(got['moments']['shape'], [64, 2, 9, 820])
        self.assertEqual(got['days']['shape'], [64])

    def test_protocol_fixes_only_two_mixed_sums_and_no_fit(self):
        with patch.dict('os.environ', FREQDUET_DIAGNOSTIC_COMMIT='fixture'):
            p = v.registration(dict(v151=True, v146=True, v172=True, main=True, window_s=3600., tails=[.1, .1]))
        self.assertEqual(p['required']['fields'], list(v.MIXED_FIELDS))
        self.assertEqual(len(p['signals']), 2)
        self.assertFalse(p['predictor_fit'])
        self.assertFalse(p['associations_estimated'])
        self.assertFalse(p['confirmation_started'])
        self.assertEqual(p['calibration_days'], list(range(2951001, 2951065)))
        self.assertEqual(v.requirements(list(v.plan.saved.VARS)), [])
        self.assertIn('innovation_rms_s', v.requirements(list(v.plan.FIELDS)))


if __name__ == '__main__':
    unittest.main()
