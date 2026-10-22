import unittest

import numpy as np

import audit_freqduet_v202_cancellation as v


class CancellationTests(unittest.TestCase):
    def test_signed_negative_allocation_and_covariance_retained(self):
        x = np.array([-1., 1., -1., 1.])
        parts = np.stack((2*x, -x), axis=1)
        r = v.partition(parts, x, ['I', 'minus_A'])
        np.testing.assert_allclose(r['signed_allocation'], [2., -1.])
        self.assertEqual(r['sum_own_variances'], 5.)
        self.assertEqual(r['cross_covariance_total'], -4.)
        self.assertEqual(r['variance'], 1.)

    def test_three_components_sum_with_offset(self):
        x = np.array([-1., 1., -1., 1.])
        parts = np.stack((x+4., 2*x-3., -.5*x+2.), axis=1)
        r = v.partition(parts, parts.sum(axis=1), ['W', 'B', 'P'])
        self.assertAlmostEqual(sum(r['signed_allocation']), 6.25)


if __name__ == '__main__':
    unittest.main()
