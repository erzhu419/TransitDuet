import unittest

import numpy as np

import audit_freqduet_v210_query_covariance as v


class QueryCovarianceReadoutTests(unittest.TestCase):
    def test_project_uses_actual_joint_cross_moments(self):
        x = np.arange(1., v.saved.N+1)
        beta = np.arange(18.).reshape(2, 9)/50.
        mass, first, second = v.project(v.saved.moment(x)*.3, beta)
        residual = x[v.prep.TARGET_IX]-beta@x[v.prep.STATE_IX]
        expected = np.r_[x[v.prep.SIGNAL_IX], residual]
        self.assertEqual(mass, .3)
        np.testing.assert_allclose(first, .3*expected)
        np.testing.assert_allclose(second, .3*np.outer(expected, expected))

    def test_full_parent_centering_keeps_unsupported_bins(self):
        mass = np.ones((64, 2, 9))*.1
        first = np.zeros((*mass.shape, 4))
        second = np.zeros((*mass.shape, 4, 4))
        mixed = np.zeros((*mass.shape, 2))
        first[..., 0] = .2
        second[..., 0, 0] = .5
        second[..., 0, 3] = .3
        keep = np.zeros((2, 9), bool)
        keep[:, 0] = True
        first[:, :, 1:, 2] = .9
        daily, variation, exposure, raw = v.profiles(mass, first, second, mixed, keep)
        self.assertTrue(np.all(daily[..., 0, 1] != 0.))
        np.testing.assert_allclose(daily[..., :2].sum(axis=-1), daily[..., 2])
        np.testing.assert_allclose(exposure, .1)
        np.testing.assert_allclose(raw[..., 0], .9)
        np.testing.assert_allclose(raw[..., 1], .1)
        self.assertTrue(np.all(variation[..., 0] > 0.))

    def test_crossed_average_is_shared_day_not_paired_halves(self):
        values = np.arange(64*3*2.).reshape(64, 3, 2)
        aw = np.array([[.2, .3, .5], [.6, .4, 0.]])
        dw = np.zeros((2, 2, 32))
        dw[0, 0, 1], dw[1, 0, 3] = 1., 1.
        dw[0, 1, 2], dw[1, 1, 4] = 1., 1.
        got = v.average(values, aw, dw)
        np.testing.assert_allclose(got[0, 0], aw[0]@values[1])
        np.testing.assert_allclose(got[0, 1], aw[0]@values[35])
        np.testing.assert_allclose(got[1, 1], aw[1]@values[36])

    def test_profiles_equal_direct_query_level_centering(self):
        queries = np.array([[1., -2., 3., 4.], [4., 1., -1., 8.], [7., 2., 6., -3.]])
        weights = np.array([.02, .03, .05])
        mass = np.full((64, 2, 9), weights.sum())
        first = np.broadcast_to(weights@queries, (*mass.shape, 4)).copy()
        second = np.broadcast_to(np.einsum('q,qi,qj->ij', weights, queries, queries), (*mass.shape, 4, 4)).copy()
        mixed = np.broadcast_to(weights@(queries[:, :2]*queries[:, 2:3]*queries[:, 3:4]), (*mass.shape, 2)).copy()
        keep = np.ones((2, 9), bool)
        daily, _, _, _ = v.profiles(mass, first, second, mixed, keep)
        means = weights@queries/weights.sum()
        direct = weights@((queries[:, :2]-means[:2])*(queries[:, 2:3]-means[2])*(queries[:, 3:4]-means[3]))
        np.testing.assert_allclose(daily[0, 0, :, 2], 9*direct)
        np.testing.assert_allclose(daily[..., 1], 0., atol=1e-12)

    def test_both_halves_and_modes_are_required(self):
        good = dict(ci=[1., 2.])
        rows = [dict(primary_family=good, halves=dict(first32=dict(total=good), last32=dict(total=good))) for _ in range(2)]
        self.assertTrue(v.replicated(rows))
        rows[1]['halves']['last32']['total'] = dict(ci=[-1., 2.])
        self.assertFalse(v.replicated(rows))
        self.assertEqual(v.sign(dict(ci=None)), 0)


if __name__ == '__main__':
    unittest.main()
