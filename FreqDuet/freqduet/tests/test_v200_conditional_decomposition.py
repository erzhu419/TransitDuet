import unittest

import numpy as np

import audit_freqduet_v200_conditional_decomposition as v


class ConditionalDecompositionTests(unittest.TestCase):
    def test_full_parent_daily_centering_includes_unsupported_mass(self):
        m = np.zeros((2, 1, 9, v.saved.SIZE))
        for day, (offset_j, offset_y) in enumerate(((0., 0.), (100., -50.))):
            for bi, mass, j, y in ((0, 1., 1., 2.), (8, 3., 9., 6.)):
                vector = np.zeros(v.saved.N)
                vector[v.J], vector[v.Y] = j+offset_j, y+offset_y
                m[day, 0, bi] = mass*v.saved.moment(vector)
        parent = m.sum(axis=2)
        values = v.centered(m, parent)
        np.testing.assert_allclose(values[0], values[1])
        self.assertAlmostEqual(values[0, 0, 0, 3], 18.)
        self.assertAlmostEqual(values[0, 0, 8, 3], 6.)
        self.assertAlmostEqual(values[0, 0, :, 3].sum(), 24.)

    def test_pure_mix_and_no_selected_mass_renormalization(self):
        first, last = np.zeros((1, 1, 9, 4)), np.zeros((1, 1, 9, 4))
        keep = np.zeros((1, 9), dtype=bool)
        keep[0, :2] = True
        for values, masses in ((first, [.3, .1]), (last, [.1, .3])):
            for bi, mass, t in zip((0, 1), masses, (2., 5.)):
                values[0, 0, bi] = [mass, 0., 0., mass*t]
            values[0, 0, 8] = [.6, 0., 0., -3.]
        e, terms = v.evaluate(first, last, keep)
        np.testing.assert_allclose(e[0, :, 0], [.4, .4])
        np.testing.assert_allclose(e[0, :, 1], [.6, .6])
        self.assertAlmostEqual(terms[0, 0], .6)
        self.assertAlmostEqual(terms[0, 1], 0.)
        self.assertAlmostEqual(e[0, 0, 3], 1.1)
        self.assertAlmostEqual(e[0, 0, 5], -1.9)

    def test_conditional_covariance_and_mean_product_change(self):
        first, last = np.zeros((1, 1, 9, 4)), np.zeros((1, 1, 9, 4))
        keep = np.zeros((1, 9), dtype=bool)
        keep[0, 0] = True
        first[0, 0, 0] = [.2, .2, .4, 1.2]
        last[0, 0, 0] = [.2, .6, .4, 2.2]
        e, t = v.evaluate(first, last, keep)
        np.testing.assert_allclose(t[0, :4], [0., 1., .2, .8])
        np.testing.assert_allclose(e[0, :, 6], [.8, 1.])
        np.testing.assert_allclose(e[0, :, 7], [.4, 1.2])

    def test_missing_resampled_cell_is_unknown_without_mask_repair(self):
        first, last = np.zeros((1, 1, 9, 4)), np.zeros((1, 1, 9, 4))
        keep = np.zeros((1, 9), dtype=bool)
        keep[0, 0] = True
        frozen = keep.copy()
        last[0, 0, 0] = [1., 0., 0., 2.]
        first[0, 0, 8] = [1., 0., 0., 4.]
        e, t = v.evaluate(first, last, keep)
        self.assertTrue(np.isnan(t).all())
        self.assertEqual(e[0, 0, 4], 4.)
        self.assertEqual(e[0, 0, 3], 0.)
        np.testing.assert_array_equal(keep, frozen)

    def test_batch_resampling_recomputes_cell_means(self):
        first, last = np.zeros((2, 1, 9, 4)), np.zeros((2, 1, 9, 4))
        keep = np.zeros((1, 9), dtype=bool)
        keep[0, 0] = True
        first[:, 0, 0] = [[.1, .2, .3, .7], [.5, 1., 2., 5.]]
        last[:, 0, 0] = [[.2, .1, .4, 1.], [.3, .9, .6, 3.]]
        e, t = v.evaluate(first, last, keep)
        for i in range(2):
            ee, tt = v.evaluate(first[i:i+1], last[i:i+1], keep)
            np.testing.assert_allclose(e[i], ee[0])
            np.testing.assert_allclose(t[i], tt[0])
        self.assertNotEqual(e[0, 0, 6], e[1, 0, 6])


if __name__ == '__main__':
    unittest.main()
