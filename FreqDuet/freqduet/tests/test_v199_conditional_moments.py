import unittest

import numpy as np

import prepare_freqduet_v199_conditional_moments as v


class ConditionalPreparationTests(unittest.TestCase):
    def test_frozen_bin_endpoints(self):
        cases = [(0., 0., (0, 0)), (1./3, 5., (1, 1)), (2./3, 10., (2, 2)), (1., 100., (2, 2)),
            (np.nextafter(1./3, 0.), np.nextafter(5., 0.), (0, 0))]
        for f, h, expected in cases:
            self.assertEqual(v.plan.stratum(f, h), expected)

    def test_all9_recover_every_second_moment_and_clock_mass(self):
        m, f = np.zeros((1, 9, v.saved.SIZE)), np.zeros((1, 9, 1+len(v.saved.FACTS)))
        parent, clocks = np.zeros(v.saved.SIZE), np.zeros(f.shape[-1])
        for i in range(9):
            vector = np.arange(v.saved.N, dtype=float)*(i-3.)
            weight = (i+1)/45.
            m[0, i] = weight*v.saved.moment(vector)
            f[0, i] = weight*np.r_[1., np.arange(len(v.saved.FACTS))]
            parent += m[0, i]
            clocks += f[0, i]
        v.verify_partition(m, f, parent[None], clocks[None])
        m[0, 0, -2] += .01
        with self.assertRaisesRegex(AssertionError, 'parent moments'):
            v.verify_partition(m, f, parent[None], clocks[None])

    def test_original_episode_denominators_not_conditional_normalization(self):
        # One episode has four eligible queries but only two admitted queries.
        source = {('a',): {'key': np.zeros((9, v.saved.SIZE))},
            ('b',): {'key': np.zeros((9, v.saved.SIZE))}}
        one = v.saved.moment(np.ones(v.saved.N))
        source[('a',)]['key'][0] = 2*one
        source[('b',)]['key'][8] = one
        result = v.saved.collapse(source, {('a',): [4, 4], ('b',): [1, 1]})['key']
        self.assertEqual(result[0, 0], .25)
        self.assertEqual(result[8, 0], .5)
        self.assertEqual(result[:, 0].sum(), .75)

    def test_19_distinct_days_in_both_halves_and_keep_unsupported_mass(self):
        mass = np.zeros((64, 1, 9))
        mass[:, 0, 8] = .2
        mass[:19, 0, 0], mass[32:51, 0, 0] = .3, .3
        mass[:18, 0, 1], mass[32:51, 0, 1] = .4, .4
        parent = np.zeros((64, 1, v.saved.SIZE))
        parent[..., 0] = mass.sum(axis=2)
        original = mass.copy()
        counts, keep, factors, bins = v.support_summary(mass, parent)
        np.testing.assert_array_equal(mass, original)
        np.testing.assert_array_equal(counts[:, 0, 0], [19, 19])
        np.testing.assert_array_equal(counts[:, 0, 1], [18, 19])
        self.assertTrue(keep[0, 0])
        self.assertFalse(keep[0, 1])
        self.assertFalse(keep[0, 2])
        self.assertTrue(all(m > 0. for m in bins[1]['balanced_mass']))
        self.assertEqual(bins[1]['retained_balanced_mass'], [0., 0.])
        self.assertEqual(factors.shape, (2, 1))

    def test_balance_is_from_full_parent_not_per_bin(self):
        mass = np.zeros((64, 1, 9))
        mass[:32, 0, 0], mass[:32, 0, 1] = .2, .1
        mass[32:, 0, 0], mass[32:, 0, 1] = .1, .5
        parent = np.zeros((64, 1, v.saved.SIZE))
        parent[..., 0] = mass.sum(axis=2)
        _, _, factors, bins = v.support_summary(mass, parent)
        np.testing.assert_allclose(factors[:, 0], [1.5, .75])
        np.testing.assert_allclose(bins[0]['balanced_mass'], [.3, .075])
        np.testing.assert_allclose(np.sum([b['balanced_mass'] for b in bins], axis=0), [.45, .45])


if __name__ == '__main__':
    unittest.main()
