from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import preregister_freqduet_v198_conditional_moments as stage


class ConditionalAvailabilityTests(unittest.TestCase):
    def test_fixed_bins_and_edges(self):
        self.assertEqual(stage.stratum(0., 0.), (0, 0))
        self.assertEqual(stage.stratum(1./3, 5.), (1, 1))
        self.assertEqual(stage.stratum(2./3, 10.), (2, 2))
        self.assertEqual(stage.stratum(1.+1e-9, 12.), (2, 2))
        with self.assertRaises(AssertionError):
            stage.stratum(.5, -1.)

    def test_parent19_days_does_not_imply_conditional19_each_half(self):
        mass = np.zeros((64, 1, 2))
        mass[:, 0, 0] = 1.
        mass[:19, 0, 1] = 1.
        mass[32:50, 0, 1] = 1.
        counts, keep = stage.supported(mass)
        self.assertTrue(keep[0, 0])
        self.assertFalse(keep[0, 1])
        self.assertEqual(counts[:, 0, 1].tolist(), [19, 18])

    def test_binary_support_is_derivable_but_conditional_JY_is_not(self):
        vectors = []
        for alternate in (False, True):
            rows = []
            for b in (0, 1):
                for j in (-1., 1.):
                    x = np.zeros(stage.saved.N)
                    y = j*(1. if (b == 0) != alternate else -1.)
                    x[stage.saved.VARS.index('first_fraction')] = .2 if b == 0 else .8
                    x[stage.saved.VARS.index('horizon_min')] = 4.
                    x[stage.saved.VARS.index('predicted_entry')] = b
                    x[stage.saved.VARS.index('signed_remaining_s')] = j
                    x[stage.saved.VARS.index('innovation_total_s')] = j
                    x[stage.saved.VARS.index('first_running_under_s')] = y
                    x[stage.saved.VARS.index('full_under_s')] = y
                    rows.append(x)
            vectors.append(np.array(rows))
        moments = [np.mean([stage.saved.moment(x) for x in rows], axis=0) for rows in vectors]
        np.testing.assert_allclose(moments[0], moments[1])
        np.testing.assert_allclose(stage.binary_mass(moments[0][None, None], 'predicted_entry'), [[[.5, .5]]])
        j, y = map(stage.saved.VARS.index, ('signed_remaining_s', 'first_running_under_s'))
        self.assertEqual(np.mean(vectors[0][:2, j]*vectors[0][:2, y]), 1.)
        self.assertEqual(np.mean(vectors[1][:2, j]*vectors[1][:2, y]), -1.)

    def test_mixture_product_partition_and_excluded_mass_stay_explicit(self):
        a, b = np.array([.2, .3]), np.array([.1, .5])
        x, y = np.array([2., -1.]), np.array([3., 4.])
        change = (b*y-a*x).sum()
        mix = ((b-a)*(x+y)/2).sum()
        conditional = ((a+b)/2*(y-x)).sum()
        self.assertAlmostEqual(change, mix+conditional)
        self.assertLess(a.sum(), 1.)
        self.assertLess(b.sum(), 1.)


if __name__ == '__main__':
    unittest.main()
