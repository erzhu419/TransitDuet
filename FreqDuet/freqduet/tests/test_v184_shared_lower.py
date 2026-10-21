from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import seal_freqduet_v184_shared_lower as seal


class SharedLowerSealTests(unittest.TestCase):
    def test_day_support_not_query_count(self):
        entry, den = seal.shared_fit(np.zeros(180), np.repeat(np.arange(18), 10), np.arange(180))
        self.assertEqual(entry['status'], 'abstain_less_than_19_days')
        self.assertIsNone(entry['delta_s'])
        self.assertEqual(entry['day_count'], 18)
        entry, den = seal.shared_fit(np.arange(19.), np.arange(19), np.arange(19))
        self.assertEqual(entry['status'], 'finite')
        self.assertEqual(entry['delta_s'], 17.)

    def test_new_sign_mass_equal_day_episode_query(self):
        days = np.array([1, 1, 1, 2])
        episodes = np.array([0, 0, 1, 2])
        contributing, within = seal.fit.hierarchy(days, episodes)
        self.assertEqual(within.tolist(), [4, 4, 2, 1])
        self.assertEqual([sum(Fraction(1, int(v)) for v in within[days == d]) for d in contributing], [1, 1])
        entry, den = seal.shared_fit(np.arange(4.), days, episodes)
        self.assertEqual(den.tolist(), [8, 8, 4, 2])

    def test_signed_delta_and_both_physical_caps(self):
        bounds = np.array([[3., 8.], [3., 8.], [3., 8.]])
        entries = [dict(status='finite', delta_s=5.), dict(status='finite', delta_s=-20.)]
        result, finite = seal.corrected(bounds, np.ones(3, bool), np.array([True, True, False]),
            np.array([0, 1, -1]), entries)
        self.assertEqual(result.tolist(), [[0., 8.], [8., 8.], [3., 8.]])
        self.assertTrue(finite.all())

    def test_unavailable_prefix_and_low_parameter_support_abstain_without_rescue(self):
        bounds = np.array([[3., 8.], [3., 8.], [np.nan, np.nan], [3., 8.]])
        entries = [dict(status='abstain_less_than_19_days', delta_s=None), dict(status='finite', delta_s=1.)]
        result, finite = seal.corrected(bounds, np.array([True, True, False, True]),
            np.array([True, True, True, False]), np.array([-1, 0, 1, -1]), entries)
        self.assertEqual(finite.tolist(), [False, False, False, True])
        self.assertTrue(np.isnan(result[:3]).all())
        self.assertEqual(result[-1].tolist(), [3., 8.])

    def test_score_identity_and_upper_tail_unchanged(self):
        bounds = np.array([[3., 8.], [4., 9.]])
        updated = np.array([[1., 8.], [6., 9.]])
        truth = np.array([2., 10.])
        values = seal.observations(bounds, updated, truth, np.ones(2, bool),
            np.ones(2, bool), np.array([2., -2.]))
        self.assertEqual(values[:, seal.METRICS.index('v172_score_delta_s')].tolist(), [-8., -2.])
        self.assertEqual(values[:, seal.METRICS.index('above')].tolist(), [0., 1.])

    def test_strict_completed_prefix_and_no_prefix_fallback(self):
        atoms = {0: ((0., 0), 100., 1, np.array([0., 1., 0.]))}
        prefix = seal.previous.Prefix(atoms)
        self.assertEqual(prefix.query(100., 1)['global']['n'], 0)
        history = prefix.query(101., 1)['global']
        self.assertEqual(history['n'], 1)
        self.assertEqual(seal.plan.prior.service_sign(history), -1)
        self.assertEqual(prefix.query(3701., 1)['global']['n'], 0)


if __name__ == '__main__':
    unittest.main()
