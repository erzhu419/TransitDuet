from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import seal_freqduet_v163_joint_intervals as fit


class JointIntervalTests(unittest.TestCase):
    def test_equal_days_then_episodes_then_queries(self):
        days = np.array([10, 10, 10, 10, 10, 11])
        episodes = np.array([0, 0, 0, 0, 1, 2])
        contributing, denominator = fit.hierarchy(days, episodes)
        self.assertEqual(contributing.tolist(), [10, 11])
        np.testing.assert_array_equal(denominator, [8, 8, 8, 8, 2, 1])
        mass = 1./(len(contributing)*denominator)
        self.assertAlmostEqual(mass[:4].sum(), .25)
        self.assertAlmostEqual(mass[4], .25)
        self.assertAlmostEqual(mass[5], .5)

    def test_day_ranks_and_exact_observed_cdf_boundaries(self):
        for n, expected in ((19, (1., 17.)), (32, (2., 29.))):
            vectors = np.column_stack((np.arange(n), np.zeros(n), np.zeros(n)))
            entry, denominator = fit.fit_key(vectors, np.arange(n), np.arange(n))
            self.assertEqual((entry['qlow_s'], entry['qhigh_s']), expected)
            np.testing.assert_array_equal(denominator, np.full(n, n))
        total = np.array([0., 1., 2., 3.])
        # Three thirds reach exactly one day; interpolation must not advance to 3.
        self.assertEqual(fit.inverse_day_mass(total, np.array([3, 3, 3, 1]), 1), 2.)

    def test_repeated_queries_do_not_create_independent_days(self):
        entry, _ = fit.fit_key(np.zeros((100, 3)), np.repeat(np.arange(10), 10), np.arange(100))
        self.assertEqual(entry['status'], 'abstain_less_than_19_days')
        self.assertIsNone(fit.interval(100., entry))
        self.assertIsNone(fit.interval(100., None))

    def test_joint_vectors_preserve_negative_component_correlation(self):
        values = np.arange(19)
        vectors = np.column_stack((values, -values, np.zeros(19)))
        entry, _ = fit.fit_key(vectors, values, values)
        self.assertEqual(fit.interval(100., entry), (100., 100.))
        self.assertIsNone(fit.interval(100., entry, native_supported=False))

    def test_signed_residual_and_physical_truncation(self):
        entry = dict(status='finite', qlow_s=-10., qhigh_s=20.)
        self.assertEqual(fit.interval(15., entry), (0., 25.))
        self.assertEqual(fit.interval(0., dict(status='finite', qlow_s=5., qhigh_s=10.)), (0., 0.))

    def test_support_reduction_is_episode_mean_not_query_ratio(self):
        cells = fit.support_cells(1, 'old', [10], np.full(4, 10), np.array([0, 0, 0, 1]),
            np.full(4, 3), np.array([True, True, True, False]), np.ones(4, dtype=bool),
            np.ones(4, dtype=bool))
        self.assertEqual(cells[0]['finite_fraction'], .5)
        self.assertEqual(cells[0]['finite_episodes'], 1)
        self.assertEqual(cells[2]['eligible_episodes'], 0)

    def readouts(self):
        return [dict(actor=a, day=d, mode=m, scope=s, finite_fraction=f, eligible_episodes=10,
            finite_episodes=8, eligible_queries=20, finite_queries=16,
            native_abstentions=2, key_abstentions=2, missing_reference=0)
            for a, f in ((1, .7), (2, 1.)) for d in (10, 11) for m in fit.protocol.MODES for s in fit.SCOPES]

    def test_release_uses_crossed_cell_average_but_requires_each_cell_finite(self):
        rows = self.readouts()
        self.assertEqual(fit.release(rows, [1, 2], [10, 11])['decision'], 'pass')
        rows[0]['finite_episodes'] = 0
        self.assertIn('old/all', fit.release(rows, [1, 2], [10, 11])['failed'])

    def test_missing_cells_and_low_support_stop_release(self):
        rows = self.readouts()
        self.assertEqual(fit.release(rows[:-1], [1, 2], [10, 11])['decision'], 'insufficient_support')
        rows[0]['finite_fraction'] = .1
        self.assertIn('old/all', fit.release(rows, [1, 2], [10, 11])['failed'])
        rows = self.readouts()
        rows[0]['missing_reference'] = 1
        self.assertIn('old/all', fit.release(rows, [1, 2], [10, 11])['failed'])


if __name__ == '__main__':
    unittest.main()
