import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from freqduet_rank_interval import completed_values, rank_interval
from freqduet_stop_phase_eta import ServiceResiduals
from audit_freqduet_v142_rank_intervals import record_interval, summarize


class RankIntervalTests(unittest.TestCase):
    def test_singleton_and_eight_samples_have_no_finite_upper_rank(self):
        for n in (1, 8):
            r = rank_interval(np.arange(n, dtype=float))
            self.assertEqual(r['status'], 'insufficient_upper_rank')
            self.assertEqual(r['lower_s'], 0.)
            self.assertIsNone(r['upper_s'])

    def test_zero_support_is_explicit(self):
        self.assertEqual(rank_interval(np.array([]))['status'], 'no_completed_survivor')

    def test_exact_integer_ranks_at_thresholds(self):
        for n, lo, hi in ((9, 1, 9), (10, 1, 10), (19, 2, 18), (29, 3, 27)):
            r = rank_interval(np.arange(1., n+1.))
            self.assertEqual((r['lower_rank'], r['upper_rank']), (lo, hi))
            self.assertEqual((r['lower_s'], r['upper_s']), (float(lo), float(hi)))

    def test_exact_rank_enumeration_without_monte_carlo(self):
        for total, covered_expected in ((10, 8), (20, 16), (30, 24)):
            ordered = np.arange(1., total+1.)
            covered = 0
            for i, truth in enumerate(ordered):
                r = rank_interval(np.delete(ordered, i))
                covered += r['lower_s'] <= truth <= r['upper_s']
            self.assertEqual(covered, covered_expected)

    def test_ties_are_not_arbitrarily_declared_missing_support(self):
        r = rank_interval(np.ones(9)*5.)
        self.assertTrue(r['finite'])
        self.assertEqual((r['lower_s'], r['upper_s']), (5., 5.))

    def test_rank_sort_does_not_mutate_points_or_input(self):
        values = np.array([9., 1., 8., 2., 7., 3., 6., 4., 5.])
        original = values.copy()
        points = dict(v140_gap_s=123.25, v141_gap_s=117.75)
        row = record_interval(rank_interval(values), dict(p10=1.8, p90=8.2, mean=4.75), 5., [1], 100., points)
        self.assertEqual(row['conditional_point_s'], 4.75)
        self.assertEqual(points, {k: row[k] for k in points})
        np.testing.assert_array_equal(values, original)

    def test_prefix_and_window_match_frozen_estimator(self):
        history = ServiceResiduals([(True, 1, 1., 10.), (True, 1, 3602., 30.), (True, 1, 3603., 999.)])
        values, latest = completed_values(history, True, 1, 3603., 20.)
        np.testing.assert_array_equal(values, [10.])
        self.assertEqual(latest, 3602.)
        past, _ = completed_values(history, True, 1, 8000., 0.)
        self.assertEqual(len(past), 3)

    def test_no_survivor_is_not_a_zero_duration_sample(self):
        values, latest = completed_values(ServiceResiduals([(True, 1, 5., 10.)]), True, 1, 100., 20.)
        self.assertEqual(len(values), 0)
        self.assertIsNone(latest)

    def test_insufficient_intervals_are_not_coverage_successes(self):
        linear = dict(p10=1.8, p90=8.2, mean=5.)
        finite = record_interval(rank_interval(np.arange(1., 10.)), linear, 5., [1], 100., {})
        insufficient = record_interval(rank_interval(np.array([5.])), linear, 5., [2], 100., {})
        r = summarize([finite, insufficient])
        self.assertEqual(r['requests'], 2)
        self.assertEqual(r['finite_n'], 1)
        self.assertEqual(r['finite_fraction'], .5)
        self.assertEqual(r['paired_rank_coverage'], 1.)
        self.assertIsNone(summarize([insufficient])['paired_rank_coverage'])


if __name__ == '__main__':
    unittest.main()
