import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v125_timing_propagation import edge_rows, moment, effect, profiles, first_change


class PropagationTests(unittest.TestCase):
    def points(self, times, holds=None):
        holds = [0.]*len(times) if holds is None else holds
        return {(i, 3, True): dict(time_s=t, holding_s=h,
            coordinates=np.asarray([t-h, h, 0., 0.])) for i, (t, h) in enumerate(zip(times, holds))}

    def test_exact_covariance_and_unequal_cohort_contributions(self):
        b = edge_rows(self.points([0., 10., 25., 60., 100.]))
        c = edge_rows(self.points([0., 12., 25., 70., 100.], [0., 2., 0., 5., 0.]))
        delta = effect(moment(b), moment(c))
        parts = profiles(b, c, lambda r: int(r['key'][0] > 1))
        self.assertAlmostEqual(sum(r['pooled_cv2_delta'] for r in parts), delta['cv_squared'])
        self.assertEqual([r['base_n'] for r in parts], [1, 3])
        self.assertAlmostEqual(sum(delta[k] for k in ('release_allocated', 'realized_holding_allocated',
            'observed_nonholding_stop_allocated', 'remaining_elapsed_allocated', 'mean_normalization')), delta['cv_squared'])

    def test_reference_cohorts_keep_cross_boundary_edge(self):
        b = edge_rows(self.points([14300., 14350., 14450.]))
        c = edge_rows(self.points([14300., 14410., 14490.]))
        reference = {0: 14300., 1: 14350., 2: 14450.}
        rows = profiles(b, c, lambda r: int(reference[r['key'][0]]//14400))
        self.assertEqual([r['candidate_n'] for r in rows], [1, 1])
        self.assertEqual(b[-1]['headway'], 100.)

    def test_first_holding_and_first_timing_change_are_distinct(self):
        b = self.points([100., 200., 300.])
        c = self.points([100., 200., 310.], [0., 5., 10.])
        self.assertEqual(first_change(b, c, 'time_s')['trip_id'], 2)
        self.assertEqual(first_change(b, c, 'holding_s')['trip_id'], 1)
        self.assertIsNone(first_change(b, b, 'time_s'))

    def test_missing_and_zero_mean_profiles_are_not_zero_effects(self):
        self.assertIsNone(moment([]))
        self.assertIsNone(moment(edge_rows(self.points([100., 100.]))))
        b = edge_rows(self.points([0., 10., 30.]))
        c = edge_rows(self.points([0., 12., 32.]))
        rows = profiles(b, c, lambda r: r['key'][0])
        self.assertTrue(all(r['effect'] is not None for r in rows))


if __name__ == '__main__':
    unittest.main()
