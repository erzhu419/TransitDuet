import unittest

import numpy as np

import prepare_freqduet_v209_query_mixed as v


class QueryMixedPreparationTests(unittest.TestCase):
    def test_frozen_beta_and_signal_order(self):
        x = np.arange(1., v.saved.N+1)
        beta = np.arange(18.).reshape(2, 9)/50.
        r = x[v.TARGET_IX]-beta@x[v.STATE_IX]
        np.testing.assert_allclose(v.mixed_sums(x, beta), x[v.SIGNAL_IX]*r.prod())
        self.assertEqual(len(v.mixed_sums(x, beta)), 2)

    def test_original_episode_denominators_not_selected_queries(self):
        source = {('a',): {'key': np.array([[2., -4.]])}, ('b',): {'key': np.array([[3., 6.]])}}
        base = {('a',): [4, 3], ('b',): [1, 1]}
        got = v.saved.collapse(source, base)['key']
        np.testing.assert_allclose(got, [[1.75, 2.5]])
        ref = [dict(episode=['a'], eligible_queries=4, finite_interval_queries=3),
            dict(episode=['b'], eligible_queries=1, finite_interval_queries=1)]
        v.same_denominators(base, ref)
        ref[0]['eligible_queries'] = 2
        with self.assertRaisesRegex(AssertionError, 'denominators differ'):
            v.same_denominators(base, ref)

    def test_per_bin_check_detects_equal_parent_reassignment(self):
        m, f = np.zeros((9, v.saved.SIZE)), np.zeros((9, 1+len(v.saved.FACTS)))
        expected = m.copy()
        m[1, 0], expected[0, 0] = 1., 1.
        self.assertEqual(m.sum(), expected.sum())
        with self.assertRaisesRegex(AssertionError, 'per-bin'):
            v.verify_cells(m, f, expected, f)
        np.testing.assert_array_equal(v.verify_cells(expected, f, expected, f), np.zeros(4))

    def test_complete_query_support_at_19_in_each_half(self):
        counts = np.zeros((64, 2, 9), dtype=int)
        counts[:19, 0, 0] = counts[32:51, 0, 0] = 2
        counts[:18, 0, 1] = counts[32:51, 0, 1] = 4
        days = np.stack([(p > 0).sum(axis=0) for p in (counts[:32], counts[32:])])
        keep = np.all(days >= 19, axis=0)
        v.verify_support(counts, days, keep)
        self.assertTrue(keep[0, 0])
        self.assertFalse(keep[0, 1])
        counts[18, 0, 1] = 1
        with self.assertRaisesRegex(AssertionError, 'support changed'):
            v.verify_support(counts, days, keep)

    def test_strict_feature_and_nonnegative_label_clocks(self):
        q, r = dict(query_time_s=100.), dict(label=dict(available_time_s=110.))
        item = [None, None, dict(n=1, latest=99.), dict(latest=98.)]
        service = dict(n=9, latest=97.)
        np.testing.assert_array_equal(v.query_clocks(q, r, item, service, [0., 0., 0.]), [2., 1., 3., 0., 10.])
        service['latest'] = 100.
        with self.assertRaisesRegex(AssertionError, 'clock violation'):
            v.query_clocks(q, r, item, service, [0., 0., 0.])
        service['latest'], service['n'] = 97., 8
        with self.assertRaisesRegex(AssertionError, 'support changed'):
            v.query_clocks(q, r, item, service, [0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
