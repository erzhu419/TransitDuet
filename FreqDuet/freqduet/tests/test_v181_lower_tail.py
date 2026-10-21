from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import preregister_freqduet_v181_lower_tail as plan


class LowerTailPlanTests(unittest.TestCase):
    def test_sign_is_service_only_and_zero_is_nonpositive(self):
        self.assertEqual(plan.service_sign(dict(n=9, mean=np.array([100., 0., -200.]))), 0)
        self.assertEqual(plan.service_sign(dict(n=9, mean=np.array([-100., 1., 0.]))), 1)
        self.assertEqual(plan.service_sign(dict(n=8, mean=np.ones(3))), -1)
        self.assertEqual(plan.service_sign(dict(n=9, mean=None)), -1)

    def test_support_counts_distinct_days_not_queries(self):
        days = np.r_[np.zeros(100, dtype=int), np.arange(1, 19)]
        codes = np.r_[np.ones(100, dtype=int), np.zeros(18, dtype=int)]
        finite, counts = plan.support_mask(np.ones(len(days), dtype=bool), np.ones(len(days), dtype=bool), codes, days)
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[0], 18)
        self.assertFalse(finite.any())
        finite, counts = plan.support_mask(np.ones(19, dtype=bool), np.ones(19, dtype=bool), np.zeros(19, dtype=int), np.arange(19))
        self.assertEqual(counts[0], 19)
        self.assertTrue(finite.all())

    def test_missing_prefix_abstains_without_fallback(self):
        finite, counts = plan.support_mask(np.ones(2, dtype=bool), np.array([True, False]), np.array([-1, -1]), np.array([1, 2]))
        np.testing.assert_array_equal(finite, [False, True])
        self.assertEqual(counts, {})

    def test_baseline_abstentions_are_never_rescued(self):
        baseline = np.zeros(20, dtype=bool)
        finite, counts = plan.support_mask(baseline, np.ones(20, dtype=bool), np.zeros(20, dtype=int), np.arange(20))
        self.assertFalse(finite.any())
        self.assertEqual(counts, {})

    def test_target_union_does_not_duplicate_overlap(self):
        np.testing.assert_array_equal(plan.target_mask(np.array([4, 4, 1, 1]), np.array([False, True, True, False])),
            [True, True, True, False])

    def test_release_cannot_drop_a_failed_scope(self):
        rows = [dict(actor=1, day=1, mode=m, scope=s, eligible_episodes=1, finite_episodes=1,
            finite_fraction=.9, baseline_fraction=.9, finite_queries=9, native_abstentions=1, key_abstentions=0)
            for m in ('old', 'handoff') for s in plan.temporal.SCOPES]
        self.assertEqual(plan.temporal.support_release(rows, [1], [1])['decision'], 'pass')
        rows[-1]['finite_fraction'] = .79
        self.assertEqual(plan.temporal.support_release(rows, [1], [1])['decision'], 'insufficient_support')


if __name__ == '__main__':
    unittest.main()
