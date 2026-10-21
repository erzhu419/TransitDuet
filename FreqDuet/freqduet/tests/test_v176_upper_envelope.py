import ast
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import seal_freqduet_v176_upper_envelope as stage


def reference(total, days, denominator, counts, rank, order):
    sort = np.argsort(total, kind='stable')
    quantiles = []
    for draw in counts:
        mass = Fraction()
        for i in sort:
            mass += Fraction(int(draw[days[i]]), int(denominator[i]))
            if mass >= rank:
                quantiles.append(float(total[i]))
                break
    return sorted(quantiles)[order-1]


class UpperEnvelopeTests(unittest.TestCase):
    def test_fixed_day_slots_keep_duplicate_multiplicity(self):
        counts = stage.day_counts(19)
        draws = np.random.RandomState(20260803).randint(0, 19, size=(5000, 19))
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(5000, 19))
        for i in (0, 1, 249, 4999):
            np.testing.assert_array_equal(counts[i], np.bincount(draws[i], minlength=19))
        self.assertTrue((counts[0] == 0).any())
        self.assertTrue((counts[0] > 1).any())

    def test_weighted_inverse_matches_fraction_reference_at_exact_ties(self):
        days = np.array([0, 0, 0, 1, 1, 2])
        den = np.array([4, 4, 2, 2, 2, 1])
        total = np.array([-5., 0., 3., -2., 4., 10.])
        counts = np.array([[3, 0, 0], [0, 3, 0], [1, 1, 1], [0, 0, 3], [2, 0, 1]])
        for order in range(1, 6):
            actual, proof = stage.block_order_statistic(total, days, den, counts, 1, order)
            self.assertEqual(actual, reference(total, days, den, counts, 1, order))
            self.assertLess(proof['draws_strictly_below'], order)
            self.assertGreaterEqual(proof['draws_at_or_below'], order)

    def test_large_common_denominator_uses_exact_integer_limbs(self):
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67])
        days = np.repeat(np.arange(19), primes)
        den = np.repeat(primes, primes)
        total = np.concatenate([np.arange(p) % 7-d for d, p in enumerate(primes)]).astype(float)
        draws = np.random.RandomState(7).randint(0, 19, size=(31, 19))
        counts = np.array([np.bincount(row, minlength=19) for row in draws])
        actual, proof = stage.block_order_statistic(total, days, den, counts, 2, 7)
        self.assertTrue(proof['limb_arithmetic'])
        self.assertEqual(actual, reference(total, days, den, counts, 2, 7))

    def test_nested_inverse_is_exactly_the_250th_of_5000_draw_quantiles(self):
        counts = stage.day_counts(19)
        total = np.arange(19, dtype=float)
        actual, proof = stage.block_order_statistic(total, np.arange(19), np.ones(19, dtype=int), counts, 2, 250)
        each = np.argmax(counts.cumsum(axis=1) >= 2, axis=1)
        self.assertEqual(actual, float(np.sort(each)[249]))
        self.assertEqual(proof['original_qlow_s'], 1.)

    def test_zero_weight_plateaus_and_duplicate_values_do_not_interpolate(self):
        total = np.array([-10., -10., 3., 20.])
        days, den = np.array([0, 0, 1, 2]), np.array([2, 2, 1, 1])
        counts = np.array([[0, 0, 3], [0, 0, 3]])
        actual, _ = stage.block_order_statistic(total, days, den, counts, 1, 1)
        self.assertEqual(actual, 20.)

    def test_envelope_reproduces_parent_and_only_changes_qlow(self):
        days, episodes = np.arange(19), np.arange(19)
        vectors = np.column_stack((np.arange(19)-9., np.zeros((19, 2))))
        entry, saved_den = stage.fit.fit_key(vectors, days, episodes)
        before = deepcopy(entry)
        result, proof = stage.envelope(entry, vectors, days, episodes, saved_den)
        self.assertEqual(entry, before)
        self.assertEqual(result, dict(before, qlow_s=result['qlow_s']))
        self.assertLessEqual(result['qlow_s'], before['qlow_s'])
        self.assertEqual(proof['original_qlow_s'], before['qlow_s'])
        self.assertEqual(result['day_count'], 19)
        with self.assertRaises(AssertionError):
            stage.envelope(dict(entry, status='abstain_less_than_19_days'), vectors, days, episodes, saved_den)
        self.assertIsNone(stage.fit.interval(100., None))

    def test_lower_non_target_and_abstention_invariants_detect_real_changes(self):
        old = np.array([[90., 110.], [80., 120.], [np.nan, np.nan], [0., 0.]])
        finite, target = np.array([True, True, False, True]), np.array([True, False, True, False])
        updated = old.copy()
        updated[0, 1] += 5.
        result = stage.invariants(old, updated, finite, target)
        self.assertEqual(result['expanded_queries'], 1)
        self.assertEqual(result['maximum_upper_expansion_s'], 5.)
        for row, column, value in ((0, 1, 109.), (0, 0, 89.), (1, 1, 121.), (2, 0, 0.)):
            wrong = updated.copy()
            wrong[row, column] = value
            if row == 2:
                wrong[row, 1] = 10.
            with self.assertRaises(AssertionError):
                stage.invariants(old, wrong, finite, target)

    def test_no_confirmation_collection_training_or_adaptive_parameters(self):
        tree = ast.parse(Path(stage.__file__).read_text())
        calls = {(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id)
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'collect', 'rollout', 'train', 'attach_labels', 'percentile', 'quantile'}))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ('v173_root', 'confirmation_root') for n in ast.walk(tree)))
        self.assertEqual((stage.DRAWS, stage.SEED, stage.ORDER), (5000, 20260803, 250))


if __name__ == '__main__':
    unittest.main()
