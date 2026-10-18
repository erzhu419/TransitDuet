from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_freqduet_v105_frozen_future_replication as run


class FrozenFutureTests(unittest.TestCase):
    def test_registered_futures_are_new_and_split_without_overlap(self):
        new = {run.future_seed(c, f) for c in range(16) for f in run.FUTURES}
        old = {204000000 + c * 100 + f for c in range(16) for f in range(4)}
        self.assertEqual(len(new), 512)
        self.assertFalse(new & old)
        self.assertFalse(set(run.COHORTS['first16']) & set(run.COHORTS['second16']))
        self.assertEqual(len(run.ITEMS), 3584)

    def test_new_labels_never_reselect_or_refit_frozen_laws(self):
        p = np.full(7, 1 / 7)
        one = np.eye(7)[0]
        contexts = [dict(p=p, laws={k: p.copy() if k == 'native' else one.copy() for k in run.METHODS})]
        y = np.arange(7, dtype=float)[None, None, :, None] * np.ones((1, 32, 7, len(run.METRICS)))
        with (patch.object(run.model, 'fit', side_effect=AssertionError('refit')),
              patch.object(run.model, 'proposal', side_effect=AssertionError('reselect'))):
            a, b = run.evaluate(contexts, y), run.evaluate(contexts, -y)
        np.testing.assert_array_equal(a, -b)
        np.testing.assert_array_equal(a[:, :, 0], 0)
        self.assertTrue(np.all(a[:, :, 1] < 0))
        np.testing.assert_array_equal(contexts[0]['laws']['separated_a'], one)

    def test_teacher_mix_is_fixed_equal_mixture(self):
        p, a, b = np.full(7, 1 / 7), np.eye(7)[0], np.eye(7)[6]
        costs = np.arange(21, dtype=float).reshape(7, 3)
        np.testing.assert_allclose(((a + b) / 2 - p) @ costs,
                                    ((a - p) @ costs + (b - p) @ costs) / 2, atol=1e-14)

    def test_future_uncertainty_does_not_resample_prefix_population(self):
        y = np.zeros((3, 32, 2, 1))
        y[:, :, :, 0] = np.arange(3)[:, None, None]
        bounds = run.future_bounds(y, draws=100)
        np.testing.assert_array_equal(bounds, 1.)
        rng = np.random.default_rng(5)
        y = rng.normal(size=(3, 32, 1, 1))
        identical = np.repeat(y, 2, axis=2)
        bounds = run.future_bounds(identical, draws=100)
        np.testing.assert_array_equal(bounds[:, 0], bounds[:, 1])

    def test_cross_split_product_separates_repeatable_from_reversing_curves(self):
        y = np.tile(np.arange(7, dtype=float)[None, :, None], (32, 1, 3))
        same = run.contrast_metrics(y[:4], y)
        self.assertTrue(np.all(same['independent_split_cross_product'] > 0))
        np.testing.assert_array_equal(same['split_difference_mse'], 0)
        np.testing.assert_array_equal(same['pooled_mean_noise'], 0)
        y[16:] *= -1
        reversed_curve = run.contrast_metrics(y[:4], y)
        self.assertTrue(np.all(reversed_curve['independent_split_cross_product'] < 0))


if __name__ == '__main__':
    unittest.main()
