from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_freqduet_v103_held_future_teacher import axes, review, teacher_law


class LocalTeacherTests(unittest.TestCase):
    def test_held_labels_do_not_affect_the_training_law(self):
        reward = np.tile(np.arange(7, dtype=float), (4, 1))
        cost = reward / 10
        law, _, _ = teacher_law(reward, cost, np.zeros(7), .1, 1., (0, 1))
        reward[2:] *= -100
        cost[2:] += 1000
        other, _, _ = teacher_law(reward, cost, np.zeros(7), .1, 1., (0, 1))
        np.testing.assert_array_equal(law, other)

    def test_exact_repeated_curve_realizes_nonnegative_local_gain(self):
        reward = np.tile(np.arange(7, dtype=float) / 10, (4, 1))
        cost, p = reward / 2, np.full(7, 1 / 7)
        charge = np.linspace(0., .2, 7)
        b, q, cq = teacher_law(reward, cost, charge, .1, .5, (0, 1))
        result = review.decompose(p, b, q, q - charge, cq, reward[2], cost[2], .1, .5)
        self.assertGreater(result['frozen_pessimism_soft_objective'], 0)
        self.assertAlmostEqual(result['predicted_frozen_objective'], result['frozen_pessimism_soft_objective'])

    def test_local_gain_does_not_imply_physical_nonharm(self):
        reward = np.tile(np.arange(7, dtype=float), (4, 1))
        p = np.full(7, 1 / 7)
        b, q, cq = teacher_law(reward, reward * 0, np.zeros(7), .1, 0., (0, 1))
        result = review.decompose(p, b, q, q, cq, reward[2], cq, .1, 0.)
        self.assertGreater(result['native_soft_objective'], 0)
        self.assertGreater(float((b - p) @ reward[2]), 0)

    def test_axis_counts_are_cluster_counts_not_future_counts(self):
        rows = [dict(seed=1, day=2, target=3, effects=dict(mc_vs_native=dict(error=1.))) for _ in range(8)]
        result = axes(rows, ('error',))
        self.assertEqual(result['seed']['mc_vs_native']['positive_counts']['error'], 1)
        self.assertEqual(result['day']['mc_vs_native']['positive_counts']['error'], 1)


if __name__ == '__main__':
    unittest.main()
