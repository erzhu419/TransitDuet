import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_freqduet_v101_action_differences import curve_audit, decompose, rank_correlation


class CachedAttributionTests(unittest.TestCase):
    def test_reward_and_cost_errors_reconstruct_gain_error(self):
        p, b = np.full(7, 1 / 7), np.arange(1, 8) / 28
        q, cq = np.arange(7) / 5, np.arange(7) / 7
        r = decompose(p, b, q, q - .1, cq, q + .3, cq + .2, .1, .5)
        self.assertAlmostEqual(r['reward_prediction_error'], 0.)
        self.assertAlmostEqual(r['cost_term_prediction_error'], 0.)

    def test_fixed_action_curve_cancels_common_future_noise(self):
        x = np.arange(7, dtype=float)
        result = curve_audit(x, x[None, :] + np.asarray([0., 10., -20., 4.])[:, None])
        self.assertAlmostEqual(result['difference_mae'], 0.)
        self.assertAlmostEqual(result['label_mean_variance'], 0.)
        self.assertEqual(result['such_pairs_with_correct_predicted_sign'], 21)
        self.assertAlmostEqual(result['split_half_rank_correlation'], 1.)

    def test_flat_cost_is_not_reported_as_a_rank_failure(self):
        self.assertIsNone(rank_correlation(np.zeros(7), np.arange(7)))

    def test_float32_network_outputs_keep_double_precision_gain_identity(self):
        p, b = np.full(7, 1 / 7), np.arange(1, 8) / 28
        q, cq = (np.arange(7) / 7).astype(np.float32), (np.arange(7) / 13).astype(np.float32)
        qp = q.astype(float) - .123
        actual = decompose(p, b, q, qp, cq, q + .3, cq + .2, .1, .34567)
        expected = decompose(p, b, q.astype(float), qp, cq.astype(float),
                             (q + .3).astype(float), (cq + .2).astype(float), .1, .34567)
        for key in actual:
            self.assertAlmostEqual(actual[key], expected[key], places=12)


if __name__ == '__main__':
    unittest.main()
