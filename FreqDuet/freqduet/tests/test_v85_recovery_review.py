import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v85_recovery_review import paired_contrasts, count_stats


class RecoveryReviewTest(unittest.TestCase):
    def test_equal_response_does_not_mean_equal_regime_cost(self):
        base = np.arange(8.)[:, None] + np.array([[0., 3.]])
        result = paired_contrasts(dict(closed_loop=base, recorded_lower=base - 5., recover_900s=base + 10.))
        self.assertEqual(result["response_difference"]["closed_loop"]["mean"], 0.)
        self.assertEqual(result["fixed_action"]["closed_loop"]["0"]["mean"], 10.)
        self.assertEqual(result["fixed_action"]["recorded_lower"]["45"]["mean"], 15.)

    def test_paired_future_noise_cancels_before_interval(self):
        base = np.arange(8.)[:, None] * 100. + np.array([[0., 1.]])
        result = paired_contrasts(dict(closed_loop=base, recorded_lower=base, recover_900s=base + [[2., 6.]]))
        self.assertEqual(result["response_difference"]["closed_loop"]["ci95"], [4., 4.])
        counts = count_stats([result["response_difference"]["closed_loop"]])
        self.assertEqual(counts["ci_positive"], 1)
        json.dumps(counts)


if __name__ == "__main__":
    unittest.main()
