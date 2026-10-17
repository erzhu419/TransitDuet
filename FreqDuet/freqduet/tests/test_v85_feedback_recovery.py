from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v85_feedback_recovery import ITEMS, RECOVERY_S, release_due, signal_counts


class FeedbackRecoveryTest(unittest.TestCase):
    def test_grid_keeps_all_prefixes_and_futures(self):
        self.assertEqual(len(ITEMS), 24 * 8 * 2)
        self.assertEqual(set(a for _, a in ITEMS), {0, 6})

    def test_release_uses_elapsed_observed_time_only(self):
        self.assertFalse(release_due(False, 9000., 7200.))
        self.assertFalse(release_due(True, 7200. + RECOVERY_S - 1., 7200.))
        self.assertTrue(release_due(True, 7200. + RECOVERY_S, 7200.))

    def test_signal_counts_exclude_native_and_remain_json_serializable(self):
        context = dict(native_index=3, effects={"closed_loop": {"projected_vs_base": {
            "cv_squared": dict(mean=-2., future_sd=1., ci95=[-3., -1.])}}},
            original_v83_effects={"closed_loop": {"projected_vs_base": {"cv_squared": 2.}}},
            action_noise={"closed_loop": {"cv_squared": dict(mean_deltas=[2.] * 7,
                paired_sd=[1.] * 7, independent_sd=[4.] * 7)}})
        result = signal_counts([context], "closed_loop", "cv_squared")
        self.assertEqual(result["action_mean_above_paired_sd"], 6)
        self.assertEqual(result["old_mean_sign_repeated"], 0)
        self.assertEqual(result["paired_to_independent_sd_median"], .25)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
