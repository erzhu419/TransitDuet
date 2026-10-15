from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_completed_window_calibration import CompletedWindows
from run_freqduet_v45_completed_calibration import observable_draws


class CompletedWindowTest(unittest.TestCase):
    def test_latent_waiting_label_is_not_a_calibration_channel(self):
        draws = dict(waiting_person_seconds=[999], onboard_person_seconds=[10],
                     backlog_trip_seconds=[20], holding_vehicle_seconds=[30])
        calibration = CompletedWindows()
        calibration.issue(0, 10, observable_draws(draws))
        self.assertNotIn("waiting_person_seconds", calibration.pending[0]["metrics"])
        calibration.release(11, lambda *args: dict(onboard_person_seconds=15,
            backlog_trip_seconds=25, holding_vehicle_seconds=35))
        self.assertEqual(len(calibration.history), 3)

    def test_partial_and_simultaneously_completed_windows_do_not_read_labels(self):
        calibration = CompletedWindows()
        calibration.issue(0, 10, {"hold": [10, 10]})

        def forbidden(*args):
            self.fail("premature future-label access")

        calibration.release(9, forbidden)
        calibration.release(10, forbidden)
        self.assertEqual(calibration.admitted, 0)
        calibration.release(11, lambda *args: {"hold": 15})
        self.assertEqual(calibration.admitted, 1)

    def test_overlapping_windows_are_not_calibration_samples(self):
        calibration = CompletedWindows()
        for origin in (0, 10, 20, 30):
            calibration.issue(origin, 20, {"hold": [10, 10]})
        calibration.release(51, lambda *args: {"hold": 15})
        self.assertEqual(calibration.admitted, 2)
        self.assertEqual(calibration.overlapping, 2)

    def test_warmup_then_bias_correction_uses_only_completed_history(self):
        calibration = CompletedWindows()
        for origin in range(0, 90, 10):
            calibration.release(origin, lambda *args: {"hold": 15})
            issued = calibration.issue(origin, 10, {"hold": [10, 10]})
            self.assertFalse(issued["hold"]["mature"])
        calibration.release(90, lambda *args: {"hold": 15})
        issued = calibration.issue(90, 10, {"hold": [10, 10]})["hold"]
        self.assertTrue(issued["mature"])
        self.assertEqual(issued["point"], 15)
        self.assertEqual(issued["latest_completion"], 80)
        self.assertEqual(issued["radius"], 5)

    def test_scores_keep_the_bias_available_at_issue_not_refitted_bias(self):
        calibration = CompletedWindows(min_windows=1)
        calibration.issue(0, 10, {"hold": [10]})
        calibration.release(11, lambda *args: {"hold": 110})
        issued = calibration.issue(20, 10, {"hold": [10]})["hold"]
        self.assertEqual(issued["bias"], 100)
        calibration.release(31, lambda *args: {"hold": 110})
        rows = calibration.history[(10, "hold")]
        self.assertEqual([row["score"] for row in rows], [100, 0])
        self.assertEqual(issued["radius"], 100)

    def test_history_is_bounded_and_does_not_cross_days(self):
        calibration = CompletedWindows()
        for origin in range(0, 300, 10):
            calibration.release(origin, lambda *args: {"hold": 15})
            calibration.issue(origin, 10, {"hold": [10]})
        self.assertEqual(len(calibration.history[(10, "hold")]), 16)
        fresh = CompletedWindows().issue(400, 10, {"hold": [10]})["hold"]
        self.assertEqual(fresh["samples"], 0)
        self.assertIsNone(fresh["low"])


if __name__ == "__main__":
    unittest.main()
