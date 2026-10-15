from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_return_uncertainty import ReturnIntervals, lead_band


class ReturnIntervalTest(unittest.TestCase):
    def history(self):
        intervals = ReturnIntervals()
        for trip in range(20):
            intervals.issue(trip * 100, trip, trip * 100 + 50)
            intervals.resolve(trip, trip * 100 + 60, trip * 100 + 61)
        return intervals

    def test_only_completed_trips_can_calibrate_a_prediction(self):
        intervals = self.history()
        before = intervals.interval(3000, 3100)
        intervals.issue(3000, 99, 3100)
        self.assertEqual(before, intervals.interval(3000, 3100))
        self.assertEqual(before["radius_s"], 10)
        self.assertLess(before["latest_completion_s"], 3000)

    def test_warmup_and_predictive_lead_bands_are_explicit(self):
        self.assertFalse(ReturnIntervals().interval(0, 100)["available"])
        self.assertEqual([lead_band(value) for value in (900, 901, 1800, 1801)],
                         ["short", "medium", "medium", "long"])

    def test_a_trip_supplies_only_one_residual_per_band(self):
        intervals = ReturnIntervals()
        intervals.issue(0, 1, 100)
        intervals.issue(50, 1, 100)
        intervals.resolve(1, 110, 111)
        self.assertEqual(len(intervals.rows), 1)
        self.assertEqual(len(intervals.completed["short"]), 1)

    def test_future_and_simultaneous_labels_are_rejected(self):
        intervals = ReturnIntervals()
        intervals.issue(0, 1, 100)
        for actual in (100, 101):
            with self.assertRaisesRegex(ValueError, "strictly earlier"):
                intervals.resolve(1, actual, 100)

    def test_coverage_is_scored_without_changing_the_issued_interval(self):
        intervals = self.history()
        intervals.issue(3000, 99, 3100)
        row = intervals.rows[-1]
        intervals.resolve(99, 3120, 3121)
        self.assertEqual(row["upper_s"], 3110)
        self.assertFalse(row["covered"])
        self.assertEqual(row["interval_score_s"], 220)


if __name__ == "__main__":
    unittest.main()
