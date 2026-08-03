import unittest
from collections import deque

from coupling.holding_feedback import HoldingFeedback
from runner_v3 import TransitDuetV2Runner


class LowerDriftSignalTest(unittest.TestCase):
    @staticmethod
    def runner(mode):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.lower_drift_signal_mode = mode
        runner.lower_drift_window = 2
        runner.lower_drift_budget_s = 12.0
        runner.freq_driftfb_norm_s = 30.0
        runner.freq_driftfb_clip = 2.0
        runner.holding_feedback = HoldingFeedback(window_size=4)
        runner._lower_drift_by_dir = {
            True: deque(maxlen=2),
            False: deque(maxlen=2),
        }
        return runner

    def test_trip_cumulative_load_does_not_mix_buses(self):
        runner = self.runner("trip_cumulative")
        runner.holding_feedback.record_action(4, 10.0)
        runner.holding_feedback.record_action(8, 40.0)

        self.assertEqual(
            runner._lower_drift_load(True, 5.0, 4), 15.0)
        self.assertEqual(
            runner._lower_drift_load(True, 2.0, 8), 42.0)

    def test_finalized_trip_action_is_not_double_counted(self):
        runner = self.runner("trip_cumulative")
        runner.holding_feedback.finalize_trip(
            4, True, actions=[10.0, 5.0])

        load = runner._lower_drift_load(
            True, 5.0, 4, action_already_recorded=True)

        self.assertEqual(load, 15.0)
        drift, excess = runner._drift_feedback_pair(True)
        self.assertAlmostEqual(drift, 0.5)
        self.assertAlmostEqual(excess, 0.1)

    def test_legacy_mode_retains_direction_action_window(self):
        runner = self.runner("rolling_action_window")

        runner._lower_drift_load(True, 5.0, 4)
        runner._lower_drift_load(True, 7.0, 8)
        load = runner._lower_drift_load(True, 11.0, 10)

        self.assertEqual(load, 18.0)
        self.assertEqual(list(runner._lower_drift_by_dir[True]), [7.0, 11.0])


if __name__ == "__main__":
    unittest.main()
