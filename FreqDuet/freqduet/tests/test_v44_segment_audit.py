from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_closed_loop_forecast import fluid_rollout
from test_v41_closed_loop import observation
from run_freqduet_v44_segment_audit import traced_event


class SegmentTraceTest(unittest.TestCase):
    def test_real_recorder_positional_and_keyword_interfaces_are_preserved(self):
        calls, rows = [], []

        def arrival(station_id, direction, arrival_time_s, trip_id, target_headway_s=None):
            calls.append((station_id, direction, arrival_time_s, trip_id, target_headway_s))
            return 123

        wrapped = traced_event(arrival, "arrival", lambda *args: rows.append(args))
        self.assertEqual(wrapped(station_id=3, direction=True, arrival_time_s=10, trip_id=2, target_headway_s=360), 123)
        wrapped(3, True, 20, 4, 360)
        self.assertEqual(rows, [("arrival", 3, True, 10, 2), ("arrival", 3, True, 20, 4)])
        self.assertEqual(len(calls), 2)

    def test_trace_does_not_change_physics(self):
        observed, rows = observation(), []
        expected = fluid_rollout(observed, 100, "response")
        actual = fluid_rollout(observed, 100, "response", event_sink=rows.append)
        self.assertEqual(expected, actual)
        self.assertEqual([row["stage"] for row in rows], ["arrival", "ready", "action", "departure", "arrival"])
        self.assertEqual(rows[2]["action_s"], 45)

    def test_service_and_action_ticks_are_distinguished(self):
        rows = []
        fluid_rollout(observation(), 100, "response", event_sink=rows.append)
        ready = next(row["time_s"] for row in rows if row["stage"] == "ready")
        action = next(row["time_s"] for row in rows if row["stage"] == "action")
        self.assertEqual(action - ready, 1)


if __name__ == "__main__":
    unittest.main()
