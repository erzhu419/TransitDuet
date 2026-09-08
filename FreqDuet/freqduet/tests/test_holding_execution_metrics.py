import unittest
from types import SimpleNamespace

import numpy as np

from env.bus import Bus, BusState
from env.evaluation import HeadwayEventRecorder


class HoldingExecutionMetricsTest(unittest.TestCase):
    def test_final_bus_action_resolves_prior_follower_forecast(self):
        recorder = HeadwayEventRecorder()
        recorder.record_follower_departure_forecast(
            station_id=3,
            direction=True,
            decision_time_s=100.0,
            current_trip_id=10,
            follower_trip_id=12,
            predicted_follower_gap_s=220.0,
            forward_departure_gap_s=120.0,
            action_s=20.0,
            action_cap_s=45.0,
            source="same_time_avl_journey_speed_eta",
        )
        recorder.record_departure(3, True, 121.0, trip_id=10)
        recorder.record_action_ready(3, True, 280.0, trip_id=12)

        bus = Bus.__new__(Bus)
        bus.trip_id = 12
        bus.passengers = np.arange(4)
        bus.holding_action_trace_mode = "all_decisions"
        bus.applied_actions = []
        bus.applied_action_loads = []
        bus.episode_commanded_hold_vehicle_seconds = 0.0
        bus.episode_commanded_hold_person_seconds = 0.0
        bus.last_station = SimpleNamespace(station_id=3)
        bus.direction = True
        bus._headway_recorder = recorder

        bus._start_dwelling(np.asarray([30.0]), current_time=281.0)

        summary = recorder.summary()
        self.assertEqual(
            summary["follower_forecast_action_resolved_count"], 1)
        self.assertEqual(
            summary["follower_forecast_follower_future_hold_s_mean"], 30.0)

    def test_pre_action_gap_requires_the_immediate_predecessor_departure(self):
        recorder = HeadwayEventRecorder()
        recorder.record(3, True, 100.0, trip_id=10)
        recorder.record_departure(3, True, 130.0, trip_id=10)
        bus = Bus.__new__(Bus)
        bus._headway_recorder = recorder
        bus.last_station = SimpleNamespace(station_id=3)
        bus.direction = True
        bus.forward_predecessor_trip_id = 10

        bus._update_pre_action_forward_headway(350.0)

        self.assertEqual(
            bus.pre_action_forward_headway_source,
            "matched_departure_event",
        )
        self.assertEqual(bus.pre_action_forward_headway, 220.0)

        recorder.record_departure(3, True, 340.0, trip_id=11)
        bus._update_pre_action_forward_headway(350.0)
        self.assertEqual(
            bus.pre_action_forward_headway_source,
            "predecessor_not_departed",
        )
        self.assertIsNone(bus.pre_action_forward_headway)

    def test_realized_holding_stops_at_the_evaluation_horizon(self):
        bus = Bus.__new__(Bus)
        bus.trip_id = 2
        bus.passengers = np.arange(10)
        bus.holding_action_trace_mode = "all_decisions"
        bus.applied_actions = []
        bus.applied_action_loads = []
        bus.episode_hold_vehicle_seconds = 0.0
        bus.episode_hold_person_seconds = 0.0
        bus.episode_commanded_hold_vehicle_seconds = 0.0
        bus.episode_commanded_hold_person_seconds = 0.0
        bus.last_station = SimpleNamespace(station_id=3)
        bus.direction = True
        bus._headway_recorder = None
        bus._stop_start_time = None
        bus._stop_station = None
        bus.state = BusState.WAITING_ACTION

        bus._start_dwelling(np.asarray([45.0]), current_time=100.0)
        for second in range(101, 111):
            bus._process_dwelling(float(second))

        self.assertEqual(bus.episode_commanded_hold_vehicle_seconds, 45.0)
        self.assertEqual(bus.episode_commanded_hold_person_seconds, 450.0)
        self.assertEqual(bus.episode_hold_vehicle_seconds, 10.0)
        self.assertEqual(bus.episode_hold_person_seconds, 100.0)
        self.assertEqual(bus.dwelling_time, 35.0)


if __name__ == "__main__":
    unittest.main()
