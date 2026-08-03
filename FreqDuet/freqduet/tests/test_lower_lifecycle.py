import unittest
from types import SimpleNamespace

import numpy as np

from coupling.holding_feedback import HoldingFeedback
from env.bus import Bus, BusState
from env.evaluation import HeadwayEventRecorder
from lower.lifecycle import LowerEpisodeLifecycle


class LowerLifecycleTest(unittest.TestCase):
    def completed_bus(self):
        return SimpleNamespace(
            bus_id=3,
            trip_id=18,
            direction=False,
            on_route=False,
            last_completed_trip_id=18,
            last_completed_direction=True,
            last_completed_reward=-0.25,
            last_completed_cost=0.125,
            last_completed_station_id=22,
            last_completed_target_headway=420.0,
            last_completed_board_wait_sum_s=900.0,
            last_completed_board_count=3,
            forward_headway=390.0,
            backward_headway=450.0,
            applied_actions=[5.0, 15.0, 25.0],
        )

    def test_trip_end_closes_state_and_finalizes_feedback_once(self):
        feedback = HoldingFeedback(window_size=10)
        feedback.record_action(18, 10.0)
        feedback.record_action(18, 20.0)
        lifecycle = LowerEpisodeLifecycle("reset", "trip_end")
        states = {3: [[3.0, 20.0]]}
        actions = {3: 15.0}
        last_actions = {3: 10.0}

        events = lifecycle.process(
            [self.completed_bus()], states, actions, last_actions, feedback
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].pending_states_dropped, 1)
        self.assertTrue(events[0].pending_action_dropped)
        self.assertTrue(events[0].feedback_finalized)
        self.assertEqual(events[0].pending_state, [3.0, 20.0])
        self.assertEqual(events[0].pending_action, 15.0)
        self.assertEqual(events[0].previous_action_s, 10.0)
        self.assertEqual(events[0].terminal_reward, -0.25)
        self.assertEqual(events[0].terminal_cost, 0.125)
        self.assertEqual(events[0].last_board_station_id, 22)
        self.assertEqual(events[0].last_board_count, 3)
        self.assertEqual(events[0].target_headway, 420.0)
        self.assertEqual(states[3], [])
        self.assertIsNone(actions[3])
        self.assertEqual(last_actions[3], 0.0)
        self.assertEqual(feedback.finalized_trip_count, 1)
        self.assertEqual(feedback.get_direction_stats(True)["n_trips"], 1)
        self.assertAlmostEqual(
            feedback.get_direction_stats(True)["rolling_mean"], 15.0)
        self.assertEqual(feedback.get_direction_stats(False)["n_trips"], 0)

        duplicate = lifecycle.process(
            [self.completed_bus()], states, actions, last_actions, feedback
        )
        self.assertEqual(duplicate, [])
        self.assertEqual(feedback.get_direction_stats(True)["n_trips"], 1)

    def test_legacy_modes_preserve_pending_agent_state(self):
        feedback = HoldingFeedback(window_size=10)
        feedback.record_action(18, 10.0)
        lifecycle = LowerEpisodeLifecycle("legacy", "episode_end")
        states = {3: [[3.0, 20.0]]}
        actions = {3: 15.0}
        last_actions = {3: 10.0}

        events = lifecycle.process(
            [self.completed_bus()], states, actions, last_actions, feedback
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(states[3], [[3.0, 20.0]])
        self.assertEqual(actions[3], 15.0)
        self.assertEqual(last_actions[3], 10.0)
        self.assertEqual(feedback.finalized_trip_count, 0)

    def test_holding_feedback_finalization_is_idempotent(self):
        feedback = HoldingFeedback(window_size=10)
        feedback.record_action(4, 12.0)
        self.assertTrue(feedback.finalize_trip(4, False))
        self.assertFalse(feedback.finalize_trip(4, False))
        self.assertEqual(feedback.get_direction_stats(False)["n_trips"], 1)
        feedback.clear()
        self.assertEqual(feedback.finalized_trip_count, 0)

    def test_episode_local_clear_removes_historical_state(self):
        feedback = HoldingFeedback(window_size=10)
        feedback.finalize_trip(4, False, actions=[12.0])
        feedback.clear(reset_history=True)
        self.assertEqual(feedback.get_direction_stats(False)["n_trips"], 0)
        self.assertEqual(feedback.get_direction_stats(False)["ema"], 0.0)

    def test_completed_trip_totals_are_physical_and_causal(self):
        feedback = HoldingFeedback(window_size=10)
        feedback.finalize_trip(4, True, actions=[5.0, 0.0, 15.0])
        feedback.finalize_trip(6, True, actions=[10.0, 20.0])

        self.assertEqual(feedback.get_trip_total(4), 20.0)
        direction = feedback.get_direction_stats(True)
        self.assertEqual(direction["n_trips"], 2)
        self.assertEqual(direction["rolling_total_mean"], 25.0)
        totals = feedback.get_direction_total_stats(True, budget_s=20.0)
        self.assertEqual(totals["rolling_mean"], 25.0)
        self.assertEqual(totals["mean_excess"], 5.0)
        self.assertEqual(totals["max"], 30.0)

        episode = feedback.episode_summary
        self.assertEqual(episode["total_holding_s"], 50.0)
        self.assertEqual(episode["trip_total_mean"], 25.0)
        self.assertEqual(episode["trip_total_max"], 30.0)

    def test_all_decisions_trace_records_zero_holding(self):
        bus = Bus.__new__(Bus)
        bus.trip_id = 4
        bus.applied_actions = []
        bus.holding_action_trace_mode = "all_decisions"
        bus.last_station = SimpleNamespace(station_id=7)
        bus.state = BusState.WAITING_ACTION

        bus._start_dwelling(0.0, current_time=100)

        self.assertEqual(bus.applied_actions, [0.0])
        self.assertEqual(bus.last_action_s, 0.0)
        self.assertEqual(bus.state, BusState.DWELLING)

    def test_positive_only_trace_preserves_legacy_semantics(self):
        bus = Bus.__new__(Bus)
        bus.trip_id = 4
        bus.applied_actions = []
        bus.holding_action_trace_mode = "positive_only"
        bus.last_station = SimpleNamespace(station_id=7)

        bus._start_dwelling(0.0, current_time=100)
        bus._start_dwelling(12.0, current_time=110)

        self.assertEqual(bus.applied_actions, [12.0])

    def test_terminal_reward_uses_regular_station_objective(self):
        bus = Bus.__new__(Bus)
        bus.forward_bus = [object()]
        bus.backward_bus = [object()]
        bus.forward_headway = 450.0
        bus.backward_headway = 330.0

        reward, cost = bus._headway_reward_cost(360.0)

        forward_reward = -90.0 / 360.0
        backward_reward = -30.0 / 360.0
        weight = 90.0 / (90.0 + 30.0 + 1e-6)
        expected_reward = (
            forward_reward * weight
            + backward_reward * (1.0 - weight)
            - (120.0 / 360.0) * 0.3
        )
        self.assertAlmostEqual(reward, expected_reward)
        self.assertAlmostEqual(cost, (90.0 / 360.0) ** 2)

    def test_lower_headway_uses_prior_same_stop_arrival_causally(self):
        recorder = HeadwayEventRecorder()
        recorder.record(7, True, 100.0, 0)
        recorder.record(7, True, 420.0, 2)
        event_count = len(recorder.events)

        bus = Bus.__new__(Bus)
        bus._headway_recorder = recorder
        bus.next_station = SimpleNamespace(station_id=7)
        bus.direction = True

        self.assertEqual(bus._recorded_forward_headway(805.0), 385.0)
        self.assertEqual(len(recorder.events), event_count)

    def test_lower_headway_has_no_cross_direction_leakage(self):
        recorder = HeadwayEventRecorder()
        recorder.record(7, False, 420.0, 1)
        bus = Bus.__new__(Bus)
        bus._headway_recorder = recorder
        bus.next_station = SimpleNamespace(station_id=7)
        bus.direction = True

        self.assertIsNone(bus._recorded_forward_headway(805.0))

    def test_unobserved_station_does_not_reuse_stale_action(self):
        bus = Bus.__new__(Bus)
        bus.trip_id = 8
        bus.next_station = SimpleNamespace(station_id=1)
        bus.effective_station = [
            SimpleNamespace(station_id=0),
            bus.next_station,
        ]
        bus.unobserved_action_mode = "zero"
        bus.dwelling_time = 17.0

        bus._prepare_for_action(current_time=100, bus_all=[], debug=False)

        self.assertEqual(bus.forward_bus, [])
        self.assertEqual(bus.backward_bus, [])
        self.assertEqual(bus.dwelling_time, 0.0)
        self.assertEqual(bus.state, BusState.DWELLING)

    def test_legacy_unobserved_station_keeps_waiting_action(self):
        bus = Bus.__new__(Bus)
        bus.trip_id = 8
        bus.next_station = SimpleNamespace(station_id=1)
        bus.effective_station = [
            SimpleNamespace(station_id=0),
            bus.next_station,
        ]
        bus.unobserved_action_mode = "legacy_stale"

        bus._prepare_for_action(current_time=100, bus_all=[], debug=False)

        self.assertEqual(bus.state, BusState.WAITING_ACTION)

    def test_passenger_exchange_preserves_onboard_and_station_order(self):
        destination_here = SimpleNamespace(station_name="B")
        destination_later = SimpleNamespace(station_name="C")
        onboard_first = SimpleNamespace(destination_station=destination_later)
        alighting = SimpleNamespace(destination_station=destination_here)
        onboard_last = SimpleNamespace(destination_station=destination_later)
        waiting = [
            SimpleNamespace(destination_station=destination_later, appear_time=10.0),
            SimpleNamespace(destination_station=destination_later, appear_time=20.0),
            SimpleNamespace(destination_station=destination_later, appear_time=30.0),
        ]
        station = SimpleNamespace(
            station_name="B",
            station_id=2,
            waiting_passengers=np.asarray(waiting, dtype=object),
        )
        bus = Bus.__new__(Bus)
        bus.next_station = station
        bus.passengers = np.asarray(
            [onboard_first, alighting, onboard_last], dtype=object)
        bus.capacity = 3
        bus.alight_num = 0.0
        bus.board_num = 0.0

        bus.exchange_passengers(current_time=100.0, debug=False)

        self.assertEqual(
            list(bus.passengers), [onboard_first, onboard_last, waiting[0]])
        self.assertEqual(list(station.waiting_passengers), waiting[1:])
        self.assertTrue(alighting.arrived)
        self.assertEqual(waiting[0].boarding_time, 100.0)


if __name__ == "__main__":
    unittest.main()
