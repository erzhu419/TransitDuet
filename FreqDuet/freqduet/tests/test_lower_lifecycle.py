import unittest
from types import SimpleNamespace

from coupling.holding_feedback import HoldingFeedback
from env.bus import Bus, BusState
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


if __name__ == "__main__":
    unittest.main()
