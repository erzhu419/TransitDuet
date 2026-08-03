import unittest
from types import SimpleNamespace

from coupling.holding_feedback import HoldingFeedback
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
        self.assertEqual(states[3], [])
        self.assertIsNone(actions[3])
        self.assertEqual(last_actions[3], 0.0)
        self.assertEqual(feedback.finalized_trip_count, 1)
        self.assertEqual(feedback.get_direction_stats(True)["n_trips"], 1)
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


if __name__ == "__main__":
    unittest.main()
