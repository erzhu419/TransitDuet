import unittest

from runner_v3 import TransitDuetV2Runner
from upper.interval_credit import UpperIntervalOutcomeTracker
from upper.plan_execution import UpperPlanExecutionContract


class UpperIntervalOutcomeTrackerTest(unittest.TestCase):
    def test_additive_credit_partitions_directional_outcomes(self):
        tracker = UpperIntervalOutcomeTracker(enabled=True)
        tracker.begin(True, 0.0)
        tracker.begin(False, 0.0)
        tracker.record_step(
            dt_s=10.0,
            waiting_by_direction={True: 10.0, False: 20.0},
            fleet_by_direction={True: 8.0, False: 4.0},
            n_fleet_target=12.0,
            headway_events=[
                {
                    "direction": True,
                    "headway_s": 540.0,
                    "target_headway_s": 360.0,
                },
                {
                    "direction": False,
                    "headway_s": 180.0,
                    "target_headway_s": 360.0,
                },
            ],
        )

        up = tracker.close(True, 10.0)
        down = tracker.close(False, 10.0)
        up_score = tracker.score(
            up,
            passengers_generated=100,
            episode_headway_samples=2,
            episode_duration_s=10.0,
            n_fleet_target=12.0,
        )
        down_score = tracker.score(
            down,
            passengers_generated=100,
            episode_headway_samples=2,
            episode_duration_s=10.0,
            n_fleet_target=12.0,
        )

        self.assertAlmostEqual(up["coverage"], 1.0)
        self.assertAlmostEqual(
            up_score["wait_cost"] + down_score["wait_cost"], 0.005)
        self.assertAlmostEqual(
            up_score["headway_cost"] + down_score["headway_cost"], 0.5)
        self.assertAlmostEqual(up_score["fleet_cost"], 1.0 / 3.0)
        self.assertAlmostEqual(down_score["fleet_cost"], 0.0)
        self.assertLess(up_score["reward"], 0.0)

    def test_local_mean_mode_scores_each_interval_directly(self):
        tracker = UpperIntervalOutcomeTracker(
            enabled=True,
            assignment_mode="local_mean",
            local_wait_queue_norm=50.0,
        )
        tracker.begin("__all__", 5.0)
        tracker.record_step(
            dt_s=20.0,
            waiting_by_direction={True: 15.0, False: 10.0},
            fleet_by_direction={True: 7.0, False: 7.0},
            n_fleet_target=12.0,
            headway_events=[{
                "direction": True,
                "headway_s": 450.0,
                "target_headway_s": 360.0,
            }],
        )
        outcome = tracker.close("__all__", 25.0)
        score = tracker.score(
            outcome,
            passengers_generated=1,
            episode_headway_samples=1,
            episode_duration_s=20.0,
            n_fleet_target=12.0,
        )

        self.assertAlmostEqual(score["wait_cost"], 0.5)
        self.assertAlmostEqual(score["headway_cost"], 0.25)
        self.assertAlmostEqual(score["fleet_cost"], 2.0 / 12.0)

    def test_headway_without_action_target_is_not_misattributed(self):
        tracker = UpperIntervalOutcomeTracker(enabled=True)
        tracker.begin(True, 0.0)
        tracker.record_step(
            dt_s=1.0,
            waiting_by_direction={True: 0.0, False: 0.0},
            fleet_by_direction={True: 0.0, False: 0.0},
            n_fleet_target=12.0,
            headway_events=[{
                "direction": True,
                "headway_s": 500.0,
                "target_headway_s": None,
            }],
        )
        outcome = tracker.close(True, 1.0)
        self.assertEqual(outcome["headway_sample_count"], 0)

    def test_v4_wait_credit_owns_only_frozen_low_frequency_mass(self):
        tracker = UpperIntervalOutcomeTracker(
            enabled=True,
            wait_ownership="frozen_low_frequency",
        )
        tracker.begin(True, 0.0)
        tracker.record_step(
            dt_s=10.0,
            waiting_by_direction={True: 10.0, False: 0.0},
            waiting_low_by_direction={True: 6.5, False: 0.0},
            onboard_by_direction={True: 4.0, False: 0.0},
            onboard_low_by_direction={True: 2.5, False: 0.0},
            fleet_by_direction={True: 0.0, False: 0.0},
            n_fleet_target=12.0,
            headway_events=[],
        )
        outcome = tracker.close(True, 10.0)

        self.assertEqual(outcome["wait_ownership"], "frozen_low_frequency")
        self.assertAlmostEqual(outcome["waiting_exposure_s"], 65.0)
        self.assertAlmostEqual(outcome["waiting_total_exposure_s"], 100.0)
        self.assertAlmostEqual(outcome["onboard_exposure_s"], 25.0)
        self.assertAlmostEqual(outcome["onboard_total_exposure_s"], 40.0)

    def test_v5_passenger_time_and_dispatch_backlog_are_priced(self):
        tracker = UpperIntervalOutcomeTracker(
            enabled=True,
            wait_weight=1.0,
            onboard_weight=1.0,
            dispatch_backlog_weight=0.5,
            headway_weight=0.0,
            fleet_weight=0.0,
            wait_reference_min=10.0,
            onboard_reference_min=10.0,
            dispatch_backlog_reference_trips=2.0,
        )
        tracker.begin(True, 0.0)
        tracker.record_step(
            dt_s=60.0,
            waiting_by_direction={True: 10.0, False: 0.0},
            onboard_by_direction={True: 20.0, False: 0.0},
            dispatch_backlog_by_direction={True: 2.0, False: 0.0},
            fleet_by_direction={True: 6.0, False: 6.0},
            n_fleet_target=12.0,
            headway_events=[],
        )
        outcome = tracker.close(True, 60.0)
        score = tracker.score(
            outcome,
            passengers_generated=100,
            episode_headway_samples=1,
            episode_duration_s=60.0,
            n_fleet_target=12.0,
        )

        self.assertAlmostEqual(score["wait_cost"], 0.01)
        self.assertAlmostEqual(score["onboard_cost"], 0.02)
        self.assertAlmostEqual(score["dispatch_backlog_cost"], 1.0)
        self.assertAlmostEqual(score["reward"], -0.53)

    def test_v4_wait_credit_rejects_missing_frequency_ownership(self):
        tracker = UpperIntervalOutcomeTracker(
            enabled=True,
            wait_ownership="frozen_low_frequency",
        )
        tracker.begin(True, 0.0)
        with self.assertRaisesRegex(ValueError, "requires frozen LF"):
            tracker.record_step(
                dt_s=1.0,
                waiting_by_direction={True: 1.0, False: 0.0},
                fleet_by_direction={True: 0.0, False: 0.0},
                n_fleet_target=12.0,
                headway_events=[],
            )

    def test_duplicate_stream_open_fails_fast(self):
        tracker = UpperIntervalOutcomeTracker(enabled=True)
        tracker.begin(True, 0.0)
        with self.assertRaisesRegex(RuntimeError, "opened twice"):
            tracker.begin(True, 1.0)


class UpperTransitionStreamTest(unittest.TestCase):
    @staticmethod
    def _pending(direction, decision_time_s):
        return {
            "s": [float(direction)],
            "a": [1.0],
            "tid": int(decision_time_s),
            "dir": direction,
            "a_eff": 1.0,
            "plan_penalty": 0.0,
            "upper_value_cost": 0.0,
            "upper_value_active": 0.0,
            "upper_residual_selector_x": None,
            "terminal_value_selector_x": None,
            "headway_value_planner_x": None,
            "decision_time_s": decision_time_s,
        }

    def test_planner_key_streams_close_independently(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.upper_transition_stream_mode = "planner_key"
        runner._episode_upper_transitions = []
        runner._prev_upper_states = {
            True: self._pending(True, 0.0),
            False: self._pending(False, 180.0),
        }
        runner.upper_plan_execution = UpperPlanExecutionContract(
            duration_discount=True,
            duration_base_s=900.0,
            duration_min_steps=0.25,
        )
        runner.upper_interval_credit = UpperIntervalOutcomeTracker()

        runner._close_previous_upper_transition(
            [2.0], done=False, decision_time_s=900.0, planner_key=True)

        self.assertEqual(len(runner._episode_upper_transitions), 1)
        self.assertAlmostEqual(
            runner._episode_upper_transitions[0]["duration_steps"], 1.0)
        self.assertIn(False, runner._prev_upper_states)

    def test_legacy_mode_maps_both_directions_to_one_stream(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.upper_transition_stream_mode = "legacy_global"
        self.assertEqual(
            runner._upper_transition_stream_key(True),
            runner._upper_transition_stream_key(False),
        )


if __name__ == "__main__":
    unittest.main()
