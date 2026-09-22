import unittest

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import (
    PointMazeRegimeDriver,
    PointMazeRegimeTask,
)
from freq_hrl.experiments.pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    make_pointmaze_environment,
)
from freq_hrl.experiments.pointmaze_plan_value_qualification import (
    build_pointmaze_plan_value_model,
    fixed_replan_steps,
    pointmaze_plan_value_dimensions,
    relocate_replan_steps,
    rollout_hrl_pointmaze_plan_value,
)


class PointMazePlanValueStageEightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gymnasium  # noqa: F401
            import gymnasium_robotics  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("gymnasium-robotics is not installed")
        cls.time_scale = PhysicalTimeScaleContract(
            dt_seconds=0.01,
            upper_period_seconds=0.25,
            history_seconds=0.32,
            fast_period_seconds=0.04,
        )
        cls.task_options = {
            "regime_dwell_seconds": (0.20, 0.35),
            "target_speed_modes": (-1.25, -0.55, 0.55, 1.25),
            "force_pulse_amplitude": 0.18,
            "force_pulse_duration_seconds": (0.04, 0.08),
            "force_pulse_gap_seconds": (0.20, 0.35),
            "distractor_amplitude": 0.50,
            "distractor_dwell_seconds": (0.15, 0.30),
        }

    def test_driver_is_prefix_causal_and_regime_is_hidden(self):
        short = PointMazeRegimeDriver(
            seed=71,
            horizon=160,
            dt_seconds=0.01,
            **self.task_options,
        )
        long = PointMazeRegimeDriver(
            seed=71,
            horizon=240,
            dt_seconds=0.01,
            **self.task_options,
        )
        for step in range(161):
            for left, right in zip(
                short.sample(step), long.sample(step), strict=True
            ):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(
                short.privileged_context(step),
                long.privileged_context(step),
            )
        self.assertTrue(short.regime_change_steps)
        self.assertTrue(short.distractor_change_steps)
        self.assertTrue(short.pulse_start_steps)

        environment = make_pointmaze_environment(
            env_id=DEFAULT_ENV_ID, horizon=160
        )
        task = PointMazeRegimeTask(
            environment, seed=71, horizon=160, **self.task_options
        )
        try:
            observation = task.reset()
            self.assertEqual(observation.task_measurement.shape, (6,))
            self.assertFalse(hasattr(observation, "regime_id"))
            context = task.privileged_context()
            self.assertEqual(context.shape, (4,))
            self.assertAlmostEqual(float(np.sum(context)), 1.0)
            self.assertFalse(
                task.observability_contract["regime_label_visible_to_candidate"]
            )
            self.assertFalse(
                task.observability_contract["external_future_visible_to_actor"]
            )
        finally:
            environment.close()

    def test_route_reflection_keeps_unwrapped_motion_after_endpoint(self):
        selected = None
        for seed in range(256):
            driver = PointMazeRegimeDriver(
                seed=seed,
                horizon=40,
                dt_seconds=0.01,
                regime_dwell_seconds=(10.0, 10.0),
                target_speed_modes=(-1.25, -0.55, 0.55, 1.25),
                force_pulse_amplitude=0.0,
                force_pulse_duration_seconds=(0.04, 0.04),
                force_pulse_gap_seconds=(10.0, 10.0),
                distractor_amplitude=0.0,
                distractor_dwell_seconds=(10.0, 10.0),
            )
            mode = int(np.argmax(driver.privileged_context(0)))
            speed = driver.target_speed_modes[mode]
            if (
                driver.start_vertex == 0 and speed < 0.0
            ) or (
                driver.start_vertex == 6 and speed > 0.0
            ):
                selected = driver
                break
        self.assertIsNotNone(selected)
        targets = np.asarray([
            selected.sample(step)[0] for step in range(12)
        ])
        displacement = np.linalg.norm(targets - targets[0], axis=1)
        self.assertGreater(displacement[10], 5.0 * displacement[1])

    def test_oracle_relocation_preserves_planning_budget(self):
        fixed = fixed_replan_steps(horizon=1200, period_steps=50)
        events = (118, 202, 300, 390, 511)
        immediate = relocate_replan_steps(
            fixed_steps=fixed,
            event_steps=events,
            delay_steps=0,
            horizon=1200,
        )
        delayed = relocate_replan_steps(
            fixed_steps=fixed,
            event_steps=events,
            delay_steps=25,
            horizon=1200,
        )
        self.assertEqual(len(immediate), len(fixed))
        self.assertEqual(len(delayed), len(fixed))
        self.assertEqual(immediate[0], 0)
        self.assertTrue(set(events).issubset(immediate))
        self.assertTrue({event + 25 for event in events}.issubset(delayed))
        self.assertEqual(tuple(sorted(immediate)), immediate)

    def test_dimensions_and_capacity_match_without_oracle_lower_leakage(self):
        dimensions = pointmaze_plan_value_dimensions(
            env_id=DEFAULT_ENV_ID,
            horizon=160,
            time_scale=self.time_scale,
            task_options=self.task_options,
        )
        self.assertEqual(dimensions.task, 6)
        self.assertEqual(dimensions.history, 192)
        self.assertEqual(dimensions.base_upper, 198)
        self.assertEqual(dimensions.oracle_upper, 202)
        self.assertEqual(dimensions.lower, 198)
        base, base_capacity = build_pointmaze_plan_value_model(
            method="hrl_regime_history",
            dimensions=dimensions,
            reference_hidden_dim=16,
            learning_rate=3e-4,
            optimizer_seed=7,
        )
        oracle, oracle_capacity = build_pointmaze_plan_value_model(
            method="hrl_regime_oracle_context",
            dimensions=dimensions,
            reference_hidden_dim=16,
            learning_rate=3e-4,
            optimizer_seed=7,
        )
        self.assertEqual(base.config.lower_state_dim, oracle.config.lower_state_dim)
        self.assertEqual(base.config.upper_state_dim + 4, oracle.config.upper_state_dim)
        self.assertLess(
            abs(float(oracle_capacity["parameter_budget_ratio"]) - 1.0),
            0.08,
        )
        self.assertFalse(base_capacity["oracle_context_visible_to_upper"])
        self.assertTrue(oracle_capacity["oracle_context_visible_to_upper"])

    def test_variable_duration_rollouts_are_accounted_and_trainable(self):
        dimensions = pointmaze_plan_value_dimensions(
            env_id=DEFAULT_ENV_ID,
            horizon=160,
            time_scale=self.time_scale,
            task_options=self.task_options,
        )
        base, base_capacity = build_pointmaze_plan_value_model(
            method="hrl_regime_history",
            dimensions=dimensions,
            reference_hidden_dim=16,
            learning_rate=3e-4,
            optimizer_seed=11,
        )
        common = {
            "env_id": DEFAULT_ENV_ID,
            "seed": 91,
            "horizon": 160,
            "parameter_budget": int(
                base_capacity["reference_parameter_budget"]
            ),
            "time_scale": self.time_scale,
            "maximum_subgoal_delta": 0.75,
            "waypoint_perturbation": 0.25,
            "event_window_seconds": 0.50,
            "task_options": self.task_options,
        }
        batch, fixed = rollout_hrl_pointmaze_plan_value(
            base,
            method="hrl_regime_history",
            schedule_mode="fixed",
            sample=True,
            **common,
        )
        self.assertEqual(fixed["protocol_valid"], 1.0)
        self.assertEqual(fixed["upper_decision_count"], 7)
        self.assertEqual(fixed["option_duration_steps_sum"], 160)
        self.assertEqual(int(np.sum(batch.upper.duration)), 160)
        self.assertEqual(batch.lower.size, 160)
        update = base.update(batch)
        self.assertTrue(np.isfinite(update["upper_loss"]))
        self.assertTrue(np.isfinite(update["lower_loss"]))

        _, event = rollout_hrl_pointmaze_plan_value(
            base,
            method="hrl_regime_history",
            schedule_mode="oracle_event_delay_000ms",
            sample=False,
            **common,
        )
        self.assertEqual(event["protocol_valid"], 1.0)
        self.assertEqual(event["upper_decision_count"], 7)
        self.assertEqual(event["planning_budget_matched"], 1.0)
        self.assertTrue(event["schedule_has_future_regime_access"])
        self.assertFalse(event["policy_has_current_regime_access"])
        self.assertGreater(event["scored_regime_change_count"], 0)
        self.assertLessEqual(
            event["causal_distinguishability_delay_seconds_max"], 0.05
        )
        self.assertEqual(
            event["causal_distinguishability_censored_rate"], 0.0
        )

        _, stale = rollout_hrl_pointmaze_plan_value(
            base,
            method="hrl_regime_history",
            schedule_mode="stale_plan",
            sample=False,
            **common,
        )
        self.assertEqual(stale["protocol_valid"], 1.0)
        self.assertEqual(stale["upper_decision_count"], 1)
        self.assertEqual(stale["planning_budget_matched"], 0.0)


if __name__ == "__main__":
    unittest.main()
