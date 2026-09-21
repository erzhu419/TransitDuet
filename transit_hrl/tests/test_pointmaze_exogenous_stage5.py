import unittest

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import (
    PointMazeExternalDriver,
    PointMazeExternalObservation,
    PointMazeExternalTask,
)
from freq_hrl.experiments.pointmaze_exogenous_validation import (
    POINTMAZE_EXOGENOUS_PROTOCOL_VERSION,
    PointMazeExogenousFeatureBuilder,
    build_pointmaze_exogenous_model,
    pointmaze_exogenous_checkpoint_rank,
    pointmaze_exogenous_dimensions,
    rollout_flat_pointmaze_exogenous,
    rollout_hrl_pointmaze_exogenous,
)
from freq_hrl.experiments.pointmaze_exogenous_analysis import (
    analyze_pointmaze_exogenous_cells,
)
from freq_hrl.experiments.pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    make_pointmaze_environment,
)


class PointMazeExogenousStageFiveTest(unittest.TestCase):
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

    def test_driver_is_seed_deterministic_and_action_independent(self):
        first_env = make_pointmaze_environment(env_id=DEFAULT_ENV_ID, horizon=32)
        second_env = make_pointmaze_environment(env_id=DEFAULT_ENV_ID, horizon=32)
        first = PointMazeExternalTask(first_env, seed=101, horizon=32)
        second = PointMazeExternalTask(second_env, seed=101, horizon=32)
        try:
            left = first.reset()
            right = second.reset()
            for step in range(32):
                np.testing.assert_array_equal(
                    left.task_measurement, right.task_measurement
                )
                left, _, _, _, left_info = first.step(
                    np.asarray([0.8, -0.6], dtype=np.float32)
                )
                right, _, _, _, right_info = second.step(
                    np.asarray([-0.7, 0.5], dtype=np.float32)
                )
                np.testing.assert_array_equal(
                    left_info["target"], right_info["target"]
                )
                np.testing.assert_array_equal(
                    left_info["force"], right_info["force"]
                )
                if step > 1:
                    self.assertFalse(np.array_equal(
                        left.physical, right.physical
                    ))
            self.assertTrue(
                first.observability_contract["external_stream_action_independent"]
            )
            self.assertFalse(
                first.observability_contract["external_future_visible_to_actor"]
            )
        finally:
            first_env.close()
            second_env.close()

    def test_driver_resolves_slow_target_and_matched_fast_force(self):
        driver = PointMazeExternalDriver(
            seed=17,
            horizon=300,
            dt_seconds=0.01,
            target_speed=1.0,
            force_rms=0.12,
        )
        targets = np.asarray([driver.sample(step)[0] for step in range(301)])
        increments = np.linalg.norm(np.diff(targets, axis=0), axis=1)
        self.assertLessEqual(float(np.max(increments)), 0.010001)
        diagnostics = driver.diagnostics()
        self.assertAlmostEqual(diagnostics["target_round_trip_period_seconds"], 12.0)
        self.assertAlmostEqual(diagnostics["force_x_rms"], 0.12, places=6)
        self.assertAlmostEqual(diagnostics["force_y_rms"], 0.12, places=6)
        self.assertLess(diagnostics["force_x_period_seconds"], 0.25)
        self.assertLess(diagnostics["force_y_period_seconds"], 0.25)

    def test_features_separate_current_physics_from_external_history(self):
        observation = PointMazeExternalObservation(
            physical=np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
            achieved_goal=np.asarray([0.1, -0.2], dtype=np.float32),
            task_measurement=np.asarray([0.5, 0.6, -0.1, 0.2], dtype=np.float32),
            target=np.asarray([0.5, 0.6], dtype=np.float32),
            force=np.asarray([-0.1, 0.2], dtype=np.float32),
        )
        changed = PointMazeExternalObservation(
            physical=np.asarray([0.7, -0.8, 0.9, -1.0], dtype=np.float32),
            achieved_goal=np.asarray([0.7, -0.8], dtype=np.float32),
            task_measurement=observation.task_measurement.copy(),
            target=observation.target.copy(),
            force=observation.force.copy(),
        )
        features = PointMazeExogenousFeatureBuilder(time_scale=self.time_scale)
        features.reset(observation)
        before = features.upper_state(observation)
        after = features.upper_state(changed)
        np.testing.assert_array_equal(after[:4], changed.physical)
        np.testing.assert_array_equal(before[6:], after[6:])
        self.assertFalse(np.array_equal(before[:6], after[:6]))
        lower = features.lower_state(
            changed, subgoal=np.asarray([0.2, 0.3], dtype=np.float32)
        )
        np.testing.assert_array_equal(lower[:4], changed.physical)
        self.assertEqual(before.size, 134)
        self.assertEqual(lower.size, 134)

    def test_external_frequency_masks_are_equal_shape_and_exact(self):
        observation = PointMazeExternalObservation(
            physical=np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
            achieved_goal=np.asarray([0.1, -0.2], dtype=np.float32),
            task_measurement=np.asarray([0.5, 0.6, -0.1, 0.2], dtype=np.float32),
            target=np.asarray([0.5, 0.6], dtype=np.float32),
            force=np.asarray([-0.1, 0.2], dtype=np.float32),
        )
        features = PointMazeExogenousFeatureBuilder(time_scale=self.time_scale)
        features.reset(observation)
        for index in range(1, 9):
            measurement = np.asarray(
                [0.5 + 0.01 * index, 0.6, (-1) ** index * 0.1, 0.2],
                dtype=np.float32,
            )
            features.update(PointMazeExternalObservation(
                physical=observation.physical,
                achieved_goal=observation.achieved_goal,
                task_measurement=measurement,
                target=measurement[:2],
                force=measurement[2:],
            ))
        snapshot = features.snapshot
        zeros_slow = np.zeros_like(snapshot.slow)
        zeros_high = np.zeros_like(snapshot.high)
        subgoal = np.asarray([0.2, 0.3], dtype=np.float32)
        routed_upper = features.upper_state(
            observation, representation="multiscale_routed_masked"
        )
        routed_lower = features.lower_state(
            observation,
            subgoal=subgoal,
            representation="multiscale_routed_masked",
        )
        swapped_upper = features.upper_state(
            observation, representation="multiscale_swapped_masked"
        )
        swapped_lower = features.lower_state(
            observation,
            subgoal=subgoal,
            representation="multiscale_swapped_masked",
        )
        all_upper = features.upper_state(
            observation, representation="multiscale_all"
        )
        np.testing.assert_array_equal(
            routed_upper[6:],
            np.concatenate((snapshot.slow, snapshot.mid, zeros_high)),
        )
        np.testing.assert_array_equal(
            routed_lower[6:],
            np.concatenate((zeros_slow, snapshot.mid, snapshot.high)),
        )
        np.testing.assert_array_equal(
            swapped_upper[6:],
            np.concatenate((zeros_slow, snapshot.mid, snapshot.high)),
        )
        np.testing.assert_array_equal(
            swapped_lower[6:],
            np.concatenate((snapshot.slow, snapshot.mid, zeros_high)),
        )
        np.testing.assert_array_equal(all_upper[6:], snapshot.multiscale)
        self.assertEqual(
            {
                routed_upper.size,
                routed_lower.size,
                swapped_upper.size,
                swapped_lower.size,
                all_upper.size,
            },
            {134},
        )

    def test_flat_and_hrl_rollouts_are_trainable_and_equal_shape(self):
        dimensions = pointmaze_exogenous_dimensions(
            env_id=DEFAULT_ENV_ID,
            horizon=64,
            time_scale=self.time_scale,
        )
        self.assertEqual(dimensions.task, 4)
        self.assertEqual(dimensions.history, 128)
        self.assertEqual(
            (dimensions.flat, dimensions.upper, dimensions.lower),
            (134, 134, 134),
        )
        flat, flat_capacity = build_pointmaze_exogenous_model(
            method="flat_exogenous_history",
            dimensions=dimensions,
            reference_hidden_dim=16,
            learning_rate=3e-4,
            optimizer_seed=7,
        )
        hrl, hrl_capacity = build_pointmaze_exogenous_model(
            method="hrl_exogenous_history",
            dimensions=dimensions,
            reference_hidden_dim=16,
            learning_rate=3e-4,
            optimizer_seed=11,
        )
        self.assertLess(
            abs(float(hrl_capacity["parameter_budget_ratio"]) - 1.0), 0.08
        )
        common = {
            "env_id": DEFAULT_ENV_ID,
            "seed": 31,
            "horizon": 64,
            "sample": True,
            "time_scale": self.time_scale,
            "target_speed": 1.0,
            "force_rms": 0.12,
            "force_period_seconds": (0.04, 0.04),
        }
        flat_batch, flat_row = rollout_flat_pointmaze_exogenous(
            flat,
            parameter_budget=int(flat_capacity["reference_parameter_budget"]),
            **common,
        )
        self.assertEqual(flat_row["protocol_valid"], 1.0)
        self.assertEqual(flat_row["episode_length"], 64)
        self.assertEqual(flat_batch.size, 64)
        self.assertTrue(np.isfinite(flat.update(flat_batch)["loss"]))

        hrl_batch, hrl_row = rollout_hrl_pointmaze_exogenous(
            hrl,
            parameter_budget=int(hrl_capacity["reference_parameter_budget"]),
            maximum_subgoal_delta=0.75,
            **common,
        )
        self.assertEqual(hrl_row["protocol_valid"], 1.0)
        self.assertEqual(hrl_row["episode_length"], 64)
        self.assertEqual(hrl_row["upper_decision_count"], 3)
        self.assertEqual(hrl_row["lower_option_boundary_count"], 3)
        self.assertEqual(hrl_batch.lower.size, 64)
        self.assertEqual(hrl_batch.upper.size, 3)
        metrics = hrl.update(hrl_batch)
        self.assertTrue(np.isfinite(metrics["upper_loss"]))
        self.assertTrue(np.isfinite(metrics["lower_loss"]))
        self.assertEqual(
            hrl_row["protocol_version"], POINTMAZE_EXOGENOUS_PROTOCOL_VERSION
        )
        self.assertEqual(hrl_row["external_stream_action_independent"], True)

    def test_checkpoint_rank_prioritizes_tracking_success(self):
        high_success = pointmaze_exogenous_checkpoint_rank([
            {"tracking_success_rate": 0.6, "episode_return": 1.0}
        ])
        high_return = pointmaze_exogenous_checkpoint_rank([
            {"tracking_success_rate": 0.5, "episode_return": 100.0}
        ])
        self.assertGreater(high_success, high_return)

    def test_checkpoint_rank_can_prioritize_dense_return(self):
        high_success = pointmaze_exogenous_checkpoint_rank(
            [{"tracking_success_rate": 0.6, "episode_return": 1.0}],
            mode="return_then_success",
        )
        high_return = pointmaze_exogenous_checkpoint_rank(
            [{"tracking_success_rate": 0.5, "episode_return": 100.0}],
            mode="return_then_success",
        )
        self.assertLess(high_success, high_return)

    def test_checkpoint_rank_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "unknown.*rank mode"):
            pointmaze_exogenous_checkpoint_rank(
                [{"tracking_success_rate": 0.5, "episode_return": 100.0}],
                mode="unregistered",
            )

    def test_analysis_requires_absolute_learning_and_paired_gain(self):
        cells = []
        for method in ("flat_exogenous_history", "hrl_exogenous_history"):
            for root in (101, 103, 107, 109):
                final_rows = []
                initial_rows = []
                for seed in (1, 2, 3):
                    shared = {
                        "protocol_valid": 1.0,
                        "algorithm_path": "exogenous_goal_conditioned_hrl_mainline",
                        "method": method,
                        "seed": seed,
                        "training_replicate_seed": root,
                        "external_stream_action_independent": True,
                        "external_stream_visible_before_action": True,
                        "current_physical_state_visible_to_both_levels": True,
                        "external_future_visible_to_actor": False,
                    }
                    hrl = method == "hrl_exogenous_history"
                    final_rows.append({
                        **shared,
                        "tracking_success_rate": 0.80 if hrl else 0.55,
                        "episode_return": 250.0 if hrl else 210.0,
                        "tracking_rmse": 0.20 if hrl else 0.35,
                        "final_tracking_distance": 0.15 if hrl else 0.30,
                    })
                    initial_rows.append({
                        **shared,
                        "tracking_success_rate": 0.15,
                        "episode_return": 105.0,
                        "tracking_rmse": 1.20,
                        "final_tracking_distance": 1.50,
                    })
                cells.append({
                    "protocol_version": POINTMAZE_EXOGENOUS_PROTOCOL_VERSION,
                    "policy": method,
                    "optimizer_seed": root,
                    "runtime_versions": {"python": "test"},
                    "evaluation_rows": final_rows,
                    "untrained_evaluation_rows": initial_rows,
                })
        analysis = analyze_pointmaze_exogenous_cells(cells)
        self.assertEqual(
            analysis["exogenous_hrl_learning_status"], "supported"
        )
        self.assertEqual(
            analysis["frequency_routing_admission_status"], "admitted"
        )
        self.assertEqual(
            analysis["learning_gain_vs_untrained"]
            ["hrl_exogenous_history"]["tracking_success_rate"]["status"],
            "supported",
        )


if __name__ == "__main__":
    unittest.main()
