import unittest

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments.pointmaze_goal_validation import DEFAULT_ENV_ID
from freq_hrl.experiments.pointmaze_multiscale_analysis import (
    analyze_pointmaze_multiscale_cells,
    render_report,
)
from freq_hrl.experiments.pointmaze_multiscale_validation import (
    CausalPointMazeStress,
    POINTMAZE_MULTISCALE_ALGORITHM_PATH,
    POINTMAZE_MULTISCALE_METHODS,
    POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
    PointMazeFeatureBuilder,
    build_pointmaze_multiscale_model,
    make_pointmaze_environment,
    pointmaze_multiscale_dimensions,
    rollout_flat_pointmaze_multiscale,
    rollout_hrl_pointmaze_multiscale,
)


class PointMazeMultiscaleStageThreeTest(unittest.TestCase):
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

    def test_raw_and_haar_flat_arms_have_equal_information_dimension(self):
        dimensions = pointmaze_multiscale_dimensions(
            env_id=DEFAULT_ENV_ID,
            horizon=64,
            time_scale=self.time_scale,
        )
        self.assertEqual(dimensions["history"].history, 128)
        self.assertEqual(dimensions["history"].flat, 134)
        self.assertEqual(dimensions["filtered"].flat, 134)
        self.assertEqual(dimensions["multiscale"].flat, 134)
        self.assertEqual(dimensions["history"].upper, 134)
        self.assertEqual(dimensions["multiscale"].upper, 38)
        self.assertEqual(dimensions["multiscale"].lower, 126)
        self.assertEqual(dimensions["multiscale"].slow, 8)
        self.assertEqual(dimensions["multiscale"].mid, 24)
        self.assertEqual(dimensions["multiscale"].high, 96)
        self.assertEqual(dimensions["multiscale"].slow_energy, 4)

    def test_both_hierarchy_levels_keep_current_physical_feedback(self):
        observation = {
            "observation": np.asarray([0.1, -0.2, 0.3, -0.4]),
            "achieved_goal": np.asarray([0.1, -0.2]),
            "desired_goal": np.asarray([1.0, 1.0]),
        }
        changed = {key: value.copy() for key, value in observation.items()}
        changed["observation"] = np.asarray([0.7, -0.8, 0.9, -1.0])
        changed["achieved_goal"] = changed["observation"][:2].copy()
        builder = PointMazeFeatureBuilder(
            physical_dim=4, time_scale=self.time_scale
        )
        builder.reset(observation)
        for representation in ("history", "multiscale"):
            upper_before = builder.upper_state(
                observation, representation=representation
            )
            upper_after = builder.upper_state(
                changed, representation=representation
            )
            np.testing.assert_allclose(upper_after[:4], changed["observation"])
            self.assertFalse(np.array_equal(upper_before[:4], upper_after[:4]))
            lower_after = builder.lower_state(
                changed,
                subgoal=np.asarray([0.2, 0.3]),
                representation=representation,
            )
            np.testing.assert_allclose(lower_after[:4], changed["observation"])

    def test_stage_four_routing_controls_preserve_information_contract(self):
        observation = {
            "observation": np.asarray([0.1, -0.2, 0.3, -0.4]),
            "achieved_goal": np.asarray([0.1, -0.2]),
            "desired_goal": np.asarray([1.0, 1.0]),
        }
        builder = PointMazeFeatureBuilder(
            physical_dim=4, time_scale=self.time_scale
        )
        builder.reset(observation)
        expected = {
            "multiscale_all": (134, 134),
            "multiscale_routed": (38, 126),
            "multiscale_swapped": (126, 38),
        }
        for representation, (upper_dim, lower_dim) in expected.items():
            dimensions = builder.dimensions(
                observation, action_dim=2, representation=representation
            )
            upper = builder.upper_state(
                observation, representation=representation
            )
            lower = builder.lower_state(
                observation,
                subgoal=np.asarray([0.2, 0.3]),
                representation=representation,
            )
            self.assertEqual(
                (dimensions.upper, dimensions.lower),
                (upper_dim, lower_dim),
            )
            self.assertEqual((upper.size, lower.size), (upper_dim, lower_dim))
            np.testing.assert_allclose(upper[:4], observation["observation"])
            np.testing.assert_allclose(lower[:4], observation["observation"])

    def test_lower_features_hide_final_goal_in_both_representations(self):
        environment = make_pointmaze_environment(
            env_id=DEFAULT_ENV_ID, horizon=64
        )
        try:
            observation, _ = environment.reset(seed=3)
            builder = PointMazeFeatureBuilder(
                physical_dim=4, time_scale=self.time_scale
            )
            builder.reset(observation)
            changed = {
                key: np.asarray(value).copy()
                for key, value in observation.items()
            }
            changed["desired_goal"] = -changed["desired_goal"]
            subgoal = np.asarray([0.2, -0.3], dtype=np.float32)
            for representation in ("history", "multiscale"):
                np.testing.assert_array_equal(
                    builder.lower_state(
                        observation,
                        subgoal=subgoal,
                        representation=representation,
                    ),
                    builder.lower_state(
                        changed,
                        subgoal=subgoal,
                        representation=representation,
                    ),
                )
        finally:
            environment.close()

    def test_hidden_stress_is_seed_paired_and_clean_is_identity(self):
        observation = {
            "observation": np.asarray([0.1, -0.2, 0.3, -0.4]),
            "achieved_goal": np.asarray([0.1, -0.2]),
            "desired_goal": np.asarray([1.0, 1.0]),
        }
        kwargs = dict(
            seed=19,
            dt_seconds=0.01,
            physical_dim=4,
            goal_dim=2,
            action_dim=2,
            horizon_steps=64,
        )
        first = CausalPointMazeStress(
            scenario="persistent_action_shift", **kwargs
        )
        second = CausalPointMazeStress(
            scenario="persistent_action_shift", **kwargs
        )
        for _ in range(4):
            visible_a, noise_a = first.observe(observation)
            visible_b, noise_b = second.observe(observation)
            np.testing.assert_array_equal(noise_a, noise_b)
            np.testing.assert_array_equal(
                visible_a["observation"], visible_b["observation"]
            )
            action_a = first.execute(
                np.zeros(2), -np.ones(2), np.ones(2)
            )
            action_b = second.execute(
                np.zeros(2), -np.ones(2), np.ones(2)
            )
            for left, right in zip(action_a, action_b):
                np.testing.assert_array_equal(left, right)
        clean = CausalPointMazeStress(scenario="clean", **kwargs)
        visible, noise = clean.observe(observation)
        executed, slow, fast, persistent = clean.execute(
            np.asarray([0.2, -0.3]), -np.ones(2), np.ones(2)
        )
        np.testing.assert_array_equal(noise, np.zeros(4))
        np.testing.assert_allclose(visible["observation"], observation["observation"])
        np.testing.assert_allclose(executed, [0.2, -0.3])
        np.testing.assert_array_equal(slow, np.zeros(2))
        np.testing.assert_array_equal(fast, np.zeros(2))
        np.testing.assert_array_equal(persistent, np.zeros(2))

    def test_stress_families_change_only_the_registered_channel(self):
        observation = {
            "observation": np.asarray([0.1, -0.2, 0.3, -0.4]),
            "achieved_goal": np.asarray([0.1, -0.2]),
            "desired_goal": np.asarray([1.0, 1.0]),
        }
        kwargs = dict(
            seed=29,
            dt_seconds=0.01,
            physical_dim=4,
            goal_dim=2,
            action_dim=2,
            horizon_steps=64,
        )
        observation_stress = CausalPointMazeStress(
            scenario="fast_observation_noise", **kwargs
        )
        _, noise = observation_stress.observe(observation)
        _, slow, fast, persistent = observation_stress.execute(
            np.zeros(2), -np.ones(2), np.ones(2)
        )
        self.assertGreater(float(np.linalg.norm(noise)), 0.0)
        np.testing.assert_array_equal(slow, np.zeros(2))
        np.testing.assert_array_equal(fast, np.zeros(2))
        np.testing.assert_array_equal(persistent, np.zeros(2))

        action_stress = CausalPointMazeStress(
            scenario="slow_drift_fast_action", **kwargs
        )
        _, noise = action_stress.observe(observation)
        _, slow, fast, persistent = action_stress.execute(
            np.zeros(2), -np.ones(2), np.ones(2)
        )
        np.testing.assert_array_equal(noise, np.zeros(4))
        self.assertGreater(float(np.linalg.norm(slow)), 0.0)
        self.assertGreater(float(np.linalg.norm(fast)), 0.0)
        np.testing.assert_array_equal(persistent, np.zeros(2))

        shift = CausalPointMazeStress(
            scenario="persistent_action_shift", **kwargs
        )
        persistent_values = [
            shift.execute(np.zeros(2), -np.ones(2), np.ones(2))[3]
            for _ in range(64)
        ]
        onset = shift.metadata()["persistent_action_shift_onset_step"]
        self.assertTrue(all(
            np.array_equal(value, np.zeros(2))
            for value in persistent_values[:onset]
        ))
        self.assertTrue(all(
            float(np.linalg.norm(value)) > 0.0
            for value in persistent_values[onset:]
        ))

    def test_five_real_rollouts_are_trainable_and_capacity_matched(self):
        dimensions = pointmaze_multiscale_dimensions(
            env_id=DEFAULT_ENV_ID,
            horizon=64,
            time_scale=self.time_scale,
        )
        for index, method in enumerate(POINTMAZE_MULTISCALE_METHODS):
            model, capacity = build_pointmaze_multiscale_model(
                method=method,
                dimensions=dimensions,
                reference_hidden_dim=16,
                learning_rate=3e-4,
                optimizer_seed=100 + index,
            )
            self.assertLess(
                abs(float(capacity["parameter_budget_ratio"]) - 1.0), 0.07
            )
            kwargs = dict(
                method=method,
                scenario="fast_observation_noise",
                env_id=DEFAULT_ENV_ID,
                seed=23,
                horizon=64,
                sample=True,
                parameter_budget=int(capacity["reference_parameter_budget"]),
                time_scale=self.time_scale,
            )
            if method.startswith("hrl_"):
                batch, row = rollout_hrl_pointmaze_multiscale(
                    model, maximum_subgoal_delta=0.75, **kwargs
                )
                self.assertEqual(batch.lower.size, 64)
                self.assertEqual(batch.upper.size, 3)
                metrics = model.update(batch)
                self.assertTrue(np.isfinite(metrics["lower_loss"]))
            else:
                batch, row = rollout_flat_pointmaze_multiscale(model, **kwargs)
                self.assertEqual(batch.size, 64)
                metrics = model.update(batch)
                self.assertTrue(np.isfinite(metrics["loss"]))
            self.assertEqual(row["protocol_valid"], 1.0)
            self.assertEqual(row["episode_length"], 64)
            self.assertEqual(row["terminated"], 0.0)
            self.assertEqual(row["truncated"], 1.0)
            self.assertGreater(row["measurement_noise_rms"], 0.0)
            self.assertEqual(row["slow_action_stress_rms"], 0.0)
            self.assertEqual(row["fast_action_stress_rms"], 0.0)
            self.assertEqual(row["persistent_action_stress_rms"], 0.0)
            self.assertEqual(
                row["protocol_version"],
                POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
            )

    def test_rollout_diagnostics_keep_stress_families_separate(self):
        dimensions = pointmaze_multiscale_dimensions(
            env_id=DEFAULT_ENV_ID,
            horizon=64,
            time_scale=self.time_scale,
        )
        model, capacity = build_pointmaze_multiscale_model(
            method="flat_history",
            dimensions=dimensions,
            reference_hidden_dim=16,
            learning_rate=3e-4,
            optimizer_seed=211,
        )
        rows = {}
        for scenario in (
            "clean",
            "fast_observation_noise",
            "slow_drift_fast_action",
            "persistent_action_shift",
        ):
            _, rows[scenario] = rollout_flat_pointmaze_multiscale(
                model,
                method="flat_history",
                scenario=scenario,
                env_id=DEFAULT_ENV_ID,
                seed=31,
                horizon=64,
                sample=False,
                parameter_budget=int(capacity["reference_parameter_budget"]),
                time_scale=self.time_scale,
            )
        channels = (
            "measurement_noise_rms",
            "slow_action_stress_rms",
            "fast_action_stress_rms",
            "persistent_action_stress_rms",
        )
        self.assertTrue(all(rows["clean"][name] == 0.0 for name in channels))
        self.assertGreater(
            rows["fast_observation_noise"]["measurement_noise_rms"], 0.0
        )
        self.assertTrue(all(
            rows["fast_observation_noise"][name] == 0.0
            for name in channels[1:]
        ))
        self.assertEqual(
            rows["slow_drift_fast_action"]["measurement_noise_rms"], 0.0
        )
        self.assertGreater(
            rows["slow_drift_fast_action"]["slow_action_stress_rms"], 0.0
        )
        self.assertGreater(
            rows["slow_drift_fast_action"]["fast_action_stress_rms"], 0.0
        )
        self.assertEqual(
            rows["slow_drift_fast_action"]["persistent_action_stress_rms"],
            0.0,
        )
        self.assertTrue(all(
            rows["persistent_action_shift"][name] == 0.0
            for name in channels[:3]
        ))
        self.assertGreater(
            rows["persistent_action_shift"]["persistent_action_stress_rms"],
            0.0,
        )

    @staticmethod
    def _synthetic_cells():
        cells = []
        for scenario in (
            "clean",
            "fast_observation_noise",
            "slow_drift_fast_action",
            "persistent_action_shift",
        ):
            for method in POINTMAZE_MULTISCALE_METHODS:
                for root in (101, 103, 107, 109):
                    rows = []
                    root_shift = (root - 105) * 0.001
                    if scenario == "clean":
                        success = {
                            "flat_history": 0.50,
                            "flat_multiscale": 0.50,
                            "flat_causal_filter": 0.50,
                            "hrl_history": 0.70,
                            "hrl_multiscale": 0.70,
                        }[method]
                    else:
                        success = {
                            "flat_history": 0.40,
                            "flat_multiscale": 0.50,
                            "flat_causal_filter": 0.55,
                            "hrl_history": 0.40,
                            "hrl_multiscale": 0.80,
                        }[method]
                    for seed in (1, 2, 3, 4):
                        rows.append({
                            "algorithm_path": POINTMAZE_MULTISCALE_ALGORITHM_PATH,
                            "protocol_valid": 1.0,
                            "training_replicate_seed": root,
                            "scenario": scenario,
                            "method": method,
                            "seed": seed,
                            "success": success + root_shift,
                            "episode_return": 100.0 * success + root_shift,
                            "final_goal_distance": 1.0 - success - root_shift,
                        })
                    cells.append({
                        "protocol_version": POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
                        "policy": method,
                        "scenario": scenario,
                        "optimizer_seed": root,
                        "runtime_versions": {"python": "test"},
                        "evaluation_rows": rows,
                    })
        return cells

    def test_analysis_separates_representation_and_interaction_claims(self):
        analysis = analyze_pointmaze_multiscale_cells(self._synthetic_cells())
        self.assertEqual(analysis["independent_training_replicate_count"], 4)
        self.assertEqual(analysis["freq_hrl_mainline_status"], "supported")
        stress = analysis["scenarios"]["fast_observation_noise"]
        self.assertAlmostEqual(
            stress["flat_representation"]["success"]["mean_improvement"],
            0.10,
        )
        self.assertAlmostEqual(
            stress["factorial_interaction"]["success"]["mean_improvement"],
            0.30,
        )
        self.assertIn("Flat representation", render_report(analysis))
        self.assertIn("causal filter", render_report(analysis))

    def test_analysis_rejects_an_incomplete_factorial(self):
        cells = self._synthetic_cells()
        cells = [cell for cell in cells if not (
            cell["scenario"] == "clean"
            and cell["policy"] == "flat_history"
            and cell["optimizer_seed"] == 101
        )]
        with self.assertRaisesRegex(ValueError, "share optimizer roots"):
            analyze_pointmaze_multiscale_cells(cells)


if __name__ == "__main__":
    unittest.main()
