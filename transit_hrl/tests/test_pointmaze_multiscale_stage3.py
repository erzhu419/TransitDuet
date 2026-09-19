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
        self.assertEqual(dimensions["multiscale"].flat, 134)
        self.assertEqual(dimensions["multiscale"].slow, 8)
        self.assertEqual(dimensions["multiscale"].mid, 24)
        self.assertEqual(dimensions["multiscale"].high, 96)
        self.assertEqual(dimensions["multiscale"].slow_energy, 4)

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
        )
        first = CausalPointMazeStress(
            scenario="mixed_causal_stress", **kwargs
        )
        second = CausalPointMazeStress(
            scenario="mixed_causal_stress", **kwargs
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
        executed, slow, fast = clean.execute(
            np.asarray([0.2, -0.3]), -np.ones(2), np.ones(2)
        )
        np.testing.assert_array_equal(noise, np.zeros(4))
        np.testing.assert_allclose(visible["observation"], observation["observation"])
        np.testing.assert_allclose(executed, [0.2, -0.3])
        np.testing.assert_array_equal(slow, np.zeros(2))
        np.testing.assert_array_equal(fast, np.zeros(2))

    def test_four_real_rollouts_are_trainable_and_capacity_matched(self):
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
                scenario="mixed_causal_stress",
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
            self.assertEqual(
                row["protocol_version"],
                POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
            )

    @staticmethod
    def _synthetic_cells():
        cells = []
        for scenario in ("clean", "mixed_causal_stress"):
            for method in POINTMAZE_MULTISCALE_METHODS:
                for root in (101, 103, 107, 109):
                    rows = []
                    root_shift = (root - 105) * 0.001
                    if scenario == "clean":
                        success = {
                            "flat_history": 0.50,
                            "flat_multiscale": 0.50,
                            "hrl_history": 0.70,
                            "hrl_multiscale": 0.70,
                        }[method]
                    else:
                        success = {
                            "flat_history": 0.40,
                            "flat_multiscale": 0.50,
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
        stress = analysis["scenarios"]["mixed_causal_stress"]
        self.assertAlmostEqual(
            stress["flat_representation"]["success"]["mean_improvement"],
            0.10,
        )
        self.assertAlmostEqual(
            stress["factorial_interaction"]["success"]["mean_improvement"],
            0.30,
        )
        self.assertIn("Flat representation", render_report(analysis))

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
