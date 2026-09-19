import unittest

import numpy as np

from freq_hrl.experiments.pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    POINTMAZE_GOAL_PROTOCOL_VERSION,
    build_pointmaze_model,
    flat_goal_state,
    lower_goal_state,
    make_pointmaze_environment,
    pointmaze_dimensions,
    pointmaze_goal_bounds,
    pointmaze_checkpoint_rank,
    pointmaze_runtime_versions,
    rollout_flat_pointmaze,
    rollout_hrl_pointmaze,
    squash_box_action,
)
from freq_hrl.experiments.pointmaze_goal_analysis import (
    analyze_pointmaze_cells,
    render_report,
)


class PointMazeGoalStageTwoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gymnasium  # noqa: F401
            import gymnasium_robotics  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("gymnasium-robotics is not installed")

    def test_goal_features_keep_final_goal_out_of_lower_state(self):
        base = {
            "observation": np.asarray([0.1, -0.2, 0.3, -0.4]),
            "achieved_goal": np.asarray([0.1, -0.2]),
            "desired_goal": np.asarray([1.0, 1.0]),
        }
        changed = dict(base)
        changed["desired_goal"] = np.asarray([-1.0, -1.0])
        subgoal = np.asarray([0.4, 0.5])
        self.assertFalse(np.array_equal(flat_goal_state(base), flat_goal_state(changed)))
        np.testing.assert_array_equal(
            lower_goal_state(base, subgoal),
            lower_goal_state(changed, subgoal),
        )

    def test_pointmaze_contract_and_bounds_are_physical(self):
        dimensions = pointmaze_dimensions(env_id=DEFAULT_ENV_ID, horizon=32)
        self.assertEqual(dimensions.physical, 4)
        self.assertEqual(dimensions.goal, 2)
        self.assertEqual(dimensions.action, 2)
        self.assertEqual(dimensions.flat, 6)
        environment = make_pointmaze_environment(
            env_id=DEFAULT_ENV_ID,
            horizon=32,
        )
        try:
            low, high = pointmaze_goal_bounds(environment)
            self.assertAlmostEqual(environment.unwrapped.point_env.dt, 0.01)
            self.assertTrue(environment.unwrapped.continuing_task)
            goal = np.zeros(2, dtype=np.float32)
            self.assertEqual(
                float(environment.unwrapped.compute_reward(goal, goal, {})),
                1.0,
            )
            self.assertFalse(
                environment.unwrapped.compute_terminated(goal, goal, {})
            )
        finally:
            environment.close()
        np.testing.assert_allclose(low, [-1.5, -1.5])
        np.testing.assert_allclose(high, [1.5, 1.5])

    def test_squashed_action_respects_asymmetric_box(self):
        action = squash_box_action(
            np.asarray([100.0, -100.0]),
            np.asarray([-2.0, 1.0]),
            np.asarray([4.0, 3.0]),
        )
        np.testing.assert_allclose(action, [4.0, 1.0], atol=1e-6)

    def test_checkpoint_rank_prioritizes_success_over_dense_return(self):
        more_success = pointmaze_checkpoint_rank([
            {"success": 1.0, "episode_return": 1.0},
            {"success": 0.0, "episode_return": 1.0},
        ])
        more_return = pointmaze_checkpoint_rank([
            {"success": 0.0, "episode_return": 100.0},
            {"success": 0.0, "episode_return": 100.0},
        ])
        self.assertGreater(more_success, more_return)

    def test_runtime_versions_cover_the_physics_stack(self):
        versions = pointmaze_runtime_versions()
        self.assertEqual(
            set(versions),
            {
                "python",
                "numpy",
                "torch",
                "gymnasium",
                "gymnasium_robotics",
                "mujoco",
                "pettingzoo",
                "scipy",
            },
        )
        self.assertTrue(all(versions.values()))

    def test_flat_and_hrl_pointmaze_rollouts_update(self):
        dimensions = pointmaze_dimensions(env_id=DEFAULT_ENV_ID, horizon=32)
        flat, flat_capacity = build_pointmaze_model(
            method="flat_goal_ppo",
            dimensions=dimensions,
            reference_hidden_dim=16,
            learning_rate=3e-4,
            optimizer_seed=7,
        )
        hrl, hrl_capacity = build_pointmaze_model(
            method="hrl_goal_ppo",
            dimensions=dimensions,
            reference_hidden_dim=16,
            learning_rate=3e-4,
            optimizer_seed=11,
        )
        self.assertLess(
            abs(float(hrl_capacity["parameter_budget_ratio"]) - 1.0),
            0.08,
        )

        flat_batch, flat_row = rollout_flat_pointmaze(
            flat,
            env_id=DEFAULT_ENV_ID,
            seed=7,
            horizon=32,
            sample=True,
            parameter_budget=int(flat_capacity["reference_parameter_budget"]),
        )
        self.assertEqual(flat_row["protocol_valid"], 1.0)
        self.assertEqual(flat_row["episode_length"], 32)
        self.assertEqual(flat_batch.size, flat_row["episode_length"])
        flat_metrics = flat.update(flat_batch)
        self.assertTrue(np.isfinite(flat_metrics["loss"]))

        hrl_batch, hrl_row = rollout_hrl_pointmaze(
            hrl,
            env_id=DEFAULT_ENV_ID,
            seed=7,
            horizon=32,
            sample=True,
            parameter_budget=int(hrl_capacity["reference_parameter_budget"]),
            upper_period_steps=8,
            maximum_subgoal_delta=0.75,
        )
        self.assertEqual(hrl_row["protocol_valid"], 1.0)
        self.assertEqual(hrl_row["episode_length"], 32)
        self.assertEqual(hrl_batch.lower.size, hrl_row["episode_length"])
        self.assertEqual(hrl_batch.upper.size, hrl_row["upper_decision_count"])
        lower_boundaries = np.flatnonzero(hrl_batch.lower.done)
        self.assertEqual(
            len(lower_boundaries),
            hrl_row["lower_option_boundary_count"],
        )
        self.assertTrue(all(
            (int(index) + 1) % 8 == 0
            or int(index) == hrl_batch.lower.size - 1
            for index in lower_boundaries
        ))
        self.assertTrue(np.isfinite(hrl_row["lower_intrinsic_return"]))
        self.assertAlmostEqual(
            float(np.sum(hrl_batch.lower.reward)),
            float(hrl_row["lower_intrinsic_return"]),
            places=5,
        )
        hrl_metrics = hrl.update(hrl_batch)
        self.assertTrue(np.isfinite(hrl_metrics["upper_loss"]))
        self.assertTrue(np.isfinite(hrl_metrics["lower_loss"]))
        self.assertEqual(hrl_row["protocol_version"], POINTMAZE_GOAL_PROTOCOL_VERSION)

    def test_analysis_uses_optimizer_roots_as_statistical_units(self):
        cells = []
        for method in ("flat_goal_ppo", "hrl_goal_ppo"):
            for root in (101, 103, 107, 109):
                rows = []
                for seed in (1, 2, 3):
                    hrl = method == "hrl_goal_ppo"
                    rows.append({
                        "algorithm_path": "goal_conditioned_hrl_mainline",
                        "protocol_valid": 1.0,
                        "training_replicate_seed": root,
                        "seed": seed,
                        "success": 0.75 if hrl else 0.25,
                        "episode_return": 100.0 if hrl else 80.0,
                        "final_goal_distance": 0.2 if hrl else 0.8,
                    })
                cells.append({
                    "policy": method,
                    "optimizer_seed": root,
                    "runtime_versions": {"python": "test"},
                    "evaluation_rows": rows,
                })
        analysis = analyze_pointmaze_cells(cells)
        self.assertEqual(analysis["independent_training_replicate_count"], 4)
        self.assertEqual(analysis["ordinary_hrl_learning_status"], "supported")
        self.assertEqual(analysis["multiscale_admission_status"], "admitted")
        self.assertEqual(analysis["hrl_vs_flat"]["joint_status"], "supported")
        self.assertIn("multiscale factorial experiment is now admitted", render_report(analysis))

    def test_analysis_rejects_unpaired_heldout_seeds(self):
        cells = []
        for method, seeds in (
            ("flat_goal_ppo", (1, 2)),
            ("hrl_goal_ppo", (1, 3)),
        ):
            cells.append({
                "policy": method,
                "optimizer_seed": 101,
                "runtime_versions": {"python": "test"},
                "evaluation_rows": [{
                    "algorithm_path": "goal_conditioned_hrl_mainline",
                    "protocol_valid": 1.0,
                    "training_replicate_seed": 101,
                    "seed": seed,
                    "success": 0.0,
                    "episode_return": 0.0,
                    "final_goal_distance": 1.0,
                } for seed in seeds],
            })
        with self.assertRaisesRegex(ValueError, "held-out seeds"):
            analyze_pointmaze_cells(cells)

    def test_analysis_rejects_runtime_drift(self):
        cells = []
        for method, runtime in (
            ("flat_goal_ppo", {"mujoco": "3.2.7"}),
            ("hrl_goal_ppo", {"mujoco": "3.6.0"}),
        ):
            cells.append({
                "policy": method,
                "optimizer_seed": 101,
                "runtime_versions": runtime,
                "evaluation_rows": [{
                    "algorithm_path": "goal_conditioned_hrl_mainline",
                    "protocol_valid": 1.0,
                    "training_replicate_seed": 101,
                    "seed": 1,
                    "success": 0.0,
                    "episode_return": 0.0,
                    "final_goal_distance": 1.0,
                }],
            })
        with self.assertRaisesRegex(ValueError, "runtime versions"):
            analyze_pointmaze_cells(cells)


if __name__ == "__main__":
    unittest.main()
