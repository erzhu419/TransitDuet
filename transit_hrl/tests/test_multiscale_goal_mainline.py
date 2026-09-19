import unittest

import numpy as np
import torch

from freq_hrl.core import (
    CausalHaarMultiscaleEncoder,
    PhysicalTimeScaleContract,
)
from freq_hrl.domains.mujoco import (
    GOAL_CONTROL_MAINLINE_CONTRACT,
    RelativeSubgoalAdapter,
    goal_environment_contract,
    parse_goal_observation,
)
from freq_hrl.domains.tracking import (
    MultiTimescaleTrackingEnv,
    TRACKING_SCENARIOS,
    tracking_scenario,
)
from freq_hrl.experiments.multiscale_tracking_validation import (
    STAGE1_METHODS,
    TrackingFeatureBuilder,
    build_stage1_model,
    rollout_flat_tracking,
    rollout_hierarchical_tracking,
    tracking_feature_dimensions,
)
from freq_hrl.experiments.multiscale_tracking_analysis import (
    analyze_stage1_cells,
)


def small_time_scale() -> PhysicalTimeScaleContract:
    return PhysicalTimeScaleContract(
        dt_seconds=0.05,
        upper_period_seconds=0.4,
        history_seconds=0.8,
        fast_period_seconds=0.1,
    )


class PhysicalMultiscaleContractTest(unittest.TestCase):
    def test_time_scales_resolve_to_physical_steps(self):
        contract = small_time_scale()
        self.assertEqual(contract.upper_period_steps, 8)
        self.assertEqual(contract.history_steps, 16)
        self.assertEqual(contract.fast_period_steps, 2)
        metadata = contract.metadata(response_seconds=2.0)
        self.assertAlmostEqual(metadata["upper_period_over_response"], 0.2)
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            PhysicalTimeScaleContract(
                dt_seconds=0.05,
                upper_period_seconds=0.4,
                history_seconds=0.9,
                fast_period_seconds=0.1,
            )

    def test_haar_and_history_use_exactly_the_same_samples(self):
        contract = small_time_scale()
        encoder = CausalHaarMultiscaleEncoder(
            feature_dim=2,
            time_scale=contract,
        )
        encoder.reset(np.asarray([0.0, 1.0]))
        for step in range(1, contract.history_steps):
            snapshot = encoder.update(np.asarray([step, -step], dtype=np.float32))
        self.assertEqual(snapshot.history.size, snapshot.multiscale.size)
        self.assertEqual(snapshot.history.size, snapshot.filtered.size)
        self.assertAlmostEqual(
            float(np.sum(np.square(snapshot.history))),
            float(np.sum(np.square(snapshot.multiscale))),
            places=3,
        )
        self.assertGreater(snapshot.slow.size, 0)
        self.assertGreater(snapshot.mid.size, 0)
        self.assertGreater(snapshot.high.size, 0)
        self.assertEqual(snapshot.slow_energy.shape, (2,))

    def test_encoder_is_prefix_causal(self):
        left = CausalHaarMultiscaleEncoder(
            feature_dim=1,
            time_scale=small_time_scale(),
        )
        right = CausalHaarMultiscaleEncoder(
            feature_dim=1,
            time_scale=small_time_scale(),
        )
        left.reset(np.asarray([0.0]))
        right.reset(np.asarray([0.0]))
        for value in (1.0, -0.5, 0.25):
            left_snapshot = left.update(np.asarray([value]))
            right_snapshot = right.update(np.asarray([value]))
        np.testing.assert_array_equal(
            left_snapshot.multiscale,
            right_snapshot.multiscale,
        )
        left.update(np.asarray([100.0]))
        right.update(np.asarray([-100.0]))
        np.testing.assert_array_equal(
            left_snapshot.multiscale,
            right_snapshot.multiscale,
        )


class GoalSemanticsTest(unittest.TestCase):
    def test_goal_observation_and_subgoal_are_not_torque_relabeling(self):
        parsed = parse_goal_observation({
            "observation": np.asarray([1.0, -0.5]),
            "achieved_goal": np.asarray([0.25, 0.5]),
            "desired_goal": np.asarray([1.25, -0.5]),
        })
        np.testing.assert_allclose(parsed.goal_error, [1.0, -1.0])
        adapter = RelativeSubgoalAdapter(
            maximum_delta=np.asarray([2.0, 1.0]),
            action_cost=0.1,
        )
        subgoal = adapter.decode(
            np.asarray([0.0, np.arctanh(0.5)]),
            parsed.achieved_goal,
        )
        np.testing.assert_allclose(subgoal, [0.25, 1.0], atol=1e-6)
        reward = adapter.intrinsic_reward(
            achieved_before=np.asarray([0.25, 0.5]),
            achieved_after=np.asarray([0.25, 0.75]),
            subgoal=subgoal,
            action=np.asarray([0.0, 0.0]),
        )
        self.assertGreater(reward, 0.0)

    def test_pointmaze_goal_contract_when_robotics_is_available(self):
        try:
            import gymnasium as gym
            import gymnasium_robotics
        except ImportError:
            self.skipTest("gymnasium-robotics is not installed")
        gym.register_envs(gymnasium_robotics)
        environment = gym.make("PointMaze_UMaze-v3", max_episode_steps=8)
        try:
            contract = goal_environment_contract(environment)
        finally:
            environment.close()
        self.assertEqual(contract["contract"], GOAL_CONTROL_MAINLINE_CONTRACT)
        self.assertEqual(contract["goal_dim"], 2)
        self.assertEqual(contract["action_dim"], 2)
        self.assertAlmostEqual(contract["env_dt_seconds"], 0.01)
        self.assertEqual(contract["timing_source"], "point_env")


class StageOneProtocolTest(unittest.TestCase):
    @staticmethod
    def _synthetic_cells():
        effects = {
            "flat_history": (0.0, 0.0),
            "flat_multiscale": (1.0, -1.0),
            "hrl_history": (0.5, -0.5),
            "hrl_multiscale": (3.0, -3.0),
            "flat_causal_filter": (0.2, -0.2),
        }
        cells = []
        for scenario in TRACKING_SCENARIOS:
            for method, (reward_gain, error_gain) in effects.items():
                rows = []
                for seed in (1, 2, 3, 4):
                    rows.append({
                        "seed": seed,
                        "algorithm_path": "multiscale_goal_hrl_mainline",
                        "protocol_valid": 1.0,
                        "episode_return": 100.0 + seed + reward_gain,
                        "tracking_rmse": 10.0 + 0.1 * seed + error_gain,
                    })
                cells.append({
                    "policy": method,
                    "scenario": scenario,
                    "evaluation_rows": rows,
                })
        return cells

    def test_scenarios_separate_truth_location_and_match_rms(self):
        contract = small_time_scale()
        target_rms = []
        for index, name in enumerate(TRACKING_SCENARIOS):
            environment = MultiTimescaleTrackingEnv(
                time_scale=contract,
                scenario=tracking_scenario(name, time_scale=contract),
                horizon=64,
                seed=100 + index,
            )
            observation = environment.reset()
            diagnostics = environment.signal_diagnostics()
            target_rms.append(diagnostics["target_rms"])
            self.assertEqual(observation.task_measurement.shape, (1,))
            self.assertFalse(
                environment.observability_contract[
                    "dynamics_force_available_before_action"
                ]
            )
            self.assertFalse(
                environment.observability_contract[
                    "measurement_noise_truth_available_to_actor"
                ]
            )
        np.testing.assert_allclose(target_rms, 0.75, atol=1e-12)

    def test_four_grid_has_information_and_capacity_contracts(self):
        contract = small_time_scale()
        dimensions = tracking_feature_dimensions(time_scale=contract)
        self.assertEqual(
            dimensions["history"].flat,
            dimensions["multiscale"].flat,
        )
        self.assertEqual(
            dimensions["history"].flat,
            dimensions["filtered"].flat,
        )
        self.assertEqual(dimensions["history"].raw_history, 16)
        for index, method in enumerate(STAGE1_METHODS):
            model, capacity = build_stage1_model(
                method=method,
                dimensions=dimensions,
                reference_hidden_dim=16,
                learning_rate=3e-4,
                optimizer_seed=200 + index,
            )
            self.assertLess(
                abs(float(capacity["parameter_budget_ratio"]) - 1.0),
                0.08,
            )
            if method.startswith("hrl_"):
                contract_row = model.mainline_contract()
                self.assertIsNone(model.lower_cost_value)
                self.assertEqual(contract_row["upper_output"], "state_space_goal")
                self.assertEqual(contract_row["lower_output"], "physical_actuator_action")
                self.assertEqual(contract_row["projector"], "disabled")

    def test_flat_and_goal_conditioned_rollouts_update(self):
        torch.manual_seed(7)
        np.random.seed(7)
        contract = small_time_scale()
        dimensions = tracking_feature_dimensions(time_scale=contract)
        flat_model, flat_capacity = build_stage1_model(
            method="flat_history",
            dimensions=dimensions,
            reference_hidden_dim=8,
            learning_rate=3e-4,
            optimizer_seed=7,
        )
        flat_batch, flat_row = rollout_flat_tracking(
            flat_model,
            method="flat_history",
            scenario="clean",
            seed=11,
            horizon=16,
            time_scale=contract,
            sample=True,
            parameter_budget=int(flat_capacity["reference_parameter_budget"]),
        )
        self.assertEqual(flat_row["protocol_valid"], 1.0)
        self.assertEqual(flat_batch.size, 16)
        flat_metrics = flat_model.update(flat_batch)
        self.assertTrue(np.isfinite(flat_metrics["loss"]))

        hrl_model, hrl_capacity = build_stage1_model(
            method="hrl_multiscale",
            dimensions=dimensions,
            reference_hidden_dim=8,
            learning_rate=3e-4,
            optimizer_seed=13,
        )
        hrl_batch, hrl_row = rollout_hierarchical_tracking(
            hrl_model,
            method="hrl_multiscale",
            scenario="slow_signal_fast_observation_noise",
            seed=17,
            horizon=16,
            time_scale=contract,
            sample=True,
            parameter_budget=int(hrl_capacity["reference_parameter_budget"]),
        )
        self.assertEqual(hrl_row["protocol_valid"], 1.0)
        self.assertEqual(hrl_batch.upper.size, 2)
        self.assertEqual(hrl_batch.lower.size, 16)
        self.assertEqual(hrl_batch.upper.action.shape[1], 1)
        self.assertEqual(hrl_batch.lower.action.shape[1], 1)
        hrl_metrics = hrl_model.update(hrl_batch)
        self.assertTrue(np.isfinite(hrl_metrics["upper_loss"]))
        self.assertTrue(np.isfinite(hrl_metrics["lower_loss"]))

    def test_factorial_analysis_uses_paired_seed_differences(self):
        analysis = analyze_stage1_cells(self._synthetic_cells())
        self.assertEqual(analysis["mainline_hrl_increment_status"], "supported")
        clean = analysis["scenarios"]["clean"]
        self.assertEqual(clean["representation_flat"]["joint_status"], "supported")
        self.assertEqual(
            clean["factorial_interaction"]["episode_return"]["status"],
            "supported",
        )
        self.assertIn("multiscale_vs_causal_filter", clean)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze_stage1_cells(self._synthetic_cells()[:-2])


if __name__ == "__main__":
    unittest.main()
