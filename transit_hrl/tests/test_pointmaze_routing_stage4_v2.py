import unittest

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments.pointmaze_goal_validation import DEFAULT_ENV_ID
from freq_hrl.experiments.pointmaze_multiscale_validation import (
    PointMazeFeatureBuilder,
    pointmaze_multiscale_dimensions,
    rollout_hrl_pointmaze_multiscale,
)
from freq_hrl.experiments.pointmaze_routing_analysis import (
    analyze_pointmaze_routing_cells,
)
from freq_hrl.experiments.pointmaze_routing_masked_attribution import (
    POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH,
    POINTMAZE_MASKED_ROUTING_METHODS,
    POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION,
    POINTMAZE_MASKED_ROUTING_REPRESENTATIONS,
    POINTMAZE_MASKED_ROUTING_SHAPE_CONTRACT,
    build_pointmaze_masked_routing_model,
)


class PointMazeMaskedRoutingStageFourTest(unittest.TestCase):
    def setUp(self):
        self.time_scale = PhysicalTimeScaleContract(
            dt_seconds=0.01,
            upper_period_seconds=0.25,
            history_seconds=0.32,
            fast_period_seconds=0.04,
        )
        representations = tuple(dict.fromkeys(
            ("history", *POINTMAZE_MASKED_ROUTING_REPRESENTATIONS.values())
        ))
        self.dimensions = pointmaze_multiscale_dimensions(
            env_id=DEFAULT_ENV_ID,
            horizon=64,
            time_scale=self.time_scale,
            representations=representations,
        )

    @staticmethod
    def _observation(step):
        physical = np.asarray([
            np.sin(step / 3.0),
            np.cos(step / 5.0),
            step / 31.0,
            (-1.0) ** step * step / 17.0,
        ], dtype=np.float32)
        return {
            "observation": physical,
            "achieved_goal": physical[:2],
            "desired_goal": np.asarray([0.75, -0.25], dtype=np.float32),
        }

    def test_masks_preserve_shapes_and_select_registered_coefficients(self):
        dimensions = list(self.dimensions.values())
        self.assertEqual(len({item.flat for item in dimensions}), 1)
        self.assertEqual(len({item.upper for item in dimensions}), 1)
        self.assertEqual(len({item.lower for item in dimensions}), 1)

        builder = PointMazeFeatureBuilder(
            physical_dim=4,
            time_scale=self.time_scale,
        )
        current = self._observation(0)
        builder.reset(current)
        for step in range(1, 32):
            current = self._observation(step)
            builder.update(current)
        snapshot = builder.snapshot
        np.testing.assert_allclose(
            snapshot.multiscale,
            np.concatenate((snapshot.slow, snapshot.mid, snapshot.high)),
        )
        zero_slow = np.zeros_like(snapshot.slow)
        zero_high = np.zeros_like(snapshot.high)
        prefix = 6
        subgoal = np.asarray([0.4, -0.1], dtype=np.float32)

        routed_upper = builder.upper_state(
            current, representation="multiscale_routed_masked"
        )[prefix:]
        routed_lower = builder.lower_state(
            current,
            subgoal=subgoal,
            representation="multiscale_routed_masked",
        )[prefix:]
        swapped_upper = builder.upper_state(
            current, representation="multiscale_swapped_masked"
        )[prefix:]
        swapped_lower = builder.lower_state(
            current,
            subgoal=subgoal,
            representation="multiscale_swapped_masked",
        )[prefix:]

        np.testing.assert_allclose(
            routed_upper,
            np.concatenate((snapshot.slow, snapshot.mid, zero_high)),
        )
        np.testing.assert_allclose(
            routed_lower,
            np.concatenate((zero_slow, snapshot.mid, snapshot.high)),
        )
        np.testing.assert_allclose(
            swapped_upper,
            np.concatenate((zero_slow, snapshot.mid, snapshot.high)),
        )
        np.testing.assert_allclose(
            swapped_lower,
            np.concatenate((snapshot.slow, snapshot.mid, zero_high)),
        )

    def test_all_methods_have_identical_models_and_initialization(self):
        models = []
        capacities = []
        for method in POINTMAZE_MASKED_ROUTING_METHODS:
            model, capacity = build_pointmaze_masked_routing_model(
                method=method,
                dimensions=self.dimensions,
                reference_hidden_dim=128,
                learning_rate=3e-4,
                optimizer_seed=811,
            )
            models.append(model)
            capacities.append(capacity)
        self.assertEqual(
            len({item["actual_parameter_count"] for item in capacities}), 1
        )
        self.assertEqual(len({item["hidden_dim"] for item in capacities}), 1)
        for module_name in (
            "upper_actor",
            "upper_value",
            "lower_actor",
            "lower_value",
        ):
            reference = list(getattr(models[0], module_name).parameters())
            for model in models[1:]:
                candidate = list(getattr(model, module_name).parameters())
                self.assertEqual(len(reference), len(candidate))
                for left, right in zip(reference, candidate):
                    np.testing.assert_array_equal(
                        left.detach().cpu().numpy(),
                        right.detach().cpu().numpy(),
                    )

    def test_all_methods_execute_the_masked_protocol(self):
        for index, method in enumerate(POINTMAZE_MASKED_ROUTING_METHODS):
            model, capacity = build_pointmaze_masked_routing_model(
                method=method,
                dimensions=self.dimensions,
                reference_hidden_dim=128,
                learning_rate=3e-4,
                optimizer_seed=901 + index,
            )
            _, row = rollout_hrl_pointmaze_multiscale(
                model,
                method=method,
                scenario="fast_observation_noise",
                env_id=DEFAULT_ENV_ID,
                seed=61,
                horizon=64,
                sample=True,
                parameter_budget=capacity["reference_parameter_budget"],
                time_scale=self.time_scale,
                maximum_subgoal_delta=0.75,
                representation_override=(
                    POINTMAZE_MASKED_ROUTING_REPRESENTATIONS[method]
                ),
                protocol_version=POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION,
                algorithm_path=POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH,
            )
            self.assertEqual(row["protocol_valid"], 1.0)
            self.assertEqual(
                row["protocol_version"],
                POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION,
            )
            self.assertEqual(
                row["algorithm_path"],
                POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH,
            )
            self.assertEqual(row["upper_decision_count"], 3)
            self.assertEqual(row["lower_option_boundary_count"], 3)
            self.assertEqual(
                POINTMAZE_MASKED_ROUTING_SHAPE_CONTRACT,
                "identical_upper_lower_state_shapes_parameters_and_initialization_per_root",
            )

    def test_analysis_accepts_only_the_masked_protocol_identity(self):
        success = {
            "clean": (0.60, 0.62, 0.70, 0.72, 0.64),
            "fast_observation_noise": (0.30, 0.45, 0.50, 0.80, 0.40),
            "slow_drift_fast_action": (0.35, 0.40, 0.50, 0.75, 0.45),
        }
        cells = []
        for scenario, values in success.items():
            for method, value in zip(
                POINTMAZE_MASKED_ROUTING_METHODS, values
            ):
                for root in (201, 203, 207, 209):
                    cells.append({
                        "protocol_version": (
                            POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION
                        ),
                        "policy": method,
                        "scenario": scenario,
                        "optimizer_seed": root,
                        "runtime_versions": {"python": "test"},
                        "evaluation_rows": [{
                            "protocol_valid": 1.0,
                            "algorithm_path": (
                                POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH
                            ),
                            "scenario": scenario,
                            "method": method,
                            "training_replicate_seed": root,
                            "seed": root + 10_000,
                            "success": value,
                            "episode_return": 100.0 * value,
                            "final_goal_distance": 2.0 - value,
                        }],
                    })
        analysis = analyze_pointmaze_routing_cells(
            cells,
            protocol_version=POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION,
            algorithm_path=POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH,
            analysis_version="test_masked_analysis",
        )
        self.assertEqual(analysis["routing_attribution_status"], "supported")
        self.assertEqual(analysis["analysis_version"], "test_masked_analysis")
        cells[0]["evaluation_rows"][0]["algorithm_path"] = "wrong_path"
        with self.assertRaises(ValueError):
            analyze_pointmaze_routing_cells(
                cells,
                protocol_version=POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION,
                algorithm_path=POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH,
            )


if __name__ == "__main__":
    unittest.main()
