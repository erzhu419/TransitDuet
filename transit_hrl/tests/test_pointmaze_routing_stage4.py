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
    render_report,
)
from freq_hrl.experiments.pointmaze_routing_attribution import (
    POINTMAZE_ROUTING_ALGORITHM_PATH,
    POINTMAZE_ROUTING_METHODS,
    POINTMAZE_ROUTING_PROTOCOL_VERSION,
    POINTMAZE_ROUTING_REPRESENTATIONS,
    POINTMAZE_ROUTING_SCENARIOS,
    build_pointmaze_routing_model,
)


class PointMazeRoutingStageFourTest(unittest.TestCase):
    def setUp(self):
        self.time_scale = PhysicalTimeScaleContract(
            dt_seconds=0.01,
            upper_period_seconds=0.25,
            history_seconds=0.32,
            fast_period_seconds=0.04,
        )
        representations = tuple(dict.fromkeys(
            ("history", *POINTMAZE_ROUTING_REPRESENTATIONS.values())
        ))
        self.dimensions = pointmaze_multiscale_dimensions(
            env_id=DEFAULT_ENV_ID,
            horizon=64,
            time_scale=self.time_scale,
            representations=representations,
        )

    def test_five_routes_are_capacity_matched_and_trainable(self):
        for index, method in enumerate(POINTMAZE_ROUTING_METHODS):
            model, capacity = build_pointmaze_routing_model(
                method=method,
                dimensions=self.dimensions,
                reference_hidden_dim=128,
                learning_rate=3e-4,
                optimizer_seed=701 + index,
            )
            self.assertGreaterEqual(capacity["parameter_budget_ratio"], 0.99)
            self.assertLessEqual(capacity["parameter_budget_ratio"], 1.01)
            batch, row = rollout_hrl_pointmaze_multiscale(
                model,
                method=method,
                scenario="fast_observation_noise",
                env_id=DEFAULT_ENV_ID,
                seed=41,
                horizon=64,
                sample=True,
                parameter_budget=capacity["reference_parameter_budget"],
                time_scale=self.time_scale,
                maximum_subgoal_delta=0.75,
                representation_override=POINTMAZE_ROUTING_REPRESENTATIONS[method],
                protocol_version=POINTMAZE_ROUTING_PROTOCOL_VERSION,
                algorithm_path=POINTMAZE_ROUTING_ALGORITHM_PATH,
            )
            self.assertIsNotNone(batch)
            self.assertEqual(row["protocol_valid"], 1.0)
            self.assertEqual(row["protocol_version"], POINTMAZE_ROUTING_PROTOCOL_VERSION)
            self.assertEqual(row["algorithm_path"], POINTMAZE_ROUTING_ALGORITHM_PATH)
            self.assertEqual(row["representation"], POINTMAZE_ROUTING_REPRESENTATIONS[method])
            self.assertEqual(row["upper_decision_count"], 3)
            self.assertEqual(row["lower_option_boundary_count"], 3)
            self.assertTrue(np.all(np.isfinite(batch.upper.reward)))
            self.assertTrue(np.all(np.isfinite(batch.lower.reward)))

    def test_routed_and_swapped_labels_select_the_registered_bands(self):
        builder = PointMazeFeatureBuilder(
            physical_dim=4,
            time_scale=self.time_scale,
        )

        def observation(step):
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

        current = observation(0)
        builder.reset(current)
        for step in range(1, 32):
            current = observation(step)
            builder.update(current)

        snapshot = builder.snapshot
        prefix = 6
        subgoal = np.asarray([0.4, -0.1], dtype=np.float32)
        routed_upper = builder.upper_state(
            current, representation="multiscale_routed"
        )
        routed_lower = builder.lower_state(
            current,
            subgoal=subgoal,
            representation="multiscale_routed",
        )
        swapped_upper = builder.upper_state(
            current, representation="multiscale_swapped"
        )
        swapped_lower = builder.lower_state(
            current,
            subgoal=subgoal,
            representation="multiscale_swapped",
        )

        np.testing.assert_allclose(
            routed_upper[prefix:],
            np.concatenate((snapshot.slow, snapshot.mid)),
        )
        np.testing.assert_allclose(
            routed_lower[prefix:],
            np.concatenate((snapshot.mid, snapshot.high)),
        )
        np.testing.assert_allclose(
            swapped_upper[prefix:],
            np.concatenate((snapshot.mid, snapshot.high)),
        )
        np.testing.assert_allclose(
            swapped_lower[prefix:],
            np.concatenate((snapshot.slow, snapshot.mid)),
        )

    @staticmethod
    def _synthetic_cells():
        success = {
            "clean": {
                "hrl_history": 0.60,
                "hrl_causal_filter": 0.65,
                "hrl_multiscale_all": 0.70,
                "hrl_multiscale_routed": 0.72,
                "hrl_multiscale_swapped": 0.62,
            },
            "fast_observation_noise": {
                "hrl_history": 0.30,
                "hrl_causal_filter": 0.45,
                "hrl_multiscale_all": 0.50,
                "hrl_multiscale_routed": 0.80,
                "hrl_multiscale_swapped": 0.40,
            },
            "slow_drift_fast_action": {
                "hrl_history": 0.35,
                "hrl_causal_filter": 0.40,
                "hrl_multiscale_all": 0.50,
                "hrl_multiscale_routed": 0.75,
                "hrl_multiscale_swapped": 0.45,
            },
        }
        cells = []
        for scenario in POINTMAZE_ROUTING_SCENARIOS:
            for method in POINTMAZE_ROUTING_METHODS:
                for root in (101, 103, 107, 109):
                    value = success[scenario][method]
                    cells.append({
                        "protocol_version": POINTMAZE_ROUTING_PROTOCOL_VERSION,
                        "policy": method,
                        "scenario": scenario,
                        "optimizer_seed": root,
                        "runtime_versions": {"python": "test"},
                        "evaluation_rows": [{
                            "protocol_valid": 1.0,
                            "algorithm_path": POINTMAZE_ROUTING_ALGORITHM_PATH,
                            "scenario": scenario,
                            "method": method,
                            "training_replicate_seed": root,
                            "seed": root + 1_000,
                            "success": value,
                            "episode_return": 100.0 * value,
                            "final_goal_distance": 2.0 - value,
                        }],
                    })
        return cells

    def test_analysis_enforces_selective_routing_gate(self):
        analysis = analyze_pointmaze_routing_cells(self._synthetic_cells())
        self.assertEqual(analysis["independent_training_replicate_count"], 4)
        self.assertEqual(analysis["routing_attribution_status"], "supported")
        self.assertIn("Routed vs all bands", render_report(analysis))

    def test_analysis_rejects_incomplete_routing_matrix(self):
        with self.assertRaises(ValueError):
            analyze_pointmaze_routing_cells(self._synthetic_cells()[:-1])


if __name__ == "__main__":
    unittest.main()
