import unittest

from freq_hrl.experiments.pointmaze_exogenous_routing_attribution import (
    POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH,
    POINTMAZE_EXOGENOUS_ROUTING_METHOD_SPECS,
    POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
    POINTMAZE_EXOGENOUS_ROUTING_SHAPE_CONTRACT,
)
from scripts.analyze_pointmaze_exogenous_multiscale_stage7 import analyze_stage7
from scripts.pointmaze_exogenous_multiscale_stage7_spec import METHODS


FINAL_SUCCESS = {
    "flat_exogenous_history": 0.40,
    "flat_exogenous_multiscale_all": 0.45,
    "hrl_exogenous_history": 0.50,
    "hrl_exogenous_multiscale_all": 0.80,
}


def _row(method, root, seed, success, episode_return):
    return {
        "protocol_version": POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH,
        "method": method,
        "training_replicate_seed": root,
        "seed": seed,
        "protocol_valid": 1.0,
        "external_stream_action_independent": True,
        "external_stream_visible_before_action": True,
        "current_physical_state_visible_to_both_levels": True,
        "external_future_visible_to_actor": False,
        "tracking_success_rate": success,
        "episode_return": episode_return,
        "tracking_rmse": 1.0 - success,
        "final_tracking_distance": 1.0 - success,
        "target_start_vertex": 1,
        "target_initial_direction": 1,
        "target_route": "perimeter",
        "target_round_trip_period_seconds": 4.0,
        "force_x_period_seconds": 0.04,
        "force_y_period_seconds": 0.04,
        "force_x_rms": 0.12,
        "force_y_rms": 0.12,
    }


def _cells():
    cells = []
    for root_index, root in enumerate((1, 2, 3, 4)):
        seeds = (10_000 + root_index * 10, 10_001 + root_index * 10)
        root_offset = (-0.01, 0.0, 0.005, 0.01)[root_index]
        for method in METHODS:
            architecture, representation = POINTMAZE_EXOGENOUS_ROUTING_METHOD_SPECS[
                method
            ]
            success = FINAL_SUCCESS[method] + root_offset
            cells.append({
                "policy": method,
                "optimizer_seed": root,
                "protocol_version": POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
                "algorithm_path": POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH,
                "architecture": architecture,
                "representation": representation,
                "routing_shape_contract": POINTMAZE_EXOGENOUS_ROUTING_SHAPE_CONTRACT,
                "projector": "disabled",
                "promotion": "disabled",
                "leakage_loss": "disabled",
                "responsibility_gauge": "disabled",
                "dimensions": {"flat": 134, "upper": 134, "lower": 134},
                "capacity": {
                    "reference_parameter_budget": 67_973,
                    "actual_parameter_count": (
                        67_973 if architecture == "flat" else 68_424
                    ),
                    "hidden_dim": 128 if architecture == "flat" else 79,
                },
                "runtime_versions": {"python": "test", "torch": "test"},
                "train_seeds": [20_000 + root_index],
                "selection_seeds": [30_000 + root_index],
                "eval_seeds": list(seeds),
                "evaluation_rows": [
                    _row(method, root, seed, success, 100.0 * success)
                    for seed in seeds
                ],
                "untrained_evaluation_rows": [
                    _row(method, root, seed, 0.10 + root_offset, 5.0)
                    for seed in seeds
                ],
            })
    return cells


class PointMazeExogenousMultiscaleStageSevenAnalysisTest(unittest.TestCase):
    def test_fresh_factorial_gate_and_interaction(self):
        analysis = analyze_stage7(_cells())
        self.assertEqual(analysis["cell_count"], 16)
        self.assertEqual(analysis["independent_training_replicate_count"], 4)
        self.assertEqual(
            analysis["multiscale_hrl_confirmation_status"], "supported"
        )
        self.assertTrue(all(analysis["claim_gate_checks"].values()))
        interaction = analysis["hierarchy_x_multiscale_interaction"][
            "tracking_success_rate"
        ]
        self.assertAlmostEqual(
            interaction["mean_interaction_improvement"], 0.25
        )
        self.assertEqual(interaction["status"], "supported")

    def test_missing_factorial_cell_is_rejected(self):
        cells = _cells()
        cells.pop()
        with self.assertRaisesRegex(ValueError, "matrix is incomplete"):
            analyze_stage7(cells)


if __name__ == "__main__":
    unittest.main()
