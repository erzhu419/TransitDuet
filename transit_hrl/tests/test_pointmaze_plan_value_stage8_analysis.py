import copy
import unittest

from freq_hrl.experiments.pointmaze_plan_value_qualification import (
    POINTMAZE_PLAN_VALUE_ALGORITHM_PATH,
    POINTMAZE_PLAN_VALUE_METHODS,
    POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION,
    POINTMAZE_PLAN_VALUE_SCHEDULES,
)
from scripts.analyze_pointmaze_plan_value_stage8 import analyze_stage8


def _loss(method: str, schedule: str) -> float:
    if method == "hrl_regime_history":
        return {
            "fixed": 10.0,
            "stale_plan": 15.0,
            "fixed_waypoint_perturbed": 13.0,
            "oracle_event_delay_000ms": 9.0,
            "oracle_event_delay_100ms": 10.0,
            "oracle_event_delay_250ms": 11.0,
            "oracle_event_delay_500ms": 13.0,
        }[schedule]
    return {
        "fixed": 8.0,
        "stale_plan": 14.0,
        "fixed_waypoint_perturbed": 12.0,
        "oracle_event_delay_000ms": 5.0,
        "oracle_event_delay_100ms": 6.0,
        "oracle_event_delay_250ms": 9.0,
        "oracle_event_delay_500ms": 12.0,
    }[schedule]


def _row(
    *,
    method: str,
    schedule: str,
    root: int,
    seed: int,
    loss: float,
) -> dict:
    oracle_method = method == "hrl_regime_oracle_context"
    oracle_schedule = schedule.startswith("oracle_event_")
    return {
        "protocol_version": POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_PLAN_VALUE_ALGORITHM_PATH,
        "method": method,
        "schedule_mode": schedule,
        "seed": seed,
        "training_replicate_seed": root,
        "protocol_valid": 1.0,
        "policy_has_current_regime_access": oracle_method,
        "policy_has_future_regime_access": False,
        "schedule_has_regime_event_access": oracle_schedule,
        "schedule_has_future_regime_access": oracle_schedule,
        "regime_label_visible_to_candidate": False,
        "external_future_visible_to_actor": False,
        "external_stream_action_independent": True,
        "planning_budget_matched": 0.0 if schedule == "stale_plan" else 1.0,
        "upper_decision_count": 1 if schedule == "stale_plan" else 24,
        "fixed_budget_upper_decision_count": 24,
        "tracking_squared_error_integral": loss,
        "episode_return": 300.0 - loss,
        "tracking_success_rate": 1.0 - loss / 100.0,
        "event_post_tracking_mse": loss / 10.0,
        "event_recovery_seconds_mean": loss / 20.0,
        "causal_distinguishability_delay_seconds_max": 0.02,
        "target_start_vertex": seed % 7,
        "target_speed_modes_world_per_second": [-1.25, -0.55, 0.55, 1.25],
        "regime_change_steps": [118, 202, 300],
        "force_pulse_start_steps": [47, 159],
        "distractor_change_steps": [31, 89, 177],
    }


def _cells() -> list[dict]:
    cells = []
    for root in (101, 103, 107, 109):
        for method in POINTMAZE_PLAN_VALUE_METHODS:
            rows = []
            untrained = []
            for seed in (11, 13, 17):
                offset = (root % 7) * 0.01 + (seed % 5) * 0.001
                for schedule in POINTMAZE_PLAN_VALUE_SCHEDULES:
                    rows.append(_row(
                        method=method,
                        schedule=schedule,
                        root=root,
                        seed=seed,
                        loss=_loss(method, schedule) + offset,
                    ))
                untrained.append(_row(
                    method=method,
                    schedule="fixed",
                    root=root,
                    seed=seed,
                    loss=20.0 + offset,
                ))
            oracle = method == "hrl_regime_oracle_context"
            cells.append({
                "policy": method,
                "optimizer_seed": root,
                "protocol_version": POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION,
                "algorithm_path": POINTMAZE_PLAN_VALUE_ALGORITHM_PATH,
                "evidence_role": "task_qualification_development",
                "belief_training": "disabled_not_yet_authorized",
                "trigger_training": "disabled_not_yet_authorized",
                "dimensions": {
                    "base_upper": 198,
                    "oracle_upper": 202,
                    "lower": 198,
                },
                "config": {
                    "upper_state_dim": 202 if oracle else 198,
                    "lower_state_dim": 198,
                },
                "capacity": {
                    "reference_parameter_budget": 1000,
                    "actual_parameter_count": 990 if oracle else 1000,
                    "parameter_budget_ratio": 0.99 if oracle else 1.0,
                },
                "runtime_versions": {"python": "test"},
                "train_seeds": [1, 2],
                "selection_seeds": [3, 4],
                "eval_seeds": [11, 13, 17],
                "evaluation_rows": rows,
                "untrained_evaluation_rows": untrained,
            })
    return cells


class PointMazePlanValueStageEightAnalysisTest(unittest.TestCase):
    def test_all_qualification_components_are_separate_and_supported(self):
        analysis = analyze_stage8(_cells())
        self.assertTrue(analysis["stage9_authorized"])
        self.assertEqual(analysis["decision"], "task_qualified_for_stage9")
        self.assertTrue(all(analysis["qualification_checks"].values()))
        for name in (
            "plan_refresh_vs_stale",
            "plan_integrity_vs_perturbed",
            "current_regime_information",
            "oracle_timing_same_budget",
            "oracle_timing_250ms_delay_cost",
        ):
            self.assertEqual(
                analysis["contrasts"][name][
                    "tracking_squared_error_integral"
                ]["status"],
                "supported",
            )

    def test_budget_mismatch_is_rejected(self):
        cells = copy.deepcopy(_cells())
        row = next(
            row
            for row in cells[0]["evaluation_rows"]
            if row["schedule_mode"] == "oracle_event_delay_000ms"
        )
        row["upper_decision_count"] = 25
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            analyze_stage8(cells)


if __name__ == "__main__":
    unittest.main()
