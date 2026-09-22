import unittest

from freq_hrl.experiments.pointmaze_compact_plan_validity import (
    POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH,
    POINTMAZE_COMPACT_PLAN_VALIDITY_POLICY,
    POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import (
    BRANCH_CATEGORIES,
)
from scripts.analyze_pointmaze_compact_plan_validity_stage8c import (
    EXPECTED_FEATURE_COUNTS,
    analyze_stage8c,
)
from scripts.pointmaze_compact_plan_validity_stage8c_spec import (
    RIDGE_ALPHA_GRID,
)


VALUES = {
    "regime_lag_010ms": -0.10,
    "regime_lag_100ms": 0.30,
    "regime_lag_250ms": 1.00,
    "force_pulse_010ms": 0.05,
    "distractor_change_010ms": 0.00,
    "neutral_matched": 0.10,
}


def _branch_row(
    root: int,
    seed: int,
    category: str,
    step: int,
    *,
    split: str,
) -> dict:
    value = VALUES[category]
    neutral = category == "neutral_matched"
    source = None if neutral else step - 1
    row = {
        "protocol_version": POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH,
        "optimizer_seed": root,
        "split": split,
        "seed": seed,
        "category": category,
        "opportunity_step": step,
        "source_step": source,
        "lag_steps": None if neutral else 1,
        "protocol_valid": True,
        "candidate_feature_has_future_access": False,
        "candidate_feature_has_regime_label": False,
        "privileged_regime_context_present": False,
        "keep_upper_calls_at_opportunity": 0,
        "renew_upper_calls_at_opportunity": 1,
        "downstream_upper_call_count_per_branch": 0,
        "lower_controller_remains_closed_loop": True,
        "prefix_max_abs_difference": 0.0,
        "feature_max_abs_difference": 0.0,
        "causal_features": [0.0] * 37,
        "keep_tracking_squared_error_integral": 2.0,
        "renew_tracking_squared_error_integral": 2.0 - value,
        "renew_ise_advantage": value,
        "keep_primitive_steps_replayed": step + 50,
        "renew_primitive_steps_replayed": step + 50,
    }
    if split == "qualification_eval":
        row.update({
            "prediction_current_compact_quadratic": -value,
            "prediction_causal_dynamic_quadratic": -0.5 * value,
            "prediction_causal_validity_interactions": value,
        })
    return row


def _predictor(seed_count: int) -> dict:
    predictors = {}
    for name, count in EXPECTED_FEATURE_COUNTS.items():
        predictors[name] = {
            "feature_count": count,
            "feature_names": [f"{name}_feature_{index}" for index in range(count)],
            "model": {
                "alpha": float(RIDGE_ALPHA_GRID[0]),
                "alpha_grid": list(RIDGE_ALPHA_GRID),
                "alpha_selection": "leave_one_path_seed_out_mean_mse",
                "group_count": seed_count,
            },
            "evaluation": {"selection_rate": 0.25},
        }
    return {
        "fit_group_count": seed_count,
        "selection_utility_is_local_counterfactual_not_closed_loop_return": True,
        "predictors": predictors,
    }


def _cells() -> list[dict]:
    cells = []
    for root in (301, 303, 307, 309):
        base = root * 100
        train = [base + 1, base + 2]
        selection = [base + 11, base + 12]
        fit_seeds = [base + 21, base + 22]
        eval_seeds = [base + 31, base + 32]
        fit_rows = []
        eval_rows = []
        manifest = {}
        for split, seeds, rows in (
            ("predictor_fit", fit_seeds, fit_rows),
            ("qualification_eval", eval_seeds, eval_rows),
        ):
            for seed in seeds:
                seed_rows = []
                for index, category in enumerate(BRANCH_CATEGORIES):
                    row = _branch_row(
                        root,
                        seed,
                        category,
                        seed + 10 * (index + 1),
                        split=split,
                    )
                    rows.append(row)
                    seed_rows.append(row)
                manifest[str(seed)] = {
                    "regime_change_steps": [
                        row["source_step"] for row in seed_rows
                        if row["category"].startswith("regime_")
                    ],
                    "force_pulse_start_steps": [
                        row["source_step"] for row in seed_rows
                        if row["category"].startswith("force_")
                    ],
                    "distractor_change_steps": [
                        row["source_step"] for row in seed_rows
                        if row["category"].startswith("distractor_")
                    ],
                }
        controller_seeds = [*fit_seeds, *eval_seeds]
        cells.append({
            "optimizer_seed": root,
            "policy": POINTMAZE_COMPACT_PLAN_VALIDITY_POLICY,
            "protocol_version": POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION,
            "algorithm_path": POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH,
            "evidence_role": "compact_plan_validity_predictor_qualification",
            "trigger_training": "disabled_qualification_only",
            "plan_validity_predictor_deployment": "disabled",
            "runtime_versions": {"python": "test"},
            "train_seeds": train,
            "selection_seeds": selection,
            "branch_fit_seeds": fit_seeds,
            "branch_eval_seeds": eval_seeds,
            "max_events_per_class": 1,
            "branch_feature_names": [f"causal_{index}" for index in range(37)],
            "branch_path_manifest": manifest,
            "branch_fit_rows": fit_rows,
            "branch_evaluation_rows": eval_rows,
            "canonical_evaluation_rows": [
                {
                    "seed": seed,
                    "protocol_valid": 1.0,
                    "training_replicate_seed": root,
                    "tracking_squared_error_integral": 5.0,
                }
                for seed in controller_seeds
            ],
            "untrained_evaluation_rows": [
                {
                    "seed": seed,
                    "protocol_valid": 1.0,
                    "training_replicate_seed": root,
                    "tracking_squared_error_integral": 10.0,
                }
                for seed in controller_seeds
            ],
            "paired_branch_transition_budget": {
                "fit_primitive_steps_replayed": sum(
                    row["keep_primitive_steps_replayed"]
                    + row["renew_primitive_steps_replayed"]
                    for row in fit_rows
                ),
                "evaluation_primitive_steps_replayed": sum(
                    row["keep_primitive_steps_replayed"]
                    + row["renew_primitive_steps_replayed"]
                    for row in eval_rows
                ),
                "counted_as_extra_supervision": True,
            },
            "plan_validity_predictor": _predictor(len(fit_seeds)),
        })
    return cells


class PointMazeCompactPlanValidityStageEightCAnalysisTest(unittest.TestCase):
    def test_registered_conjunction_is_evaluated_by_optimizer_root(self):
        analysis = analyze_stage8c(_cells())
        self.assertTrue(analysis["stage9_authorized"])
        self.assertTrue(all(analysis["qualification_checks"].values()))
        self.assertEqual(analysis["independent_optimizer_root_count"], 4)

    def test_privileged_context_is_rejected(self):
        cells = _cells()
        cells[0]["branch_evaluation_rows"][0]["oracle_regime_context"] = [1.0]
        with self.assertRaisesRegex(ValueError, "branch row is invalid"):
            analyze_stage8c(cells)

    def test_incomplete_expected_matrix_is_rejected(self):
        cells = _cells()
        roots = [cell["optimizer_seed"] for cell in cells]
        with self.assertRaisesRegex(ValueError, "matrix is incomplete"):
            analyze_stage8c(cells[:-1], expected_roots=roots)


if __name__ == "__main__":
    unittest.main()
