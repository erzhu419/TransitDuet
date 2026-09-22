import unittest

from freq_hrl.experiments.pointmaze_plan_validity_branching import (
    BRANCH_CATEGORIES,
    POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH,
    POINTMAZE_PLAN_VALIDITY_POLICY,
    POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION,
)
from scripts.analyze_pointmaze_plan_validity_stage8b import analyze_stage8b


VALUES = {
    "regime_lag_010ms": 0.10,
    "regime_lag_100ms": 0.30,
    "regime_lag_250ms": 1.00,
    "force_pulse_010ms": 0.00,
    "distractor_change_010ms": -0.10,
    "neutral_matched": -0.05,
}


def _branch_row(
    root: int,
    seed: int,
    category: str,
    step: int,
    *,
    split: str = "qualification_eval",
) -> dict:
    value = VALUES[category]
    neutral = category == "neutral_matched"
    lag = None if neutral else 1
    source = None if neutral else step - int(lag)
    return {
        "protocol_version": POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH,
        "optimizer_seed": root,
        "split": split,
        "seed": seed,
        "category": category,
        "opportunity_step": step,
        "source_step": source,
        "lag_steps": lag,
        "protocol_valid": True,
        "candidate_feature_has_future_access": False,
        "candidate_feature_has_regime_label": False,
        "oracle_regime_used_only_by_diagnostic_predictor": True,
        "keep_upper_calls_at_opportunity": 0,
        "renew_upper_calls_at_opportunity": 1,
        "downstream_upper_call_count_per_branch": 0,
        "lower_controller_remains_closed_loop": True,
        "prefix_max_abs_difference": 0.0,
        "feature_max_abs_difference": 0.0,
        "feature_names": ["plan_age_fraction", "signal"],
        "feature_masks": {
            "age_only": [0],
            "plan_state": [0],
            "change_magnitude": [0, 1],
            "causal_history": [0, 1],
        },
        "causal_features": [step / 100.0, value],
        "keep_tracking_squared_error_integral": 2.0,
        "renew_tracking_squared_error_integral": 2.0 - value,
        "renew_ise_advantage": value,
        "keep_primitive_steps_replayed": step + 50,
        "renew_primitive_steps_replayed": step + 50,
        "prediction_age_only": -value,
        "prediction_plan_state": -value,
        "prediction_change_magnitude": value * 0.5,
        "prediction_causal_history": value,
        "prediction_causal_history_plus_regime": value,
    }


def _cells() -> list[dict]:
    cells = []
    for root in (101, 103, 107, 109):
        seed_base = root * 100
        train_seeds = [seed_base + 1, seed_base + 2]
        selection_seeds = [seed_base + 11, seed_base + 12]
        fit_seeds = [seed_base + 21, seed_base + 22]
        eval_seeds = [seed_base + 31, seed_base + 32]
        branch_rows = []
        for seed in eval_seeds:
            for index, category in enumerate(BRANCH_CATEGORIES):
                branch_rows.append(_branch_row(
                    root, seed, category, 10 + index * 10 + seed
                ))
        fit_rows = []
        for seed in fit_seeds:
            for index, category in enumerate(BRANCH_CATEGORIES):
                fit_rows.append(_branch_row(
                    root,
                    seed,
                    category,
                    10 + index * 10 + seed,
                    split="predictor_fit",
                ))
        controller_eval_seeds = [*fit_seeds, *eval_seeds]
        manifest = {}
        for seed in (*fit_seeds, *eval_seeds):
            seed_rows = [
                row for row in (*fit_rows, *branch_rows)
                if row["seed"] == seed
            ]
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
        cells.append({
            "optimizer_seed": root,
            "policy": POINTMAZE_PLAN_VALIDITY_POLICY,
            "protocol_version": POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION,
            "algorithm_path": POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH,
            "evidence_role": "counterfactual_plan_validity_qualification",
            "trigger_training": "disabled_qualification_only",
            "plan_validity_predictor_deployment": "disabled",
            "max_events_per_class": 1,
            "runtime_versions": {"python": "test"},
            "train_seeds": train_seeds,
            "selection_seeds": selection_seeds,
            "branch_fit_seeds": fit_seeds,
            "branch_eval_seeds": eval_seeds,
            "canonical_evaluation_rows": [
                {
                    "seed": seed,
                    "protocol_valid": 1.0,
                    "training_replicate_seed": root,
                    "tracking_squared_error_integral": 5.0,
                }
                for seed in controller_eval_seeds
            ],
            "untrained_evaluation_rows": [
                {
                    "seed": seed,
                    "protocol_valid": 1.0,
                    "training_replicate_seed": root,
                    "tracking_squared_error_integral": 10.0,
                }
                for seed in controller_eval_seeds
            ],
            "branch_evaluation_rows": branch_rows,
            "branch_fit_rows": fit_rows,
            "branch_feature_names": ["plan_age_fraction", "signal"],
            "branch_feature_masks": {
                "age_only": [0],
                "plan_state": [0],
                "change_magnitude": [0, 1],
                "causal_history": [0, 1],
            },
            "branch_path_manifest": manifest,
            "paired_branch_transition_budget": {
                "fit_primitive_steps_replayed": sum(
                    row["keep_primitive_steps_replayed"]
                    + row["renew_primitive_steps_replayed"]
                    for row in fit_rows
                ),
                "evaluation_primitive_steps_replayed": sum(
                    row["keep_primitive_steps_replayed"]
                    + row["renew_primitive_steps_replayed"]
                    for row in branch_rows
                ),
                "counted_as_extra_supervision": True,
            },
            "plan_validity_predictor": {
                "predictors": {
                    "causal_history": {
                        "evaluation": {"selection_rate": 0.25}
                    }
                }
            },
        })
    return cells


class PointMazePlanValidityStageEightBAnalysisTest(unittest.TestCase):
    def test_registered_conjunction_is_evaluated_at_optimizer_root_level(self):
        analysis = analyze_stage8b(_cells())
        self.assertTrue(analysis["stage9_authorized"])
        self.assertEqual(
            analysis["decision"],
            "plan_validity_trigger_development_authorized",
        )
        self.assertTrue(all(analysis["qualification_checks"].values()))
        self.assertEqual(analysis["independent_optimizer_root_count"], 4)

    def test_seed_reuse_across_optimizer_roots_is_rejected(self):
        cells = _cells()
        cells[1]["branch_eval_seeds"] = cells[0]["branch_eval_seeds"]
        with self.assertRaisesRegex(ValueError, "reused across optimizer roots"):
            analyze_stage8b(cells)

    def test_incomplete_expected_root_matrix_is_rejected(self):
        cells = _cells()
        expected_roots = [cell["optimizer_seed"] for cell in cells]
        with self.assertRaisesRegex(ValueError, "matrix is incomplete"):
            analyze_stage8b(cells[:-1], expected_roots=expected_roots)

    def test_duplicate_seed_inside_role_is_rejected(self):
        cells = _cells()
        cells[0]["train_seeds"] = [10101, 10101]
        with self.assertRaisesRegex(ValueError, "contains duplicates"):
            analyze_stage8b(cells)


if __name__ == "__main__":
    unittest.main()
