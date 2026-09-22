import unittest

import numpy as np

from freq_hrl.experiments.pointmaze_compact_plan_validity import (
    COMPACT_PLAN_VALIDITY_PREDICTORS,
    _grouped_ridge_fit_predict,
    compact_plan_validity_feature_views,
    fit_compact_plan_validity_predictors,
)


FEATURE_NAMES = (
    "physical_0", "physical_1", "physical_2", "physical_3",
    "target_error_0", "target_error_1",
    "waypoint_error_0", "waypoint_error_1",
    "tracking_distance", "waypoint_distance",
    "plan_age_seconds", "plan_age_fraction",
    "target_current_0", "target_current_1",
    "force_current_0", "force_current_1",
    "distractor_current_0", "distractor_current_1",
    "target_velocity_001_0", "target_velocity_001_1",
    "target_velocity_010_0", "target_velocity_010_1",
    "target_velocity_025_0", "target_velocity_025_1",
    "target_velocity_050_0", "target_velocity_050_1",
    "force_rms_010_0", "force_rms_010_1",
    "distractor_delta_010_0", "distractor_delta_010_1",
    "force_rms_025_0", "force_rms_025_1",
    "distractor_delta_025_0", "distractor_delta_025_1",
    "target_velocity_change_norm", "force_current_norm",
    "distractor_change_norm",
)


def _rows(*, seed_start: int, seed_count: int, rows_per_seed: int) -> list[dict]:
    rng = np.random.default_rng(seed_start)
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    rows = []
    for seed in range(seed_start, seed_start + seed_count):
        for item in range(rows_per_seed):
            features = rng.normal(size=len(FEATURE_NAMES))
            features[index["plan_age_fraction"]] = rng.uniform(0.0, 1.0)
            features[index["tracking_distance"]] = abs(
                features[index["tracking_distance"]]
            )
            features[index["waypoint_distance"]] = abs(
                features[index["waypoint_distance"]]
            )
            target_error = features[[
                index["target_error_0"], index["target_error_1"]
            ]]
            velocity = features[[
                index["target_velocity_025_0"],
                index["target_velocity_025_1"],
            ]]
            target = (
                0.8 * float(target_error @ velocity)
                + 0.5
                * features[index["plan_age_fraction"]]
                * float(np.linalg.norm(velocity))
                + rng.normal(scale=0.02)
            )
            rows.append({
                "seed": seed,
                "category": f"category_{item % 3}",
                "feature_names": list(FEATURE_NAMES),
                "causal_features": features.tolist(),
                "candidate_feature_has_future_access": False,
                "candidate_feature_has_regime_label": False,
                "renew_ise_advantage": target,
            })
    return rows


class PointMazeCompactPlanValidityTest(unittest.TestCase):
    def test_feature_views_have_frozen_dimensions(self):
        views = compact_plan_validity_feature_views(
            _rows(seed_start=10, seed_count=2, rows_per_seed=4)
        )
        self.assertEqual(set(views), set(COMPACT_PLAN_VALIDITY_PREDICTORS))
        self.assertEqual(views["current_compact_quadratic"][0].shape, (8, 170))
        self.assertEqual(views["causal_dynamic_quadratic"][0].shape, (8, 170))
        self.assertEqual(views["causal_validity_interactions"][0].shape, (8, 39))

    def test_grouped_ridge_rejects_single_fit_path(self):
        with self.assertRaisesRegex(ValueError, "contract is invalid"):
            _grouped_ridge_fit_predict(
                np.ones((4, 2)),
                np.ones(4),
                np.ones(4),
                np.ones((2, 2)),
                alpha_grid=(1.0,),
            )

    def test_causal_interactions_recover_held_out_nonlinear_value(self):
        fit_rows = _rows(seed_start=20, seed_count=6, rows_per_seed=24)
        eval_rows = _rows(seed_start=40, seed_count=3, rows_per_seed=24)
        result = fit_compact_plan_validity_predictors(
            fit_rows,
            eval_rows,
            ridge_alpha_grid=(0.1, 1.0, 10.0, 100.0),
            selection_rate=0.25,
        )
        predictors = result["predictors"]
        self.assertEqual(set(predictors), set(COMPACT_PLAN_VALIDITY_PREDICTORS))
        self.assertEqual(result["fit_group_count"], 6)
        candidate = predictors["causal_validity_interactions"]
        baseline = predictors["current_compact_quadratic"]
        self.assertGreater(candidate["evaluation"]["spearman"], 0.9)
        self.assertGreater(
            candidate["evaluation"]["selected_mean_renew_ise_advantage"],
            baseline["evaluation"]["selected_mean_renew_ise_advantage"],
        )
        self.assertEqual(candidate["feature_count"], 39)
        self.assertEqual(candidate["model"]["group_count"], 6)
        for row in eval_rows:
            for name in COMPACT_PLAN_VALIDITY_PREDICTORS:
                self.assertTrue(np.isfinite(row[f"prediction_{name}"]))


if __name__ == "__main__":
    unittest.main()
