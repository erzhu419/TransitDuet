import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.audit_protocol_v6_follower_forecast_calibration import (
    CONFIGS,
    ACTION_METRICS,
    DEPARTURE_METRICS,
    EVAL_SEEDS,
    EXACT_RMSE_METRICS,
    RESOLVED_METRICS,
    TRAIN_EPISODES,
    TRAIN_SEEDS,
    V13_ANCHOR,
    audit_follower_forecast_calibration,
)


class FollowerForecastCalibrationAuditTest(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        target_mae: float = 2.0,
        false_positive: float = 0.01,
        false_negative: float = 0.01,
        follower_hold: float = 2.0,
        follower_hold_rate: float = 0.1,
    ) -> Path:
        manifest = {
            "strict_complete": True,
            "run_manifests_verified": True,
            "common_random_numbers_verified": True,
            "configs": CONFIGS,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "train_episodes": TRAIN_EPISODES,
            "checkpoint_ep": TRAIN_EPISODES - 1,
            "stage": "exploratory",
            "independent_confirmation": False,
            "reference": V13_ANCHOR,
            "run_git_provenance": {
                "commit": "fixture-commit",
                "tracked_dirty": False,
            },
        }
        rows = []
        for config in CONFIGS:
            for train_seed in TRAIN_SEEDS:
                for eval_seed in EVAL_SEEDS:
                    row = {
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "lower_policy_frozen": 1,
                        "lower_critic_frozen": 1,
                        "upper_policy_frozen": 1,
                        "follower_forecast_decision_count": 100,
                        "follower_forecast_registered_count": 80,
                        "follower_forecast_resolved_count": 70,
                        "follower_forecast_action_resolved_count": 68,
                        "follower_forecast_departure_resolved_count": 65,
                    }
                    row.update({column: 1.0 for column in RESOLVED_METRICS})
                    row.update({column: 1.0 for column in EXACT_RMSE_METRICS})
                    row.update({column: 1.0 for column in ACTION_METRICS})
                    row.update({column: 1.0 for column in DEPARTURE_METRICS})
                    row[
                        "follower_forecast_target_action_prediction_mae_s"
                    ] = target_mae
                    row[
                        "follower_forecast_hold_need_false_positive_mean"
                    ] = false_positive
                    row[
                        "follower_forecast_hold_need_false_negative_mean"
                    ] = false_negative
                    row[
                        "follower_forecast_follower_future_hold_s_mean"
                    ] = follower_hold
                    row[
                        "follower_forecast_follower_future_hold_positive_rate"
                    ] = follower_hold_rate
                    rows.append(row)
        (root / "matrix_manifest.json").write_text(json.dumps(manifest))
        pd.DataFrame(rows).to_csv(root / "frozen_per_eval.csv", index=False)
        return root

    def test_strict_fixture_passes_and_pools_event_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit_follower_forecast_calibration(
                self._fixture(Path(tmp)))
        self.assertEqual(result["status"], "mechanical_pass")
        self.assertFalse(result["effect_evidence"])
        self.assertEqual(
            result["diagnosis"], "local_surrogate_mismatch_beyond_forecast")
        rollout_count = len(CONFIGS) * len(TRAIN_SEEDS) * len(EVAL_SEEDS)
        self.assertEqual(
            result["pooled"]["follower_forecast_resolved_count"],
            70 * rollout_count,
        )
        self.assertAlmostEqual(result["pooled"]["resolution_rate"], 0.875)

    def test_pooled_rmse_combines_squared_error_moments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp))
            path = root / "frozen_per_eval.csv"
            frame = pd.read_csv(path)
            column = "follower_forecast_raw_gap_prediction_rmse_s"
            frame[column] = 3.0
            frame.loc[0, column] = 4.0
            frame.to_csv(path, index=False)
            result = audit_follower_forecast_calibration(root)
        rollout_count = len(CONFIGS) * len(TRAIN_SEEDS) * len(EVAL_SEEDS)
        expected = (
            ((rollout_count - 1) * 3.0 ** 2 + 4.0 ** 2) / rollout_count
        ) ** 0.5
        self.assertAlmostEqual(result["pooled"][column], expected)

    def test_material_forecast_error_and_future_hold_are_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit_follower_forecast_calibration(self._fixture(
                Path(tmp),
                target_mae=8.0,
                follower_hold=10.0,
                follower_hold_rate=0.5,
            ))
        self.assertEqual(
            result["diagnosis"], "forecast_error_and_sequential_holding")

    def test_missing_metric_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp))
            path = root / "frozen_per_eval.csv"
            frame = pd.read_csv(path).drop(columns=[RESOLVED_METRICS[0]])
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing follower"):
                audit_follower_forecast_calibration(root)


if __name__ == "__main__":
    unittest.main()
