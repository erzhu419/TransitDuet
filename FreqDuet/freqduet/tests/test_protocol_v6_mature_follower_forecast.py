import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.audit_protocol_v6_follower_forecast_calibration import (
    ACTION_METRICS,
    COUNT_COLUMNS,
    DEPARTURE_METRICS,
    EXACT_RMSE_METRICS,
    RESOLVED_METRICS,
)
from scripts.audit_protocol_v6_mature_follower_forecast import (
    AUDIT_VERSION,
    CHECKPOINT_EP,
    CHECKPOINT_SOURCE_COMMIT,
    CONFIGS,
    EVAL_SEEDS,
    ORIGIN_KEY,
    ORIGIN_VERSION,
    PROTOCOL_VERSION,
    TRAIN_EPISODES,
    TRAIN_SEEDS,
    V24,
    assert_behavior_unchanged,
    audit_mature_follower_forecast,
    validate_checkpoint_origin,
)
from scripts.run_freqduet_protocol_v2_matrix import (
    METRICS,
    strict_protocol_metrics,
)
from scripts.analysis_provenance import csv_artifact_record


TELEMETRY_COMMIT = "telemetry-fixture-commit"


def _evaluation_row(
    eval_seed: int,
    *,
    policy_digest: str = "fixture-policy",
    target_mae: float = 2.0,
    follower_hold: float = 8.0,
    follower_hold_rate: float = 0.5,
) -> dict[str, object]:
    row = {
        "protocol_version": PROTOCOL_VERSION,
        "eval_seed": eval_seed,
        "checkpoint_ep": CHECKPOINT_EP,
        "policy_digest": policy_digest,
        "scenario_tape_id": f"tape-{eval_seed}",
        "lower_causal_guard_evidence_mode": "pre_action_departure_v6",
        "lower_policy_frozen": 1,
        "lower_critic_frozen": 1,
        "upper_policy_frozen": 1,
        "wall_env_s": 1.0,
        "wall_train_s": 0.0,
    }
    row.update({metric: 1.0 for metric in METRICS})
    row.update({
        metric: 1.0 for metric in strict_protocol_metrics(PROTOCOL_VERSION)
    })
    row.update({
        "follower_forecast_decision_count": 100,
        "follower_forecast_registered_count": 80,
        "follower_forecast_resolved_count": 70,
        "follower_forecast_action_resolved_count": 70,
        "follower_forecast_departure_resolved_count": 70,
    })
    row.update({metric: 1.0 for metric in RESOLVED_METRICS})
    row.update({metric: 1.0 for metric in EXACT_RMSE_METRICS})
    row.update({metric: 1.0 for metric in ACTION_METRICS})
    row.update({metric: 1.0 for metric in DEPARTURE_METRICS})
    row["follower_forecast_target_action_prediction_mae_s"] = target_mae
    row["follower_forecast_hold_need_false_positive_mean"] = 0.01
    row["follower_forecast_hold_need_false_negative_mean"] = 0.01
    row["follower_forecast_follower_future_hold_s_mean"] = follower_hold
    row[
        "follower_forecast_follower_future_hold_positive_rate"
    ] = follower_hold_rate
    return row


def _evaluation_manifest(
    config: str,
    train_seed: int,
    evaluation_path: Path,
    *,
    include_origin: bool,
) -> dict[str, object]:
    frame = pd.read_csv(evaluation_path)
    manifest = {
        "manifest_version": "freqduet-evaluation-manifest-v2",
        "protocol_version": PROTOCOL_VERSION,
        "config_name": config,
        "training_seed": train_seed,
        "checkpoint_ep": CHECKPOINT_EP,
        "scenario_seeds": EVAL_SEEDS,
        "n_episodes": len(EVAL_SEEDS),
        "policy_digest": f"policy-{config}-{train_seed}",
        "artifacts": {
            "evaluation_csv": csv_artifact_record(
                evaluation_path, frame, ["eval_seed"]),
        },
    }
    if include_origin:
        manifest[ORIGIN_KEY] = {
            "version": ORIGIN_VERSION,
            "checkpoint_source_commit": CHECKPOINT_SOURCE_COMMIT,
            "telemetry_source_commit": TELEMETRY_COMMIT,
            "checkpoint_ep": CHECKPOINT_EP,
            "checkpoint_config_fingerprint_sha256": "resolved-fixture",
            "behavior_invariance_verified": True,
            "behavior_invariance_compared_columns": ["headway_cv"],
        }
    return manifest


class MatureFollowerForecastAuditTest(unittest.TestCase):
    def _aggregate_fixture(self, root: Path) -> Path:
        for config in CONFIGS:
            for train_seed in TRAIN_SEEDS:
                destination = (
                    root / f"{config}_seed{train_seed}"
                    / "frozen_evaluation")
                destination.mkdir(parents=True)
                target_mae = 6.0 if config == V24 else 2.0
                rows = [
                    _evaluation_row(
                        eval_seed,
                        policy_digest=f"policy-{config}-{train_seed}",
                        target_mae=target_mae,
                    )
                    for eval_seed in EVAL_SEEDS
                ]
                pd.DataFrame(rows).to_csv(
                    destination / "evaluation.csv", index=False)
                (destination / "evaluation_manifest.json").write_text(
                    json.dumps(_evaluation_manifest(
                        config,
                        train_seed,
                        destination / "evaluation.csv",
                        include_origin=True,
                    )))
        return root

    def test_registered_mature_diagnosis_preserves_policy_heterogeneity(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit_mature_follower_forecast(
                self._aggregate_fixture(Path(tmp)),
                telemetry_commit=TELEMETRY_COMMIT,
            )
        self.assertEqual(result["schema_version"], AUDIT_VERSION)
        self.assertEqual(result["status"], "mechanical_pass")
        self.assertFalse(result["effect_evidence"])
        self.assertEqual(
            result["diagnosis"],
            "sequential_holding_with_policy_specific_forecast_error",
        )
        authorization = result["successor_authorization"]
        self.assertTrue(authorization["delayed_control_state_objective"])
        self.assertTrue(authorization["forecast_calibration_ablation"])
        self.assertEqual(authorization["forecast_error_configs"], [V24])
        self.assertEqual(
            result["pooled"]["follower_forecast_resolved_count"],
            len(CONFIGS) * len(TRAIN_SEEDS) * len(EVAL_SEEDS) * 70,
        )

    def test_sequential_gate_requires_three_checkpoint_seeds_per_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._aggregate_fixture(Path(tmp))
            config = CONFIGS[0]
            for train_seed in TRAIN_SEEDS[:2]:
                path = (
                    root / f"{config}_seed{train_seed}"
                    / "frozen_evaluation" / "evaluation.csv")
                frame = pd.read_csv(path)
                frame[
                    "follower_forecast_follower_future_hold_s_mean"
                ] = 1.0
                frame[
                    "follower_forecast_follower_future_hold_positive_rate"
                ] = 0.1
                frame.to_csv(path, index=False)
                (path.parent / "evaluation_manifest.json").write_text(
                    json.dumps(_evaluation_manifest(
                        config,
                        train_seed,
                        path,
                        include_origin=True,
                    )))
            result = audit_mature_follower_forecast(
                root, telemetry_commit=TELEMETRY_COMMIT)
        self.assertFalse(
            result["successor_authorization"][
                "delayed_control_state_objective"])
        self.assertEqual(
            result["diagnosis"],
            "forecast_error_without_general_sequential_holding",
        )

    def test_behavior_invariance_ignores_runtime_but_not_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_path = root / "old.csv"
            new_path = root / "new.csv"
            old = pd.DataFrame([_evaluation_row(EVAL_SEEDS[0])])
            new = old.copy()
            new["wall_env_s"] = 99.0
            new["new_follower_metric"] = 7.0
            old.to_csv(old_path, index=False)
            new.to_csv(new_path, index=False)
            columns = assert_behavior_unchanged(old_path, new_path)
            self.assertNotIn("wall_env_s", columns)
            new["headway_cv"] = 9.0
            new.to_csv(new_path, index=False)
            with self.assertRaisesRegex(ValueError, "changed a frozen outcome"):
                assert_behavior_unchanged(old_path, new_path)

    def test_checkpoint_origin_validates_dual_source_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            for stem in ("lower", "upper", "runner"):
                (checkpoint_dir / f"{stem}_ep{CHECKPOINT_EP}.pt").touch()
            meta = {
                "protocol_version": PROTOCOL_VERSION,
                "config_name": CONFIGS[0],
                "seed": TRAIN_SEEDS[0],
                "latest_episode": CHECKPOINT_EP,
                "config_fingerprint_sha256": "resolved-fixture",
            }
            (checkpoint_dir / "checkpoint_meta.json").write_text(
                json.dumps(meta))
            run_manifest = {
                "protocol_version": PROTOCOL_VERSION,
                "config_name": CONFIGS[0],
                "train_seed": TRAIN_SEEDS[0],
                "train_episodes": TRAIN_EPISODES,
                "checkpoint_ep": CHECKPOINT_EP,
                "eval_seeds": EVAL_SEEDS,
                "stage": "exploratory",
                "git": {
                    "commit": CHECKPOINT_SOURCE_COMMIT,
                    "tracked_dirty": False,
                },
            }
            (run_dir / "protocol_run_manifest.json").write_text(
                json.dumps(run_manifest))
            destination = run_dir / "frozen_evaluation"
            destination.mkdir()
            rows = [
                _evaluation_row(
                    eval_seed,
                    policy_digest=(
                        f"policy-{CONFIGS[0]}-{TRAIN_SEEDS[0]}"),
                )
                for eval_seed in EVAL_SEEDS
            ]
            pd.DataFrame(rows).to_csv(
                destination / "evaluation.csv", index=False)
            (destination / "evaluation_manifest.json").write_text(json.dumps(
                _evaluation_manifest(
                    CONFIGS[0],
                    TRAIN_SEEDS[0],
                    destination / "evaluation.csv",
                    include_origin=False,
                )))
            _, old_path = validate_checkpoint_origin(
                run_dir,
                config=CONFIGS[0],
                train_seed=TRAIN_SEEDS[0],
                resolved_fingerprint="resolved-fixture",
            )
            self.assertEqual(old_path, destination / "evaluation.csv")
            meta["config_fingerprint_sha256"] = "wrong"
            (checkpoint_dir / "checkpoint_meta.json").write_text(
                json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "config_fingerprint"):
                validate_checkpoint_origin(
                    run_dir,
                    config=CONFIGS[0],
                    train_seed=TRAIN_SEEDS[0],
                    resolved_fingerprint="resolved-fixture",
                )


if __name__ == "__main__":
    unittest.main()
