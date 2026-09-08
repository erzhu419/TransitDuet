#!/usr/bin/env python3
"""Evaluate and audit follower forecasts on frozen mature V6 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner_v3 import (  # noqa: E402
    TransitDuetV2Runner,
    config_fingerprint as resolved_config_fingerprint,
    load_config,
)
from scripts.audit_protocol_v6_follower_forecast_calibration import (  # noqa: E402
    COUNT_COLUMNS,
    FOLLOWER_HOLD_MATERIAL_RATE,
    FOLLOWER_HOLD_MATERIAL_S,
    HOLD_SIGN_ERROR_MATERIAL_RATE,
    TARGET_ERROR_MATERIAL_S,
    _summarize,
)
from scripts.run_freqduet_protocol_v2_matrix import (  # noqa: E402
    git_provenance,
    validate_evaluation_frame,
    validate_evaluation_manifest,
)


V13 = "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_hiro"
V23 = "F_freqduet_protocol_v6_v23_jointproj_r036_p075_hiro"
V24 = "F_freqduet_protocol_v6_v24_jointproj_rkl_s4_hiro"
CONFIGS = [V13, V23, V24]
TRAIN_SEEDS = [29013, 29031, 29053, 29077]
EVAL_SEEDS = [62017, 62041, 62059, 62083]
TRAIN_EPISODES = 40
CHECKPOINT_EP = TRAIN_EPISODES - 1
CHECKPOINT_SOURCE_COMMIT = "b59f478597445b88ea61c459d7a2647594651fe3"
PROTOCOL_VERSION = "freqduet-eval-v6"
ORIGIN_KEY = "mature_checkpoint_forecast_audit"
ORIGIN_VERSION = "freqduet-v25-mature-checkpoint-origin-v1"
AUDIT_VERSION = "freqduet-v25-mature-follower-forecast-audit-v1"


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_equal(source: Path, actual: object, expected: object,
                   field: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{source}: {field}={actual!r}, expected {expected!r}")


def validate_checkpoint_origin(
    checkpoint_run_dir: Path,
    *,
    config: str,
    train_seed: int,
    resolved_fingerprint: str,
    checkpoint_source_commit: str = CHECKPOINT_SOURCE_COMMIT,
) -> tuple[dict[str, object], Path]:
    """Validate the frozen training run and its original evaluation."""
    checkpoint_run_dir = Path(checkpoint_run_dir).resolve()
    checkpoint_dir = checkpoint_run_dir / "checkpoints"
    required_checkpoints = [
        checkpoint_dir / f"lower_ep{CHECKPOINT_EP}.pt",
        checkpoint_dir / f"upper_ep{CHECKPOINT_EP}.pt",
        checkpoint_dir / f"runner_ep{CHECKPOINT_EP}.pt",
    ]
    missing = [str(path) for path in required_checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing mature checkpoints: {missing}")

    meta_path = checkpoint_dir / "checkpoint_meta.json"
    meta = _read_json(meta_path)
    for field, expected in {
        "protocol_version": PROTOCOL_VERSION,
        "config_name": config,
        "seed": int(train_seed),
        "latest_episode": CHECKPOINT_EP,
        "config_fingerprint_sha256": resolved_fingerprint,
    }.items():
        _require_equal(meta_path, meta.get(field), expected, field)

    run_manifest_path = checkpoint_run_dir / "protocol_run_manifest.json"
    run_manifest = _read_json(run_manifest_path)
    for field, expected in {
        "protocol_version": PROTOCOL_VERSION,
        "config_name": config,
        "train_seed": int(train_seed),
        "train_episodes": TRAIN_EPISODES,
        "checkpoint_ep": CHECKPOINT_EP,
        "eval_seeds": EVAL_SEEDS,
        "stage": "exploratory",
    }.items():
        _require_equal(run_manifest_path, run_manifest.get(field), expected, field)
    run_git = run_manifest.get("git") or {}
    _require_equal(
        run_manifest_path,
        run_git.get("commit"),
        checkpoint_source_commit,
        "git.commit",
    )
    _require_equal(
        run_manifest_path, run_git.get("tracked_dirty"), False,
        "git.tracked_dirty")

    old_evaluation_path = (
        checkpoint_run_dir / "frozen_evaluation" / "evaluation.csv")
    old_manifest_path = old_evaluation_path.parent / "evaluation_manifest.json"
    old_manifest = _read_json(old_manifest_path)
    for field, expected in {
        "protocol_version": PROTOCOL_VERSION,
        "config_name": config,
        "training_seed": int(train_seed),
        "checkpoint_ep": CHECKPOINT_EP,
        "scenario_seeds": EVAL_SEEDS,
        "n_episodes": len(EVAL_SEEDS),
    }.items():
        _require_equal(old_manifest_path, old_manifest.get(field), expected, field)
    old_frame = pd.read_csv(old_evaluation_path)
    validate_evaluation_frame(
        old_frame,
        EVAL_SEEDS,
        old_evaluation_path,
        checkpoint_ep=CHECKPOINT_EP,
        protocol_version=PROTOCOL_VERSION,
    )
    validate_evaluation_manifest(
        old_evaluation_path,
        old_frame,
        config,
        int(train_seed),
        EVAL_SEEDS,
        checkpoint_ep=CHECKPOINT_EP,
        protocol_version=PROTOCOL_VERSION,
    )
    _require_equal(
        old_manifest_path,
        old_manifest.get("policy_digest"),
        str(old_frame["policy_digest"].iloc[0]),
        "policy_digest",
    )
    return {
        "checkpoint_meta": meta,
        "run_manifest": run_manifest,
        "old_evaluation_manifest": old_manifest,
    }, old_evaluation_path


def assert_behavior_unchanged(
    old_evaluation_path: Path,
    new_evaluation_path: Path,
) -> list[str]:
    """Require the telemetry replay to preserve every old non-runtime cell."""
    old = pd.read_csv(old_evaluation_path).sort_values("eval_seed")
    new = pd.read_csv(new_evaluation_path).sort_values("eval_seed")
    excluded = {"wall_env_s", "wall_train_s"}
    columns = [column for column in old.columns if column not in excluded]
    missing = sorted(set(columns).difference(new.columns))
    if missing:
        raise ValueError(
            f"new telemetry evaluation dropped old columns: {missing}")
    try:
        pd.testing.assert_frame_equal(
            old[columns].reset_index(drop=True),
            new[columns].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-9,
        )
    except AssertionError as exc:
        raise ValueError(
            "behavior-neutral telemetry replay changed a frozen outcome"
        ) from exc
    return columns


def evaluate_checkpoint(
    *,
    config: str,
    train_seed: int,
    checkpoint_run_dir: Path,
    output_root: Path,
    telemetry_commit: str,
) -> Path:
    if config not in CONFIGS:
        raise ValueError(f"unregistered mature config: {config}")
    if int(train_seed) not in TRAIN_SEEDS:
        raise ValueError(f"unregistered mature train seed: {train_seed}")

    provenance = git_provenance()
    if provenance.get("commit") != telemetry_commit:
        raise ValueError(
            "telemetry source commit mismatch: "
            f"{provenance.get('commit')} != {telemetry_commit}")
    if provenance.get("tracked_dirty") is not False:
        raise ValueError("telemetry source must have a clean tracked worktree")

    cfg_path = ROOT / "configs_freqduet" / f"{config}.yaml"
    cfg = load_config(str(cfg_path))
    cfg["seed"] = int(train_seed)
    resolved_fingerprint = resolved_config_fingerprint(cfg)
    _, old_evaluation_path = validate_checkpoint_origin(
        checkpoint_run_dir,
        config=config,
        train_seed=int(train_seed),
        resolved_fingerprint=resolved_fingerprint,
    )

    destination = (
        Path(output_root).resolve()
        / f"{config}_seed{int(train_seed)}"
        / "frozen_evaluation"
    )
    if destination.exists():
        raise FileExistsError(
            f"refusing to reuse mature audit output root: {destination}")

    runner = TransitDuetV2Runner(cfg, device="cpu")
    loaded_ep = runner.load_checkpoint(
        checkpoint_dir=Path(checkpoint_run_dir) / "checkpoints",
        ep=CHECKPOINT_EP,
        require_deployment_state=True,
    )
    if loaded_ep != CHECKPOINT_EP:
        raise ValueError(f"loaded checkpoint {loaded_ep}, expected {CHECKPOINT_EP}")
    _, destination = runner.evaluate(
        EVAL_SEEDS,
        output_dir=destination,
        policy_ep=max(CHECKPOINT_EP, runner.upper_warmup),
    )
    evaluation_path = destination / "evaluation.csv"
    compared_columns = assert_behavior_unchanged(
        old_evaluation_path, evaluation_path)

    manifest_path = destination / "evaluation_manifest.json"
    manifest = _read_json(manifest_path)
    manifest[ORIGIN_KEY] = {
        "version": ORIGIN_VERSION,
        "checkpoint_source_commit": CHECKPOINT_SOURCE_COMMIT,
        "telemetry_source_commit": telemetry_commit,
        "checkpoint_run_dir": str(Path(checkpoint_run_dir).resolve()),
        "checkpoint_config_fingerprint_sha256": resolved_fingerprint,
        "checkpoint_ep": CHECKPOINT_EP,
        "behavior_invariance_verified": True,
        "behavior_invariance_compared_columns": compared_columns,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "mature_checkpoint_evaluation_complete",
        "config": config,
        "train_seed": int(train_seed),
        "checkpoint_ep": CHECKPOINT_EP,
        "eval_seeds": EVAL_SEEDS,
        "output": str(destination),
    }, sort_keys=True))
    return destination


def _forecast_material(summary: dict[str, float | int]) -> bool:
    return bool(
        summary["follower_forecast_target_action_prediction_mae_s"]
        > TARGET_ERROR_MATERIAL_S
        or summary["follower_forecast_hold_need_false_positive_mean"]
        + summary["follower_forecast_hold_need_false_negative_mean"]
        > HOLD_SIGN_ERROR_MATERIAL_RATE
    )


def _sequential_material(summary: dict[str, float | int]) -> bool:
    return bool(
        summary["follower_forecast_follower_future_hold_s_mean"]
        > FOLLOWER_HOLD_MATERIAL_S
        and summary["follower_forecast_follower_future_hold_positive_rate"]
        > FOLLOWER_HOLD_MATERIAL_RATE
    )


def audit_mature_follower_forecast(
    evaluation_root: Path,
    *,
    telemetry_commit: str,
) -> dict[str, object]:
    evaluation_root = Path(evaluation_root).resolve()
    records = []
    origin_records = []
    for config in CONFIGS:
        for train_seed in TRAIN_SEEDS:
            destination = (
                evaluation_root / f"{config}_seed{train_seed}"
                / "frozen_evaluation")
            evaluation_path = destination / "evaluation.csv"
            manifest_path = destination / "evaluation_manifest.json"
            manifest = _read_json(manifest_path)
            origin = manifest.get(ORIGIN_KEY) or {}
            expected_origin = {
                "version": ORIGIN_VERSION,
                "checkpoint_source_commit": CHECKPOINT_SOURCE_COMMIT,
                "telemetry_source_commit": telemetry_commit,
                "checkpoint_ep": CHECKPOINT_EP,
                "behavior_invariance_verified": True,
            }
            for field, expected in expected_origin.items():
                _require_equal(manifest_path, origin.get(field), expected, field)
            for field, expected in {
                "protocol_version": PROTOCOL_VERSION,
                "config_name": config,
                "training_seed": train_seed,
                "checkpoint_ep": CHECKPOINT_EP,
                "scenario_seeds": EVAL_SEEDS,
                "n_episodes": len(EVAL_SEEDS),
            }.items():
                _require_equal(manifest_path, manifest.get(field), expected, field)
            frame = pd.read_csv(evaluation_path)
            validate_evaluation_frame(
                frame,
                EVAL_SEEDS,
                evaluation_path,
                checkpoint_ep=CHECKPOINT_EP,
                protocol_version=PROTOCOL_VERSION,
            )
            validate_evaluation_manifest(
                evaluation_path,
                frame,
                config,
                train_seed,
                EVAL_SEEDS,
                checkpoint_ep=CHECKPOINT_EP,
                protocol_version=PROTOCOL_VERSION,
            )
            _require_equal(
                manifest_path,
                manifest.get("policy_digest"),
                str(frame["policy_digest"].iloc[0]),
                "policy_digest",
            )
            frame["config"] = config
            frame["train_seed"] = train_seed
            records.append(frame)
            origin_records.append(origin)

    per_eval = pd.concat(records, ignore_index=True)
    expected_rows = len(CONFIGS) * len(TRAIN_SEEDS) * len(EVAL_SEEDS)
    key_columns = ["config", "train_seed", "eval_seed"]
    exact_keys = (
        len(per_eval) == expected_rows
        and not per_eval.duplicated(key_columns).any()
    )
    common_tapes = all(
        group["scenario_tape_id"].astype(str).nunique() == 1
        for _, group in per_eval.groupby("eval_seed")
    )
    policies_frozen = bool(
        (pd.to_numeric(per_eval["lower_policy_frozen"]) == 1).all()
        and (pd.to_numeric(per_eval["lower_critic_frozen"]) == 1).all()
        and (pd.to_numeric(per_eval["upper_policy_frozen"]) == 1).all()
    )
    exact_resolution = bool(
        (pd.to_numeric(per_eval[COUNT_COLUMNS[2]]) > 0).all()
        and (pd.to_numeric(per_eval[COUNT_COLUMNS[3]])
             == pd.to_numeric(per_eval[COUNT_COLUMNS[2]])).all()
        and (pd.to_numeric(per_eval[COUNT_COLUMNS[4]])
             == pd.to_numeric(per_eval[COUNT_COLUMNS[2]])).all()
    )
    strict_checks = {
        "exact_48_unique_rollouts": bool(exact_keys),
        "common_random_numbers_verified": bool(common_tapes),
        "policies_frozen": policies_frozen,
        "all_forecasts_action_and_departure_resolved": exact_resolution,
        "checkpoint_source_commit_locked": all(
            item.get("checkpoint_source_commit")
            == CHECKPOINT_SOURCE_COMMIT for item in origin_records),
        "telemetry_source_commit_locked": all(
            item.get("telemetry_source_commit") == telemetry_commit
            for item in origin_records),
        "checkpoint_config_fingerprints_verified": all(
            bool(item.get("checkpoint_config_fingerprint_sha256"))
            for item in origin_records),
        "behavior_invariance_verified": all(
            item.get("behavior_invariance_verified") is True
            for item in origin_records),
    }
    if not all(strict_checks.values()):
        raise ValueError(f"mature follower audit failed: {strict_checks}")

    by_config = {}
    by_checkpoint = {}
    for config in CONFIGS:
        config_frame = per_eval.loc[per_eval["config"] == config]
        summary = _summarize(config_frame)
        by_config[config] = summary
        by_checkpoint[config] = {}
        for train_seed in TRAIN_SEEDS:
            seed_summary = _summarize(config_frame.loc[
                config_frame["train_seed"] == train_seed])
            by_checkpoint[config][str(train_seed)] = seed_summary

    material = {}
    for config in CONFIGS:
        stable_seed_count = sum(
            _sequential_material(by_checkpoint[config][str(train_seed)])
            for train_seed in TRAIN_SEEDS
        )
        material[config] = {
            "forecast_error": _forecast_material(by_config[config]),
            "sequential_holding": _sequential_material(by_config[config]),
            "sequential_seed_pass_count": int(stable_seed_count),
            "stable_sequential_holding": bool(
                _sequential_material(by_config[config])
                and stable_seed_count >= 3),
        }

    stable_sequential = [
        config for config in CONFIGS
        if material[config]["stable_sequential_holding"]
    ]
    forecast_error = [
        config for config in CONFIGS if material[config]["forecast_error"]
    ]
    general_sequential = len(stable_sequential) == len(CONFIGS)
    if general_sequential and len(forecast_error) == len(CONFIGS):
        diagnosis = "forecast_error_and_sequential_holding"
    elif general_sequential and forecast_error:
        diagnosis = (
            "sequential_holding_with_policy_specific_forecast_error")
    elif general_sequential:
        diagnosis = "sequential_holding_primary"
    elif forecast_error:
        diagnosis = "forecast_error_without_general_sequential_holding"
    else:
        diagnosis = "local_surrogate_mismatch_beyond_forecast"

    return {
        "schema_version": AUDIT_VERSION,
        "status": "mechanical_pass",
        "effect_evidence": False,
        "checkpoint_source_commit": CHECKPOINT_SOURCE_COMMIT,
        "telemetry_source_commit": telemetry_commit,
        "configs": CONFIGS,
        "train_seeds": TRAIN_SEEDS,
        "eval_seeds": EVAL_SEEDS,
        "checkpoint_ep": CHECKPOINT_EP,
        "strict_checks": strict_checks,
        "thresholds": {
            "target_error_material_s": TARGET_ERROR_MATERIAL_S,
            "hold_sign_error_material_rate": HOLD_SIGN_ERROR_MATERIAL_RATE,
            "follower_hold_material_s": FOLLOWER_HOLD_MATERIAL_S,
            "follower_hold_material_rate": FOLLOWER_HOLD_MATERIAL_RATE,
            "stable_sequential_min_checkpoint_seeds": 3,
        },
        "material_by_config": material,
        "by_config": by_config,
        "by_checkpoint": by_checkpoint,
        "pooled": _summarize(per_eval),
        "diagnosis": diagnosis,
        "successor_authorization": {
            "delayed_control_state_objective": general_sequential,
            "forecast_calibration_ablation": bool(forecast_error),
            "forecast_error_configs": forecast_error,
            "abandon_one_step_surrogate": bool(
                not general_sequential and not forecast_error),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True, choices=CONFIGS)
    evaluate_parser.add_argument("--train-seed", required=True, type=int)
    evaluate_parser.add_argument("--checkpoint-run-dir", required=True,
                                 type=Path)
    evaluate_parser.add_argument("--output-root", required=True, type=Path)
    evaluate_parser.add_argument("--telemetry-commit", required=True)
    audit_parser = subparsers.add_parser("aggregate")
    audit_parser.add_argument("--evaluation-root", required=True, type=Path)
    audit_parser.add_argument("--telemetry-commit", required=True)
    audit_parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate_checkpoint(
            config=args.config,
            train_seed=args.train_seed,
            checkpoint_run_dir=args.checkpoint_run_dir,
            output_root=args.output_root,
            telemetry_commit=args.telemetry_commit,
        )
        return

    result = audit_mature_follower_forecast(
        args.evaluation_root,
        telemetry_commit=args.telemetry_commit,
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
