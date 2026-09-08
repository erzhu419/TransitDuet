#!/usr/bin/env python3
"""Audit causal follower-gap calibration before defining a V25 objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NOGUARD = "F_freqduet_protocol_v6_noguard_hiro"
CONFIRMED_MAIN = "F_freqduet_protocol_v6_confirmed_main_hiro"
V13_ANCHOR = "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_hiro"
V24_RKL4 = "F_freqduet_protocol_v6_v24_jointproj_rkl_s4_hiro"
CONFIGS = [CONFIRMED_MAIN, NOGUARD, V13_ANCHOR, V24_RKL4]
TRAIN_SEEDS = [30003]
EVAL_SEEDS = [63011, 63029, 63047, 63071]
TRAIN_EPISODES = 2
TARGET_ERROR_MATERIAL_S = 5.0
HOLD_SIGN_ERROR_MATERIAL_RATE = 0.10
FOLLOWER_HOLD_MATERIAL_S = 5.0
FOLLOWER_HOLD_MATERIAL_RATE = 0.25

COUNT_COLUMNS = (
    "follower_forecast_decision_count",
    "follower_forecast_registered_count",
    "follower_forecast_resolved_count",
    "follower_forecast_departure_resolved_count",
)
RESOLVED_METRICS = (
    "follower_forecast_predicted_follower_gap_s_mean",
    "follower_forecast_actual_follower_gap_s_mean",
    "follower_forecast_raw_gap_prediction_error_s_mean",
    "follower_forecast_raw_gap_prediction_mae_s",
    "follower_forecast_post_hold_gap_prediction_error_s_mean",
    "follower_forecast_predicted_target_action_s_mean",
    "follower_forecast_realized_target_action_s_mean",
    "follower_forecast_target_action_prediction_error_s_mean",
    "follower_forecast_target_action_prediction_mae_s",
    "follower_forecast_departure_timing_error_s_mean",
    "follower_forecast_hold_need_false_positive_mean",
    "follower_forecast_hold_need_false_negative_mean",
)
EXACT_RMSE_METRICS = (
    "follower_forecast_raw_gap_prediction_rmse_s",
)
DEPARTURE_METRICS = (
    "follower_forecast_follower_future_hold_s_mean",
    "follower_forecast_follower_future_hold_positive_rate",
)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite follower forecast metric: {column}")
    return values


def _weighted_mean(
    frame: pd.DataFrame,
    column: str,
    weight_column: str,
) -> float:
    values = _numeric(frame, column)
    weights = _numeric(frame, weight_column)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"no observations available for {column}")
    return float(np.dot(values, weights) / total)


def _summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    counts = {
        column: int(round(float(_numeric(frame, column).sum())))
        for column in COUNT_COLUMNS
    }
    decisions = counts[COUNT_COLUMNS[0]]
    registered = counts[COUNT_COLUMNS[1]]
    resolved = counts[COUNT_COLUMNS[2]]
    departed = counts[COUNT_COLUMNS[3]]
    if not 0 < resolved <= registered <= decisions or not 0 < departed <= resolved:
        raise ValueError(f"invalid follower forecast counts: {counts}")
    result: dict[str, float | int] = {
        **counts,
        "valid_rate": float(registered / decisions),
        "resolution_rate": float(resolved / registered),
        "departure_resolution_rate": float(departed / resolved),
    }
    result.update({
        column: _weighted_mean(
            frame, column, "follower_forecast_resolved_count")
        for column in RESOLVED_METRICS
    })
    for column in EXACT_RMSE_METRICS:
        values = _numeric(frame, column)
        weights = _numeric(frame, "follower_forecast_resolved_count")
        result[column] = float(np.sqrt(
            np.dot(values ** 2, weights) / float(weights.sum())))
    result.update({
        column: _weighted_mean(
            frame, column,
            "follower_forecast_departure_resolved_count",
        )
        for column in DEPARTURE_METRICS
    })
    return result


def audit_follower_forecast_calibration(
    aggregate_dir: Path,
) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    manifest_path = aggregate_dir / "matrix_manifest.json"
    per_eval_path = aggregate_dir / "frozen_per_eval.csv"
    if not manifest_path.is_file() or not per_eval_path.is_file():
        raise FileNotFoundError(
            "follower forecast audit requires matrix_manifest.json and "
            "frozen_per_eval.csv")
    manifest = json.loads(manifest_path.read_text())
    per_eval = pd.read_csv(per_eval_path)
    required = {
        "config", "train_seed", "eval_seed", "lower_policy_frozen",
        "lower_critic_frozen", "upper_policy_frozen", *COUNT_COLUMNS,
        *RESOLVED_METRICS, *EXACT_RMSE_METRICS, *DEPARTURE_METRICS,
    }
    missing = sorted(required.difference(per_eval.columns))
    if missing:
        raise ValueError(f"missing follower forecast columns: {missing}")

    expected_rows = len(CONFIGS) * len(TRAIN_SEEDS) * len(EVAL_SEEDS)
    expected_keys = {
        (config, train_seed, eval_seed)
        for config in CONFIGS
        for train_seed in TRAIN_SEEDS
        for eval_seed in EVAL_SEEDS
    }
    actual_keys = set(zip(
        per_eval["config"].astype(str),
        pd.to_numeric(per_eval["train_seed"], errors="coerce").astype(int),
        pd.to_numeric(per_eval["eval_seed"], errors="coerce").astype(int),
    ))
    strict_checks = {
        "strict_complete": manifest.get("strict_complete") is True,
        "run_manifests_verified": manifest.get(
            "run_manifests_verified") is True,
        "common_random_numbers_verified": manifest.get(
            "common_random_numbers_verified") is True,
        "exact_configs": manifest.get("configs") == CONFIGS,
        "exact_train_seeds": manifest.get("train_seeds") == TRAIN_SEEDS,
        "exact_eval_seeds": manifest.get("eval_seeds") == EVAL_SEEDS,
        "two_training_episodes": (
            manifest.get("train_episodes") == TRAIN_EPISODES
            and manifest.get("checkpoint_ep") == TRAIN_EPISODES - 1),
        "exploratory_nonconfirmation": (
            manifest.get("stage") == "exploratory"
            and manifest.get("independent_confirmation") is False),
        "reference_is_v13": manifest.get("reference") == V13_ANCHOR,
        "source_is_clean_and_identified": (
            manifest.get("run_git_provenance", {}).get(
                "tracked_dirty") is False
            and bool(manifest.get("run_git_provenance", {}).get("commit"))),
        "exact_unique_rollouts": (
            len(per_eval) == expected_rows
            and len(actual_keys) == expected_rows
            and actual_keys == expected_keys),
        "policies_frozen": bool(
            (pd.to_numeric(
                per_eval["lower_policy_frozen"], errors="coerce") == 1).all()
            and (pd.to_numeric(
                per_eval["lower_critic_frozen"], errors="coerce") == 1).all()
            and (pd.to_numeric(
                per_eval["upper_policy_frozen"], errors="coerce") == 1).all()),
    }
    if not all(strict_checks.values()):
        raise ValueError(
            f"follower forecast strict checks failed: {strict_checks}")

    by_config = {
        config: _summarize(
            per_eval.loc[per_eval["config"].astype(str) == config])
        for config in CONFIGS
    }
    pooled = _summarize(per_eval)
    forecast_material = bool(
        pooled["follower_forecast_target_action_prediction_mae_s"]
        > TARGET_ERROR_MATERIAL_S
        or pooled["follower_forecast_hold_need_false_positive_mean"]
        + pooled["follower_forecast_hold_need_false_negative_mean"]
        > HOLD_SIGN_ERROR_MATERIAL_RATE)
    sequential_material = bool(
        pooled["follower_forecast_follower_future_hold_s_mean"]
        > FOLLOWER_HOLD_MATERIAL_S
        and pooled[
            "follower_forecast_follower_future_hold_positive_rate"]
        > FOLLOWER_HOLD_MATERIAL_RATE)
    if forecast_material and sequential_material:
        diagnosis = "forecast_error_and_sequential_holding"
    elif forecast_material:
        diagnosis = "forecast_error_primary"
    elif sequential_material:
        diagnosis = "sequential_holding_primary"
    else:
        diagnosis = "local_surrogate_mismatch_beyond_forecast"
    return {
        "schema_version": "freqduet-v25-follower-forecast-audit-v1",
        "status": "mechanical_pass",
        "effect_evidence": False,
        "strict_checks": strict_checks,
        "thresholds": {
            "target_error_material_s": TARGET_ERROR_MATERIAL_S,
            "hold_sign_error_material_rate": (
                HOLD_SIGN_ERROR_MATERIAL_RATE),
            "follower_hold_material_s": FOLLOWER_HOLD_MATERIAL_S,
            "follower_hold_material_rate": FOLLOWER_HOLD_MATERIAL_RATE,
        },
        "by_config": by_config,
        "pooled": pooled,
        "diagnosis": diagnosis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit_follower_forecast_calibration(args.aggregate_dir)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
