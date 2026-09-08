#!/usr/bin/env python3
"""Audit the preregistered V26 historical follower-calibration screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_v26_historical_calibration_common import (
    CANDIDATES,
    CONTROLS,
    PRIORITY,
    V13_ANCHOR,
    candidate_evaluation_checks,
    candidate_training_checks,
    control_evaluation_checks,
    load_aggregate,
    load_training_rows,
    strict_matrix_checks,
    weighted_mean,
)


CONFIGS = [*CONTROLS, *CANDIDATES]
TRAIN_SEEDS = [32013, 32031, 32053, 32077]
EVAL_SEEDS = [65017, 65041, 65059, 65083]
TRAIN_EPISODES = 40
TARGET_MAE_IMPROVEMENT_S = 0.5
TARGET_MAE_SEED_IMPROVEMENT_S = 0.25
SIGN_ERROR_TOLERANCE = 0.005
HEADWAY_CV_IMPROVEMENT = -0.003
JOURNEY_NONINFERIORITY_MIN = 0.10
SERVICE_COST_NONINFERIORITY = 0.003
UNSERVED_RATE_NONINFERIORITY = 0.002
ACTION_MEAN_NONINFERIORITY_S = 1.0


def _paired_rows(
    per_eval: pd.DataFrame, candidate: str, reference: str
) -> pd.DataFrame:
    candidate_rows = per_eval.loc[
        per_eval["config"] == candidate].copy()
    reference_rows = per_eval.loc[
        per_eval["config"] == reference].copy()
    return candidate_rows.merge(
        reference_rows,
        on=["train_seed", "eval_seed"],
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )


def _paired_delta(merged: pd.DataFrame, metric: str) -> float:
    candidate = pd.to_numeric(
        merged[f"{metric}_candidate"], errors="coerce")
    reference = pd.to_numeric(
        merged[f"{metric}_reference"], errors="coerce")
    values = (candidate - reference).to_numpy(dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"invalid V26 paired outcome metric {metric}")
    return float(values.mean())


def _forecast_checks(
    rows: pd.DataFrame,
) -> tuple[dict[str, bool], dict[str, object]]:
    effective_mae = weighted_mean(
        rows, "follower_forecast_target_action_prediction_mae_s")
    base_mae = weighted_mean(
        rows, "follower_forecast_base_target_action_prediction_mae_s")
    effective_sign_error = (
        weighted_mean(rows, "follower_forecast_hold_need_false_positive_mean")
        + weighted_mean(
            rows, "follower_forecast_hold_need_false_negative_mean"))
    base_sign_error = (
        weighted_mean(
            rows, "follower_forecast_base_hold_need_false_positive_mean")
        + weighted_mean(
            rows, "follower_forecast_base_hold_need_false_negative_mean"))

    seed_diagnostics = {}
    seed_mae_improvements = []
    seed_sign_regressions = []
    for train_seed, seed_rows in rows.groupby("train_seed"):
        seed_effective_mae = weighted_mean(
            seed_rows,
            "follower_forecast_target_action_prediction_mae_s")
        seed_base_mae = weighted_mean(
            seed_rows,
            "follower_forecast_base_target_action_prediction_mae_s")
        seed_effective_sign = (
            weighted_mean(
                seed_rows,
                "follower_forecast_hold_need_false_positive_mean")
            + weighted_mean(
                seed_rows,
                "follower_forecast_hold_need_false_negative_mean"))
        seed_base_sign = (
            weighted_mean(
                seed_rows,
                "follower_forecast_base_hold_need_false_positive_mean")
            + weighted_mean(
                seed_rows,
                "follower_forecast_base_hold_need_false_negative_mean"))
        mae_improvement = seed_base_mae - seed_effective_mae
        sign_regression = seed_effective_sign - seed_base_sign
        seed_mae_improvements.append(mae_improvement)
        seed_sign_regressions.append(sign_regression)
        seed_diagnostics[str(int(train_seed))] = {
            "effective_target_mae_s": seed_effective_mae,
            "base_target_mae_s": seed_base_mae,
            "target_mae_improvement_s": mae_improvement,
            "effective_sign_error_rate": seed_effective_sign,
            "base_sign_error_rate": seed_base_sign,
            "sign_error_regression": sign_regression,
        }

    checks = {
        "pooled_target_mae_improves_by_half_second": bool(
            base_mae - effective_mae >= TARGET_MAE_IMPROVEMENT_S),
        "pooled_sign_error_is_noninferior": bool(
            effective_sign_error
            <= base_sign_error + SIGN_ERROR_TOLERANCE),
        "target_mae_improves_in_at_least_three_seeds": bool(
            sum(value >= TARGET_MAE_SEED_IMPROVEMENT_S
                for value in seed_mae_improvements) >= 3),
        "sign_error_noninferior_in_at_least_three_seeds": bool(
            sum(value <= SIGN_ERROR_TOLERANCE
                for value in seed_sign_regressions) >= 3),
    }
    diagnostics = {
        "effective_target_mae_s": effective_mae,
        "base_target_mae_s": base_mae,
        "target_mae_improvement_s": base_mae - effective_mae,
        "effective_sign_error_rate": effective_sign_error,
        "base_sign_error_rate": base_sign_error,
        "sign_error_regression": effective_sign_error - base_sign_error,
        "by_train_seed": seed_diagnostics,
    }
    return checks, diagnostics


def _outcome_checks(
    per_eval: pd.DataFrame,
    candidate: str,
) -> tuple[dict[str, bool], dict[str, object]]:
    merged = _paired_rows(per_eval, candidate, V13_ANCHOR)
    expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
    metrics = {
        "headway_cv": "headway_cv",
        "journey_min": "restricted_total_journey_horizon_min",
        "service_cost": "service_cost",
        "unserved_rate": "passenger_unserved_rate",
        "lower_action_mean_s": "lower_action_mean",
    }
    deltas = {
        label: _paired_delta(merged, metric)
        for label, metric in metrics.items()
    }
    cv_by_seed = (
        pd.to_numeric(
            merged["headway_cv_candidate"], errors="coerce")
        - pd.to_numeric(
            merged["headway_cv_reference"], errors="coerce")
    ).groupby(merged["train_seed"]).mean()
    checks = {
        "paired_rollouts_complete": len(merged) == expected_pairs,
        "headway_cv_improves_v13": (
            deltas["headway_cv"] <= HEADWAY_CV_IMPROVEMENT),
        "headway_cv_improves_in_at_least_three_seeds": bool(
            (cv_by_seed < 0.0).sum() >= 3),
        "journey_noninferior_to_v13": (
            deltas["journey_min"] <= JOURNEY_NONINFERIORITY_MIN),
        "service_cost_noninferior_to_v13": (
            deltas["service_cost"] <= SERVICE_COST_NONINFERIORITY),
        "unserved_rate_noninferior_to_v13": (
            deltas["unserved_rate"] <= UNSERVED_RATE_NONINFERIORITY),
        "lower_action_noninferior_to_v13": (
            deltas["lower_action_mean_s"]
            <= ACTION_MEAN_NONINFERIORITY_S),
    }
    diagnostics = {
        "paired_delta_vs_v13": deltas,
        "headway_cv_delta_by_train_seed": {
            str(int(seed)): float(value)
            for seed, value in cv_by_seed.items()
        },
    }
    return checks, diagnostics


def evaluate_v26_historical_calibration_screen(
    aggregate_dir: Path,
    log_roots: list[Path],
) -> dict[str, object]:
    manifest, per_eval = load_aggregate(aggregate_dir)
    strict_checks = strict_matrix_checks(
        manifest,
        per_eval,
        configs=CONFIGS,
        train_seeds=TRAIN_SEEDS,
        eval_seeds=EVAL_SEEDS,
        train_episodes=TRAIN_EPISODES,
        reference=V13_ANCHOR,
    )
    if not all(strict_checks.values()):
        raise ValueError(
            f"V26 screen strict checks failed: {strict_checks}")

    training = load_training_rows(
        log_roots,
        train_seeds=TRAIN_SEEDS,
        train_episodes=TRAIN_EPISODES,
    )
    control_checks = {
        control: control_evaluation_checks(
            per_eval.loc[per_eval["config"] == control].copy())
        for control in CONTROLS
    }
    if not all(all(checks.values()) for checks in control_checks.values()):
        raise ValueError(
            f"V26 control identity checks failed: {control_checks}")

    candidate_results = {}
    passing = set()
    for candidate in CANDIDATES:
        train_rows = training.loc[training["config"] == candidate].copy()
        eval_rows = per_eval.loc[per_eval["config"] == candidate].copy()
        training_checks, training_diagnostics = candidate_training_checks(
            train_rows,
            candidate=candidate,
            train_seeds=TRAIN_SEEDS,
            train_episodes=TRAIN_EPISODES,
        )
        evaluation_checks, evaluation_diagnostics = (
            candidate_evaluation_checks(
                eval_rows,
                candidate=candidate,
                train_seeds=TRAIN_SEEDS,
                eval_seeds=EVAL_SEEDS,
                train_episodes=TRAIN_EPISODES,
            ))
        forecast_checks, forecast_diagnostics = _forecast_checks(eval_rows)
        outcome_checks, outcome_diagnostics = _outcome_checks(
            per_eval, candidate)
        passes = bool(
            all(training_checks.values())
            and all(evaluation_checks.values())
            and all(forecast_checks.values())
            and all(outcome_checks.values()))
        if passes:
            passing.add(candidate)
        candidate_results[candidate] = {
            "training_checks": training_checks,
            "evaluation_checks": evaluation_checks,
            "forecast_checks": forecast_checks,
            "outcome_checks": outcome_checks,
            "training_diagnostics": training_diagnostics,
            "evaluation_diagnostics": evaluation_diagnostics,
            "forecast_diagnostics": forecast_diagnostics,
            "outcome_diagnostics": outcome_diagnostics,
            "passes": passes,
        }

    selected = next(
        (candidate for candidate in PRIORITY if candidate in passing), None)
    return {
        "gate_version": "freqduet-v26-historical-calibration-screen-v1",
        "status": (
            "exploratory_candidate_selected" if selected else "no_pass"),
        "claim_eligible": False,
        "selected_for_confirmation": selected,
        "registered_priority": PRIORITY,
        "thresholds": {
            "target_mae_improvement_s": TARGET_MAE_IMPROVEMENT_S,
            "target_mae_seed_improvement_s": (
                TARGET_MAE_SEED_IMPROVEMENT_S),
            "sign_error_tolerance": SIGN_ERROR_TOLERANCE,
            "headway_cv_delta_max": HEADWAY_CV_IMPROVEMENT,
            "journey_delta_min_max": JOURNEY_NONINFERIORITY_MIN,
            "service_cost_delta_max": SERVICE_COST_NONINFERIORITY,
            "unserved_rate_delta_max": UNSERVED_RATE_NONINFERIORITY,
            "lower_action_mean_delta_s_max": (
                ACTION_MEAN_NONINFERIORITY_S),
        },
        "strict_checks": strict_checks,
        "control_checks": control_checks,
        "candidate_results": candidate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument(
        "--logs-root", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-selection", action="store_true")
    args = parser.parse_args()
    result = evaluate_v26_historical_calibration_screen(
        args.aggregate_dir, args.logs_root)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)
    if args.require_selection and result["selected_for_confirmation"] is None:
        raise SystemExit("V26 screen selected no confirmation candidate")
    print("DONE V26 historical-calibration screen gate")


if __name__ == "__main__":
    main()
