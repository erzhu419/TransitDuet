#!/usr/bin/env python3
"""Audit the preregistered V27 causal multi-step value screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_v27_multistep_value_common import (
    CANDIDATES,
    CONTROLS,
    PRIORITY,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    candidate_evaluation_checks,
    candidate_training_checks,
    control_evaluation_checks,
    load_aggregate,
    load_training_rows,
    strict_matrix_checks,
)


CONFIGS = [*CONTROLS, *CANDIDATES]
TRAIN_SEEDS = [31013, 31031, 31057, 31081]
EVAL_SEEDS = [65011, 65029, 65047, 65071]
TRAIN_EPISODES = 40
V13_CV_DELTA_MAX = -0.002
V13_JOURNEY_DELTA_MAX_MIN = 0.05
V13_SERVICE_COST_DELTA_MAX = 0.003
V13_ACTION_DELTA_MAX_S = 0.25
V13_HOLDING_RATIO_MAX = 1.03
V13_UNSERVED_DELTA_MAX = 0.0
V19_JOURNEY_DELTA_MAX_MIN = -0.05
V19_CV_DELTA_MAX = 0.001


def _paired_rows(
    per_eval: pd.DataFrame, candidate: str, reference: str
) -> pd.DataFrame:
    return per_eval.loc[per_eval["config"] == candidate].merge(
        per_eval.loc[per_eval["config"] == reference],
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
        raise ValueError(f"invalid V27 paired outcome metric {metric}")
    return float(values.mean())


def _mean(rows: pd.DataFrame, config: str, metric: str) -> float:
    values = pd.to_numeric(
        rows.loc[rows["config"] == config, metric], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"invalid V27 outcome metric {metric} for {config}")
    return float(values.mean())


def outcome_checks(
    per_eval: pd.DataFrame,
    candidate: str,
) -> tuple[dict[str, bool], dict[str, object]]:
    expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
    v13 = _paired_rows(per_eval, candidate, V13_ANCHOR)
    v19 = _paired_rows(per_eval, candidate, V13_ZERO_HOLD_ADVANTAGE)
    metrics = {
        "headway_cv": "headway_cv",
        "journey_min": "restricted_total_journey_horizon_min",
        "service_cost": "service_cost",
        "unserved_rate": "passenger_unserved_rate",
        "lower_action_mean_s": "lower_action_mean",
        "holding_vehicle_s": "holding_vehicle_seconds",
    }
    v13_deltas = {
        label: _paired_delta(v13, metric) for label, metric in metrics.items()
    }
    v19_deltas = {
        label: _paired_delta(v19, metric) for label, metric in metrics.items()
    }
    cv_by_seed = (
        pd.to_numeric(v13["headway_cv_candidate"], errors="coerce")
        - pd.to_numeric(v13["headway_cv_reference"], errors="coerce")
    ).groupby(v13["train_seed"]).mean()
    candidate_holding = _mean(
        per_eval, candidate, "holding_vehicle_seconds")
    v13_holding = _mean(
        per_eval, V13_ANCHOR, "holding_vehicle_seconds")
    v19_holding = _mean(
        per_eval, V13_ZERO_HOLD_ADVANTAGE, "holding_vehicle_seconds")
    checks = {
        "paired_v13_rollouts_complete": len(v13) == expected_pairs,
        "paired_v19_rollouts_complete": len(v19) == expected_pairs,
        "headway_cv_improves_v13": (
            v13_deltas["headway_cv"] <= V13_CV_DELTA_MAX),
        "headway_cv_improves_v13_in_at_least_three_train_seeds": bool(
            (cv_by_seed <= V13_CV_DELTA_MAX).sum() >= 3),
        "journey_noninferior_to_v13": (
            v13_deltas["journey_min"] <= V13_JOURNEY_DELTA_MAX_MIN),
        "service_cost_noninferior_to_v13": (
            v13_deltas["service_cost"] <= V13_SERVICE_COST_DELTA_MAX),
        "lower_action_noninferior_to_v13": (
            v13_deltas["lower_action_mean_s"] <= V13_ACTION_DELTA_MAX_S),
        "holding_noninferior_to_v13": (
            candidate_holding <= V13_HOLDING_RATIO_MAX * v13_holding),
        "unserved_rate_does_not_increase_vs_v13": (
            v13_deltas["unserved_rate"] <= V13_UNSERVED_DELTA_MAX),
        "journey_improves_v19": (
            v19_deltas["journey_min"] <= V19_JOURNEY_DELTA_MAX_MIN),
        "headway_cv_noninferior_to_v19": (
            v19_deltas["headway_cv"] <= V19_CV_DELTA_MAX),
        "holding_does_not_increase_vs_v19": (
            candidate_holding <= v19_holding),
    }
    diagnostics = {
        "paired_delta_vs_v13": v13_deltas,
        "paired_delta_vs_v19": v19_deltas,
        "headway_cv_delta_vs_v13_by_train_seed": {
            str(int(seed)): float(value) for seed, value in cv_by_seed.items()
        },
        "holding_vehicle_seconds_mean": candidate_holding,
        "v13_holding_vehicle_seconds_mean": v13_holding,
        "v19_holding_vehicle_seconds_mean": v19_holding,
    }
    return checks, diagnostics


def evaluate_v27_multistep_value_screen(
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
        raise ValueError(f"V27 screen strict checks failed: {strict_checks}")

    required_outcomes = {
        "headway_cv", "restricted_total_journey_horizon_min",
        "service_cost", "passenger_unserved_rate", "lower_action_mean",
        "holding_vehicle_seconds",
    }
    missing = sorted(required_outcomes - set(per_eval.columns))
    if missing:
        raise ValueError(f"V27 outcome metrics are missing: {missing}")
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
            f"V27 control identity checks failed: {control_checks}")

    candidate_results = {}
    passing = set()
    for candidate in CANDIDATES:
        training_checks, training_diagnostics = candidate_training_checks(
            training.loc[training["config"] == candidate].copy(),
            candidate=candidate,
            train_seeds=TRAIN_SEEDS,
            train_episodes=TRAIN_EPISODES,
        )
        evaluation_checks, evaluation_diagnostics = (
            candidate_evaluation_checks(
                per_eval.loc[per_eval["config"] == candidate].copy(),
                candidate=candidate,
                train_seeds=TRAIN_SEEDS,
                eval_seeds=EVAL_SEEDS,
            ))
        candidate_outcome_checks, outcome_diagnostics = outcome_checks(
            per_eval, candidate)
        passes = bool(
            all(training_checks.values())
            and all(evaluation_checks.values())
            and all(candidate_outcome_checks.values()))
        if passes:
            passing.add(candidate)
        candidate_results[candidate] = {
            "training_checks": training_checks,
            "evaluation_checks": evaluation_checks,
            "outcome_checks": candidate_outcome_checks,
            "training_diagnostics": training_diagnostics,
            "evaluation_diagnostics": evaluation_diagnostics,
            "outcome_diagnostics": outcome_diagnostics,
            "passes": passes,
        }

    selected = next(
        (candidate for candidate in PRIORITY if candidate in passing), None)
    return {
        "gate_version": "freqduet-v27-causal-multistep-value-screen-v1",
        "status": (
            "exploratory_candidate_selected" if selected else "no_pass"),
        "claim_eligible": False,
        "selected_for_confirmation": selected,
        "registered_priority": PRIORITY,
        "thresholds": {
            "v13_headway_cv_delta_max": V13_CV_DELTA_MAX,
            "v13_headway_cv_train_seed_blocks_required": 3,
            "v13_journey_delta_min_max": V13_JOURNEY_DELTA_MAX_MIN,
            "v13_service_cost_delta_max": V13_SERVICE_COST_DELTA_MAX,
            "v13_lower_action_delta_s_max": V13_ACTION_DELTA_MAX_S,
            "v13_holding_ratio_max": V13_HOLDING_RATIO_MAX,
            "v13_unserved_rate_delta_max": V13_UNSERVED_DELTA_MAX,
            "v19_journey_delta_min_max": V19_JOURNEY_DELTA_MAX_MIN,
            "v19_headway_cv_delta_max": V19_CV_DELTA_MAX,
            "v19_holding_delta_max": 0.0,
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
    result = evaluate_v27_multistep_value_screen(
        args.aggregate_dir, args.logs_root)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)
    if args.require_selection and result["selected_for_confirmation"] is None:
        raise SystemExit("V27 screen selected no confirmation candidate")
    print("DONE V27 causal-multistep-value screen gate")


if __name__ == "__main__":
    main()
