#!/usr/bin/env python3
"""Analyze the frozen v14.25 bias-matched critic preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts import mujoco_v14_25_bias_matched_critic_preflight_spec as spec


ROOT = Path(__file__).resolve().parents[1]


def cell_relative_dir(run_name: str, environment: str, seed: int) -> Path:
    return Path("results") / run_name / "cells" / environment / f"replicate_{seed}"


def _validate_payload(payload: dict[str, Any], environment: str, seed: int) -> None:
    expected = {
        "probe_version": spec.PROBE_VERSION,
        "environment": str(environment),
        "optimizer_seed": int(seed),
        "critic_train_roots": list(spec.CRITIC_TRAIN_ROOTS),
        "critic_holdout_roots": list(spec.CRITIC_HOLDOUT_ROOTS),
        "design_roots": list(spec.DESIGN_ROOTS),
        "validation_roots": list(spec.VALIDATION_ROOTS),
        "critic_train_base_path_count": spec.EXPECTED_CRITIC_TRAIN_BASE_PATH_COUNT,
        "critic_holdout_base_path_count": spec.EXPECTED_CRITIC_HOLDOUT_BASE_PATH_COUNT,
        "critic_train_path_count": spec.EXPECTED_CRITIC_TRAIN_PATH_COUNT,
        "critic_holdout_path_count": spec.EXPECTED_CRITIC_HOLDOUT_PATH_COUNT,
        "design_path_count": spec.EXPECTED_DESIGN_PATH_COUNT,
        "validation_path_count": spec.EXPECTED_VALIDATION_PATH_COUNT,
        "critic_collection_mode": spec.CRITIC_COLLECTION_MODE,
        "critic_intervention_bias_rms": spec.CRITIC_INTERVENTION_BIAS_RMS,
        "critic_intervention_variants": list(spec.CRITIC_INTERVENTION_VARIANTS),
        "actor_update_scope": spec.ACTOR_UPDATE_SCOPE,
        "critic_seeds": list(spec.CRITIC_ENSEMBLE_SEEDS),
        "critic_hidden_dim": spec.CRITIC_HIDDEN_DIM,
        "critic_epochs": spec.CRITIC_EPOCHS,
        "critic_minibatch_size": spec.CRITIC_MINIBATCH_SIZE,
        "critic_learning_rate": spec.CRITIC_LEARNING_RATE,
        "cost_return_horizon_decisions": {
            "upper": spec.UPPER_COST_RETURN_HORIZON_DECISIONS,
            "lower": spec.LOWER_COST_RETURN_HORIZON_DECISIONS,
        },
        "critic_minimum_holdout_r2": spec.CRITIC_MINIMUM_HOLDOUT_R2,
        "critic_minimum_action_permutation_mse_increase": (
            spec.CRITIC_MINIMUM_ACTION_PERMUTATION_MSE_INCREASE
        ),
        "minimum_gradient_median_cosine": spec.MINIMUM_GRADIENT_MEDIAN_COSINE,
        "actor_state_limit": spec.ACTOR_STATE_LIMIT_PER_LEVEL,
        "actor_step_rms_values": list(spec.ACTOR_STEP_RMS_VALUES),
        "minimum_reduction": spec.MINIMUM_REDUCTION,
        "funnel_multiplier": spec.FUNNEL_MULTIPLIER,
        "workers": spec.WORKERS,
        "risk_mode": spec.RISK_MODE,
        "cvar_alpha": spec.CVAR_ALPHA,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v14.25 payload contract mismatch: {mismatches}")
    if int(payload.get("actor_update_parameter_count", 0)) < 1:
        raise ValueError("v14.25 payload lacks output-bias parameters")
    metrics = payload.get("critic_metrics")
    gradients = payload.get("gradient_metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"upper", "lower"}:
        raise ValueError("v14.25 payload lacks both critic metric blocks")
    gate = payload.get("gradient_error") is None
    for level in ("upper", "lower"):
        gate = gate and (
            float(metrics[level]["ensemble_holdout_r2"])
            > spec.CRITIC_MINIMUM_HOLDOUT_R2
            and float(metrics[level]["action_permutation_mse_increase_fraction"])
            > spec.CRITIC_MINIMUM_ACTION_PERMUTATION_MSE_INCREASE
        )
        if payload.get("gradient_error") is None:
            if not isinstance(gradients, dict) or level not in gradients:
                raise ValueError("v14.25 payload lacks actor-gradient metrics")
            gate = gate and (
                float(gradients[level]["median_gradient_cosine"])
                > spec.MINIMUM_GRADIENT_MEDIAN_COSINE
            )
    if bool(payload.get("critic_gate_pass")) != bool(gate):
        raise ValueError("v14.25 critic gate does not reproduce from metrics")
    expected_candidates = spec.EXPECTED_CANDIDATE_COUNT if gate else 0
    if int(payload.get("candidate_count", -1)) != expected_candidates:
        raise ValueError("v14.25 candidate count violates critic gate")
    if (
        payload.get("selected_design_candidate") is None
        and payload.get("validation_candidate") is not None
    ):
        raise ValueError("v14.25 validation candidate lacks design selection")


def _metric(snapshot: dict[str, Any] | None, key: str) -> float | int | None:
    if snapshot is None:
        return None
    value = snapshot[key]
    return int(value) if key.endswith("count") else float(value)


def analyze_run(run_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    rows = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            path = ROOT / cell_relative_dir(run_name, environment, seed) / "probe.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing v14.25 probe result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            _validate_payload(payload, environment, seed)
            selected = payload.get("selected_design_candidate")
            design_candidate = None if selected is None else selected["snapshot"]
            validation_baseline = payload.get("validation_baseline")
            validation_candidate = payload.get("validation_candidate")
            baseline_merit = _metric(validation_baseline, "frequency_violation_merit")
            candidate_merit = _metric(validation_candidate, "frequency_violation_merit")
            relative_reduction = None
            if baseline_merit is not None and candidate_merit is not None and baseline_merit > 0:
                relative_reduction = (baseline_merit - candidate_merit) / baseline_merit
            rows.append({
                "environment": environment,
                "optimizer_seed": int(seed),
                "actor_update_parameter_count": int(payload["actor_update_parameter_count"]),
                "upper_holdout_r2": float(payload["critic_metrics"]["upper"]["ensemble_holdout_r2"]),
                "lower_holdout_r2": float(payload["critic_metrics"]["lower"]["ensemble_holdout_r2"]),
                "upper_action_permutation_mse_increase": float(payload["critic_metrics"]["upper"]["action_permutation_mse_increase_fraction"]),
                "lower_action_permutation_mse_increase": float(payload["critic_metrics"]["lower"]["action_permutation_mse_increase_fraction"]),
                "upper_gradient_median_cosine": (
                    None if "upper" not in payload["gradient_metrics"]
                    else float(payload["gradient_metrics"]["upper"]["median_gradient_cosine"])
                ),
                "lower_gradient_median_cosine": (
                    None if "lower" not in payload["gradient_metrics"]
                    else float(payload["gradient_metrics"]["lower"]["median_gradient_cosine"])
                ),
                "critic_gate_pass": bool(payload["critic_gate_pass"]),
                "design_eligible_candidate_count": int(payload["design_eligible_candidate_count"]),
                "selected_step_rms": None if selected is None else float(selected["step_rms"]),
                "design_baseline_merit": float(payload["design_baseline"]["frequency_violation_merit"]),
                "design_candidate_merit": _metric(design_candidate, "frequency_violation_merit"),
                "validation_baseline_merit": baseline_merit,
                "validation_candidate_merit": candidate_merit,
                "validation_relative_merit_reduction": relative_reduction,
                "validation_reward_violation_count": _metric(validation_candidate, "reward_violation_count"),
                "validation_frequency_violation_count": _metric(validation_candidate, "frequency_violation_count"),
                "validation_worst_frequency_violation": _metric(validation_candidate, "worst_frequency_violation"),
                "validation_supported": bool(payload["validation_supported"]),
            })
    supported = sum(row["validation_supported"] for row in rows)
    result = {
        "analysis_version": "mujoco_v14_25_bias_matched_critic_preflight_v1",
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(rows),
        "supported_cell_count": supported,
        "status": (
            "bias_matched_critic_preflight_supported"
            if supported == spec.EXPECTED_CELL_COUNT
            else "bias_matched_critic_preflight_not_supported"
        ),
        "cells": rows,
    }
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "bias_matched_critic_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (target / "bias_matched_critic_preflight.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_name, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "supported_cell_count": result["supported_cell_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
