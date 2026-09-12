#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v24 policy-mean upper-target development screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    mujoco_v24_policy_mean_upper_projection_target_development_spec as spec,
)
from scripts.submit_mujoco_v24_policy_mean_upper_projection_target_development_scheduleurm import (  # noqa: E402
    cell_relative_dir,
)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _path_registry(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(row["disturbance_mode"]), int(row["seed"])) for row in rows
    }


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _weight_summary(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    payload = summary.get("projection_consistency_weight_training")
    if not isinstance(payload, dict) or set(payload) != {"upper", "lower"}:
        raise ValueError("v24 consistency summary is missing")
    keys = (
        "active_iteration_count",
        "unweighted_mse_mean",
        "weighted_mse_mean",
        "weight_mean",
        "weight_max",
    )
    normalized: dict[str, dict[str, float]] = {}
    for level in ("upper", "lower"):
        values = payload[level]
        if not isinstance(values, dict):
            raise ValueError("v24 consistency summary level is invalid")
        normalized[level] = {key: float(values[key]) for key in keys}
        numeric = np.asarray(list(normalized[level].values()), dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
            raise ValueError("v24 consistency diagnostics are invalid")
    return normalized


def _policy_mean_summary(summary: dict[str, Any]) -> dict[str, float]:
    payload = summary.get("upper_policy_mean_projection_target_training")
    if not isinstance(payload, dict):
        raise ValueError("v24 policy-mean target summary is missing")
    keys = (
        "active_iteration_count",
        "target_count",
        "sampled_target_delta_rms_mean",
    )
    normalized = {key: float(payload[key]) for key in keys}
    values = np.asarray(list(normalized.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("v24 policy-mean target diagnostics are invalid")
    return normalized


def _load_cell(
    run_name: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = ROOT / cell_relative_dir(
        run_name, environment, arm, optimizer_seed
    )
    summary = json.loads(
        (directory / "cell_summary.json").read_text(encoding="utf-8")
    )
    with (directory / "evaluation_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    return summary, rows


def _validate_cell(
    environment: str,
    arm: str,
    optimizer_seed: int,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    arm_spec = spec.ARMS[str(arm)]
    expected_paths = {
        (mode, int(seed))
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.EVALUATION_SEEDS
    }
    mismatches = {
        "protocol_version": (
            summary.get("protocol_version"),
            spec.FROZEN_CORE_PROTOCOL_VERSION,
        ),
        "code_revision": (
            summary.get("code_revision"),
            spec.FROZEN_ALGORITHM_REVISION,
        ),
        "environment": (summary.get("environment"), str(environment)),
        "optimizer_seed": (
            int(summary.get("optimizer_seed", -1)),
            int(optimizer_seed),
        ),
        "train_seed_roots": (
            tuple(summary.get("rollout_seed_roots", ())),
            spec.TRAIN_SEEDS,
        ),
        "selection_seed_roots": (
            tuple(summary.get("checkpoint_selection_seed_roots", ())),
            spec.SELECTION_SEEDS,
        ),
        "evaluation_seed_roots": (
            tuple(summary.get("eval_seeds", ())),
            spec.EVALUATION_SEEDS,
        ),
        "method": (summary.get("method"), arm_spec["method"]),
        "ppo_clip_ratio": (
            float(summary.get("ppo_clip_ratio", -1.0)),
            spec.PPO_CLIP_RATIO,
        ),
        "terminal_context": (
            bool(summary.get("terminal_reserve_context_enabled")),
            bool(arm_spec["terminal_reserve_context"]),
        ),
        "terminal_projection": (
            bool(summary.get("terminal_reserve_projection_enabled")),
            bool(arm_spec["terminal_reserve_projection"]),
        ),
        "upper_consistency": (
            float(summary.get("upper_projection_consistency_coef", -1.0)),
            float(arm_spec["upper_projection_consistency_coef"]),
        ),
        "lower_consistency": (
            float(summary.get("lower_projection_consistency_coef", -1.0)),
            float(arm_spec["lower_projection_consistency_coef"]),
        ),
        "upper_target_aggregation": (
            summary.get("upper_projection_target_aggregation"),
            arm_spec["upper_projection_target_aggregation"],
        ),
        "consistency_update_mode": (
            summary.get("projection_consistency_update_mode"),
            arm_spec["projection_consistency_update_mode"],
        ),
        "consistency_weighting": (
            summary.get("projection_consistency_weighting"),
            arm_spec["projection_consistency_weighting"],
        ),
        "consistency_schedule": (
            summary.get("projection_consistency_training_schedule"),
            arm_spec["projection_consistency_training_schedule"],
        ),
        "consistency_warmup": (
            float(summary.get("projection_consistency_warmup_fraction", -1.0)),
            float(arm_spec["projection_consistency_warmup_fraction"]),
        ),
        "consistency_ramp": (
            float(summary.get("projection_consistency_ramp_fraction", -1.0)),
            float(arm_spec["projection_consistency_ramp_fraction"]),
        ),
        "iterations": (int(summary.get("iterations", -1)), spec.ITERATIONS),
        "steps": (int(summary.get("steps", -1)), spec.STEPS),
        "checkpoint_minimum_iteration": (
            int(summary.get("checkpoint_minimum_eligible_iteration", -2)),
            spec.CHECKPOINT_MINIMUM_ITERATION,
        ),
        "upper_window": (
            int(summary.get("terminal_reserve_upper_window", -1)),
            spec.TERMINAL_RESERVE_UPPER_WINDOW,
        ),
        "lower_window": (
            int(summary.get("terminal_reserve_lower_window", -1)),
            spec.TERMINAL_RESERVE_LOWER_WINDOW,
        ),
        "checkpoint_score": (
            summary.get("checkpoint_score_mode"),
            spec.CHECKPOINT_SCORE_MODE,
        ),
    }
    drift = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in mismatches.items()
        if observed != expected
    }
    if drift:
        raise ValueError(
            f"v24 cell contract mismatch {environment}/{arm}/"
            f"{optimizer_seed}: " + json.dumps(drift, sort_keys=True)
        )
    if int(summary.get("selected_checkpoint_iteration", -1)) < (
        spec.CHECKPOINT_MINIMUM_ITERATION
    ):
        raise ValueError(
            f"v24 checkpoint selected too early: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    _weight_summary(summary)
    policy_mean_training = _policy_mean_summary(summary)
    expects_policy_mean = arm == spec.POLICY_MEAN_UNIFORM_010
    target_active = (
        policy_mean_training["active_iteration_count"] > 0.0
        and policy_mean_training["target_count"] > 0.0
        and policy_mean_training["sampled_target_delta_rms_mean"] > 0.0
    )
    if target_active != expects_policy_mean:
        raise ValueError(
            f"v24 policy-mean training audit mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    if len(rows) != spec.EXPECTED_EVALUATION_ROWS_PER_CELL:
        raise ValueError(
            f"v24 heldout row count mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    if _path_registry(rows) != expected_paths:
        raise ValueError(
            f"v24 heldout path mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    if not all(_as_bool(row["protocol_valid"]) for row in rows):
        raise ValueError(
            f"v24 invalid rollout protocol: {environment}/{arm}/{optimizer_seed}"
        )
    if not all(
        _as_bool(row["terminal_reserve_context_enabled"])
        and _as_bool(row["terminal_reserve_projection_enabled"])
        for row in rows
    ):
        raise ValueError(
            f"v24 row-level terminal contract mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    evaluation_target_active = all(
        float(row["terminal_reserve_upper_policy_mean_target_count"]) > 0.0
        and abs(float(row[
            "terminal_reserve_upper_policy_mean_target_delta_rms_mean"
        ])) <= 1e-12
        for row in rows
    )
    evaluation_target_disabled = all(
        float(row["terminal_reserve_upper_policy_mean_target_count"]) == 0.0
        and float(row[
            "terminal_reserve_upper_policy_mean_target_delta_rms_mean"
        ]) == 0.0
        for row in rows
    )
    if (
        expects_policy_mean and not evaluation_target_active
    ) or (
        not expects_policy_mean and not evaluation_target_disabled
    ):
        raise ValueError(
            f"v24 policy-mean evaluation audit mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )


def _summarize_cell(
    summary: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "reward": _mean(rows, "episode_return"),
        "selected_checkpoint_iteration": int(
            summary["selected_checkpoint_iteration"]
        ),
        "parameter_count": int(summary["capacity_actual_parameter_count"]),
        "weight_training": _weight_summary(summary),
        "policy_mean_target_training": _policy_mean_summary(summary),
        "certificate_violation_count": sum(
            float(row["terminal_reserve_certificate_violation_count"])
            for row in rows
        ),
        "component_correction_rms": _mean(
            rows, "terminal_reserve_component_correction_rms_mean"
        ),
        "total_correction_rms": _mean(
            rows, "terminal_reserve_correction_rms_mean"
        ),
        "total_action_change_rate": _mean(
            rows, "terminal_reserve_total_action_change_rate"
        ),
        "projection_converged_rate": _mean(
            rows, "terminal_reserve_projection_converged_rate"
        ),
        "recursive_fallback_rate": _mean(
            rows, "terminal_reserve_recursive_fallback_rate"
        ),
        "evaluation_policy_mean_target_count": sum(
            float(row["terminal_reserve_upper_policy_mean_target_count"])
            for row in rows
        ),
        "evaluation_policy_mean_target_delta_rms_max": max(
            float(row[
                "terminal_reserve_upper_policy_mean_target_delta_rms_mean"
            ])
            for row in rows
        ),
        "upper_prefix_power_max": max(
            float(row["terminal_reserve_upper_prefix_power_max"])
            for row in rows
        ),
        "lower_prefix_power_max": max(
            float(row["terminal_reserve_lower_prefix_power_max"])
            for row in rows
        ),
    }


def _reward_delta(
    candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> float:
    candidate_mean = statistics.fmean(float(row["reward"]) for row in candidate)
    baseline_mean = statistics.fmean(float(row["reward"]) for row in baseline)
    return (candidate_mean - baseline_mean) / max(abs(baseline_mean), 1.0)


def _reduction(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    key: str,
) -> float:
    candidate_mean = statistics.fmean(float(row[key]) for row in candidate)
    baseline_mean = statistics.fmean(float(row[key]) for row in baseline)
    return 1.0 - candidate_mean / max(baseline_mean, 1e-12)


def _validity(
    registry: dict[tuple[str, str, int], dict[str, Any]], arm: str
) -> dict[str, Any]:
    cells = [
        registry[(environment, arm, int(seed))]
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]
    gates = {
        "zero_certificate_violations": all(
            float(cell["certificate_violation_count"]) == 0.0 for cell in cells
        ),
        "upper_prefix_budget": all(
            float(cell["upper_prefix_power_max"])
            <= spec.UPPER_HF_RMS_BUDGET**2 + spec.POWER_TOLERANCE
            for cell in cells
        ),
        "lower_prefix_budget": all(
            float(cell["lower_prefix_power_max"])
            <= spec.LOWER_LF_RMS_BUDGET**2 + spec.POWER_TOLERANCE
            for cell in cells
        ),
        "recursive_fallback_bounded": all(
            float(cell["recursive_fallback_rate"])
            <= spec.MAXIMUM_RECURSIVE_FALLBACK_RATE
            for cell in cells
        ),
    }
    return {
        "gates": gates,
        "supported": bool(all(gates.values())),
        "projection_step_tolerance_diagnostic_only": True,
        "minimum_projection_converged_rate": min(
            float(cell["projection_converged_rate"]) for cell in cells
        ),
        "maximum_recursive_fallback_rate": max(
            float(cell["recursive_fallback_rate"]) for cell in cells
        ),
    }


def _training_audit(
    registry: dict[tuple[str, str, int], dict[str, Any]], arm: str
) -> dict[str, Any]:
    cells = [
        registry[(environment, arm, int(seed))]
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]
    levels: dict[str, Any] = {}
    for level in ("upper", "lower"):
        active = float(spec.ARMS[arm][f"{level}_projection_consistency_coef"]) > 0.0
        rows = [cell["weight_training"][level] for cell in cells]
        gates = {
            "activity_matches_coefficient": all(
                (float(row["active_iteration_count"]) > 0.0) == active
                for row in rows
            ),
            "active_mse_positive": (
                not active
                or all(
                    float(row["unweighted_mse_mean"]) > 0.0
                    and float(row["weighted_mse_mean"]) > 0.0
                    for row in rows
                )
            ),
            "active_uniform_weights": (
                not active
                or all(
                    abs(float(row["weight_mean"]) - 1.0) <= 1e-5
                    and abs(float(row["weight_max"]) - 1.0) <= 1e-5
                    for row in rows
                )
            ),
        }
        levels[level] = {
            "active_expected": active,
            "gates": gates,
            "supported": bool(all(gates.values())),
        }
    expects_policy_mean = arm == spec.POLICY_MEAN_UNIFORM_010
    policy_mean_rows = [cell["policy_mean_target_training"] for cell in cells]
    policy_mean_gates = {
        "activity_matches_target_estimator": all(
            (
                float(row["active_iteration_count"]) == spec.ITERATIONS
                and float(row["target_count"]) > 0.0
                and float(row["sampled_target_delta_rms_mean"]) > 0.0
            )
            if expects_policy_mean else (
                float(row["active_iteration_count"]) == 0.0
                and float(row["target_count"]) == 0.0
                and float(row["sampled_target_delta_rms_mean"]) == 0.0
            )
            for row in policy_mean_rows
        ),
        "deterministic_evaluation_matches_policy_mean": all(
            (
                float(cell["evaluation_policy_mean_target_count"]) > 0.0
                and float(cell[
                    "evaluation_policy_mean_target_delta_rms_max"
                ]) <= 1e-12
            )
            if expects_policy_mean else (
                float(cell["evaluation_policy_mean_target_count"]) == 0.0
                and float(cell[
                    "evaluation_policy_mean_target_delta_rms_max"
                ]) == 0.0
            )
            for cell in cells
        ),
    }
    return {
        "levels": levels,
        "policy_mean_target_expected": expects_policy_mean,
        "policy_mean_target_gates": policy_mean_gates,
        "supported": bool(
            all(row["supported"] for row in levels.values())
            and all(policy_mean_gates.values())
        ),
    }


def _candidate_result(
    registry: dict[tuple[str, str, int], dict[str, Any]],
    candidate_arm: str,
    *,
    common_valid: bool,
    training_valid: bool,
) -> dict[str, Any]:
    environments: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        choice = [
            registry[(environment, candidate_arm, int(seed))]
            for seed in spec.OPTIMIZER_SEEDS
        ]
        controls = {
            "zero": [
                registry[(
                    environment,
                    spec.PRIMARY_PROJECTED_BASELINE,
                    int(seed),
                )]
                for seed in spec.OPTIMIZER_SEEDS
            ],
            "causal_first_sample": [
                registry[(
                    environment,
                    spec.PRIMARY_CAUSAL_CONTROL,
                    int(seed),
                )]
                for seed in spec.OPTIMIZER_SEEDS
            ],
            "hindsight_macro_mean": [
                registry[(
                    environment,
                    spec.DIAGNOSTIC_HINDSIGHT_CONTROL,
                    int(seed),
                )]
                for seed in spec.OPTIMIZER_SEEDS
            ],
        }
        candidate_upper_mse = statistics.fmean(
            float(row["weight_training"]["upper"]["unweighted_mse_mean"])
            for row in choice
        )
        first_sample_upper_mse = statistics.fmean(
            float(row["weight_training"]["upper"]["unweighted_mse_mean"])
            for row in controls["causal_first_sample"]
        )
        environments.append({
            "environment": environment,
            "mean_reward": {
                **{
                    name: statistics.fmean(
                        float(row["reward"]) for row in rows
                    )
                    for name, rows in controls.items()
                },
                "candidate": statistics.fmean(
                    float(row["reward"]) for row in choice
                ),
            },
            "reward_delta": {
                name: _reward_delta(choice, rows)
                for name, rows in controls.items()
            },
            "reward_wins_vs_causal_first_sample": sum(
                float(selected["reward"]) > float(control["reward"])
                for selected, control in zip(
                    choice,
                    controls["causal_first_sample"],
                    strict=True,
                )
            ),
            "component_reduction": {
                name: _reduction(
                    choice, rows, "component_correction_rms"
                )
                for name, rows in controls.items()
            },
            "total_reduction": {
                name: _reduction(choice, rows, "total_correction_rms")
                for name, rows in controls.items()
            },
            "upper_consistency_mse": {
                "causal_first_sample": first_sample_upper_mse,
                "candidate": candidate_upper_mse,
                "reduction_vs_causal_first_sample": (
                    1.0
                    - candidate_upper_mse
                    / max(first_sample_upper_mse, 1e-12)
                ),
            },
            "mean_total_correction_rms": statistics.fmean(
                float(row["total_correction_rms"]) for row in choice
            ),
            "mean_total_action_change_rate": statistics.fmean(
                float(row["total_action_change_rate"]) for row in choice
            ),
        })

    reward_wins = sum(
        row["reward_wins_vs_causal_first_sample"] for row in environments
    )
    reward_improved_environments = sum(
        row["reward_delta"]["causal_first_sample"] > 0.0
        for row in environments
    )
    component_improved_environments = sum(
        row["component_reduction"]["zero"]
        >= spec.MINIMUM_CORRECTION_REDUCTION_FRACTION
        for row in environments
    )
    total_improved_environments = sum(
        row["total_reduction"]["zero"]
        >= spec.MINIMUM_CORRECTION_REDUCTION_FRACTION
        for row in environments
    )
    upper_mse_improved_environments = sum(
        row["upper_consistency_mse"]["reduction_vs_causal_first_sample"]
        >= spec.MINIMUM_UPPER_MSE_REDUCTION_FRACTION
        for row in environments
    )
    hopper = next(
        row for row in environments if row["environment"] == "Hopper-v5"
    )
    gates = {
        "all_matched_arms_valid": bool(common_valid),
        "training_activity_audit_supported": bool(training_valid),
        "upper_mse_reduces_in_two_environments": (
            upper_mse_improved_environments
            >= spec.MINIMUM_UPPER_MSE_IMPROVED_ENVIRONMENTS
        ),
        "upper_mse_noninferior_in_every_environment": all(
            row["upper_consistency_mse"][
                "reduction_vs_causal_first_sample"
            ] >= -spec.MAXIMUM_UPPER_MSE_REGRESSION_FRACTION
            for row in environments
        ),
        "reward_floor_vs_all_controls_in_every_environment": all(
            all(
                delta >= -spec.MAXIMUM_REWARD_REGRESSION_FRACTION
                for delta in row["reward_delta"].values()
            )
            for row in environments
        ),
        "reward_wins_in_every_environment": all(
            row["reward_wins_vs_causal_first_sample"]
            >= spec.MINIMUM_REWARD_WINS_PER_ENVIRONMENT
            for row in environments
        ),
        "total_reward_wins": reward_wins >= spec.MINIMUM_TOTAL_REWARD_WINS,
        "reward_improves_vs_causal_in_two_environments": (
            reward_improved_environments
            >= spec.MINIMUM_REWARD_IMPROVED_ENVIRONMENTS
        ),
        "component_reduces_in_two_environments": (
            component_improved_environments
            >= spec.MINIMUM_CORRECTION_IMPROVED_ENVIRONMENTS
        ),
        "total_reduces_in_two_environments": (
            total_improved_environments
            >= spec.MINIMUM_CORRECTION_IMPROVED_ENVIRONMENTS
        ),
        "component_noninferior_vs_all_controls": all(
            all(
                reduction >= -spec.MAXIMUM_CORRECTION_REGRESSION_FRACTION
                for reduction in row["component_reduction"].values()
            )
            for row in environments
        ),
        "total_noninferior_vs_all_controls": all(
            all(
                reduction >= -spec.MAXIMUM_CORRECTION_REGRESSION_FRACTION
                for reduction in row["total_reduction"].values()
            )
            for row in environments
        ),
        "hopper_correction_burden_bounded": (
            hopper["mean_total_correction_rms"]
            <= spec.MAXIMUM_MEAN_TOTAL_CORRECTION_RMS
        ),
    }
    return {
        "arm": candidate_arm,
        "supported": bool(all(gates.values())),
        "gates": gates,
        "reward_wins_vs_causal_first_sample": reward_wins,
        "reward_improved_environment_count": reward_improved_environments,
        "component_improved_environment_count": component_improved_environments,
        "total_improved_environment_count": total_improved_environments,
        "upper_mse_improved_environment_count": upper_mse_improved_environments,
        "environment_results": environments,
    }


def analyze(run_name: str) -> dict[str, Any]:
    registry: dict[tuple[str, str, int], dict[str, Any]] = {}
    path_sets: dict[tuple[str, int], set[frozenset[tuple[str, int]]]] = {}
    parameter_counts: dict[tuple[str, int], set[int]] = {}
    for environment in spec.ENVIRONMENTS:
        for arm in spec.ARMS:
            for optimizer_seed in spec.OPTIMIZER_SEEDS:
                summary, rows = _load_cell(
                    run_name, environment, arm, optimizer_seed
                )
                _validate_cell(environment, arm, optimizer_seed, summary, rows)
                cell = _summarize_cell(summary, rows)
                registry[(environment, arm, int(optimizer_seed))] = cell
                key = (environment, int(optimizer_seed))
                path_sets.setdefault(key, set()).add(
                    frozenset(_path_registry(rows))
                )
                parameter_counts.setdefault(key, set()).add(
                    int(cell["parameter_count"])
                )
    if any(len(values) != 1 for values in path_sets.values()):
        raise ValueError("v24 paired arms do not share heldout paths")
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise ValueError("v24 capacity-matched parameter counts differ")

    validity = {arm: _validity(registry, arm) for arm in spec.ARMS}
    training_audits = {
        arm: _training_audit(registry, arm) for arm in spec.ARMS
    }
    common_valid = all(row["supported"] for row in validity.values())
    training_valid = all(row["supported"] for row in training_audits.values())
    candidates = {
        arm: _candidate_result(
            registry,
            arm,
            common_valid=common_valid,
            training_valid=training_valid,
        )
        for arm in spec.CANDIDATES
    }
    selected = (
        spec.POLICY_MEAN_UNIFORM_010
        if candidates[spec.POLICY_MEAN_UNIFORM_010]["supported"]
        else None
    )
    advances = selected is not None
    return {
        "analysis_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "evidence_role": spec.EVIDENCE_ROLE,
        "cell_count": len(registry),
        "status": spec.ADVANCES_STATUS if advances else spec.STOPS_STATUS,
        "selected_candidate": selected,
        "advance_gate": advances,
        "validity": validity,
        "training_audits": training_audits,
        "candidate_results": candidates,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cells": [
            {"environment": environment, "arm": arm, "optimizer_seed": seed, **cell}
            for (environment, arm, seed), cell in sorted(registry.items())
        ],
    }


def _write_readme(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# MuJoCo v24 Policy-Mean Upper-Target Development",
        "",
        f"- Status: `{result['status']}`",
        f"- Cells: {result['cell_count']}",
        f"- Selected candidate: `{result['selected_candidate']}`",
        f"- Evidence role: `{result['evidence_role']}`",
    ]
    for arm in spec.CANDIDATES:
        candidate = result["candidate_results"][arm]
        lines.extend([
            "",
            f"## {arm}",
            "",
            f"- Advances: `{str(candidate['supported']).lower()}`",
            f"- Reward wins versus causal first-sample control: "
            f"{candidate['reward_wins_vs_causal_first_sample']}/12",
            "",
            "| Environment | Zero reward | Causal reward | Hindsight reward | "
            "Candidate reward | Candidate reward delta zero/causal/hindsight | "
            "Wins vs causal | Upper MSE causal/candidate/reduction | "
            "Component reduction zero/causal/hindsight | "
            "Total reduction zero/causal/hindsight | Total correction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in candidate["environment_results"]:
            reward = row["mean_reward"]
            reward_delta = row["reward_delta"]
            upper_mse = row["upper_consistency_mse"]
            component = row["component_reduction"]
            total = row["total_reduction"]
            lines.append(
                f"| {row['environment']} | {reward['zero']:.3f} | "
                f"{reward['causal_first_sample']:.3f} | "
                f"{reward['hindsight_macro_mean']:.3f} | "
                f"{reward['candidate']:.3f} | "
                f"{reward_delta['zero']:.4f}/"
                f"{reward_delta['causal_first_sample']:.4f}/"
                f"{reward_delta['hindsight_macro_mean']:.4f} | "
                f"{row['reward_wins_vs_causal_first_sample']}/4 | "
                f"{upper_mse['causal_first_sample']:.4f}/"
                f"{upper_mse['candidate']:.4f}/"
                f"{upper_mse['reduction_vs_causal_first_sample']:.4f} | "
                f"{component['zero']:.4f}/"
                f"{component['causal_first_sample']:.4f}/"
                f"{component['hindsight_macro_mean']:.4f} | "
                f"{total['zero']:.4f}/"
                f"{total['causal_first_sample']:.4f}/"
                f"{total['hindsight_macro_mean']:.4f} | "
                f"{row['mean_total_correction_rms']:.4f} |"
            )
        lines.extend(["", "### Frozen Gates", ""])
        lines.extend(
            f"- {name}: `{str(value).lower()}`"
            for name, value in candidate["gates"].items()
        )
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        spec.SELECTION_CONTRACT["claim_boundary"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    result = analyze(args.run_name)
    target = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "results" / args.run_name / "analysis"
    )
    target.mkdir(parents=True, exist_ok=True)
    (target / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(result, target / "README.md")
    print(json.dumps({
        "cell_count": result["cell_count"],
        "selected_candidate": result["selected_candidate"],
        "status": result["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
