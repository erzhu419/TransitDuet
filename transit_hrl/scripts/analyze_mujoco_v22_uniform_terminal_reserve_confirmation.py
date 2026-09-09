#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v22 terminal-reserve confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    mujoco_v22_uniform_terminal_reserve_confirmation_spec as spec,
)
from scripts.submit_mujoco_v22_uniform_terminal_reserve_confirmation_scheduleurm import (  # noqa: E402
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


def _weight_summary(summary: dict[str, Any]) -> dict[str, Any]:
    payload = summary.get("projection_consistency_weight_training")
    if not isinstance(payload, dict) or set(payload) != {"upper", "lower"}:
        raise ValueError("v22 consistency summary is missing")
    keys = (
        "active_iteration_count",
        "unweighted_mse_mean",
        "weighted_mse_mean",
        "weight_mean",
        "weight_max",
    )
    normalized: dict[str, Any] = {}
    for level in ("upper", "lower"):
        values = payload[level]
        if not isinstance(values, dict):
            raise ValueError("v22 consistency summary level is invalid")
        normalized[level] = {key: float(values[key]) for key in keys}
        numeric = np.asarray(list(normalized[level].values()), dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
            raise ValueError("v22 consistency diagnostics are invalid")
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
            float(summary.get(
                "projection_consistency_warmup_fraction", -1.0
            )),
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
            f"v22 cell contract mismatch {environment}/{arm}/"
            f"{optimizer_seed}: " + json.dumps(drift, sort_keys=True)
        )
    if int(summary.get("selected_checkpoint_iteration", -1)) < (
        spec.CHECKPOINT_MINIMUM_ITERATION
    ):
        raise ValueError(
            f"v22 checkpoint selected too early: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    _weight_summary(summary)
    if len(rows) != spec.EXPECTED_EVALUATION_ROWS_PER_CELL:
        raise ValueError(
            f"v22 heldout row count mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    if _path_registry(rows) != expected_paths:
        raise ValueError(
            f"v22 heldout path mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    if not all(_as_bool(row["protocol_valid"]) for row in rows):
        raise ValueError(
            f"v22 invalid rollout protocol: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    expected_projection = bool(arm_spec["terminal_reserve_projection"])
    if not all(
        _as_bool(row["terminal_reserve_context_enabled"])
        and (
            _as_bool(row["terminal_reserve_projection_enabled"])
            == expected_projection
        )
        for row in rows
    ):
        raise ValueError(
            f"v22 row-level terminal contract mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )


def _summarize_cell(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    projected: bool,
) -> dict[str, Any]:
    result = {
        "reward": _mean(rows, "episode_return"),
        "selected_checkpoint_iteration": int(
            summary["selected_checkpoint_iteration"]
        ),
        "parameter_count": int(summary["capacity_actual_parameter_count"]),
        "weight_training": _weight_summary(summary),
    }
    if projected:
        result.update({
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
            "fixed_total_rate": _mean(
                rows, "terminal_reserve_fixed_total_rate"
            ),
            "projection_converged_rate": _mean(
                rows, "terminal_reserve_projection_converged_rate"
            ),
            "recursive_fallback_rate": _mean(
                rows, "terminal_reserve_recursive_fallback_rate"
            ),
            "upper_prefix_power_max": max(
                float(row["terminal_reserve_upper_prefix_power_max"])
                for row in rows
            ),
            "lower_prefix_power_max": max(
                float(row["terminal_reserve_lower_prefix_power_max"])
                for row in rows
            ),
        })
    else:
        result.update({
            "raw_prefix_budget_violation_count": sum(
                float(row[
                    "terminal_reserve_raw_prefix_budget_violation_count"
                ])
                for row in rows
            ),
            "raw_upper_prefix_power_max": max(
                float(row["terminal_reserve_raw_upper_prefix_power_max"])
                for row in rows
            ),
            "raw_lower_prefix_power_max": max(
                float(row["terminal_reserve_raw_lower_prefix_power_max"])
                for row in rows
            ),
        })
    return result


def _reward_delta(candidate: np.ndarray, baseline: np.ndarray) -> float:
    base = float(np.mean(baseline))
    return float((np.mean(candidate) - base) / max(abs(base), 1.0))


def _reduction(candidate: np.ndarray, baseline: np.ndarray) -> float:
    base = float(np.mean(baseline))
    return float(1.0 - np.mean(candidate) / max(base, 1e-12))


def _paired_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    confidence: float,
    seed_offset: int,
) -> dict[str, float]:
    choice = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(baseline, dtype=np.float64)
    expected = (len(spec.OPTIMIZER_SEEDS),)
    if choice.shape != expected or control.shape != expected:
        raise ValueError("v22 bootstrap requires one pair per optimizer root")
    rng = np.random.default_rng(spec.BOOTSTRAP_SEED + int(seed_offset))
    indices = rng.integers(
        0,
        choice.size,
        size=(spec.BOOTSTRAP_DRAWS, choice.size),
    )
    draws = np.asarray([
        statistic(choice[index], control[index]) for index in indices
    ])
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "estimate": statistic(choice, control),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "confidence": float(confidence),
    }


def _pooled_statistic(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    metric: str,
) -> float:
    choice = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(baseline, dtype=np.float64)
    if choice.ndim != 2 or choice.shape != control.shape:
        raise ValueError("v22 pooled matrices must align")
    if metric == "reward":
        scales = np.maximum(np.abs(np.mean(control, axis=0)), 1.0)
        return float(np.mean(
            (np.mean(choice, axis=0) - np.mean(control, axis=0)) / scales
        ))
    if metric == "reduction":
        scales = np.maximum(np.mean(control, axis=0), 1e-12)
        return float(np.mean(1.0 - np.mean(choice, axis=0) / scales))
    raise ValueError(f"unknown pooled metric: {metric}")


def _paired_pooled_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    metric: str,
    seed_offset: int,
) -> dict[str, float]:
    choice = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(baseline, dtype=np.float64)
    expected = (len(spec.OPTIMIZER_SEEDS), len(spec.ENVIRONMENTS))
    if choice.shape != expected or control.shape != expected:
        raise ValueError("v22 pooled bootstrap shape mismatch")
    rng = np.random.default_rng(spec.BOOTSTRAP_SEED + int(seed_offset))
    indices = rng.integers(
        0,
        choice.shape[0],
        size=(spec.BOOTSTRAP_DRAWS, choice.shape[0]),
    )
    draws = np.asarray([
        _pooled_statistic(choice[index], control[index], metric=metric)
        for index in indices
    ])
    alpha = (1.0 - spec.CONFIDENCE) / 2.0
    return {
        "estimate": _pooled_statistic(choice, control, metric=metric),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "confidence": float(spec.CONFIDENCE),
    }


def _validity(
    registry: dict[tuple[str, str, int], dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    cells = [
        registry[(environment, arm, int(seed))]
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]
    gates = {
        "zero_certificate_violations": all(
            float(cell["certificate_violation_count"]) == 0.0
            for cell in cells
        ),
        "upper_prefix_budget": all(
            float(cell["upper_prefix_power_max"])
            <= spec.UPPER_HF_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
            for cell in cells
        ),
        "lower_prefix_budget": all(
            float(cell["lower_prefix_power_max"])
            <= spec.LOWER_LF_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
            for cell in cells
        ),
        "projection_converged": all(
            float(cell["projection_converged_rate"])
            >= spec.MINIMUM_PROJECTION_CONVERGED_RATE
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
        "maximum_certificate_violation_count": max(
            float(cell["certificate_violation_count"]) for cell in cells
        ),
        "minimum_projection_converged_rate": min(
            float(cell["projection_converged_rate"]) for cell in cells
        ),
        "maximum_recursive_fallback_rate": max(
            float(cell["recursive_fallback_rate"]) for cell in cells
        ),
    }


def _uniform_audit(
    candidate_cells: list[dict[str, Any]],
    baseline_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    for level in ("upper", "lower"):
        candidate = [cell["weight_training"][level] for cell in candidate_cells]
        baseline = [cell["weight_training"][level] for cell in baseline_cells]
        gates = {
            "candidate_active": all(
                float(row["active_iteration_count"]) > 0.0
                for row in candidate
            ),
            "candidate_mse_positive": all(
                float(row["unweighted_mse_mean"]) > 0.0
                and float(row["weighted_mse_mean"]) > 0.0
                for row in candidate
            ),
            "candidate_uniform_weights": all(
                abs(float(row["weight_mean"]) - 1.0)
                <= spec.UNIFORM_WEIGHT_TOLERANCE
                and abs(float(row["weight_max"]) - 1.0)
                <= spec.UNIFORM_WEIGHT_TOLERANCE
                for row in candidate
            ),
            "baseline_consistency_inactive": all(
                float(row["active_iteration_count"]) == 0.0
                for row in baseline
            ),
        }
        levels[level] = {
            "gates": gates,
            "supported": bool(all(gates.values())),
        }
    return {
        "levels": levels,
        "supported": bool(all(row["supported"] for row in levels.values())),
    }


def analyze(run_name: str) -> dict[str, Any]:
    registry: dict[tuple[str, str, int], dict[str, Any]] = {}
    path_sets: dict[tuple[str, int], set[frozenset[tuple[str, int]]]] = {}
    parameter_counts: dict[tuple[str, int], set[int]] = {}
    for environment in spec.ENVIRONMENTS:
        for arm, arm_spec in spec.ARMS.items():
            for optimizer_seed in spec.OPTIMIZER_SEEDS:
                summary, rows = _load_cell(
                    run_name, environment, arm, optimizer_seed
                )
                _validate_cell(environment, arm, optimizer_seed, summary, rows)
                cell = _summarize_cell(
                    summary,
                    rows,
                    projected=bool(arm_spec["terminal_reserve_projection"]),
                )
                registry[(environment, arm, int(optimizer_seed))] = cell
                key = (environment, int(optimizer_seed))
                path_sets.setdefault(key, set()).add(
                    frozenset(_path_registry(rows))
                )
                parameter_counts.setdefault(key, set()).add(
                    int(cell["parameter_count"])
                )
    if any(len(values) != 1 for values in path_sets.values()):
        raise ValueError("v22 paired arms do not share heldout paths")
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise ValueError("v22 capacity-matched parameter counts differ")

    validity = {
        arm: _validity(registry, arm)
        for arm in (spec.PRIMARY_PROJECTED_BASELINE, spec.CANDIDATE)
    }
    environment_results: list[dict[str, Any]] = []
    candidate_matrices = {
        key: np.zeros(
            (len(spec.OPTIMIZER_SEEDS), len(spec.ENVIRONMENTS)),
            dtype=np.float64,
        )
        for key in ("reward", "component_correction_rms", "total_correction_rms")
    }
    baseline_matrices = {
        key: np.zeros_like(value) for key, value in candidate_matrices.items()
    }
    offset = 0
    for environment_index, environment in enumerate(spec.ENVIRONMENTS):
        candidate = [
            registry[(environment, spec.CANDIDATE, int(seed))]
            for seed in spec.OPTIMIZER_SEEDS
        ]
        baseline = [
            registry[(
                environment,
                spec.PRIMARY_PROJECTED_BASELINE,
                int(seed),
            )]
            for seed in spec.OPTIMIZER_SEEDS
        ]
        raw = [
            registry[(environment, spec.PRIMARY_RAW_REFERENCE, int(seed))]
            for seed in spec.OPTIMIZER_SEEDS
        ]
        intervals: dict[str, dict[str, float]] = {}
        for name, key, statistic in (
            ("reward", "reward", _reward_delta),
            (
                "component_correction",
                "component_correction_rms",
                _reduction,
            ),
            ("total_correction", "total_correction_rms", _reduction),
        ):
            choice = np.asarray([float(row[key]) for row in candidate])
            control = np.asarray([float(row[key]) for row in baseline])
            intervals[name] = _paired_bootstrap(
                choice,
                control,
                statistic,
                confidence=spec.ENVIRONMENT_FAMILY_CONFIDENCE,
                seed_offset=offset,
            )
            offset += 1
            candidate_matrices[key][:, environment_index] = choice
            baseline_matrices[key][:, environment_index] = control
        environment_results.append({
            "environment": environment,
            "mean_reward": {
                "raw": statistics.fmean(float(row["reward"]) for row in raw),
                "reserve": statistics.fmean(
                    float(row["reward"]) for row in baseline
                ),
                "uniform": statistics.fmean(
                    float(row["reward"]) for row in candidate
                ),
            },
            "reward_relative_delta": intervals["reward"],
            "reward_root_wins": sum(
                float(choice["reward"]) > float(control["reward"])
                for choice, control in zip(candidate, baseline, strict=True)
            ),
            "component_correction_reduction": intervals[
                "component_correction"
            ],
            "component_root_wins": sum(
                float(choice["component_correction_rms"])
                < float(control["component_correction_rms"])
                for choice, control in zip(candidate, baseline, strict=True)
            ),
            "total_correction_reduction": intervals["total_correction"],
            "total_root_wins": sum(
                float(choice["total_correction_rms"])
                < float(control["total_correction_rms"])
                for choice, control in zip(candidate, baseline, strict=True)
            ),
            "mean_total_correction_rms": statistics.fmean(
                float(row["total_correction_rms"]) for row in candidate
            ),
            "mean_total_action_change_rate": statistics.fmean(
                float(row["total_action_change_rate"]) for row in candidate
            ),
        })

    pooled = {
        "reward_relative_delta": _paired_pooled_bootstrap(
            candidate_matrices["reward"],
            baseline_matrices["reward"],
            metric="reward",
            seed_offset=offset,
        ),
        "component_correction_reduction": _paired_pooled_bootstrap(
            candidate_matrices["component_correction_rms"],
            baseline_matrices["component_correction_rms"],
            metric="reduction",
            seed_offset=offset + 1,
        ),
        "total_correction_reduction": _paired_pooled_bootstrap(
            candidate_matrices["total_correction_rms"],
            baseline_matrices["total_correction_rms"],
            metric="reduction",
            seed_offset=offset + 2,
        ),
    }
    candidate_cells = [
        registry[(environment, spec.CANDIDATE, int(seed))]
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]
    baseline_cells = [
        registry[(environment, spec.PRIMARY_PROJECTED_BASELINE, int(seed))]
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]
    uniform_audit = _uniform_audit(candidate_cells, baseline_cells)
    gates = {
        "projected_controls_valid": all(
            result["supported"] for result in validity.values()
        ),
        "uniform_training_audit_supported": bool(uniform_audit["supported"]),
        "reward_noninferior_in_every_environment": all(
            row["reward_relative_delta"]["ci_low"]
            >= -spec.REWARD_NONINFERIORITY_MARGIN
            for row in environment_results
        ),
        "pooled_reward_noninferior": (
            pooled["reward_relative_delta"]["ci_low"]
            >= -spec.REWARD_NONINFERIORITY_MARGIN
        ),
        "pooled_component_reduction_supported": (
            pooled["component_correction_reduction"]["ci_low"]
            >= spec.MINIMUM_CORRECTION_REDUCTION_FRACTION
        ),
        "pooled_total_reduction_supported": (
            pooled["total_correction_reduction"]["ci_low"]
            >= spec.MINIMUM_CORRECTION_REDUCTION_FRACTION
        ),
        "component_improves_in_two_environments": sum(
            row["component_correction_reduction"]["estimate"]
            >= spec.MINIMUM_CORRECTION_REDUCTION_FRACTION
            for row in environment_results
        ) >= spec.MINIMUM_IMPROVED_ENVIRONMENTS,
        "total_improves_in_two_environments": sum(
            row["total_correction_reduction"]["estimate"]
            >= spec.MINIMUM_CORRECTION_REDUCTION_FRACTION
            for row in environment_results
        ) >= spec.MINIMUM_IMPROVED_ENVIRONMENTS,
        "component_noninferior_in_every_environment": all(
            row["component_correction_reduction"]["ci_low"]
            >= -spec.MAXIMUM_CORRECTION_REGRESSION_FRACTION
            for row in environment_results
        ),
        "total_noninferior_in_every_environment": all(
            row["total_correction_reduction"]["ci_low"]
            >= -spec.MAXIMUM_CORRECTION_REGRESSION_FRACTION
            for row in environment_results
        ),
        "correction_magnitude_bounded": all(
            row["mean_total_correction_rms"]
            <= spec.MAXIMUM_MEAN_TOTAL_CORRECTION_RMS
            for row in environment_results
        ),
    }
    supported = bool(all(gates.values()))
    secondary = {
        "pooled_reward_superiority": (
            pooled["reward_relative_delta"]["ci_low"] > 0.0
        ),
        "reward_superior_environment_count": sum(
            row["reward_relative_delta"]["ci_low"] > 0.0
            for row in environment_results
        ),
    }
    return {
        "analysis_version": spec.CONFIRMATION_PROTOCOL_VERSION,
        "evidence_role": spec.EVIDENCE_ROLE,
        "cell_count": len(registry),
        "status": spec.SUPPORTED_STATUS if supported else spec.NOT_SUPPORTED_STATUS,
        "selected_candidate": spec.CANDIDATE if supported else None,
        "support_gate": supported,
        "gates": gates,
        "secondary_reward_results": secondary,
        "environment_results": environment_results,
        "pooled_intervals": pooled,
        "validity": validity,
        "uniform_training_audit": uniform_audit,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cells": [
            {"environment": environment, "arm": arm, "optimizer_seed": seed, **cell}
            for (environment, arm, seed), cell in sorted(registry.items())
        ],
    }


def _write_readme(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# MuJoCo v22 Uniform Terminal-Reserve Confirmation",
        "",
        f"- Status: `{result['status']}`",
        f"- Cells: {result['cell_count']}",
        f"- Selected candidate: `{result['selected_candidate']}`",
        f"- Evidence role: `{result['evidence_role']}`",
        "",
        "| Environment | Reserve reward | Uniform reward | Reward delta | "
        "Reward wins | Component reduction | Total reduction | Total correction | Change rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["environment_results"]:
        reward = row["reward_relative_delta"]
        component = row["component_correction_reduction"]
        total = row["total_correction_reduction"]
        lines.append(
            f"| {row['environment']} | {row['mean_reward']['reserve']:.3f} | "
            f"{row['mean_reward']['uniform']:.3f} | "
            f"{reward['estimate']:.4f} "
            f"[{reward['ci_low']:.4f}, {reward['ci_high']:.4f}] | "
            f"{row['reward_root_wins']}/{len(spec.OPTIMIZER_SEEDS)} | "
            f"{component['estimate']:.4f} "
            f"[{component['ci_low']:.4f}, {component['ci_high']:.4f}] | "
            f"{total['estimate']:.4f} "
            f"[{total['ci_low']:.4f}, {total['ci_high']:.4f}] | "
            f"{row['mean_total_correction_rms']:.4f} | "
            f"{row['mean_total_action_change_rate']:.4f} |"
        )
    lines.extend(["", "## Registered Gates", ""])
    lines.extend(
        f"- {name}: `{str(value).lower()}`"
        for name, value in result["gates"].items()
    )
    lines.extend([
        "",
        "## Pooled Paired Intervals",
        "",
    ])
    for name, interval in result["pooled_intervals"].items():
        lines.append(
            f"- {name}: {interval['estimate']:.4f} "
            f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
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
