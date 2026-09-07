#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v21 reward-selective mechanism preflight."""

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
    mujoco_v21_reward_selective_feasible_action_preflight_spec as spec,
)
from scripts.submit_mujoco_v21_reward_selective_feasible_action_preflight_scheduleurm import (  # noqa: E402
    cell_relative_dir,
)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _path_registry(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(row["disturbance_mode"]), int(row["seed"]))
        for row in rows
    }


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _weight_summary(summary: dict[str, Any]) -> dict[str, Any]:
    payload = summary.get("projection_consistency_weight_training")
    if not isinstance(payload, dict) or set(payload) != {"upper", "lower"}:
        raise ValueError("v21 consistency-weight summary is missing")
    normalized: dict[str, Any] = {}
    keys = (
        "active_iteration_count",
        "unweighted_mse_mean",
        "weighted_mse_mean",
        "weight_mean",
        "weight_max",
    )
    for level in ("upper", "lower"):
        values = payload[level]
        if not isinstance(values, dict):
            raise ValueError("v21 consistency-weight level is invalid")
        normalized[level] = {key: float(values[key]) for key in keys}
        numeric = np.asarray(list(normalized[level].values()), dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
            raise ValueError("v21 consistency-weight diagnostics are invalid")
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
        "advantage_temperature": (
            float(summary.get(
                "projection_consistency_advantage_temperature", -1.0
            )),
            float(arm_spec[
                "projection_consistency_advantage_temperature"
            ]),
        ),
        "advantage_weight_clip": (
            float(summary.get(
                "projection_consistency_advantage_weight_clip", -1.0
            )),
            float(arm_spec[
                "projection_consistency_advantage_weight_clip"
            ]),
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
            float(summary.get(
                "projection_consistency_ramp_fraction", -1.0
            )),
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
            f"v21 cell contract mismatch {environment}/{arm}/"
            f"{optimizer_seed}: " + json.dumps(drift, sort_keys=True)
        )
    if int(summary.get("selected_checkpoint_iteration", -1)) < (
        spec.CHECKPOINT_MINIMUM_ITERATION
    ):
        raise ValueError(
            f"v21 checkpoint selected too early: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    _weight_summary(summary)
    if len(rows) != spec.EXPECTED_EVALUATION_ROWS_PER_CELL:
        raise ValueError(
            f"v21 heldout row count mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    if _path_registry(rows) != expected_paths:
        raise ValueError(
            f"v21 heldout path mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    if not all(_as_bool(row["protocol_valid"]) for row in rows):
        raise ValueError(
            f"v21 invalid rollout protocol: "
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
            f"v21 row-level terminal contract mismatch: "
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


def _paired_interval(
    values: list[float],
    *,
    seed_offset: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(spec.OPTIMIZER_SEEDS),):
        raise ValueError("v21 bootstrap requires one value per optimizer root")
    rng = np.random.default_rng(spec.BOOTSTRAP_SEED + int(seed_offset))
    indices = rng.integers(
        0,
        array.size,
        size=(spec.BOOTSTRAP_DRAWS, array.size),
    )
    draws = np.mean(array[indices], axis=1)
    alpha = (1.0 - spec.CONFIDENCE) / 2.0
    return {
        "estimate": float(np.mean(array)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
    }


def _normalized_delta(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    key: str,
    *,
    denominator_floor: float = 1e-12,
) -> float:
    base = float(baseline[key])
    return (
        float(candidate[key]) - base
    ) / max(abs(base), float(denominator_floor))


def _reduction(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    key: str,
) -> float:
    return -_normalized_delta(candidate, baseline, key)


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


def _weight_audit(
    selective_cells: list[dict[str, Any]],
    uniform_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    for level in ("upper", "lower"):
        selected = [
            cell["weight_training"][level] for cell in selective_cells
        ]
        uniform = [cell["weight_training"][level] for cell in uniform_cells]
        gates = {
            "active_in_every_candidate_cell": all(
                float(row["active_iteration_count"]) > 0.0
                for row in selected
            ),
            "candidate_unit_mean": all(
                abs(float(row["weight_mean"]) - 1.0)
                <= spec.WEIGHT_MEAN_TOLERANCE
                for row in selected
            ),
            "candidate_is_selective": all(
                float(row["weight_max"])
                >= spec.MINIMUM_SELECTIVE_WEIGHT_MAX
                for row in selected
            ),
            "candidate_mse_finite_positive": all(
                float(row["unweighted_mse_mean"]) > 0.0
                and float(row["weighted_mse_mean"]) > 0.0
                for row in selected
            ),
            "uniform_weights_exact": all(
                abs(float(row["weight_mean"]) - 1.0)
                <= spec.WEIGHT_MEAN_TOLERANCE
                and abs(float(row["weight_max"]) - 1.0)
                <= spec.WEIGHT_MEAN_TOLERANCE
                for row in uniform
            ),
        }
        levels[level] = {
            "gates": gates,
            "supported": bool(all(gates.values())),
            "candidate_weight_mean": statistics.fmean(
                float(row["weight_mean"]) for row in selected
            ),
            "candidate_weight_max": max(
                float(row["weight_max"]) for row in selected
            ),
            "candidate_unweighted_mse_mean": statistics.fmean(
                float(row["unweighted_mse_mean"]) for row in selected
            ),
            "candidate_weighted_mse_mean": statistics.fmean(
                float(row["weighted_mse_mean"]) for row in selected
            ),
        }
    return {
        "levels": levels,
        "supported": bool(
            all(result["supported"] for result in levels.values())
        ),
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
                _validate_cell(
                    environment, arm, optimizer_seed, summary, rows
                )
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
        raise ValueError("v21 paired arms do not share heldout paths")
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise ValueError("v21 capacity-matched arm parameter counts differ")

    projected_arms = (
        spec.PRIMARY_MECHANISM_BASELINE,
        spec.PRIMARY_UNIFORM_BASELINE,
        *spec.CANDIDATES,
    )
    validity = {
        arm: _validity(registry, arm) for arm in projected_arms
    }

    candidate = spec.DELAYED_REWARD_SELECTIVE_010
    environment_results: list[dict[str, Any]] = []
    pooled_by_root: dict[str, list[float]] = {
        "reward_vs_uniform": [],
        "component_vs_reserve": [],
        "total_vs_reserve": [],
        "component_vs_uniform": [],
        "total_vs_uniform": [],
    }
    offset = 0
    for environment in spec.ENVIRONMENTS:
        raw = [
            registry[(environment, spec.PRIMARY_RAW_REFERENCE, int(seed))]
            for seed in spec.OPTIMIZER_SEEDS
        ]
        reserve = [
            registry[(environment, spec.PRIMARY_MECHANISM_BASELINE, int(seed))]
            for seed in spec.OPTIMIZER_SEEDS
        ]
        uniform = [
            registry[(environment, spec.PRIMARY_UNIFORM_BASELINE, int(seed))]
            for seed in spec.OPTIMIZER_SEEDS
        ]
        selected = [
            registry[(environment, candidate, int(seed))]
            for seed in spec.OPTIMIZER_SEEDS
        ]
        metrics = {
            "reward_vs_uniform": [
                _normalized_delta(
                    choice,
                    base,
                    "reward",
                    denominator_floor=1.0,
                )
                for choice, base in zip(selected, uniform, strict=True)
            ],
            "component_vs_reserve": [
                _reduction(choice, base, "component_correction_rms")
                for choice, base in zip(selected, reserve, strict=True)
            ],
            "total_vs_reserve": [
                _reduction(choice, base, "total_correction_rms")
                for choice, base in zip(selected, reserve, strict=True)
            ],
            "component_vs_uniform": [
                _reduction(choice, base, "component_correction_rms")
                for choice, base in zip(selected, uniform, strict=True)
            ],
            "total_vs_uniform": [
                _reduction(choice, base, "total_correction_rms")
                for choice, base in zip(selected, uniform, strict=True)
            ],
        }
        intervals: dict[str, dict[str, float]] = {}
        for name, values in metrics.items():
            intervals[name] = _paired_interval(values, seed_offset=offset)
            offset += 1
        mean_total_correction = statistics.fmean(
            float(cell["total_correction_rms"]) for cell in selected
        )
        mean_action_change = statistics.fmean(
            float(cell["total_action_change_rate"]) for cell in selected
        )
        environment_results.append({
            "environment": environment,
            "mean_reward": {
                "raw": statistics.fmean(float(row["reward"]) for row in raw),
                "reserve": statistics.fmean(
                    float(row["reward"]) for row in reserve
                ),
                "uniform": statistics.fmean(
                    float(row["reward"]) for row in uniform
                ),
                "reward_selective": statistics.fmean(
                    float(row["reward"]) for row in selected
                ),
            },
            "reward_normalized_delta_vs_uniform": intervals[
                "reward_vs_uniform"
            ],
            "component_reduction_vs_reserve": intervals[
                "component_vs_reserve"
            ],
            "total_reduction_vs_reserve": intervals["total_vs_reserve"],
            "component_reduction_vs_uniform": intervals[
                "component_vs_uniform"
            ],
            "total_reduction_vs_uniform": intervals["total_vs_uniform"],
            "physical_burden": {
                "mean_total_correction_rms": mean_total_correction,
                "mean_total_action_change_rate": mean_action_change,
                "supported": bool(
                    mean_total_correction
                    <= spec.MAXIMUM_MEAN_TOTAL_CORRECTION_RMS
                    and mean_action_change
                    <= spec.MAXIMUM_MEAN_TOTAL_ACTION_CHANGE_RATE
                ),
            },
        })

    for optimizer_seed in spec.OPTIMIZER_SEEDS:
        for name, key, baseline_arm in (
            ("reward_vs_uniform", "reward", spec.PRIMARY_UNIFORM_BASELINE),
            (
                "component_vs_reserve",
                "component_correction_rms",
                spec.PRIMARY_MECHANISM_BASELINE,
            ),
            (
                "total_vs_reserve",
                "total_correction_rms",
                spec.PRIMARY_MECHANISM_BASELINE,
            ),
            (
                "component_vs_uniform",
                "component_correction_rms",
                spec.PRIMARY_UNIFORM_BASELINE,
            ),
            (
                "total_vs_uniform",
                "total_correction_rms",
                spec.PRIMARY_UNIFORM_BASELINE,
            ),
        ):
            values = []
            for environment in spec.ENVIRONMENTS:
                choice = registry[
                    (environment, candidate, int(optimizer_seed))
                ]
                base = registry[
                    (environment, baseline_arm, int(optimizer_seed))
                ]
                values.append(
                    _normalized_delta(
                        choice,
                        base,
                        key,
                        denominator_floor=1.0,
                    )
                    if name == "reward_vs_uniform"
                    else _reduction(choice, base, key)
                )
            pooled_by_root[name].append(statistics.fmean(values))

    pooled = {}
    for name, values in pooled_by_root.items():
        pooled[name] = _paired_interval(values, seed_offset=offset)
        offset += 1

    reward_improved_environment_count = sum(
        row["reward_normalized_delta_vs_uniform"]["estimate"] > 0.0
        for row in environment_results
    )
    component_improved_environment_count = sum(
        row["component_reduction_vs_reserve"]["estimate"]
        >= spec.MINIMUM_CORRECTION_REDUCTION_FRACTION_VS_RESERVE
        for row in environment_results
    )
    total_improved_environment_count = sum(
        row["total_reduction_vs_reserve"]["estimate"]
        >= spec.MINIMUM_CORRECTION_REDUCTION_FRACTION_VS_RESERVE
        for row in environment_results
    )

    selective_cells = [
        registry[(environment, candidate, int(seed))]
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]
    uniform_cells = [
        registry[
            (environment, spec.PRIMARY_UNIFORM_BASELINE, int(seed))
        ]
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]
    weight_audit = _weight_audit(selective_cells, uniform_cells)

    gates = {
        "projected_controls_valid": all(
            validity[arm]["supported"] for arm in projected_arms
        ),
        "reward_improves_in_two_environments": (
            reward_improved_environment_count
            >= spec.MINIMUM_REWARD_IMPROVED_ENVIRONMENTS
        ),
        "pooled_normalized_reward_positive": (
            pooled["reward_vs_uniform"]["estimate"]
            > spec.MINIMUM_POOLED_NORMALIZED_REWARD_DELTA
        ),
        "reward_floor_in_every_environment": all(
            row["reward_normalized_delta_vs_uniform"]["estimate"]
            >= -spec.MAXIMUM_REWARD_REGRESSION_FRACTION_VS_UNIFORM
            for row in environment_results
        ),
        "component_reduction_vs_reserve": (
            component_improved_environment_count
            >= spec.MINIMUM_CORRECTION_IMPROVED_ENVIRONMENTS
        ),
        "total_reduction_vs_reserve": (
            total_improved_environment_count
            >= spec.MINIMUM_CORRECTION_IMPROVED_ENVIRONMENTS
        ),
        "component_noninferior_to_uniform": all(
            row["component_reduction_vs_uniform"]["estimate"]
            >= -spec.MAXIMUM_CORRECTION_REGRESSION_FRACTION_VS_UNIFORM
            for row in environment_results
        ),
        "total_noninferior_to_uniform": all(
            row["total_reduction_vs_uniform"]["estimate"]
            >= -spec.MAXIMUM_CORRECTION_REGRESSION_FRACTION_VS_UNIFORM
            for row in environment_results
        ),
        "physical_burden_supported": all(
            row["physical_burden"]["supported"]
            for row in environment_results
        ),
        "weight_audit_supported": bool(weight_audit["supported"]),
    }
    advance = bool(all(gates.values()))
    result = {
        "candidate": candidate,
        "projection_consistency_weighting": spec.ARMS[candidate][
            "projection_consistency_weighting"
        ],
        "environment_results": environment_results,
        "pooled_intervals": pooled,
        "reward_improved_environment_count": (
            reward_improved_environment_count
        ),
        "component_improved_environment_count": (
            component_improved_environment_count
        ),
        "total_improved_environment_count": (
            total_improved_environment_count
        ),
        "weight_audit": weight_audit,
        "gates": gates,
        "eligible": advance,
    }
    cells = [
        {
            "environment": environment,
            "arm": arm,
            "optimizer_seed": int(seed),
            **registry[(environment, arm, int(seed))],
        }
        for environment in spec.ENVIRONMENTS
        for arm in spec.ARMS
        for seed in spec.OPTIMIZER_SEEDS
    ]
    return {
        "analysis_version": (
            "mujoco_v21_reward_selective_feasible_action_preflight_analysis_v1"
        ),
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "bootstrap_contract": {
            "unit": "paired_optimizer_root",
            "draws": spec.BOOTSTRAP_DRAWS,
            "confidence": spec.CONFIDENCE,
            "seed": spec.BOOTSTRAP_SEED,
            "role": "descriptive_preflight_interval_not_claim_gate",
        },
        "cell_count": len(cells),
        "validity": validity,
        "candidate_results": [result],
        "selected_candidate": candidate if advance else None,
        "support_gate": advance,
        "status": (
            spec.SUPPORTED_STATUS if advance else spec.NOT_SUPPORTED_STATUS
        ),
        "cells": cells,
    }


def _markdown(result: dict[str, Any]) -> str:
    candidate = result["candidate_results"][0]
    lines = [
        "# MuJoCo v21 Reward-Selective Feasible-Action Preflight",
        "",
        f"- Status: `{result['status']}`",
        f"- Cells: {result['cell_count']}",
        (
            "- Advance candidate: "
            f"`{result['selected_candidate']}`"
        ),
        "- Evidence role: development preflight only",
        "",
        "| Environment | Raw reward | Reserve reward | Uniform reward | "
        "Selective reward | Reward delta | Component vs reserve | "
        "Component vs uniform |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate["environment_results"]:
        reward = row["mean_reward"]
        lines.append(
            "| {environment} | {raw:.3f} | {reserve:.3f} | "
            "{uniform:.3f} | {selective:.3f} | {reward_delta:.4f} | "
            "{reserve_reduction:.4f} | {uniform_reduction:.4f} |".format(
                environment=row["environment"],
                raw=reward["raw"],
                reserve=reward["reserve"],
                uniform=reward["uniform"],
                selective=reward["reward_selective"],
                reward_delta=row[
                    "reward_normalized_delta_vs_uniform"
                ]["estimate"],
                reserve_reduction=row[
                    "component_reduction_vs_reserve"
                ]["estimate"],
                uniform_reduction=row[
                    "component_reduction_vs_uniform"
                ]["estimate"],
            )
        )
    lines.extend([
        "",
        "## Gates",
        "",
    ])
    for name, supported in candidate["gates"].items():
        lines.append(f"- {name}: `{str(bool(supported)).lower()}`")
    lines.extend([
        "",
        "A failed gate stops this weighting mechanism. These roots cannot be "
        "reused for temperature, clip, coefficient, or schedule tuning.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = analyze(args.run_name)
    target = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / "results" / args.run_name / "analysis"
    )
    target.mkdir(parents=True, exist_ok=True)
    (target / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "README.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "selected_candidate": result["selected_candidate"],
        "cell_count": result["cell_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
