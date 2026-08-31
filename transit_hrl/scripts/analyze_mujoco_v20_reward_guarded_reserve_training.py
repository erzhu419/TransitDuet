#!/usr/bin/env python3
"""Analyze the frozen fresh-seed MuJoCo v20 development panel."""

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
    mujoco_v20_reward_guarded_reserve_training_spec as spec,
)
from scripts.submit_mujoco_v20_reward_guarded_reserve_training_scheduleurm import (  # noqa: E402
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


def _projection_guard_summary(summary: dict[str, Any]) -> dict[str, Any]:
    payload = summary.get("projection_consistency_guard_training")
    if not isinstance(payload, dict) or set(payload) != {"upper", "lower"}:
        raise ValueError("v20 projection-guard summary is missing or incomplete")
    normalized: dict[str, Any] = {}
    for level in ("upper", "lower"):
        values = payload[level]
        if not isinstance(values, dict):
            raise ValueError("v20 projection-guard level summary is invalid")
        normalized[level] = {
            key: float(values[key])
            for key in (
                "active_iteration_count",
                "attempted_mass",
                "accepted_mass",
                "acceptance_rate",
                "reward_loss_delta_max",
                "native_constraint_loss_delta_max",
                "consistency_loss_delta_mean",
                "gradient_conflict_rate",
            )
        }
        numeric = np.asarray(list(normalized[level].values()), dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("v20 projection-guard diagnostics must be finite")
        if (
            normalized[level]["active_iteration_count"] < 0.0
            or normalized[level]["attempted_mass"] < 0.0
            or normalized[level]["accepted_mass"] < 0.0
            or normalized[level]["accepted_mass"]
            > normalized[level]["attempted_mass"] + 1e-12
            or not 0.0 <= normalized[level]["acceptance_rate"] <= 1.0
        ):
            raise ValueError("v20 projection-guard diagnostics are inconsistent")
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
            summary.get("protocol_version"), spec.FROZEN_CORE_PROTOCOL_VERSION
        ),
        "code_revision": (
            summary.get("code_revision"), spec.FROZEN_ALGORITHM_REVISION
        ),
        "environment": (summary.get("environment"), str(environment)),
        "optimizer_seed": (
            int(summary.get("optimizer_seed", -1)), int(optimizer_seed)
        ),
        "method": (summary.get("method"), arm_spec["method"]),
        "ppo_clip_ratio": (
            float(summary.get("ppo_clip_ratio", -1.0)), spec.PPO_CLIP_RATIO
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
        "consistency_step_scale": (
            float(summary.get("projection_consistency_step_scale", -1.0)),
            float(arm_spec["projection_consistency_step_scale"]),
        ),
        "consistency_max_backtracks": (
            int(summary.get("projection_consistency_max_backtracks", -1)),
            int(arm_spec["projection_consistency_max_backtracks"]),
        ),
        "consistency_reward_tolerance": (
            float(summary.get(
                "projection_consistency_reward_tolerance", -1.0
            )),
            float(arm_spec["projection_consistency_reward_tolerance"]),
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
            summary.get("checkpoint_score_mode"), spec.CHECKPOINT_SCORE_MODE
        ),
    }
    drift = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in mismatches.items()
        if observed != expected
    }
    if drift:
        raise ValueError(
            f"v20 cell contract mismatch {environment}/{arm}/{optimizer_seed}: "
            + json.dumps(drift, sort_keys=True)
        )
    if int(summary.get("selected_checkpoint_iteration", -1)) < (
        spec.CHECKPOINT_MINIMUM_ITERATION
    ):
        raise ValueError(
            f"v20 checkpoint selected too early: {environment}/{arm}/{optimizer_seed}"
        )
    if len(rows) != spec.EXPECTED_EVALUATION_ROWS_PER_CELL:
        raise ValueError(
            f"v20 heldout row count mismatch: {environment}/{arm}/{optimizer_seed}"
        )
    if _path_registry(rows) != expected_paths:
        raise ValueError(
            f"v20 heldout path mismatch: {environment}/{arm}/{optimizer_seed}"
        )
    if not all(_as_bool(row["protocol_valid"]) for row in rows):
        raise ValueError(
            f"v20 invalid rollout protocol: {environment}/{arm}/{optimizer_seed}"
        )
    expected_projection = bool(arm_spec["terminal_reserve_projection"])
    if not all(
        _as_bool(row["terminal_reserve_context_enabled"])
        and _as_bool(row["terminal_reserve_projection_enabled"])
        == expected_projection
        for row in rows
    ):
        raise ValueError(
            f"v20 row-level terminal contract mismatch: {environment}/{arm}/{optimizer_seed}"
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
        "projection_guard_training": _projection_guard_summary(summary),
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
        raise ValueError("v20 bootstrap requires one value per optimizer root")
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


def _relative_reduction(baseline: list[float], candidate: list[float]) -> float:
    baseline_mean = statistics.fmean(baseline)
    candidate_mean = statistics.fmean(candidate)
    return float(
        (baseline_mean - candidate_mean) / max(abs(baseline_mean), 1e-12)
    )


def _guard_audit(
    cells: list[dict[str, Any]],
    *,
    required: bool,
) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    for level in ("upper", "lower"):
        rows = [cell["projection_guard_training"][level] for cell in cells]
        attempted = sum(float(row["attempted_mass"]) for row in rows)
        accepted = sum(float(row["accepted_mass"]) for row in rows)
        if required:
            gates = {
                "active_in_every_cell": all(
                    float(row["active_iteration_count"]) > 0.0
                    and float(row["attempted_mass"]) > 0.0
                    for row in rows
                ),
                "accepted_in_every_cell": all(
                    float(row["accepted_mass"]) > 0.0 for row in rows
                ),
                "reward_surrogate_nonworsening": all(
                    float(row["reward_loss_delta_max"]) <= 1e-10
                    for row in rows
                ),
                "native_constraint_nonworsening": all(
                    float(row["native_constraint_loss_delta_max"]) <= 1e-10
                    for row in rows
                ),
                "consistency_loss_decreased": all(
                    float(row["consistency_loss_delta_mean"]) < 0.0
                    for row in rows
                ),
            }
        else:
            gates = {
                "guard_disabled_exactly": all(
                    float(row["active_iteration_count"]) == 0.0
                    and float(row["attempted_mass"]) == 0.0
                    and float(row["accepted_mass"]) == 0.0
                    for row in rows
                )
            }
        levels[level] = {
            "gates": gates,
            "supported": bool(all(gates.values())),
            "attempted_mass": float(attempted),
            "accepted_mass": float(accepted),
            "acceptance_rate": (
                float(accepted / attempted) if attempted > 0.0 else 0.0
            ),
            "maximum_reward_loss_delta": max(
                float(row["reward_loss_delta_max"]) for row in rows
            ),
            "maximum_native_constraint_loss_delta": max(
                float(row["native_constraint_loss_delta_max"])
                for row in rows
            ),
            "mean_consistency_loss_delta": statistics.fmean(
                float(row["consistency_loss_delta_mean"]) for row in rows
            ),
        }
    return {
        "required": bool(required),
        "levels": levels,
        "supported": bool(
            all(level["supported"] for level in levels.values())
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
        raise ValueError("v20 paired arms do not share heldout paths")
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise ValueError("v20 capacity-matched arm parameter counts differ")

    projected_arms = (
        spec.PRIMARY_MECHANISM_BASELINE,
        *spec.CANDIDATES,
    )
    validity: dict[str, dict[str, Any]] = {}
    for arm in projected_arms:
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
        validity[arm] = {
            "gates": gates,
            "supported": bool(all(gates.values())),
            "maximum_certificate_violation_count": max(
                float(cell["certificate_violation_count"])
                for cell in cells
            ),
            "minimum_projection_converged_rate": min(
                float(cell["projection_converged_rate"])
                for cell in cells
            ),
            "maximum_recursive_fallback_rate": max(
                float(cell["recursive_fallback_rate"])
                for cell in cells
            ),
        }

    candidate_results: list[dict[str, Any]] = []
    offset = 0
    for candidate in spec.CANDIDATES:
        environment_results: list[dict[str, Any]] = []
        pooled_reward_vs_reserve: list[float] = []
        pooled_reward_vs_raw: list[float] = []
        pooled_reward_delta: list[float] = []
        pooled_component_improvement: list[float] = []
        pooled_total_improvement: list[float] = []
        pooled_reserve_total: list[float] = []
        pooled_selected_total: list[float] = []
        candidate_cells: list[dict[str, Any]] = []
        for environment in spec.ENVIRONMENTS:
            reserve = [
                registry[(environment, spec.PRIMARY_MECHANISM_BASELINE, int(seed))]
                for seed in spec.OPTIMIZER_SEEDS
            ]
            raw = [
                registry[(environment, spec.PRIMARY_RAW_BASELINE, int(seed))]
                for seed in spec.OPTIMIZER_SEEDS
            ]
            selected = [
                registry[(environment, candidate, int(seed))]
                for seed in spec.OPTIMIZER_SEEDS
            ]
            candidate_cells.extend(selected)
            reward_vs_reserve_values = [
                float(choice["reward"])
                - float(base["reward"])
                + spec.REWARD_NONINFERIORITY_FRACTION_VS_RESERVE
                * max(abs(float(base["reward"])), 1.0)
                for choice, base in zip(selected, reserve, strict=True)
            ]
            reward_vs_raw_values = [
                float(choice["reward"])
                - float(base["reward"])
                + spec.REWARD_NONINFERIORITY_FRACTION_VS_RAW
                * max(abs(float(base["reward"])), 1.0)
                for choice, base in zip(selected, raw, strict=True)
            ]
            component_values = [
                float(base["component_correction_rms"])
                - float(choice["component_correction_rms"])
                - spec.MINIMUM_COMPONENT_CORRECTION_RELATIVE_REDUCTION
                * float(base["component_correction_rms"])
                for choice, base in zip(selected, reserve, strict=True)
            ]
            total_values = [
                float(base["total_correction_rms"])
                - float(choice["total_correction_rms"])
                - spec.MINIMUM_TOTAL_CORRECTION_RELATIVE_REDUCTION
                * float(base["total_correction_rms"])
                for choice, base in zip(selected, reserve, strict=True)
            ]
            reward_reserve_interval = _paired_interval(
                reward_vs_reserve_values, seed_offset=offset
            )
            offset += 1
            reward_raw_interval = _paired_interval(
                reward_vs_raw_values, seed_offset=offset
            )
            offset += 1
            component_interval = _paired_interval(
                component_values, seed_offset=offset
            )
            offset += 1
            total_interval = _paired_interval(
                total_values, seed_offset=offset
            )
            offset += 1
            mean_total_correction = statistics.fmean(
                float(cell["total_correction_rms"]) for cell in selected
            )
            mean_action_change = statistics.fmean(
                float(cell["total_action_change_rate"]) for cell in selected
            )
            environment_results.append({
                "environment": environment,
                "reward_noninferiority_vs_reserve": {
                    **reward_reserve_interval,
                    "supported": reward_reserve_interval["ci_low"] >= 0.0,
                },
                "reward_noninferiority_vs_raw": {
                    **reward_raw_interval,
                    "supported": reward_raw_interval["ci_low"] >= 0.0,
                },
                "component_correction_minimum_reduction": {
                    **component_interval,
                    "estimated_relative_reduction": _relative_reduction(
                        [float(cell["component_correction_rms"]) for cell in reserve],
                        [float(cell["component_correction_rms"]) for cell in selected],
                    ),
                    "supported": component_interval["ci_low"] > 0.0,
                },
                "total_correction_minimum_reduction": {
                    **total_interval,
                    "estimated_relative_reduction": _relative_reduction(
                        [float(cell["total_correction_rms"]) for cell in reserve],
                        [float(cell["total_correction_rms"]) for cell in selected],
                    ),
                    "supported": total_interval["ci_low"] > 0.0,
                },
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
            reserve_rows = [
                registry[(environment, spec.PRIMARY_MECHANISM_BASELINE, int(
                    optimizer_seed
                ))]
                for environment in spec.ENVIRONMENTS
            ]
            raw_rows = [
                registry[(environment, spec.PRIMARY_RAW_BASELINE, int(
                    optimizer_seed
                ))]
                for environment in spec.ENVIRONMENTS
            ]
            selected_rows = [
                registry[(environment, candidate, int(optimizer_seed))]
                for environment in spec.ENVIRONMENTS
            ]
            reserve_reward = statistics.fmean(
                float(row["reward"]) for row in reserve_rows
            )
            raw_reward = statistics.fmean(
                float(row["reward"]) for row in raw_rows
            )
            selected_reward = statistics.fmean(
                float(row["reward"]) for row in selected_rows
            )
            reserve_component = statistics.fmean(
                float(row["component_correction_rms"])
                for row in reserve_rows
            )
            selected_component = statistics.fmean(
                float(row["component_correction_rms"])
                for row in selected_rows
            )
            reserve_total = statistics.fmean(
                float(row["total_correction_rms"]) for row in reserve_rows
            )
            selected_total = statistics.fmean(
                float(row["total_correction_rms"])
                for row in selected_rows
            )
            pooled_reward_delta.append(selected_reward - reserve_reward)
            pooled_reward_vs_reserve.append(
                selected_reward
                - reserve_reward
                + spec.REWARD_NONINFERIORITY_FRACTION_VS_RESERVE
                * max(abs(reserve_reward), 1.0)
            )
            pooled_reward_vs_raw.append(
                selected_reward
                - raw_reward
                + spec.REWARD_NONINFERIORITY_FRACTION_VS_RAW
                * max(abs(raw_reward), 1.0)
            )
            pooled_component_improvement.append(
                reserve_component
                - selected_component
                - spec.MINIMUM_COMPONENT_CORRECTION_RELATIVE_REDUCTION
                * reserve_component
            )
            pooled_total_improvement.append(
                reserve_total
                - selected_total
                - spec.MINIMUM_TOTAL_CORRECTION_RELATIVE_REDUCTION
                * reserve_total
            )
            pooled_reserve_total.append(reserve_total)
            pooled_selected_total.append(selected_total)
        pooled_reward_reserve_interval = _paired_interval(
            pooled_reward_vs_reserve, seed_offset=offset
        )
        offset += 1
        pooled_reward_raw_interval = _paired_interval(
            pooled_reward_vs_raw, seed_offset=offset
        )
        offset += 1
        pooled_component_interval = _paired_interval(
            pooled_component_improvement, seed_offset=offset
        )
        offset += 1
        pooled_total_interval = _paired_interval(
            pooled_total_improvement, seed_offset=offset
        )
        offset += 1
        component_supported_count = sum(
            bool(row["component_correction_minimum_reduction"]["supported"])
            for row in environment_results
        )
        reward_supported = all(
            bool(row["reward_noninferiority_vs_reserve"]["supported"])
            and bool(row["reward_noninferiority_vs_raw"]["supported"])
            for row in environment_results
        )
        component_supported = bool(
            pooled_component_interval["ci_low"] > 0.0
            and component_supported_count >= spec.MINIMUM_SUPPORTED_ENVIRONMENTS
        )
        total_supported = bool(pooled_total_interval["ci_low"] > 0.0)
        physical_burden_supported = all(
            bool(row["physical_burden"]["supported"])
            for row in environment_results
        )
        guard_audit = _guard_audit(
            candidate_cells,
            required=(
                spec.ARMS[candidate]["projection_consistency_update_mode"]
                == "reward_guarded_projection"
            ),
        )
        gates = {
            "mechanism_baseline_valid": bool(
                validity[spec.PRIMARY_MECHANISM_BASELINE]["supported"]
            ),
            "candidate_valid": bool(validity[candidate]["supported"]),
            "reward_noninferior_all_environments": reward_supported,
            "component_correction_supported": component_supported,
            "total_correction_supported": total_supported,
            "physical_burden_supported": physical_burden_supported,
            "guard_audit_supported": bool(guard_audit["supported"]),
        }
        relative_by_environment = [
            float(row["component_correction_minimum_reduction"][
                "estimated_relative_reduction"
            ])
            for row in environment_results
        ]
        candidate_results.append({
            "candidate": candidate,
            "consistency_coef": float(
                spec.ARMS[candidate]["upper_projection_consistency_coef"]
            ),
            "projection_consistency_update_mode": str(
                spec.ARMS[candidate]["projection_consistency_update_mode"]
            ),
            "environment_results": environment_results,
            "guard_audit": guard_audit,
            "pooled_reward_noninferiority_vs_reserve": {
                **pooled_reward_reserve_interval,
                "supported": pooled_reward_reserve_interval["ci_low"] >= 0.0,
            },
            "pooled_reward_noninferiority_vs_raw": {
                **pooled_reward_raw_interval,
                "supported": pooled_reward_raw_interval["ci_low"] >= 0.0,
            },
            "pooled_reward_delta_vs_reserve_estimate": statistics.fmean(
                pooled_reward_delta
            ),
            "pooled_component_correction_minimum_reduction": {
                **pooled_component_interval,
                "supported": pooled_component_interval["ci_low"] > 0.0,
            },
            "pooled_total_correction_minimum_reduction": {
                **pooled_total_interval,
                "estimated_relative_reduction": _relative_reduction(
                    pooled_reserve_total, pooled_selected_total
                ),
                "supported": pooled_total_interval["ci_low"] > 0.0,
            },
            "component_correction_supported_environment_count": (
                component_supported_count
            ),
            "worst_environment_estimated_relative_reduction": min(
                relative_by_environment
            ),
            "mean_environment_estimated_relative_reduction": statistics.fmean(
                relative_by_environment
            ),
            "gates": gates,
            "eligible": bool(all(gates.values())),
        })

    eligible = [row for row in candidate_results if bool(row["eligible"])]
    selected = (
        min(
            eligible,
            key=lambda row: (
                -float(row["worst_environment_estimated_relative_reduction"]),
                -float(row["pooled_total_correction_minimum_reduction"][
                    "estimated_relative_reduction"
                ]),
                -float(row["pooled_reward_delta_vs_reserve_estimate"]),
                0
                if row["projection_consistency_update_mode"]
                == "reward_guarded_projection"
                else 1,
            ),
        )
        if eligible else None
    )
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
    support_gate = selected is not None
    return {
        "analysis_version": (
            "mujoco_v20_reward_guarded_reserve_training_analysis_v1"
        ),
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "bootstrap_contract": {
            "unit": "paired_optimizer_seed_cluster",
            "draws": spec.BOOTSTRAP_DRAWS,
            "confidence": spec.CONFIDENCE,
            "seed": spec.BOOTSTRAP_SEED,
        },
        "cell_count": len(cells),
        "validity": validity,
        "candidate_results": candidate_results,
        "selected_candidate": (
            None if selected is None else str(selected["candidate"])
        ),
        "selected_consistency_coef": (
            None if selected is None else float(selected["consistency_coef"])
        ),
        "selected_projection_consistency_update_mode": (
            None
            if selected is None
            else str(selected["projection_consistency_update_mode"])
        ),
        "support_gate": support_gate,
        "status": (
            spec.SUPPORTED_STATUS if support_gate else spec.NOT_SUPPORTED_STATUS
        ),
        "cells": cells,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# MuJoCo v20 Reward-Guarded Reserve Development",
        "",
        f"- Status: `{result['status']}`",
        f"- Cells: {result['cell_count']}",
        f"- Selected candidate: `{result['selected_candidate']}`",
        f"- Selected coefficient: `{result['selected_consistency_coef']}`",
        "- Evidence role: development only, not confirmatory or manuscript evidence",
        "",
        "| Candidate | Mode | Eligible | Worst component reduction | Total reduction | Supported envs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["candidate_results"]:
        lines.append(
            "| {candidate} | {mode} | {eligible} | {reduction:.4f} | "
            "{total:.4f} | {count}/3 |".format(
                candidate=row["candidate"],
                mode=row["projection_consistency_update_mode"],
                eligible=str(bool(row["eligible"])).lower(),
                reduction=float(
                    row["worst_environment_estimated_relative_reduction"]
                ),
                total=float(
                    row["pooled_total_correction_minimum_reduction"][
                        "estimated_relative_reduction"
                    ]
                ),
                count=int(
                    row["component_correction_supported_environment_count"]
                ),
            )
        )
    lines.extend([
        "",
        "A failed gate ends this development panel. Its roots cannot be reused "
        "for a revised coefficient screen or confirmation.",
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
