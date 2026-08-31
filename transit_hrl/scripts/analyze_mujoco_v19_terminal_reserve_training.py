#!/usr/bin/env python3
"""Analyze the frozen fresh-seed MuJoCo v19 development panel."""

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

from scripts import mujoco_v19_terminal_reserve_training_spec as spec  # noqa: E402
from scripts.submit_mujoco_v19_terminal_reserve_training_scheduleurm import (  # noqa: E402
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
            f"v19 cell contract mismatch {environment}/{arm}/{optimizer_seed}: "
            + json.dumps(drift, sort_keys=True)
        )
    if int(summary.get("selected_checkpoint_iteration", -1)) < (
        spec.CHECKPOINT_MINIMUM_ITERATION
    ):
        raise ValueError(
            f"v19 checkpoint selected too early: {environment}/{arm}/{optimizer_seed}"
        )
    if len(rows) != spec.EXPECTED_EVALUATION_ROWS_PER_CELL:
        raise ValueError(
            f"v19 heldout row count mismatch: {environment}/{arm}/{optimizer_seed}"
        )
    if _path_registry(rows) != expected_paths:
        raise ValueError(
            f"v19 heldout path mismatch: {environment}/{arm}/{optimizer_seed}"
        )
    if not all(_as_bool(row["protocol_valid"]) for row in rows):
        raise ValueError(
            f"v19 invalid rollout protocol: {environment}/{arm}/{optimizer_seed}"
        )
    expected_projection = bool(arm_spec["terminal_reserve_projection"])
    if not all(
        _as_bool(row["terminal_reserve_context_enabled"])
        and _as_bool(row["terminal_reserve_projection_enabled"])
        == expected_projection
        for row in rows
    ):
        raise ValueError(
            f"v19 row-level terminal contract mismatch: {environment}/{arm}/{optimizer_seed}"
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
        raise ValueError("v19 bootstrap requires one value per optimizer root")
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
        raise ValueError("v19 paired arms do not share heldout paths")
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise ValueError("v19 capacity-matched arm parameter counts differ")

    projected_arms = (
        spec.PRIMARY_MECHANISM_BASELINE,
        *spec.CONSISTENCY_CANDIDATES,
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
    for candidate in spec.CONSISTENCY_CANDIDATES:
        environment_results: list[dict[str, Any]] = []
        pooled_reward_vs_reserve: list[float] = []
        pooled_reward_vs_raw: list[float] = []
        pooled_correction_improvement: list[float] = []
        for environment in spec.ENVIRONMENTS:
            reserve = [
                registry[(environment, spec.PRIMARY_MECHANISM_BASELINE, int(seed))]
                for seed in spec.OPTIMIZER_SEEDS
            ]
            raw = [
                registry[(environment, spec.RAW_CONTEXT_BASELINE, int(seed))]
                for seed in spec.OPTIMIZER_SEEDS
            ]
            selected = [
                registry[(environment, candidate, int(seed))]
                for seed in spec.OPTIMIZER_SEEDS
            ]
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
            correction_values = [
                float(base["component_correction_rms"])
                - float(choice["component_correction_rms"])
                - spec.MINIMUM_COMPONENT_CORRECTION_RELATIVE_REDUCTION
                * float(base["component_correction_rms"])
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
            correction_interval = _paired_interval(
                correction_values, seed_offset=offset
            )
            offset += 1
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
                    **correction_interval,
                    "estimated_relative_reduction": _relative_reduction(
                        [float(cell["component_correction_rms"]) for cell in reserve],
                        [float(cell["component_correction_rms"]) for cell in selected],
                    ),
                    "supported": correction_interval["ci_low"] > 0.0,
                },
            })
        for index in range(len(spec.OPTIMIZER_SEEDS)):
            reserve_reward = statistics.fmean(
                float(registry[(environment, spec.PRIMARY_MECHANISM_BASELINE, int(
                    spec.OPTIMIZER_SEEDS[index]
                ))]["reward"])
                for environment in spec.ENVIRONMENTS
            )
            raw_reward = statistics.fmean(
                float(registry[(environment, spec.RAW_CONTEXT_BASELINE, int(
                    spec.OPTIMIZER_SEEDS[index]
                ))]["reward"])
                for environment in spec.ENVIRONMENTS
            )
            selected_reward = statistics.fmean(
                float(registry[(environment, candidate, int(
                    spec.OPTIMIZER_SEEDS[index]
                ))]["reward"])
                for environment in spec.ENVIRONMENTS
            )
            reserve_correction = statistics.fmean(
                float(registry[(environment, spec.PRIMARY_MECHANISM_BASELINE, int(
                    spec.OPTIMIZER_SEEDS[index]
                ))]["component_correction_rms"])
                for environment in spec.ENVIRONMENTS
            )
            selected_correction = statistics.fmean(
                float(registry[(environment, candidate, int(
                    spec.OPTIMIZER_SEEDS[index]
                ))]["component_correction_rms"])
                for environment in spec.ENVIRONMENTS
            )
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
            pooled_correction_improvement.append(
                reserve_correction
                - selected_correction
                - spec.MINIMUM_COMPONENT_CORRECTION_RELATIVE_REDUCTION
                * reserve_correction
            )
        pooled_reward_reserve_interval = _paired_interval(
            pooled_reward_vs_reserve, seed_offset=offset
        )
        offset += 1
        pooled_reward_raw_interval = _paired_interval(
            pooled_reward_vs_raw, seed_offset=offset
        )
        offset += 1
        pooled_correction_interval = _paired_interval(
            pooled_correction_improvement, seed_offset=offset
        )
        offset += 1
        correction_supported_count = sum(
            bool(row["component_correction_minimum_reduction"]["supported"])
            for row in environment_results
        )
        reward_supported = all(
            bool(row["reward_noninferiority_vs_reserve"]["supported"])
            and bool(row["reward_noninferiority_vs_raw"]["supported"])
            for row in environment_results
        )
        correction_supported = bool(
            pooled_correction_interval["ci_low"] > 0.0
            and correction_supported_count
            >= spec.MINIMUM_SUPPORTED_ENVIRONMENTS
        )
        gates = {
            "mechanism_baseline_valid": bool(
                validity[spec.PRIMARY_MECHANISM_BASELINE]["supported"]
            ),
            "candidate_valid": bool(validity[candidate]["supported"]),
            "reward_noninferior_all_environments": reward_supported,
            "component_correction_supported": correction_supported,
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
            "environment_results": environment_results,
            "pooled_reward_noninferiority_vs_reserve": {
                **pooled_reward_reserve_interval,
                "supported": pooled_reward_reserve_interval["ci_low"] >= 0.0,
            },
            "pooled_reward_noninferiority_vs_raw": {
                **pooled_reward_raw_interval,
                "supported": pooled_reward_raw_interval["ci_low"] >= 0.0,
            },
            "pooled_component_correction_minimum_reduction": {
                **pooled_correction_interval,
                "supported": pooled_correction_interval["ci_low"] > 0.0,
            },
            "component_correction_supported_environment_count": (
                correction_supported_count
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
                -float(row["mean_environment_estimated_relative_reduction"]),
                float(row["consistency_coef"]),
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
        "analysis_version": "mujoco_v19_terminal_reserve_training_analysis_v1",
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
        "support_gate": support_gate,
        "status": (
            spec.SUPPORTED_STATUS if support_gate else spec.NOT_SUPPORTED_STATUS
        ),
        "cells": cells,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# MuJoCo v19 Terminal-Reserve Development",
        "",
        f"- Status: `{result['status']}`",
        f"- Cells: {result['cell_count']}",
        f"- Selected candidate: `{result['selected_candidate']}`",
        f"- Selected coefficient: `{result['selected_consistency_coef']}`",
        "- Evidence role: development only, not confirmatory or manuscript evidence",
        "",
        "| Candidate | Eligible | Worst correction reduction | Supported envs |",
        "|---|---:|---:|---:|",
    ]
    for row in result["candidate_results"]:
        lines.append(
            "| {candidate} | {eligible} | {reduction:.4f} | {count}/3 |".format(
                candidate=row["candidate"],
                eligible=str(bool(row["eligible"])).lower(),
                reduction=float(
                    row["worst_environment_estimated_relative_reduction"]
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
