#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v17 raw-action architecture screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v17_zero_dc_plan_screen_spec as spec  # noqa: E402
from scripts.submit_mujoco_v17_zero_dc_plan_screen_scheduleurm import (  # noqa: E402
    cell_relative_dir,
)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _relative_reduction(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(abs(baseline), 1e-12))


def _path_registry(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(row["disturbance_mode"]), int(row["seed"]))
        for row in rows
    }


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
    router_mode = str(arm_spec["lower_action_router_mode"])
    expected_router_strength = (
        0.0 if router_mode == "direct" else float(arm_spec["lower_action_router_strength"])
    )
    if (
        summary.get("protocol_version") != spec.FROZEN_CORE_PROTOCOL_VERSION
        or summary.get("code_revision") != spec.FROZEN_ALGORITHM_REVISION
        or summary.get("source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
        or summary.get("environment") != str(environment)
        or int(summary.get("optimizer_seed", -1)) != int(optimizer_seed)
        or summary.get("method") != arm_spec["method"]
        or summary.get("responsibility_mode") != arm_spec["responsibility_mode"]
        or summary.get("upper_action_decoder_mode")
        != arm_spec["upper_action_decoder_mode"]
        or summary.get("lower_action_router_mode") != router_mode
        or float(summary.get("lower_action_router_alpha", -1.0))
        != float(arm_spec["lower_action_router_alpha"])
        or float(summary.get("lower_action_router_strength", -1.0))
        != expected_router_strength
        or bool(summary.get("lower_action_router_observe_strength"))
        != bool(arm_spec["lower_action_router_observe_strength"])
        or summary.get("leakage_constraint_scope")
        != arm_spec["leakage_constraint_scope"]
        or summary.get("leakage_constraint_cost_mode")
        != arm_spec["leakage_cost_mode"]
        or summary.get("checkpoint_score_mode") != spec.CHECKPOINT_SCORE_MODE
        or int(summary.get("selected_checkpoint_iteration", -1))
        < spec.CHECKPOINT_MINIMUM_ITERATION
        or len(rows) != spec.EXPECTED_EVALUATION_ROWS_PER_CELL
        or not all(bool(int(float(row["protocol_valid"]))) for row in rows)
    ):
        raise ValueError(
            "v17 cell contract mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    if arm != spec.HOLD_DIRECT_CONTROL and (
        summary.get("upper_action_decoder_contract") != spec.SMOOTH_PLAN_CONTRACT
    ):
        raise ValueError(f"v17 smooth-plan contract mismatch: {environment}/{arm}")
    if arm == spec.ZERO_DC_PLAN_CANDIDATE and (
        summary.get("lower_action_router_contract") != spec.ZERO_DC_ROUTER_CONTRACT
    ):
        raise ValueError(f"v17 zero-DC router contract mismatch: {environment}/{arm}")
    expected_paths = {
        (mode, int(seed))
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.EVALUATION_SEEDS
    }
    if _path_registry(rows) != expected_paths:
        raise ValueError(
            "v17 heldout path mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )


def _summarize(
    summary: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    upper = _mean(rows, "UpperHFPowerAbs")
    raw_lower = _mean(rows, "RawLowerLFDriftAbs")
    latent_lower = _mean(rows, "LatentLowerLFDriftAbs")
    return {
        "reward": _mean(rows, "episode_return"),
        "upper_hf_power": upper,
        "raw_lower_lf_power": raw_lower,
        "latent_lower_lf_power": latent_lower,
        "raw_joint_merit": (
            upper / spec.UPPER_HF_RMS_BUDGET ** 2
            + raw_lower / spec.LOWER_LF_RMS_BUDGET ** 2
        ),
        "mean_macro_projection_rate": _mean(
            rows, "LowerRouterMacroProjectionRate"
        ),
        "mean_macro_debt_rms": _mean(rows, "LowerRouterMacroDebtRMSMean"),
        "max_macro_completion_error": max(
            float(row["LowerRouterMacroCompletionErrorMax"]) for row in rows
        ),
        "max_responsibility_reconstruction_rms": max(
            float(row["ResponsibilityReconstructionRMS"]) for row in rows
        ),
        "mean_router_clip_rate": _mean(rows, "LowerRouterClipRate"),
        "mean_additive_action_clip_rate": _mean(
            rows, "AdditiveActionClipRate"
        ),
        "selected_checkpoint_iteration": int(
            summary["selected_checkpoint_iteration"]
        ),
        "parameter_count": int(summary["capacity_actual_parameter_count"]),
    }


def analyze(run_name: str) -> dict[str, Any]:
    registry: dict[
        tuple[str, str, int], tuple[dict[str, Any], list[dict[str, Any]]]
    ] = {}
    for environment in spec.ENVIRONMENTS:
        for arm in spec.ARMS:
            for seed in spec.OPTIMIZER_SEEDS:
                summary, rows = _load_cell(run_name, environment, arm, seed)
                _validate_cell(environment, arm, seed, summary, rows)
                registry[(environment, arm, int(seed))] = (summary, rows)

    cells: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            entries = {
                arm: registry[(environment, arm, int(seed))]
                for arm in spec.ARMS
            }
            if len({
                frozenset(_path_registry(rows))
                for _, rows in entries.values()
            }) != 1:
                raise ValueError("v17 arms do not share heldout paths")
            summaries = {
                arm: _summarize(summary, rows)
                for arm, (summary, rows) in entries.items()
            }
            if len({row["parameter_count"] for row in summaries.values()}) != 1:
                raise ValueError("v17 arm parameter counts are not matched")
            hold = summaries[spec.HOLD_DIRECT_CONTROL]
            smooth = summaries[spec.SMOOTH_DIRECT_CONTROL]
            candidate = summaries[spec.ZERO_DC_PLAN_CANDIDATE]
            reward_floor = hold["reward"] - (
                spec.REWARD_NONINFERIORITY_FRACTION
                * max(abs(hold["reward"]), 1.0)
            )
            upper_reduction = _relative_reduction(
                hold["upper_hf_power"], candidate["upper_hf_power"]
            )
            smooth_upper_reduction = _relative_reduction(
                hold["upper_hf_power"], smooth["upper_hf_power"]
            )
            lower_reduction_vs_smooth = _relative_reduction(
                smooth["raw_lower_lf_power"], candidate["raw_lower_lf_power"]
            )
            lower_reduction_vs_latent = _relative_reduction(
                candidate["latent_lower_lf_power"],
                candidate["raw_lower_lf_power"],
            )
            joint_reduction = _relative_reduction(
                hold["raw_joint_merit"], candidate["raw_joint_merit"]
            )
            gates = {
                "trained_checkpoint": bool(
                    candidate["selected_checkpoint_iteration"]
                    >= spec.CHECKPOINT_MINIMUM_ITERATION
                ),
                "reward_noninferior": bool(candidate["reward"] >= reward_floor),
                "smooth_upper_ablation": bool(
                    smooth_upper_reduction
                    >= spec.MINIMUM_UPPER_HF_RELATIVE_REDUCTION
                ),
                "candidate_upper_hf_reduction": bool(
                    upper_reduction >= spec.MINIMUM_UPPER_HF_RELATIVE_REDUCTION
                ),
                "candidate_upper_hf_budget": bool(
                    candidate["upper_hf_power"]
                    <= spec.UPPER_HF_RMS_BUDGET ** 2
                ),
                "raw_lower_lf_reduction_vs_smooth": bool(
                    lower_reduction_vs_smooth
                    >= spec.MINIMUM_LOWER_LF_RELATIVE_REDUCTION
                ),
                "raw_lower_lf_reduction_vs_latent": bool(
                    lower_reduction_vs_latent
                    >= spec.MINIMUM_LOWER_LF_RELATIVE_REDUCTION
                ),
                "raw_joint_merit_reduction": bool(
                    joint_reduction
                    >= spec.MINIMUM_JOINT_MERIT_RELATIVE_REDUCTION
                ),
                "complete_macro_zero_sum": bool(
                    candidate["max_macro_completion_error"]
                    <= spec.MACRO_COMPLETION_ERROR_TOLERANCE
                ),
                "projection_active": bool(
                    candidate["mean_macro_projection_rate"]
                    > spec.MINIMUM_PROJECTION_RATE
                ),
                "responsibility_reconstruction_exact": bool(
                    candidate["max_responsibility_reconstruction_rms"]
                    <= spec.RESPONSIBILITY_RECONSTRUCTION_TOLERANCE
                ),
            }
            cells.append({
                "environment": environment,
                "optimizer_seed": int(seed),
                "arm_summaries": summaries,
                "reward_floor": reward_floor,
                "smooth_upper_hf_relative_reduction_vs_hold": smooth_upper_reduction,
                "candidate_upper_hf_relative_reduction_vs_hold": upper_reduction,
                "candidate_raw_lower_lf_relative_reduction_vs_smooth": (
                    lower_reduction_vs_smooth
                ),
                "candidate_raw_lower_lf_relative_reduction_vs_latent": (
                    lower_reduction_vs_latent
                ),
                "candidate_raw_joint_merit_relative_reduction_vs_hold": (
                    joint_reduction
                ),
                "candidate_mean_additive_action_clip_rate": candidate[
                    "mean_additive_action_clip_rate"
                ],
                "gates": gates,
                "supported": bool(all(gates.values())),
            })

    environment_results = []
    for environment in spec.ENVIRONMENTS:
        rows = [row for row in cells if row["environment"] == environment]
        supported_count = sum(bool(row["supported"]) for row in rows)
        environment_results.append({
            "environment": environment,
            "cell_count": len(rows),
            "supported_count": supported_count,
            "environment_gate": bool(
                supported_count >= spec.MINIMUM_SUPPORTED_SEEDS_PER_ENVIRONMENT
            ),
            "median_candidate_upper_hf_relative_reduction_vs_hold": (
                statistics.median(
                    float(row["candidate_upper_hf_relative_reduction_vs_hold"])
                    for row in rows
                )
            ),
            "median_candidate_raw_lower_lf_relative_reduction_vs_smooth": (
                statistics.median(
                    float(row[
                        "candidate_raw_lower_lf_relative_reduction_vs_smooth"
                    ])
                    for row in rows
                )
            ),
            "median_candidate_raw_joint_merit_relative_reduction_vs_hold": (
                statistics.median(
                    float(row[
                        "candidate_raw_joint_merit_relative_reduction_vs_hold"
                    ])
                    for row in rows
                )
            ),
        })
    support_gate = bool(all(row["environment_gate"] for row in environment_results))
    gate_counts = {
        gate: sum(bool(row["gates"][gate]) for row in cells)
        for gate in cells[0]["gates"]
    }
    return {
        "analysis_version": "mujoco_v17_zero_dc_plan_screen_analysis_v1",
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(cells),
        "supported_cell_count": sum(bool(row["supported"]) for row in cells),
        "gate_counts": gate_counts,
        "cells": cells,
        "environment_results": environment_results,
        "support_gate": support_gate,
        "status": spec.SUPPORTED_STATUS if support_gate else spec.NOT_SUPPORTED_STATUS,
    }


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
    (target / "zero_dc_plan_screen.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = (
        "environment",
        "optimizer_seed",
        "supported",
        "smooth_upper_hf_relative_reduction_vs_hold",
        "candidate_upper_hf_relative_reduction_vs_hold",
        "candidate_raw_lower_lf_relative_reduction_vs_smooth",
        "candidate_raw_lower_lf_relative_reduction_vs_latent",
        "candidate_raw_joint_merit_relative_reduction_vs_hold",
        "candidate_mean_additive_action_clip_rate",
    )
    with (target / "zero_dc_plan_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result["cells"]:
            writer.writerow({key: row[key] for key in fields})
    print(json.dumps({
        "status": result["status"],
        "cell_count": result["cell_count"],
        "supported_cell_count": result["supported_cell_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
