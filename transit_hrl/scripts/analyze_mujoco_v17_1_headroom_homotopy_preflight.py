#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v17.1 development preflight."""

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

from freq_hrl.experiments.mujoco.control_validation import (  # noqa: E402
    lower_action_router_training_strength,
)
from scripts import (  # noqa: E402
    mujoco_v17_1_headroom_homotopy_preflight_spec as spec,
)
from scripts.submit_mujoco_v17_1_headroom_homotopy_preflight_scheduleurm import (  # noqa: E402
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


def _expected_training_strengths(arm_spec: dict[str, Any]) -> list[float]:
    return [
        lower_action_router_training_strength(
            iteration=iteration,
            total_iterations=spec.ITERATIONS,
            target_strength=float(arm_spec["lower_action_router_strength"]),
            schedule=str(arm_spec["lower_action_router_training_schedule"]),
            warmup_fraction=float(
                arm_spec["lower_action_router_warmup_fraction"]
            ),
            ramp_fraction=float(arm_spec["lower_action_router_ramp_fraction"]),
        )
        for iteration in range(spec.ITERATIONS)
    ]


def _validate_cell(
    environment: str,
    arm: str,
    optimizer_seed: int,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    arm_spec = spec.ARMS[str(arm)]
    candidate = arm in spec.CANDIDATE_ARMS
    expected_strengths = _expected_training_strengths(arm_spec)
    observed_strengths = [
        float(value)
        for value in summary.get(
            "lower_action_router_training_strengths_by_iteration", []
        )
    ]
    if (
        summary.get("protocol_version")
        != arm_spec["control_protocol_version"]
        or summary.get("code_revision") != spec.FROZEN_ALGORITHM_REVISION
        or summary.get("source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
        or summary.get("environment") != str(environment)
        or int(summary.get("optimizer_seed", -1)) != int(optimizer_seed)
        or summary.get("method") != arm_spec["method"]
        or summary.get("responsibility_mode") != arm_spec["responsibility_mode"]
        or summary.get("upper_action_decoder_mode")
        != arm_spec["upper_action_decoder_mode"]
        or float(summary.get("upper_promotion_gain", -1.0))
        != float(arm_spec["upper_promotion_gain"])
        or summary.get("lower_action_router_mode")
        != arm_spec["lower_action_router_mode"]
        or float(summary.get("lower_action_router_alpha", -1.0))
        != float(arm_spec["lower_action_router_alpha"])
        or float(summary.get("lower_action_router_strength", -1.0))
        != float(arm_spec["lower_action_router_strength"])
        or summary.get("lower_action_router_training_schedule")
        != arm_spec["lower_action_router_training_schedule"]
        or float(summary.get("lower_action_router_warmup_fraction", -1.0))
        != float(arm_spec["lower_action_router_warmup_fraction"])
        or float(summary.get("lower_action_router_ramp_fraction", -1.0))
        != float(arm_spec["lower_action_router_ramp_fraction"])
        or bool(summary.get("lower_action_router_observe_strength")) is not True
        or observed_strengths != expected_strengths
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
            "v17.1 cell contract mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    if summary.get("upper_action_decoder_contract") != spec.SMOOTH_PLAN_CONTRACT:
        raise ValueError(f"v17.1 smooth-plan contract mismatch: {environment}/{arm}")
    if candidate and (
        summary.get("lower_action_router_contract")
        != spec.HEADROOM_ROUTER_CONTRACT
        or summary.get("lower_action_headroom_contract")
        != spec.HEADROOM_ACTION_CONTRACT
    ):
        raise ValueError(
            f"v17.1 headroom router contract mismatch: {environment}/{arm}"
        )
    expected_paths = {
        (mode, int(seed))
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.EVALUATION_SEEDS
    }
    if _path_registry(rows) != expected_paths:
        raise ValueError(
            "v17.1 heldout path mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    expected_row_strength = float(arm_spec["lower_action_router_strength"])
    expected_promotion_gain = float(arm_spec["upper_promotion_gain"])
    if any(
        float(row["LowerActionRouterStrength"]) != expected_row_strength
        or float(row["UpperPromotionGain"]) != expected_promotion_gain
        for row in rows
    ):
        raise ValueError(
            f"v17.1 heldout target mismatch: {environment}/{arm}"
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
        "max_macro_completion_error": max(
            float(row["LowerRouterMacroCompletionErrorMax"]) for row in rows
        ),
        "max_responsibility_reconstruction_rms": max(
            float(row["ResponsibilityReconstructionRMS"]) for row in rows
        ),
        "max_additive_action_clip_rate": max(
            float(row["AdditiveActionClipRate"]) for row in rows
        ),
        "mean_headroom_clip_rate": _mean(
            rows, "LowerRouterHeadroomClipRate"
        ),
        "mean_upper_promotion_rms": _mean(rows, "UpperPromotionRMS"),
        "mean_upper_promotion_activation_rate": _mean(
            rows, "UpperPromotionActivationRate"
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
                raise ValueError("v17.1 arms do not share heldout paths")
            summaries = {
                arm: _summarize(summary, rows)
                for arm, (summary, rows) in entries.items()
            }
            if len({row["parameter_count"] for row in summaries.values()}) != 1:
                raise ValueError("v17.1 arm parameter counts are not matched")
            control = summaries[spec.SMOOTH_DIRECT_CONTROL]
            reward_floor = control["reward"] - (
                spec.REWARD_NONINFERIORITY_FRACTION
                * max(abs(control["reward"]), 1.0)
            )
            for arm in spec.CANDIDATE_ARMS:
                candidate = summaries[arm]
                upper_reduction = _relative_reduction(
                    control["upper_hf_power"], candidate["upper_hf_power"]
                )
                lower_reduction = _relative_reduction(
                    control["raw_lower_lf_power"],
                    candidate["raw_lower_lf_power"],
                )
                lower_reduction_vs_latent = _relative_reduction(
                    candidate["latent_lower_lf_power"],
                    candidate["raw_lower_lf_power"],
                )
                joint_reduction = _relative_reduction(
                    control["raw_joint_merit"],
                    candidate["raw_joint_merit"],
                )
                configured_gain = float(spec.ARMS[arm]["upper_promotion_gain"])
                gates = {
                    "trained_checkpoint": bool(
                        candidate["selected_checkpoint_iteration"]
                        >= spec.CHECKPOINT_MINIMUM_ITERATION
                    ),
                    "reward_noninferior": bool(
                        candidate["reward"] >= reward_floor
                    ),
                    "upper_hf_nonworsening": bool(
                        upper_reduction
                        >= -spec.MAXIMUM_UPPER_HF_RELATIVE_INCREASE
                    ),
                    "raw_lower_lf_reduction": bool(
                        lower_reduction
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
                    "nominal_action_headroom_exact": bool(
                        candidate["max_additive_action_clip_rate"]
                        <= spec.ADDITIVE_CLIP_RATE_TOLERANCE
                    ),
                    "promotion_contract_active": bool(
                        candidate["mean_upper_promotion_rms"]
                        > spec.MINIMUM_PROMOTION_RMS
                        if configured_gain > 0.0
                        else candidate["mean_upper_promotion_rms"] == 0.0
                    ),
                }
                cells.append({
                    "environment": environment,
                    "optimizer_seed": int(seed),
                    "arm": arm,
                    "control_summary": control,
                    "candidate_summary": candidate,
                    "reward_floor": reward_floor,
                    "reward_margin_above_floor": (
                        candidate["reward"] - reward_floor
                    ) / max(abs(control["reward"]), 1.0),
                    "upper_hf_relative_reduction": upper_reduction,
                    "raw_lower_lf_relative_reduction": lower_reduction,
                    "raw_lower_lf_relative_reduction_vs_latent": (
                        lower_reduction_vs_latent
                    ),
                    "raw_joint_merit_relative_reduction": joint_reduction,
                    "gates": gates,
                    "supported": bool(all(gates.values())),
                })

    candidate_results: list[dict[str, Any]] = []
    mechanism_gates = (
        "trained_checkpoint",
        "complete_macro_zero_sum",
        "projection_active",
        "responsibility_reconstruction_exact",
        "nominal_action_headroom_exact",
        "promotion_contract_active",
    )
    for arm in spec.CANDIDATE_ARMS:
        rows = [row for row in cells if row["arm"] == arm]
        gate_counts = {
            gate: sum(bool(row["gates"][gate]) for row in rows)
            for gate in rows[0]["gates"]
        }
        eligible = bool(
            all(gate_counts[gate] == len(spec.ENVIRONMENTS) for gate in mechanism_gates)
            and gate_counts["reward_noninferior"] == len(spec.ENVIRONMENTS)
            and gate_counts["upper_hf_nonworsening"]
            >= spec.MINIMUM_PERFORMANCE_ENVIRONMENTS
            and gate_counts["raw_lower_lf_reduction"]
            >= spec.MINIMUM_PERFORMANCE_ENVIRONMENTS
            and gate_counts["raw_lower_lf_reduction_vs_latent"]
            >= spec.MINIMUM_PERFORMANCE_ENVIRONMENTS
            and gate_counts["raw_joint_merit_reduction"]
            >= spec.MINIMUM_PERFORMANCE_ENVIRONMENTS
        )
        candidate_results.append({
            "arm": arm,
            "eligible_for_fresh_multiseed": eligible,
            "gate_counts": gate_counts,
            "worst_environment_reward_margin_above_floor": min(
                float(row["reward_margin_above_floor"]) for row in rows
            ),
            "median_raw_joint_merit_relative_reduction": statistics.median(
                float(row["raw_joint_merit_relative_reduction"])
                for row in rows
            ),
            "supported_environment_count": sum(
                bool(row["supported"]) for row in rows
            ),
        })
    eligible = [
        row for row in candidate_results
        if row["eligible_for_fresh_multiseed"]
    ]
    selected = (
        max(
            eligible,
            key=lambda row: (
                float(row["worst_environment_reward_margin_above_floor"]),
                float(row["median_raw_joint_merit_relative_reduction"]),
            ),
        )
        if eligible else None
    )
    support_gate = selected is not None
    return {
        "analysis_version": (
            "mujoco_v17_1_headroom_homotopy_preflight_analysis_v1"
        ),
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(cells),
        "cells": cells,
        "candidate_results": candidate_results,
        "selected_arm_for_fresh_multiseed": (
            selected["arm"] if selected is not None else None
        ),
        "support_gate": support_gate,
        "status": (
            spec.SUPPORTED_STATUS if support_gate else spec.NOT_SUPPORTED_STATUS
        ),
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
    (target / "headroom_homotopy_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = (
        "environment",
        "optimizer_seed",
        "arm",
        "supported",
        "reward_margin_above_floor",
        "upper_hf_relative_reduction",
        "raw_lower_lf_relative_reduction",
        "raw_lower_lf_relative_reduction_vs_latent",
        "raw_joint_merit_relative_reduction",
    )
    with (target / "headroom_homotopy_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result["cells"]:
            writer.writerow({key: row[key] for key in fields})
    print(json.dumps({
        "status": result["status"],
        "cell_count": result["cell_count"],
        "selected_arm": result["selected_arm_for_fresh_multiseed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
