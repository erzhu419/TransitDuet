#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v17.3 paired audit-optimal preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    mujoco_v17_3_audit_optimal_macro_gauge_preflight_spec as spec,
)
from scripts.run_mujoco_v17_3_paired_gauge_cell import (  # noqa: E402
    paired_intervention_audit,
)
from scripts.submit_mujoco_v17_3_audit_optimal_macro_gauge_preflight_scheduleurm import (  # noqa: E402
    cell_relative_dir,
)


NUMERIC_LATENT_METRICS = (
    "LatentUpperHFPowerAbs",
    "LatentLowerLFDriftAbs",
)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _relative_reduction(control: float, candidate: float) -> float:
    return float((control - candidate) / max(abs(control), 1e-12))


def _path_registry(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(row["disturbance_mode"]), int(row["seed"])) for row in rows
    }


def _explicit_protocol_row_valid(row: dict[str, Any]) -> bool:
    episode_length = int(float(row["episode_length"]))
    upper_decisions = int(float(row["upper_decision_count"]))
    upper_transitions = int(float(row["upper_transition_count"]))
    lower_transitions = int(float(row["lower_transition_count"]))
    finite_metrics = (
        "LowerRouterActionReconstructionRMS",
        "ResponsibilityReconstructionRMS",
        "UpperHFPowerAbs",
        "LowerLFDriftAbs",
        "episode_return",
    )
    return bool(
        episode_length > 0
        and lower_transitions == episode_length
        and upper_transitions == upper_decisions
        and 0 < upper_transitions < lower_transitions
        and all(math.isfinite(float(row[key])) for key in finite_metrics)
    )


def _load_cell(
    root: Path,
    run_name: str,
    environment: str,
    optimizer_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = Path(root) / cell_relative_dir(
        run_name, environment, optimizer_seed
    )
    summary = json.loads(
        (directory / "cell_summary.json").read_text(encoding="utf-8")
    )
    with (directory / "evaluation_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    return summary, rows


def _validate_preregistration(root: Path, run_name: str) -> dict[str, Any]:
    run_root = Path(root) / "results" / str(run_name)
    preregistration = json.loads(
        (run_root / "preregistration.json").read_text(encoding="utf-8")
    )
    sync = json.loads(
        (run_root / "run_scoped_result_sync.json").read_text(encoding="utf-8")
    )
    if (
        preregistration.get("status") != spec.PREREGISTRATION_STATUS
        or preregistration.get("evidence_role") != spec.EVIDENCE_ROLE
        or preregistration.get("frozen_algorithm_revision")
        != spec.FROZEN_ALGORITHM_REVISION
        or preregistration.get("frozen_source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
        or preregistration.get("environments") != list(spec.ENVIRONMENTS)
        or float(preregistration.get("router_alpha", -1.0))
        != spec.ROUTER_ALPHA
        or preregistration.get("optimizer_seeds")
        != list(spec.OPTIMIZER_SEEDS)
        or preregistration.get("train_seeds") != list(spec.TRAIN_SEEDS)
        or preregistration.get("selection_seeds")
        != list(spec.SELECTION_SEEDS)
        or preregistration.get("evaluation_seeds")
        != list(spec.EVALUATION_SEEDS)
        or preregistration.get("selection_contract")
        != spec.SELECTION_CONTRACT
        or int(sync.get("cell_count", -1)) != spec.EXPECTED_CELL_COUNT
        or sync.get("artifact_contract") != "small_results_only_v1"
    ):
        raise ValueError("v17.3 preregistration or result-sync contract mismatch")
    return sync


def _split_interventions(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    control = [
        row for row in rows
        if row.get("paired_intervention") == spec.CONTROL_INTERVENTION
    ]
    candidate = [
        row for row in rows
        if row.get("paired_intervention") == spec.CANDIDATE_INTERVENTION
    ]
    if (
        len(control) != spec.EXPECTED_PATHS_PER_INTERVENTION
        or len(candidate) != spec.EXPECTED_PATHS_PER_INTERVENTION
    ):
        raise ValueError("v17.3 paired intervention row counts are incomplete")
    return control, candidate


def _validate_cell(
    environment: str,
    optimizer_seed: int,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if (
        summary.get("development_protocol_version")
        != spec.DEVELOPMENT_PROTOCOL_VERSION
        or summary.get("evidence_role") != spec.EVIDENCE_ROLE
        or summary.get("protocol_version") != spec.FROZEN_CORE_PROTOCOL_VERSION
        or summary.get("code_revision") != spec.FROZEN_ALGORITHM_REVISION
        or summary.get("source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
        or summary.get("environment") != str(environment)
        or int(summary.get("optimizer_seed", -1)) != int(optimizer_seed)
        or summary.get("method") != "freq_hrl"
        or summary.get("responsibility_mode") != "additive"
        or summary.get("upper_action_decoder_mode")
        != "causal_smoothstep_plan"
        or summary.get("upper_action_decoder_contract")
        != spec.SMOOTH_PLAN_CONTRACT
        or summary.get("lower_action_router_mode")
        != "causal_audit_optimal_macro_gauge"
        or float(summary.get("lower_action_router_alpha", -1.0))
        != spec.ROUTER_ALPHA
        or float(summary.get("lower_action_router_strength", -1.0))
        != spec.CONTROL_STRENGTH
        or bool(summary.get("lower_action_router_observe_strength"))
        or summary.get("lower_action_router_contract") != spec.ROUTER_CONTRACT
        or summary.get("policy_filter_state_contract")
        != spec.POLICY_STATE_CONTRACT
        or summary.get("lower_action_router_training_schedule") != "constant"
        or summary.get("checkpoint_score_mode") != spec.CHECKPOINT_SCORE_MODE
        or int(summary.get("selected_checkpoint_iteration", -1))
        < spec.CHECKPOINT_MINIMUM_ITERATION
        or int(summary.get("heldout_evaluation_pass_count", -1)) != 2
        or len(rows) != spec.EXPECTED_EVALUATION_ROWS_PER_CELL
        or not all(_explicit_protocol_row_valid(row) for row in rows)
    ):
        raise ValueError(
            f"v17.3 cell contract mismatch: {environment}/{optimizer_seed}"
        )
    control, candidate = _split_interventions(rows)
    expected_paths = {
        (mode, int(seed))
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.EVALUATION_SEEDS
    }
    if (
        _path_registry(control) != expected_paths
        or _path_registry(candidate) != expected_paths
    ):
        raise ValueError(f"v17.3 heldout path registry mismatch: {environment}")
    if any(
        float(row["LowerActionRouterStrength"]) != spec.CONTROL_STRENGTH
        for row in control
    ) or any(
        float(row["LowerActionRouterStrength"]) != spec.CANDIDATE_STRENGTH
        for row in candidate
    ):
        raise ValueError("v17.3 paired intervention strength mismatch")
    audit = paired_intervention_audit(control, candidate)
    if audit != summary.get("paired_intervention_audit"):
        raise ValueError("v17.3 stored paired audit does not reproduce")
    return control, candidate, audit


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    upper = _mean(rows, "UpperHFPowerAbs")
    lower = _mean(rows, "LowerLFDriftAbs")
    return {
        "reward": _mean(rows, "episode_return"),
        "upper_hf_power": upper,
        "lower_lf_power": lower,
        "joint_merit": (
            upper / spec.UPPER_HF_RMS_BUDGET ** 2
            + lower / spec.LOWER_LF_RMS_BUDGET ** 2
        ),
        "max_router_reconstruction_rms": max(
            float(row["LowerRouterActionReconstructionRMS"]) for row in rows
        ),
        "max_responsibility_reconstruction_rms": max(
            float(row["ResponsibilityReconstructionRMS"]) for row in rows
        ),
        "mean_component_clip_rate": _mean(
            rows, "LowerRouterHeadroomClipRate"
        ),
        "max_additive_clip_excess": max(
            float(row["AdditiveActionClipExcessMax"]) for row in rows
        ),
        "legacy_protocol_valid_rate": statistics.fmean(
            float(row["protocol_valid"]) for row in rows
        ),
    }


def analyze(run_name: str, *, root: Path = ROOT) -> dict[str, Any]:
    sync = _validate_preregistration(Path(root), str(run_name))
    cells: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        for optimizer_seed in spec.OPTIMIZER_SEEDS:
            summary, rows = _load_cell(
                Path(root), run_name, environment, optimizer_seed
            )
            control_rows, candidate_rows, audit = _validate_cell(
                environment,
                optimizer_seed,
                summary,
                rows,
            )
            control = _summarize(control_rows)
            candidate = _summarize(candidate_rows)
            upper_reduction = _relative_reduction(
                control["upper_hf_power"], candidate["upper_hf_power"]
            )
            lower_reduction = _relative_reduction(
                control["lower_lf_power"], candidate["lower_lf_power"]
            )
            joint_reduction = _relative_reduction(
                control["joint_merit"], candidate["joint_merit"]
            )
            metric_differences = audit["metric_max_abs_difference"]
            gates = {
                "reward_trace_exact": (
                    audit["trace_mismatches"]["RewardTraceSHA256"]
                    == spec.TRACE_MATCH_TOLERANCE
                ),
                "executed_action_trace_exact": (
                    audit["trace_mismatches"]["ExecutedActionTraceSHA256"]
                    == spec.TRACE_MATCH_TOLERANCE
                ),
                "latent_policy_trace_exact": (
                    audit["trace_mismatches"]["LatentPolicyTraceSHA256"]
                    == spec.TRACE_MATCH_TOLERANCE
                ),
                "reward_numeric_exact": (
                    float(metric_differences["episode_return"])
                    <= spec.REWARD_ABSOLUTE_TOLERANCE
                ),
                "latent_metrics_exact": all(
                    float(metric_differences[key])
                    <= spec.LATENT_METRIC_ABSOLUTE_TOLERANCE
                    for key in NUMERIC_LATENT_METRICS
                ),
                "router_reconstruction_exact": (
                    candidate["max_router_reconstruction_rms"]
                    <= spec.RECONSTRUCTION_RMS_TOLERANCE
                ),
                "responsibility_reconstruction_exact": (
                    candidate["max_responsibility_reconstruction_rms"]
                    <= spec.RECONSTRUCTION_RMS_TOLERANCE
                ),
                "component_projection_bounded": (
                    candidate["mean_component_clip_rate"]
                    <= spec.MAXIMUM_COMPONENT_CLIP_RATE
                ),
                "explicit_protocol_structure_valid": all(
                    _explicit_protocol_row_valid(row)
                    for row in [*control_rows, *candidate_rows]
                ),
                "upper_hf_reduction": (
                    upper_reduction
                    >= spec.MINIMUM_UPPER_HF_RELATIVE_REDUCTION
                ),
                "lower_lf_reduction": (
                    lower_reduction
                    >= spec.MINIMUM_LOWER_LF_RELATIVE_REDUCTION
                ),
                "joint_merit_reduction": (
                    joint_reduction
                    >= spec.MINIMUM_JOINT_MERIT_RELATIVE_REDUCTION
                ),
            }
            cells.append({
                "environment": environment,
                "optimizer_seed": int(optimizer_seed),
                "control_summary": control,
                "candidate_summary": candidate,
                "paired_audit": audit,
                "upper_hf_relative_reduction": upper_reduction,
                "lower_lf_relative_reduction": lower_reduction,
                "joint_merit_relative_reduction": joint_reduction,
                "gates": gates,
                "supported": bool(all(gates.values())),
            })

    eligible = bool(
        len(cells) == spec.EXPECTED_CELL_COUNT
        and all(bool(row["supported"]) for row in cells)
    )
    gate_counts = {
        gate: sum(bool(row["gates"][gate]) for row in cells)
        for gate in cells[0]["gates"]
    }
    return {
        "status": spec.SUPPORTED_STATUS if eligible else spec.NOT_SUPPORTED_STATUS,
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(cells),
        "paired_path_count": (
            len(cells) * spec.EXPECTED_PATHS_PER_INTERVENTION
        ),
        "evaluation_row_count": (
            len(cells) * spec.EXPECTED_EVALUATION_ROWS_PER_CELL
        ),
        "gate_counts": gate_counts,
        "cells": cells,
        "eligible_for_leakage_active_multiseed": eligible,
        "scheduler_tasks": sync.get("tasks", {}),
        "scheduler_node_counts": sync.get("nodes", {}),
        "claim_boundary": (
            "development paired intervention only; no reward improvement, "
            "training-seed uncertainty, or confirmatory claim"
        ),
        "analysis_contract": (
            "pre_registered_transition_structure_and_reconstruction_rms_"
            "recomputed_directly_legacy_protocol_flag_diagnostic_only_v1"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = [{
        "environment": row["environment"],
        "optimizer_seed": row["optimizer_seed"],
        "upper_hf_relative_reduction": row["upper_hf_relative_reduction"],
        "lower_lf_relative_reduction": row["lower_lf_relative_reduction"],
        "joint_merit_relative_reduction": row[
            "joint_merit_relative_reduction"
        ],
        "supported": row["supported"],
        **{f"gate_{key}": value for key, value in row["gates"].items()},
    } for row in rows]
    fields = list(flattened[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = analyze(args.run_name)
    output = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / "results" / args.run_name / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_optimal_macro_gauge_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "audit_optimal_macro_gauge_cells.csv", result["cells"])
    print(json.dumps({
        "status": result["status"],
        "eligible_for_leakage_active_multiseed": result[
            "eligible_for_leakage_active_multiseed"
        ],
        "cell_count": result["cell_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
