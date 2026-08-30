#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v16.2 macro-hold gauge screen."""

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

from scripts import mujoco_v16_2_macro_hold_gauge_screen_spec as spec  # noqa: E402
from scripts.submit_mujoco_v16_2_macro_hold_gauge_screen_scheduleurm import (  # noqa: E402
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
    if (
        summary.get("protocol_version") != spec.FROZEN_CORE_PROTOCOL_VERSION
        or summary.get("code_revision") != spec.FROZEN_ALGORITHM_REVISION
        or summary.get("source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
        or summary.get("environment") != str(environment)
        or int(summary.get("optimizer_seed", -1)) != int(optimizer_seed)
        or summary.get("method") != arm_spec["method"]
        or summary.get("responsibility_mode")
        != arm_spec["responsibility_mode"]
        or summary.get("lower_action_router_mode")
        != arm_spec["lower_action_router_mode"]
        or float(summary.get("lower_action_router_alpha", -1.0))
        != float(arm_spec["lower_action_router_alpha"])
        or float(summary.get("lower_action_router_strength", -1.0))
        != float(arm_spec["lower_action_router_strength"])
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
            "v16.2 cell contract mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )
    expected_paths = {
        (mode, int(seed))
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.EVALUATION_SEEDS
    }
    if _path_registry(rows) != expected_paths:
        raise ValueError(
            "v16.2 heldout path mismatch: "
            f"{environment}/{arm}/{optimizer_seed}"
        )


def _summarize(
    summary: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    upper = _mean(rows, "UpperHFPowerAbs")
    lower = _mean(rows, "LowerLFDriftAbs")
    latent_upper = _mean(rows, "LatentUpperHFPowerAbs")
    latent_lower = _mean(rows, "LatentLowerLFDriftAbs")
    return {
        "reward": _mean(rows, "episode_return"),
        "upper_hf_power": upper,
        "lower_lf_power": lower,
        "latent_upper_hf_power": latent_upper,
        "latent_lower_lf_power": latent_lower,
        "joint_merit": (
            upper / spec.UPPER_HF_RMS_BUDGET ** 2
            + lower / spec.LOWER_LF_RMS_BUDGET ** 2
        ),
        "latent_joint_merit": (
            latent_upper / spec.UPPER_HF_RMS_BUDGET ** 2
            + latent_lower / spec.LOWER_LF_RMS_BUDGET ** 2
        ),
        "max_router_clip_rate": max(
            float(row["LowerRouterClipRate"]) for row in rows
        ),
        "max_router_reconstruction_rms": max(
            float(row["LowerRouterActionReconstructionRMS"])
            for row in rows
        ),
        "max_responsibility_reconstruction_rms": max(
            float(row["ResponsibilityReconstructionRMS"])
            for row in rows
        ),
        "audit_alpha_mean": _mean(rows, "LowerRouterAuditAlphaMean"),
        "audit_alpha_final_mean": _mean(rows, "LowerRouterAuditAlphaFinal"),
        "selected_checkpoint_iteration": int(
            summary["selected_checkpoint_iteration"]
        ),
        "parameter_count": int(summary["parameter_count"]),
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
                raise ValueError("v16.2 arms do not share heldout paths")
            summaries = {
                arm: _summarize(summary, rows)
                for arm, (summary, rows) in entries.items()
            }
            if len({row["parameter_count"] for row in summaries.values()}) != 1:
                raise ValueError("v16.2 arm parameter counts are not matched")
            direct = summaries[spec.DIRECT_CONTROL]
            primitive = summaries[spec.PRIMITIVE_GAUGE_CONTROL]
            candidate = summaries[spec.MACRO_HOLD_CANDIDATE]
            reward_floor = direct["reward"] - (
                spec.REWARD_NONINFERIORITY_FRACTION
                * max(abs(direct["reward"]), 1.0)
            )
            lower_reduction = _relative_reduction(
                candidate["latent_lower_lf_power"],
                candidate["lower_lf_power"],
            )
            joint_reduction = _relative_reduction(
                candidate["latent_joint_merit"], candidate["joint_merit"]
            )
            reconstruction_exact = bool(
                candidate["max_router_reconstruction_rms"]
                <= spec.RECONSTRUCTION_RMS_TOLERANCE
                and candidate["max_responsibility_reconstruction_rms"]
                <= spec.RECONSTRUCTION_RMS_TOLERANCE
            )
            gates = {
                "trained_checkpoint": bool(
                    candidate["selected_checkpoint_iteration"]
                    >= spec.CHECKPOINT_MINIMUM_ITERATION
                ),
                "reward_noninferior": bool(candidate["reward"] >= reward_floor),
                "reconstruction_exact": reconstruction_exact,
                "router_clip_free": bool(
                    candidate["max_router_clip_rate"]
                    <= spec.MAXIMUM_ROUTER_CLIP_RATE
                ),
                "upper_hf_budget": bool(
                    candidate["upper_hf_power"]
                    <= spec.UPPER_HF_RMS_BUDGET ** 2
                ),
                "lower_lf_reduction": bool(
                    lower_reduction
                    >= spec.MINIMUM_LOWER_LF_RELATIVE_REDUCTION
                ),
                "joint_merit_reduction": bool(
                    joint_reduction
                    >= spec.MINIMUM_JOINT_MERIT_RELATIVE_REDUCTION
                ),
            }
            cells.append({
                "environment": environment,
                "optimizer_seed": int(seed),
                "arm_summaries": summaries,
                "reward_floor": reward_floor,
                "macro_lower_lf_relative_reduction_vs_latent": lower_reduction,
                "macro_joint_merit_relative_reduction_vs_latent": joint_reduction,
                "macro_upper_hf_relative_reduction_vs_primitive": (
                    _relative_reduction(
                        primitive["upper_hf_power"],
                        candidate["upper_hf_power"],
                    )
                ),
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
            "median_macro_lower_lf_relative_reduction_vs_latent": (
                statistics.median(
                    float(row[
                        "macro_lower_lf_relative_reduction_vs_latent"
                    ])
                    for row in rows
                )
            ),
            "median_macro_joint_merit_relative_reduction_vs_latent": (
                statistics.median(
                    float(row[
                        "macro_joint_merit_relative_reduction_vs_latent"
                    ])
                    for row in rows
                )
            ),
            "median_macro_upper_hf_relative_reduction_vs_primitive": (
                statistics.median(
                    float(row[
                        "macro_upper_hf_relative_reduction_vs_primitive"
                    ])
                    for row in rows
                )
            ),
        })
    support_gate = bool(all(row["environment_gate"] for row in environment_results))
    return {
        "analysis_version": "mujoco_v16_2_macro_hold_gauge_screen_analysis_v1",
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(cells),
        "supported_cell_count": sum(bool(row["supported"]) for row in cells),
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
    (target / "macro_hold_gauge_screen.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = (
        "environment",
        "optimizer_seed",
        "supported",
        "macro_lower_lf_relative_reduction_vs_latent",
        "macro_joint_merit_relative_reduction_vs_latent",
        "macro_upper_hf_relative_reduction_vs_primitive",
    )
    with (target / "macro_hold_gauge_cells.csv").open(
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
