#!/usr/bin/env python3
"""Analyze the frozen v16 training-time gauge development preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.domains.mujoco import lower_action_router_contract
from scripts import mujoco_v16_gauge_training_preflight_spec as spec
from scripts.submit_mujoco_v16_gauge_training_preflight_scheduleurm import (
    cell_relative_dir,
)

def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _frequency_merit(rows: list[dict[str, Any]], *, latent: bool) -> float:
    lower_key = "LatentLowerLFDriftAbs" if latent else "LowerLFDriftAbs"
    upper_key = "LatentUpperHFPowerAbs" if latent else "UpperHFPowerAbs"
    return float(
        _mean(rows, lower_key) / (spec.LOWER_LF_RMS_BUDGET ** 2)
        + _mean(rows, upper_key) / (spec.UPPER_HF_RMS_BUDGET ** 2)
    )


def _relative_reduction(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(abs(baseline), 1e-12))


def _path_registry(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(row["disturbance_mode"]), int(row["seed"]))
        for row in rows
    }


def _validate_cell(
    environment: str,
    arm: str,
    optimizer_seed: int,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    arm_spec = spec.ARMS[arm]
    expected_paths = {
        (mode, seed)
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.EVALUATION_SEEDS
    }
    expected_contract = lower_action_router_contract(
        str(arm_spec["lower_action_router_mode"])
    )
    valid = bool(
        summary.get("environment") == environment
        and int(summary.get("optimizer_seed", -1)) == int(optimizer_seed)
        and summary.get("method") == "freq_hrl"
        and summary.get("code_revision") == spec.FROZEN_ALGORITHM_REVISION
        and summary.get("source_manifest_sha256")
        == spec.FROZEN_SOURCE_MANIFEST_SHA256
        and summary.get("lower_action_router_mode")
        == arm_spec["lower_action_router_mode"]
        and float(summary.get("lower_action_router_strength", -1.0))
        == float(arm_spec["lower_action_router_strength"])
        and summary.get("lower_action_router_contract") == expected_contract
        and summary.get("leakage_constraint_scope")
        == arm_spec["leakage_constraint_scope"]
        and summary.get("leakage_constraint_cost_mode")
        == arm_spec["leakage_cost_mode"]
        and summary.get("upper_constraint_mode") == "primal_dual"
        and float(summary.get("upper_dual_lr", -1.0))
        == float(arm_spec["upper_dual_lr"])
        and float(summary.get("lower_dual_lr", -1.0))
        == float(arm_spec["lower_dual_lr"])
        and len(rows) == spec.EXPECTED_EVALUATION_ROWS_PER_CELL
        and _path_registry(rows) == expected_paths
        and all(float(row["protocol_valid"]) == 1.0 for row in rows)
        and all(row["environment"] == environment for row in rows)
        and all(row["LowerActionRouterMode"] == arm_spec["lower_action_router_mode"] for row in rows)
    )
    if not valid:
        raise ValueError(
            f"invalid or incomplete v16 cell: {(environment, arm, optimizer_seed)}"
        )


def analyze_cells(
    cells: Iterable[
        tuple[str, str, int, dict[str, Any], list[dict[str, Any]]]
    ],
) -> dict[str, Any]:
    registry: dict[
        tuple[str, str, int], tuple[dict[str, Any], list[dict[str, Any]]]
    ] = {}
    for environment, arm, seed, summary, rows in cells:
        key = (str(environment), str(arm), int(seed))
        if key in registry:
            raise ValueError(f"duplicate v16 cell: {key}")
        _validate_cell(*key, summary, rows)
        registry[key] = (summary, rows)

    expected = {
        (environment, arm, int(seed))
        for environment in spec.ENVIRONMENTS
        for arm in spec.ARMS
        for seed in spec.OPTIMIZER_SEEDS
    }
    if set(registry) != expected:
        raise ValueError("v16 cell registry is incomplete")

    cell_results = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            arm_rows = {
                arm: registry[(environment, arm, int(seed))][1]
                for arm in spec.ARMS
            }
            paths = {frozenset(_path_registry(rows)) for rows in arm_rows.values()}
            if len(paths) != 1:
                raise ValueError("v16 paired evaluation paths are not aligned")
            summaries = {
                arm: {
                    "reward": _mean(rows, "episode_return"),
                    "canonical_merit": _frequency_merit(rows, latent=False),
                    "latent_merit": _frequency_merit(rows, latent=True),
                    "max_responsibility_reconstruction_rms": max(
                        float(row["ResponsibilityReconstructionRMS"])
                        for row in rows
                    ),
                    "max_router_reconstruction_rms": max(
                        float(row["LowerRouterActionReconstructionRMS"])
                        for row in rows
                    ),
                }
                for arm, rows in arm_rows.items()
            }
            candidate = summaries[spec.GAUGE_PD_CANDIDATE]
            joint = summaries[spec.JOINT_PD_CONTROL]
            gauge_reward = summaries[spec.GAUGE_REWARD_CONTROL]
            reward_floor_joint = (
                joint["reward"]
                - spec.REWARD_NONINFERIORITY_FRACTION
                * max(abs(joint["reward"]), 1.0)
            )
            reward_floor_gauge = (
                gauge_reward["reward"]
                - spec.REWARD_NONINFERIORITY_FRACTION
                * max(abs(gauge_reward["reward"]), 1.0)
            )
            canonical_reduction = _relative_reduction(
                joint["canonical_merit"], candidate["canonical_merit"]
            )
            latent_reduction = _relative_reduction(
                gauge_reward["latent_merit"], candidate["latent_merit"]
            )
            latent_joint_limit = joint["latent_merit"] * (
                1.0 + spec.LATENT_NONINFERIORITY_FRACTION
            )
            reconstruction_exact = bool(
                candidate["max_responsibility_reconstruction_rms"]
                <= spec.RECONSTRUCTION_RMS_TOLERANCE
                and candidate["max_router_reconstruction_rms"]
                <= spec.RECONSTRUCTION_RMS_TOLERANCE
            )
            cell_results.append({
                "environment": environment,
                "optimizer_seed": int(seed),
                "arm_summaries": summaries,
                "reward_noninferior": bool(
                    candidate["reward"] >= reward_floor_joint
                    and candidate["reward"] >= reward_floor_gauge
                ),
                "canonical_relative_reduction_vs_joint": canonical_reduction,
                "canonical_reduction_supported": bool(
                    canonical_reduction >= spec.CANONICAL_MIN_RELATIVE_REDUCTION
                ),
                "latent_relative_reduction_vs_gauge_reward": latent_reduction,
                "latent_constraint_improvement": bool(
                    latent_reduction >= spec.LATENT_MIN_RELATIVE_REDUCTION
                ),
                "latent_noninferior_vs_joint": bool(
                    candidate["latent_merit"] <= latent_joint_limit
                ),
                "reconstruction_exact": reconstruction_exact,
            })

    environment_results = []
    for environment in spec.ENVIRONMENTS:
        rows = [row for row in cell_results if row["environment"] == environment]
        improvements = [
            float(row["latent_relative_reduction_vs_gauge_reward"])
            for row in rows
        ]
        environment_results.append({
            "environment": environment,
            "cell_count": len(rows),
            "latent_improvement_count": sum(
                bool(row["latent_constraint_improvement"]) for row in rows
            ),
            "median_latent_relative_reduction": statistics.median(improvements),
            "latent_environment_gate": bool(
                sum(bool(row["latent_constraint_improvement"]) for row in rows)
                >= 2
            ),
        })

    latent_improvement_count = sum(
        bool(row["latent_constraint_improvement"]) for row in cell_results
    )
    universal_gate = bool(
        all(row["reward_noninferior"] for row in cell_results)
        and all(row["canonical_reduction_supported"] for row in cell_results)
        and all(row["latent_noninferior_vs_joint"] for row in cell_results)
        and all(row["reconstruction_exact"] for row in cell_results)
        and latent_improvement_count >= spec.MINIMUM_LATENT_IMPROVEMENT_CELLS
        and all(row["latent_environment_gate"] for row in environment_results)
    )
    return {
        "analysis_version": "mujoco_v16_gauge_training_preflight_analysis_v1",
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(cell_results),
        "latent_improvement_count": latent_improvement_count,
        "status": spec.SUPPORTED_STATUS if universal_gate else spec.NOT_SUPPORTED_STATUS,
        "support_gate": universal_gate,
        "environment_results": environment_results,
        "cells": cell_results,
    }


def analyze_run(run_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    cells = []
    for environment in spec.ENVIRONMENTS:
        for arm in spec.ARMS:
            for seed in spec.OPTIMIZER_SEEDS:
                directory = ROOT / cell_relative_dir(run_name, environment, arm, seed)
                summary_path = directory / "cell_summary.json"
                rows_path = directory / "evaluation_rows.csv"
                if not summary_path.is_file() or not rows_path.is_file():
                    raise FileNotFoundError(f"missing v16 result cell: {directory}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                with rows_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                cells.append((environment, arm, seed, summary, rows))
    result = analyze_cells(cells)
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "gauge_training_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (target / "gauge_training_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "environment",
            "optimizer_seed",
            "reward_noninferior",
            "canonical_relative_reduction_vs_joint",
            "canonical_reduction_supported",
            "latent_relative_reduction_vs_gauge_reward",
            "latent_constraint_improvement",
            "latent_noninferior_vs_joint",
            "reconstruction_exact",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["cells"]:
            writer.writerow({key: row[key] for key in fieldnames})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_name, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "cell_count": result["cell_count"],
        "latent_improvement_count": result["latent_improvement_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
