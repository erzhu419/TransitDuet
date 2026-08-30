#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v16.1 paired audit-gauge preflight."""

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

from freq_hrl.domains.mujoco import lower_action_router_contract  # noqa: E402
from scripts import mujoco_v16_1_audit_gauge_paired_preflight_spec as spec  # noqa: E402
from scripts.submit_mujoco_v16_1_audit_gauge_paired_preflight_scheduleurm import (  # noqa: E402
    cell_relative_dir,
)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _path_registry(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(row["disturbance_mode"]), int(row["seed"]))
        for row in rows
    }


def _frequency_merit(rows: list[dict[str, Any]], *, latent: bool) -> float:
    lower = "LatentLowerLFDriftAbs" if latent else "LowerLFDriftAbs"
    upper = "LatentUpperHFPowerAbs" if latent else "UpperHFPowerAbs"
    return float(
        _mean(rows, lower) / (spec.LOWER_LF_RMS_BUDGET ** 2)
        + _mean(rows, upper) / (spec.UPPER_HF_RMS_BUDGET ** 2)
    )


def _relative_reduction(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(abs(baseline), 1e-12))


def _selection_constraints_feasible(summary: dict[str, Any]) -> bool:
    diagnostics = summary.get("selected_checkpoint_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    constraints = diagnostics.get("constraints")
    expected = len(spec.TRAINING_DISTURBANCE_MODES) * 6
    return bool(
        isinstance(constraints, list)
        and len(constraints) == expected
        and all(
            float(item.get("normalized_violation", float("inf"))) <= 1e-10
            for item in constraints
        )
    )


def _validate_cell(
    phase: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    is_anchor = str(phase) == "anchor"
    arm_spec = spec.ANCHOR_SPEC if is_anchor else spec.ARMS[str(arm)]
    expected_paths = {
        (mode, seed)
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.EVALUATION_SEEDS
    }
    continuation = summary.get("paired_checkpoint_continuation") or {}
    continuation_valid = bool(
        not is_anchor
        and continuation.get("enabled") is True
        and continuation.get("checkpoint_environment") == environment
        and int(continuation.get("checkpoint_optimizer_seed", -1))
        == int(optimizer_seed)
        and continuation.get("checkpoint_router_mode")
        == spec.ANCHOR_SPEC["lower_action_router_mode"]
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
        and float(summary.get("lower_action_router_alpha", -1.0))
        == float(arm_spec["lower_action_router_alpha"])
        and float(summary.get("lower_action_router_strength", -1.0))
        == float(arm_spec["lower_action_router_strength"])
        and summary.get("lower_action_router_contract")
        == lower_action_router_contract(str(arm_spec["lower_action_router_mode"]))
        and summary.get("leakage_constraint_scope")
        == arm_spec["leakage_constraint_scope"]
        and summary.get("leakage_constraint_cost_mode")
        == arm_spec["leakage_cost_mode"]
        and summary.get("upper_constraint_mode") == "primal_dual"
        and float(summary.get("upper_dual_lr", -1.0))
        == float(arm_spec["upper_dual_lr"])
        and float(summary.get("lower_dual_lr", -1.0))
        == float(arm_spec["lower_dual_lr"])
        and summary.get("checkpoint_score_mode")
        == arm_spec["checkpoint_score_mode"]
        and len(rows) == spec.EXPECTED_EVALUATION_ROWS_PER_CELL
        and _path_registry(rows) == expected_paths
        and all(float(row["protocol_valid"]) == 1.0 for row in rows)
        and all(row["environment"] == environment for row in rows)
        and all(
            row["LowerActionRouterMode"]
            == arm_spec["lower_action_router_mode"]
            for row in rows
        )
        and (is_anchor or continuation_valid)
    )
    if not valid:
        raise ValueError(
            "invalid or incomplete v16.1 cell: "
            f"{(phase, environment, arm, optimizer_seed)}"
        )


def analyze_cells(
    cells: Iterable[
        tuple[
            str,
            str,
            str,
            int,
            dict[str, Any],
            list[dict[str, Any]],
        ]
    ],
) -> dict[str, Any]:
    registry: dict[
        tuple[str, str, str, int],
        tuple[dict[str, Any], list[dict[str, Any]]],
    ] = {}
    for phase, environment, arm, seed, summary, rows in cells:
        key = (str(phase), str(environment), str(arm), int(seed))
        if key in registry:
            raise ValueError(f"duplicate v16.1 cell: {key}")
        _validate_cell(*key, summary, rows)
        registry[key] = (summary, rows)

    expected = {
        ("anchor", environment, spec.ANCHOR_ARM, int(seed))
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    } | {
        ("continuation", environment, arm, int(seed))
        for environment in spec.ENVIRONMENTS
        for arm in spec.ARMS
        for seed in spec.OPTIMIZER_SEEDS
    }
    if set(registry) != expected:
        raise ValueError("v16.1 cell registry is incomplete")

    cell_results: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            anchor_summary, anchor_rows = registry[
                ("anchor", environment, spec.ANCHOR_ARM, int(seed))
            ]
            arm_entries = {
                arm: registry[("continuation", environment, arm, int(seed))]
                for arm in spec.ARMS
            }
            path_sets = {
                frozenset(_path_registry(rows))
                for _, rows in ([(spec.ANCHOR_ARM, anchor_rows)] + [
                    (arm, entry[1]) for arm, entry in arm_entries.items()
                ])
            }
            if len(path_sets) != 1:
                raise ValueError("v16.1 heldout evaluation paths are not paired")

            def summarize(
                summary: dict[str, Any], rows: list[dict[str, Any]]
            ) -> dict[str, Any]:
                return {
                    "reward": _mean(rows, "episode_return"),
                    "canonical_merit": _frequency_merit(rows, latent=False),
                    "latent_merit": _frequency_merit(rows, latent=True),
                    "audit_alpha_mean": _mean(
                        rows, "LowerRouterAuditAlphaMean"
                    ),
                    "audit_alpha_final_mean": _mean(
                        rows, "LowerRouterAuditAlphaFinal"
                    ),
                    "max_responsibility_reconstruction_rms": max(
                        float(row["ResponsibilityReconstructionRMS"])
                        for row in rows
                    ),
                    "max_router_reconstruction_rms": max(
                        float(row["LowerRouterActionReconstructionRMS"])
                        for row in rows
                    ),
                    "selected_checkpoint_iteration": int(
                        summary["selected_checkpoint_iteration"]
                    ),
                }

            summaries = {
                spec.ANCHOR_ARM: summarize(anchor_summary, anchor_rows),
                **{
                    arm: summarize(summary, rows)
                    for arm, (summary, rows) in arm_entries.items()
                },
            }
            candidate_summary, _ = arm_entries[spec.PRIMAL_DUAL_CANDIDATE]
            candidate = summaries[spec.PRIMAL_DUAL_CANDIDATE]
            control = summaries[spec.REWARD_CONTINUATION_CONTROL]
            anchor = summaries[spec.ANCHOR_ARM]
            reward_floors = [
                baseline["reward"]
                - spec.REWARD_NONINFERIORITY_FRACTION
                * max(abs(baseline["reward"]), 1.0)
                for baseline in (anchor, control)
            ]
            canonical_reduction = _relative_reduction(
                control["canonical_merit"], candidate["canonical_merit"]
            )
            latent_limit = control["latent_merit"] * (
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
                "candidate_selected_trained_checkpoint": bool(
                    candidate["selected_checkpoint_iteration"] >= 0
                ),
                "selection_constraints_feasible": (
                    _selection_constraints_feasible(candidate_summary)
                ),
                "reward_noninferior": bool(
                    candidate["reward"] >= max(reward_floors)
                ),
                "canonical_relative_reduction_vs_control": canonical_reduction,
                "canonical_reduction_supported": bool(
                    canonical_reduction
                    >= spec.CANONICAL_MIN_RELATIVE_REDUCTION
                ),
                "latent_noninferior_vs_control": bool(
                    candidate["latent_merit"] <= latent_limit
                ),
                "reconstruction_exact": reconstruction_exact,
                "adaptive_cutoff_active": bool(
                    0.0 < candidate["audit_alpha_mean"] < 1.0
                    and abs(
                        candidate["audit_alpha_final_mean"]
                        - float(spec.ANCHOR_SPEC["lower_action_router_alpha"])
                    ) > 1e-5
                ),
            })

    environment_results: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        rows = [
            row for row in cell_results if row["environment"] == environment
        ]
        count = sum(bool(row["canonical_reduction_supported"]) for row in rows)
        environment_results.append({
            "environment": environment,
            "cell_count": len(rows),
            "canonical_improvement_count": count,
            "median_canonical_relative_reduction": statistics.median(
                float(row["canonical_relative_reduction_vs_control"])
                for row in rows
            ),
            "environment_gate": bool(
                count >= spec.MINIMUM_ENVIRONMENT_IMPROVEMENT_CELLS
            ),
        })

    improvement_count = sum(
        bool(row["canonical_reduction_supported"]) for row in cell_results
    )
    support_gate = bool(
        all(row["candidate_selected_trained_checkpoint"] for row in cell_results)
        and all(row["selection_constraints_feasible"] for row in cell_results)
        and all(row["reward_noninferior"] for row in cell_results)
        and all(row["latent_noninferior_vs_control"] for row in cell_results)
        and all(row["reconstruction_exact"] for row in cell_results)
        and all(row["adaptive_cutoff_active"] for row in cell_results)
        and improvement_count >= spec.MINIMUM_CANONICAL_IMPROVEMENT_CELLS
        and all(row["environment_gate"] for row in environment_results)
    )
    return {
        "analysis_version": (
            "mujoco_v16_1_audit_gauge_paired_preflight_analysis_v1"
        ),
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(cell_results),
        "canonical_improvement_count": improvement_count,
        "status": spec.SUPPORTED_STATUS if support_gate else spec.NOT_SUPPORTED_STATUS,
        "support_gate": support_gate,
        "environment_results": environment_results,
        "cells": cell_results,
    }


def analyze_run(run_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    cells = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            entries = [("anchor", spec.ANCHOR_ARM), *(
                ("continuation", arm) for arm in spec.ARMS
            )]
            for phase, arm in entries:
                directory = ROOT / cell_relative_dir(
                    run_name, phase, environment, arm, seed
                )
                summary_path = directory / "cell_summary.json"
                rows_path = directory / "evaluation_rows.csv"
                if not summary_path.is_file() or not rows_path.is_file():
                    raise FileNotFoundError(
                        f"missing v16.1 result cell: {directory}"
                    )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                with rows_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                cells.append((phase, environment, arm, seed, summary, rows))
    result = analyze_cells(cells)
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "audit_gauge_paired_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (target / "audit_gauge_paired_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "environment",
            "optimizer_seed",
            "candidate_selected_trained_checkpoint",
            "selection_constraints_feasible",
            "reward_noninferior",
            "canonical_relative_reduction_vs_control",
            "canonical_reduction_supported",
            "latent_noninferior_vs_control",
            "reconstruction_exact",
            "adaptive_cutoff_active",
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
        "canonical_improvement_count": result[
            "canonical_improvement_count"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
