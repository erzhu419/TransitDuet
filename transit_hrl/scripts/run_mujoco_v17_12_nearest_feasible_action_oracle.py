#!/usr/bin/env python3
"""Construct nearest feasible component targets on the reused MuJoCo panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.mujoco.nearest_feasible_component_projection import (  # noqa: E402
    project_nearest_feasible_components,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import (  # noqa: E402
    mujoco_v17_12_nearest_feasible_action_oracle_spec as spec,
)
from scripts.train_mujoco_v17_8_causal_fir import (  # noqa: E402
    load_reused_panel,
)


def _projection(
    upper: np.ndarray,
    lower: np.ndarray,
    *,
    deployment_aligned: bool,
):
    return project_nearest_feasible_components(
        upper,
        lower,
        upper_rms_budget=spec.UPPER_RMS_BUDGET,
        lower_rms_budget=spec.LOWER_RMS_BUDGET,
        upper_action_limit=spec.UPPER_ACTION_LIMIT,
        lower_action_limit=spec.LOWER_ACTION_LIMIT,
        total_action_limit=spec.TOTAL_ACTION_LIMIT,
        include_total_action_box=bool(deployment_aligned),
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
        convergence_tolerance=spec.CONVERGENCE_TOLERANCE,
        feasibility_tolerance=spec.FEASIBILITY_TOLERANCE,
        max_iterations=spec.MAX_PROJECTION_ITERATIONS,
    )


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("v17.12 distribution requires finite values")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _path_target(
    target_root: Path,
    row: dict[str, Any],
    frequency_result: Any,
) -> Path:
    path = (
        Path(target_root)
        / str(row["environment"])
        / str(row["disturbance_mode"])
        / f"seed_{int(row['evaluation_seed'])}"
        / "nearest_feasible_target.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    reference_total = np.asarray(row["total_action"], dtype=np.float64)
    target_upper = np.asarray(frequency_result.upper, dtype=np.float64)
    target_lower = np.asarray(frequency_result.lower, dtype=np.float64)
    target_total = target_upper + target_lower
    np.savez_compressed(
        path,
        reference_total_action=reference_total,
        target_upper_action=target_upper,
        target_lower_action=target_lower,
        target_total_action=target_total,
        total_action_correction=target_total - reference_total,
    )
    return path


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != spec.EXPECTED_PATH_COUNT:
        raise ValueError("v17.12 path count does not match frozen design")
    actor_floor = [row for row in rows if not row["reference_joint_feasible"]]
    preserved = [row for row in rows if row["reference_joint_feasible"]]
    floor_total_rms = [
        float(row["frequency_only"]["total_action_correction_rms"])
        for row in actor_floor
    ]
    floor_total_abs = [
        float(row["frequency_total_correction_abs_max"])
        for row in actor_floor
    ]
    gate = {
        "expected_reference_feasible_path_count": bool(
            len(preserved) == spec.EXPECTED_ORACLE_FEASIBLE_PATH_COUNT
        ),
        "expected_actor_floor_path_count": bool(
            len(actor_floor) == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
        ),
        "all_frequency_only_targets_feasible": bool(all(
            row["frequency_only"]["feasible"] for row in rows
        )),
        "all_reference_feasible_paths_preserved_exactly": bool(all(
            float(row["frequency_only"]["component_correction_rms"])
            <= spec.PRESERVED_PATH_CORRECTION_TOLERANCE
            for row in preserved
        )),
        "all_actor_floor_targets_change_total_action": bool(all(
            value > spec.PRESERVED_PATH_CORRECTION_TOLERANCE
            for value in floor_total_rms
        )),
        "actor_floor_total_correction_rms_gate": bool(
            floor_total_rms
            and max(floor_total_rms)
            <= spec.MAX_ACTOR_FLOOR_TOTAL_CORRECTION_RMS
        ),
        "actor_floor_total_correction_abs_gate": bool(
            floor_total_abs
            and max(floor_total_abs)
            <= spec.MAX_ACTOR_FLOOR_TOTAL_CORRECTION_ABS
        ),
        "all_actor_floor_deployment_targets_feasible": bool(all(
            row["deployment_aligned"]["feasible"] for row in actor_floor
        )),
        "server_target_count_complete": bool(sum(
            bool(row["target_written"]) for row in rows
        ) == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT),
    }
    advances = bool(all(gate.values()))
    by_environment = {}
    for environment in spec.ENVIRONMENTS:
        selected = [row for row in rows if row["environment"] == environment]
        selected_floor = [
            row for row in selected if not row["reference_joint_feasible"]
        ]
        by_environment[environment] = {
            "path_count": len(selected),
            "reference_feasible_path_count": sum(
                bool(row["reference_joint_feasible"]) for row in selected
            ),
            "actor_floor_path_count": len(selected_floor),
            "frequency_target_feasible_path_count": sum(
                bool(row["frequency_only"]["feasible"]) for row in selected
            ),
            "actor_floor_total_correction_rms": (
                _distribution([
                    float(row["frequency_only"]
                          ["total_action_correction_rms"])
                    for row in selected_floor
                ]) if selected_floor else None
            ),
        }
    return {
        "status": (
            "nearest_feasible_targets_authorize_causal_actor_adapter"
            if advances
            else "nearest_feasible_targets_stop_actor_adapter"
        ),
        "path_count": len(rows),
        "reference_feasible_path_count": len(preserved),
        "actor_floor_path_count": len(actor_floor),
        "frequency_target_feasible_path_count": sum(
            bool(row["frequency_only"]["feasible"]) for row in rows
        ),
        "deployment_target_feasible_actor_floor_path_count": sum(
            bool(row["deployment_aligned"]["feasible"])
            for row in actor_floor
        ),
        "actor_floor_total_correction_rms": _distribution(floor_total_rms),
        "actor_floor_total_correction_abs_max": _distribution(
            floor_total_abs
        ),
        "actor_floor_component_correction_rms": _distribution([
            float(row["frequency_only"]["component_correction_rms"])
            for row in actor_floor
        ]),
        "actor_floor_deployment_total_correction_rms": _distribution([
            float(row["deployment_aligned"]
                  ["total_action_correction_rms"])
            for row in actor_floor
        ]),
        "by_environment": by_environment,
        "advancement_gate": gate,
        "causal_actor_adapter_authorized": advances,
    }


def run(dataset_root: Path, target_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    panel = load_reused_panel(Path(dataset_root))
    rows: list[dict[str, Any]] = []
    for row in panel:
        reference_total = np.asarray(row["total_action"], dtype=np.float64)
        reference_upper = np.asarray(
            row["oracle_upper_action"], dtype=np.float64
        )
        reference_lower = reference_total - reference_upper
        frequency = _projection(
            reference_upper, reference_lower, deployment_aligned=False
        )
        actor_floor = not bool(row["oracle_joint_feasible"])
        deployment = (
            _projection(
                reference_upper,
                reference_lower,
                deployment_aligned=True,
            )
            if actor_floor else None
        )
        target_path = (
            _path_target(target_root, row, frequency)
            if actor_floor else None
        )
        frequency_total_delta = (
            np.asarray(frequency.upper + frequency.lower, dtype=np.float64)
            - reference_total
        )
        nominal_total_excess = np.maximum(
            np.abs(reference_total) - spec.TOTAL_ACTION_LIMIT, 0.0
        )
        rows.append({
            "environment": str(row["environment"]),
            "disturbance_mode": str(row["disturbance_mode"]),
            "evaluation_seed": int(row["evaluation_seed"]),
            "trajectory_length": int(reference_total.shape[0]),
            "action_dimension": int(reference_total.shape[1]),
            "reference_joint_feasible": bool(row["oracle_joint_feasible"]),
            "reference_upper_power": float(row["oracle_upper_power"]),
            "reference_lower_power": float(row["oracle_lower_power"]),
            "reference_nominal_total_box_excess_max": float(np.max(
                nominal_total_excess
            )),
            "reference_nominal_total_box_excess_rate": float(np.mean(
                nominal_total_excess > 0.0
            )),
            "frequency_only": frequency.summary(),
            "frequency_total_correction_abs_max": float(np.max(np.abs(
                frequency_total_delta
            ))),
            "deployment_aligned": (
                deployment.summary() if deployment is not None else None
            ),
            "target_written": target_path is not None,
            "target_relative_path": (
                str(target_path.relative_to(target_root))
                if target_path is not None else None
            ),
        })
    summary = summarize_rows(rows)
    summary.update({
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": (
            spec.FROZEN_SOURCE_MANIFEST_SHA256
        ),
        "source_dataset_run": spec.SOURCE_DATASET_RUN,
        "dataset_root": str(dataset_root),
        "server_target_root": str(target_root),
        "selection_contract": spec.SELECTION_CONTRACT,
        "rows": rows,
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=spec.FROZEN_CORE_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        require_verified=True,
    )
    summary = run(args.dataset_root, args.target_root)
    summary["source_identity"] = source_identity
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "nearest_feasible_action_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "server_target_location.json").write_text(
        json.dumps({
            "artifact_policy": "actor_floor_targets_server_only_v1",
            "server_target_root": str(args.target_root),
            "target_file_count": int(summary["actor_floor_path_count"]),
            "exported_files": [
                "nearest_feasible_action_summary.json",
                "server_target_location.json",
            ],
            "server_only_glob": "**/nearest_feasible_target.npz",
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE v17.12 status={summary['status']} "
        f"actor_floor={summary['actor_floor_path_count']} "
        f"target_feasible={summary['frequency_target_feasible_path_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
