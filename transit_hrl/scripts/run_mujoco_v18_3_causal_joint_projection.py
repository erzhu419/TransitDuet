#!/usr/bin/env python3
"""Evaluate the frozen v18.3 causal joint projection screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.core import CausalJointFrequencyProjector  # noqa: E402
from freq_hrl.experiments.mujoco import (  # noqa: E402
    full_horizon_responsibility_oracle as oracle,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import (  # noqa: E402
    mujoco_v18_3_causal_joint_projection_spec as spec,
)
from scripts.train_mujoco_v17_8_causal_fir import (  # noqa: E402
    load_reused_panel,
)


def evaluate_path(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    source, candidate_id = payload
    config = dict(spec.CANDIDATES[str(candidate_id)])
    total = np.asarray(source["total_action"], dtype=np.float64)
    proposed_upper = np.asarray(
        source["baseline_upper_action"], dtype=np.float64
    )
    proposed_lower = total - proposed_upper
    projector = CausalJointFrequencyProjector(
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
        upper_rms_budget=spec.UPPER_RMS_BUDGET,
        lower_rms_budget=spec.LOWER_RMS_BUDGET,
        upper_action_limit=spec.UPPER_ACTION_LIMIT,
        lower_action_limit=spec.LOWER_ACTION_LIMIT,
        budget_mode=str(config["budget_mode"]),
    )
    projector.reset(total.shape[1])
    projected = [
        projector.project(upper_row, lower_row)
        for upper_row, lower_row in zip(
            proposed_upper, proposed_lower, strict=True
        )
    ]
    upper = np.stack([row["upper"] for row in projected])
    lower = np.stack([row["lower"] for row in projected])
    corrected_total = upper + lower
    correction = corrected_total - total
    executed_correction = (
        np.clip(
            corrected_total,
            -spec.EXECUTED_ACTION_LIMIT,
            spec.EXECUTED_ACTION_LIMIT,
        )
        - np.clip(
            total,
            -spec.EXECUTED_ACTION_LIMIT,
            spec.EXECUTED_ACTION_LIMIT,
        )
    )
    upper_power, lower_power = oracle.responsibility_frequency_powers(
        corrected_total,
        upper,
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
    )
    exact = oracle.solve_full_horizon_responsibility_oracle(
        corrected_total,
        upper_rms_budget=spec.UPPER_RMS_BUDGET,
        lower_rms_budget=spec.LOWER_RMS_BUDGET,
        upper_action_limit=spec.UPPER_ACTION_LIMIT,
        lower_action_limit=spec.LOWER_ACTION_LIMIT,
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
        power_tolerance=spec.POWER_TOLERANCE,
    ).summary()
    bound_violation = float(max(
        np.max(np.maximum(np.abs(upper) - spec.UPPER_ACTION_LIMIT, 0.0)),
        np.max(np.maximum(np.abs(lower) - spec.LOWER_ACTION_LIMIT, 0.0)),
    ))
    reconstruction = float(np.max(np.abs(upper + lower - corrected_total)))
    direct_joint = bool(
        upper_power
        <= spec.UPPER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
        and lower_power
        <= spec.LOWER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
    )
    component_feasible = all(
        bool(row["component_feasible"]) for row in projected
    )
    finite = bool(
        np.all(np.isfinite(upper))
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(correction))
    )
    valid = bool(
        finite
        and component_feasible
        and bound_violation <= spec.BOUND_TOLERANCE
        and reconstruction <= spec.RECONSTRUCTION_TOLERANCE
    )
    actor_floor = not bool(source["oracle_joint_feasible"])
    return {
        "candidate_id": str(candidate_id),
        "environment": str(source["environment"]),
        "disturbance_mode": str(source["disturbance_mode"]),
        "evaluation_seed": int(source["evaluation_seed"]),
        "trajectory_length": int(total.shape[0]),
        "action_dimension": int(total.shape[1]),
        "reference_feasible": bool(source["oracle_joint_feasible"]),
        "actor_floor": bool(actor_floor),
        "valid": valid,
        "component_feasible": component_feasible,
        "direct_joint_feasible": direct_joint,
        "exact_oracle_joint_feasible": bool(exact["joint_feasible"]),
        "upper_power": float(upper_power),
        "lower_power": float(lower_power),
        "exact_oracle_upper_power": float(exact["upper_power"]),
        "exact_oracle_lower_power": float(exact["lower_power"]),
        "correction_rms": _rms(correction),
        "correction_abs_max": float(np.max(np.abs(correction))),
        "executed_correction_rms": _rms(executed_correction),
        "component_correction_rms": _rms(np.concatenate((
            upper - proposed_upper,
            lower - proposed_lower,
        ), axis=1)),
        "total_changed_step_count": sum(
            bool(row["total_action_changed"]) for row in projected
        ),
        "fixed_total_feasible_step_count": sum(
            bool(row["fixed_total_feasible"]) for row in projected
        ),
        "projection_nonconverged_step_count": sum(
            not bool(row["projection_converged"]) for row in projected
        ),
        "bound_violation_max": bound_violation,
        "reconstruction_error_max": reconstruction,
    }


def summarize_candidate(
    candidate_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = [row for row in rows if row["reference_feasible"]]
    floor = [row for row in rows if row["actor_floor"]]
    floor_by_seed = {
        str(seed): {
            "path_count": sum(
                int(row["evaluation_seed"]) == int(seed) for row in floor
            ),
            "recovered_path_count": sum(
                int(row["evaluation_seed"]) == int(seed)
                and bool(row["direct_joint_feasible"])
                and bool(row["exact_oracle_joint_feasible"])
                for row in floor
            ),
        }
        for seed in spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
    }
    return {
        "candidate_id": str(candidate_id),
        "config": dict(spec.CANDIDATES[str(candidate_id)]),
        "path_count": len(rows),
        "valid_path_count": sum(bool(row["valid"]) for row in rows),
        "direct_joint_feasible_path_count": sum(
            bool(row["direct_joint_feasible"]) for row in rows
        ),
        "exact_oracle_joint_feasible_path_count": sum(
            bool(row["exact_oracle_joint_feasible"]) for row in rows
        ),
        "reference_feasible_path_count": len(reference),
        "reference_feasible_preserved_path_count": sum(
            bool(row["direct_joint_feasible"])
            and bool(row["exact_oracle_joint_feasible"])
            for row in reference
        ),
        "actor_floor_path_count": len(floor),
        "actor_floor_recovered_path_count": sum(
            bool(row["direct_joint_feasible"])
            and bool(row["exact_oracle_joint_feasible"])
            for row in floor
        ),
        "actor_floor_by_seed": floor_by_seed,
        "actor_floor_executed_nonzero_path_count": sum(
            float(row["executed_correction_rms"])
            >= spec.EXECUTED_CORRECTION_RMS_MIN_GATE
            for row in floor
        ),
        "correction_abs_maximum": max(
            float(row["correction_abs_max"]) for row in rows
        ),
        "reference_feasible_correction_rms_mean": _mean([
            float(row["correction_rms"]) for row in reference
        ]),
        "reference_feasible_correction_rms_maximum": max(
            float(row["correction_rms"]) for row in reference
        ),
        "actor_floor_correction_rms_mean": _mean([
            float(row["correction_rms"]) for row in floor
        ]),
        "actor_floor_correction_rms_maximum": max(
            float(row["correction_rms"]) for row in floor
        ),
        "total_changed_step_count": sum(
            int(row["total_changed_step_count"]) for row in rows
        ),
        "projection_nonconverged_step_count": sum(
            int(row["projection_nonconverged_step_count"])
            for row in rows
        ),
    }


def selection_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(summary["valid_path_count"]),
        -int(summary["direct_joint_feasible_path_count"]),
        -int(summary["exact_oracle_joint_feasible_path_count"]),
        -int(summary["actor_floor_recovered_path_count"]),
        -int(summary["reference_feasible_preserved_path_count"]),
        float(summary["reference_feasible_correction_rms_maximum"]),
        float(summary["actor_floor_correction_rms_maximum"]),
        str(summary["candidate_id"]),
    )


def advancement_gate(summary: dict[str, Any]) -> dict[str, bool]:
    floor_by_seed = dict(summary.get("actor_floor_by_seed") or {})
    return {
        "all_paths_valid": bool(
            int(summary["valid_path_count"]) == spec.EXPECTED_PATH_COUNT
        ),
        "all_paths_directly_joint_feasible": bool(
            int(summary["direct_joint_feasible_path_count"])
            == spec.EXPECTED_PATH_COUNT
        ),
        "all_paths_exact_oracle_feasible": bool(
            int(summary["exact_oracle_joint_feasible_path_count"])
            == spec.EXPECTED_PATH_COUNT
        ),
        "all_reference_feasible_paths_preserved": bool(
            int(summary["reference_feasible_preserved_path_count"])
            == spec.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
        ),
        "all_actor_floor_paths_recovered": bool(
            int(summary["actor_floor_recovered_path_count"])
            == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
        ),
        "all_actor_floor_seed_groups_recovered": bool(all(
            int(floor_by_seed[str(seed)]["recovered_path_count"])
            == int(expected)
            for seed, expected
            in spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED.items()
        )),
        "all_actor_floor_paths_change_executed_action": bool(
            int(summary["actor_floor_executed_nonzero_path_count"])
            == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
        ),
        "global_correction_abs_gate": bool(
            float(summary["correction_abs_maximum"])
            <= spec.CORRECTION_ABS_MAX_GATE
        ),
        "reference_feasible_trust_region_gate": bool(
            float(summary["reference_feasible_correction_rms_maximum"])
            <= spec.REFERENCE_CORRECTION_RMS_MAX_GATE
        ),
        "actor_floor_trust_region_gate": bool(
            float(summary["actor_floor_correction_rms_maximum"])
            <= spec.ACTOR_FLOOR_CORRECTION_RMS_MAX_GATE
        ),
    }


def run(reference_root: Path, workers: int) -> dict[str, Any]:
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=spec.FROZEN_CORE_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        require_verified=True,
    )
    panel = load_reused_panel(Path(reference_root))
    if (
        len(panel) != spec.EXPECTED_PATH_COUNT
        or sum(bool(row["oracle_joint_feasible"]) for row in panel)
        != spec.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
    ):
        raise RuntimeError("v18.3 reused panel does not match its freeze")
    payloads = [
        (row, candidate_id)
        for candidate_id in spec.CANDIDATES
        for row in panel
    ]
    started = time.perf_counter()
    if int(workers) == 1:
        rows = [evaluate_path(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            rows = list(pool.map(evaluate_path, payloads, chunksize=1))
    runtime = time.perf_counter() - started
    rows_by_candidate = {
        candidate_id: [
            row for row in rows if row["candidate_id"] == candidate_id
        ]
        for candidate_id in spec.CANDIDATES
    }
    summaries = {
        candidate_id: summarize_candidate(candidate_id, candidate_rows)
        for candidate_id, candidate_rows in rows_by_candidate.items()
    }
    selected = min(summaries.values(), key=selection_key)
    gate = advancement_gate(selected)
    passes = bool(all(gate.values()))
    return {
        "status": (
            "causal_joint_projection_authorizes_fresh_closed_loop_freeze"
            if passes
            else "causal_joint_projection_stops_before_fresh_path_access"
        ),
        "integrity_status": "valid",
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": (
            spec.FROZEN_SOURCE_MANIFEST_SHA256
        ),
        "source_identity": source_identity,
        "reference_dataset_run": spec.REFERENCE_DATASET_RUN,
        "path_count": len(panel),
        "candidate_count": len(spec.CANDIDATES),
        "worker_count": int(workers),
        "runtime_seconds": float(runtime),
        "target_labels_accessed": False,
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_candidate": selected,
        "candidate_summaries": summaries,
        "selected_path_rows": rows_by_candidate[str(selected["candidate_id"])],
        "advancement_gate": gate,
        "fresh_validation_paths_accessed": False,
        "fresh_path_access_allowed": passes,
        "support_gate": passes,
        "selection_contract": spec.SELECTION_CONTRACT,
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if int(args.workers) < 1:
        raise SystemExit("v18.3 workers must be positive")
    summary = run(args.reference_root, int(args.workers))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE v18.3 joint projection status={summary['status']} "
        f"selected={summary['selected_candidate_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
