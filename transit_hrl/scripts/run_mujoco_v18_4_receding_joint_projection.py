#!/usr/bin/env python3
"""Evaluate the frozen v18.4 causal receding-horizon projection screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.core import (  # noqa: E402
    CausalRecedingHorizonJointProjector,
)
from freq_hrl.experiments.mujoco import (  # noqa: E402
    full_horizon_responsibility_oracle as oracle,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import (  # noqa: E402
    mujoco_v18_4_receding_joint_projection_spec as spec,
)
from scripts.train_mujoco_v17_8_causal_fir import (  # noqa: E402
    load_reused_panel,
)


def project_path(
    source: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    config = dict(spec.CANDIDATES[str(candidate_id)])
    total = np.asarray(source["total_action"], dtype=np.float64)
    proposed_upper = np.asarray(
        source["baseline_upper_action"], dtype=np.float64
    )
    proposed_lower = total - proposed_upper
    projector = CausalRecedingHorizonJointProjector(
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
        upper_rms_budget=spec.UPPER_RMS_BUDGET,
        lower_rms_budget=spec.LOWER_RMS_BUDGET,
        upper_action_limit=spec.UPPER_ACTION_LIMIT,
        lower_action_limit=spec.LOWER_ACTION_LIMIT,
        planning_horizon=int(config["planning_horizon"]),
        forecast_mode=str(config["forecast_mode"]),
        velocity_alpha=spec.VELOCITY_ALPHA,
        velocity_decay=spec.VELOCITY_DECAY,
        projection_tolerance=spec.PROJECTION_TOLERANCE,
        feasibility_tolerance=spec.FEASIBILITY_TOLERANCE,
        maximum_projection_iterations=(
            spec.MAXIMUM_PROJECTION_ITERATIONS
        ),
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
    return {
        "total": total,
        "proposed_upper": proposed_upper,
        "proposed_lower": proposed_lower,
        "upper": upper,
        "lower": lower,
        "corrected_total": upper + lower,
        "projected": projected,
    }


def evaluate_direct_path(
    payload: tuple[dict[str, Any], str],
) -> dict[str, Any]:
    source, candidate_id = payload
    path = project_path(source, candidate_id)
    total = path["total"]
    proposed_upper = path["proposed_upper"]
    proposed_lower = path["proposed_lower"]
    upper = path["upper"]
    lower = path["lower"]
    corrected_total = path["corrected_total"]
    projected = path["projected"]
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
    forecast_feasible = all(
        bool(row["projected_forecast_feasible"]) for row in projected
    )
    finite = bool(
        np.all(np.isfinite(upper))
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(correction))
    )
    valid = bool(
        finite
        and forecast_feasible
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
        "forecast_feasible": forecast_feasible,
        "direct_joint_feasible": direct_joint,
        "upper_power": float(upper_power),
        "lower_power": float(lower_power),
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
        "fixed_total_forecast_feasible_step_count": sum(
            bool(row["fixed_total_forecast_feasible"])
            for row in projected
        ),
        "projection_nonconverged_step_count": sum(
            not bool(row["projection_converged"]) for row in projected
        ),
        "upper_prefix_power_maximum": max(
            float(row["upper_prefix_power"]) for row in projected
        ),
        "lower_prefix_power_maximum": max(
            float(row["lower_prefix_power"]) for row in projected
        ),
        "prefix_budget_violation_step_count": sum(
            float(row["upper_prefix_power"])
            > spec.UPPER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
            or float(row["lower_prefix_power"])
            > spec.LOWER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
            for row in projected
        ),
        "bound_violation_max": bound_violation,
        "reconstruction_error_max": reconstruction,
        "exact_oracle_audited": False,
    }


def evaluate_exact_path(
    payload: tuple[dict[str, Any], str],
) -> dict[str, Any]:
    source, candidate_id = payload
    path = project_path(source, candidate_id)
    exact = oracle.solve_full_horizon_responsibility_oracle(
        path["corrected_total"],
        upper_rms_budget=spec.UPPER_RMS_BUDGET,
        lower_rms_budget=spec.LOWER_RMS_BUDGET,
        upper_action_limit=spec.UPPER_ACTION_LIMIT,
        lower_action_limit=spec.LOWER_ACTION_LIMIT,
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
        power_tolerance=spec.POWER_TOLERANCE,
    ).summary()
    return {
        "candidate_id": str(candidate_id),
        "environment": str(source["environment"]),
        "disturbance_mode": str(source["disturbance_mode"]),
        "evaluation_seed": int(source["evaluation_seed"]),
        "exact_oracle_audited": True,
        "exact_oracle_joint_feasible": bool(exact["joint_feasible"]),
        "exact_oracle_upper_power": float(exact["upper_power"]),
        "exact_oracle_lower_power": float(exact["lower_power"]),
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
            "direct_recovered_path_count": sum(
                int(row["evaluation_seed"]) == int(seed)
                and bool(row["direct_joint_feasible"])
                for row in floor
            ),
            "exact_recovered_path_count": sum(
                int(row["evaluation_seed"]) == int(seed)
                and bool(row.get("exact_oracle_audited"))
                and bool(row.get("exact_oracle_joint_feasible"))
                for row in floor
            ),
        }
        for seed in spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
    }
    audited = [
        row for row in rows if bool(row.get("exact_oracle_audited"))
    ]
    return {
        "candidate_id": str(candidate_id),
        "config": dict(spec.CANDIDATES[str(candidate_id)]),
        "path_count": len(rows),
        "valid_path_count": sum(bool(row["valid"]) for row in rows),
        "direct_joint_feasible_path_count": sum(
            bool(row["direct_joint_feasible"]) for row in rows
        ),
        "direct_reference_feasible_preserved_path_count": sum(
            bool(row["direct_joint_feasible"]) for row in reference
        ),
        "direct_actor_floor_recovered_path_count": sum(
            bool(row["direct_joint_feasible"]) for row in floor
        ),
        "actor_floor_by_seed": floor_by_seed,
        "actor_floor_executed_nonzero_path_count": sum(
            float(row["executed_correction_rms"])
            >= spec.EXECUTED_CORRECTION_RMS_MIN_GATE
            for row in floor
        ),
        "exact_oracle_audited_path_count": len(audited),
        "exact_oracle_joint_feasible_path_count": sum(
            bool(row.get("exact_oracle_joint_feasible")) for row in audited
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
        "prefix_budget_violation_step_count": sum(
            int(row["prefix_budget_violation_step_count"])
            for row in rows
        ),
        "upper_prefix_power_maximum": max(
            float(row["upper_prefix_power_maximum"]) for row in rows
        ),
        "lower_prefix_power_maximum": max(
            float(row["lower_prefix_power_maximum"]) for row in rows
        ),
    }


def selection_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(summary["valid_path_count"]),
        -int(summary["direct_joint_feasible_path_count"]),
        -int(summary["direct_actor_floor_recovered_path_count"]),
        -int(summary["direct_reference_feasible_preserved_path_count"]),
        float(summary["reference_feasible_correction_rms_maximum"]),
        float(summary["actor_floor_correction_rms_maximum"]),
        float(summary["correction_abs_maximum"]),
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
        "selected_candidate_exactly_audited_on_all_paths": bool(
            int(summary["exact_oracle_audited_path_count"])
            == spec.EXPECTED_PATH_COUNT
        ),
        "all_paths_exact_oracle_feasible": bool(
            int(summary["exact_oracle_joint_feasible_path_count"])
            == spec.EXPECTED_PATH_COUNT
        ),
        "all_reference_feasible_paths_preserved": bool(
            int(summary["direct_reference_feasible_preserved_path_count"])
            == spec.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
        ),
        "all_actor_floor_paths_recovered": bool(
            int(summary["direct_actor_floor_recovered_path_count"])
            == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
        ),
        "all_actor_floor_seed_groups_recovered": bool(all(
            int(floor_by_seed[str(seed)]["direct_recovered_path_count"])
            == int(expected)
            and int(floor_by_seed[str(seed)]["exact_recovered_path_count"])
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
    _validate_panel(panel)
    direct_payloads = [
        (row, candidate_id)
        for candidate_id in spec.CANDIDATES
        for row in panel
    ]
    started = time.perf_counter()
    direct_started = time.perf_counter()
    direct_rows = _parallel_map(
        evaluate_direct_path,
        direct_payloads,
        workers=workers,
        progress_label="v18.4 direct",
    )
    direct_runtime = time.perf_counter() - direct_started
    direct_rows.sort(key=_row_key)
    rows_by_candidate = {
        candidate_id: [
            row
            for row in direct_rows
            if row["candidate_id"] == candidate_id
        ]
        for candidate_id in spec.CANDIDATES
    }
    summaries = {
        candidate_id: summarize_candidate(candidate_id, candidate_rows)
        for candidate_id, candidate_rows in rows_by_candidate.items()
    }
    selected = min(summaries.values(), key=selection_key)
    selected_id = str(selected["candidate_id"])

    exact_started = time.perf_counter()
    exact_rows = _parallel_map(
        evaluate_exact_path,
        [(row, selected_id) for row in panel],
        workers=workers,
        progress_label="v18.4 exact",
    )
    exact_runtime = time.perf_counter() - exact_started
    exact_by_path = {_path_key(row): row for row in exact_rows}
    selected_rows = []
    for row in rows_by_candidate[selected_id]:
        merged = dict(row)
        merged.update(exact_by_path[_path_key(row)])
        selected_rows.append(merged)
    selected_rows.sort(key=_row_key)
    rows_by_candidate[selected_id] = selected_rows
    summaries[selected_id] = summarize_candidate(
        selected_id, selected_rows
    )
    selected = summaries[selected_id]
    gate = advancement_gate(selected)
    passes = bool(all(gate.values()))
    runtime = time.perf_counter() - started
    return {
        "status": (
            "receding_joint_projection_authorizes_fresh_closed_loop_freeze"
            if passes
            else "receding_joint_projection_stops_before_fresh_path_access"
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
        "direct_audit_candidate_count": len(spec.CANDIDATES),
        "exact_oracle_audit_candidate_count": 1,
        "worker_count": int(workers),
        "runtime_seconds": float(runtime),
        "direct_audit_runtime_seconds": float(direct_runtime),
        "exact_oracle_runtime_seconds": float(exact_runtime),
        "actor_correction_targets_accessed": False,
        "reference_feasibility_labels_used_for_evaluation": True,
        "selected_candidate_id": selected_id,
        "selected_candidate": selected,
        "candidate_summaries": summaries,
        "selected_path_rows": selected_rows,
        "advancement_gate": gate,
        "fresh_validation_paths_accessed": False,
        "fresh_path_access_allowed": passes,
        "support_gate": passes,
        "selection_contract": spec.SELECTION_CONTRACT,
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }


def _parallel_map(
    function: Callable[[Any], dict[str, Any]],
    payloads: list[Any],
    *,
    workers: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    if int(workers) == 1:
        rows = []
        for index, payload in enumerate(payloads, start=1):
            rows.append(function(payload))
            if index % 20 == 0 or index == len(payloads):
                print(
                    f"PROGRESS {progress_label} {index}/{len(payloads)}",
                    flush=True,
                )
        return rows
    rows = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(function, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 20 == 0 or index == len(payloads):
                print(
                    f"PROGRESS {progress_label} {index}/{len(payloads)}",
                    flush=True,
                )
    return rows


def _validate_panel(panel: list[dict[str, Any]]) -> None:
    if len(panel) != spec.EXPECTED_PATH_COUNT:
        raise RuntimeError("v18.4 reused panel path count does not match freeze")
    if (
        sum(bool(row["oracle_joint_feasible"]) for row in panel)
        != spec.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
    ):
        raise RuntimeError(
            "v18.4 reused panel reference feasibility does not match freeze"
        )
    keys = {_path_key(row) for row in panel}
    if len(keys) != spec.EXPECTED_PATH_COUNT:
        raise RuntimeError("v18.4 reused panel contains duplicate path keys")
    expected_keys = {
        (str(environment), str(mode), int(seed))
        for environment in spec.ENVIRONMENTS
        for mode in spec.DISTURBANCE_MODES
        for seed in spec.REUSED_SELECTION_SEEDS
    }
    if keys != expected_keys:
        raise RuntimeError("v18.4 reused panel cells do not match freeze")
    floor_by_seed = {
        int(seed): sum(
            int(row["evaluation_seed"]) == int(seed)
            and not bool(row["oracle_joint_feasible"])
            for row in panel
        )
        for seed in spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
    }
    if floor_by_seed != spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED:
        raise RuntimeError("v18.4 actor-floor seed counts do not match freeze")


def _path_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["environment"]),
        str(row["disturbance_mode"]),
        int(row["evaluation_seed"]),
    )


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["candidate_id"]), *_path_key(row))


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if int(args.workers) < 1:
        raise SystemExit("v18.4 workers must be positive")
    summary = run(args.reference_root, int(args.workers))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE v18.4 receding projection status={summary['status']} "
        f"selected={summary['selected_candidate_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
