#!/usr/bin/env python3
"""Evaluate frozen causal actor-floor signals on reused MuJoCo paths."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.core import (  # noqa: E402
    CausalRecedingHorizonResponsibilityPlanner,
)
from freq_hrl.experiments.mujoco import (  # noqa: E402
    full_horizon_responsibility_oracle as oracle,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import mujoco_v18_5_actor_floor_signal_spec as spec  # noqa: E402
from scripts.train_mujoco_v17_8_causal_fir import (  # noqa: E402
    load_reused_panel,
)


SCORE_FIELDS = (
    "floor_ratio_mean",
    "floor_ratio_p95",
    "floor_ratio_maximum",
    "floor_ratio_ema_maximum",
    "floor_power_excess_mean",
    "forecast_joint_infeasible_rate",
)


def evaluate_path(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    source, candidate_id = payload
    config = dict(spec.CANDIDATES[str(candidate_id)])
    total = np.asarray(source["total_action"], dtype=np.float64)
    planner = CausalRecedingHorizonResponsibilityPlanner(
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
        upper_rms_budget=spec.UPPER_RMS_BUDGET,
        lower_rms_budget=spec.LOWER_RMS_BUDGET,
        planning_horizon=int(config["planning_horizon"]),
        forecast_mode=str(config["forecast_mode"]),
        velocity_alpha=spec.VELOCITY_ALPHA,
        velocity_decay=spec.VELOCITY_DECAY,
        coordinate_sweeps=spec.COORDINATE_SWEEPS,
        multiplier_bisection_steps=spec.MULTIPLIER_BISECTION_STEPS,
        power_tolerance=spec.POWER_TOLERANCE,
        use_budget_ledger=True,
        enforce_prefix_upper_budget=True,
    )
    planner.reset(total.shape[1])
    projected = [
        planner.split(
            row,
            upper_limit=spec.UPPER_ACTION_LIMIT,
            lower_limit=spec.LOWER_ACTION_LIMIT,
        )
        for row in total
    ]
    upper = np.stack([row["upper"] for row in projected]).astype(
        np.float64
    )
    lower = np.stack([row["lower"] for row in projected]).astype(
        np.float64
    )
    upper_power, lower_power = oracle.responsibility_frequency_powers(
        total,
        upper,
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
    )
    ratios = np.asarray([
        float(row["actor_floor_ratio_excess_squared"])
        for row in projected
    ])
    excess = np.asarray([
        float(row["actor_floor_power_excess"]) for row in projected
    ])
    forecast_errors = np.asarray([
        float(row["forecast_error_rms"]) for row in projected
    ])
    ema = _causal_ema(ratios, alpha=0.05)
    reconstruction = float(np.max(np.abs(upper + lower - total)))
    bound_violation = float(max(
        np.max(np.maximum(np.abs(upper) - spec.UPPER_ACTION_LIMIT, 0.0)),
        np.max(np.maximum(np.abs(lower) - spec.LOWER_ACTION_LIMIT, 0.0)),
    ))
    finite = bool(
        np.all(np.isfinite(upper))
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(ratios))
        and np.all(np.isfinite(excess))
    )
    return {
        "candidate_id": str(candidate_id),
        "environment": str(source["environment"]),
        "disturbance_mode": str(source["disturbance_mode"]),
        "evaluation_seed": int(source["evaluation_seed"]),
        "actor_floor": not bool(source["oracle_joint_feasible"]),
        "reference_feasible": bool(source["oracle_joint_feasible"]),
        "trajectory_length": int(total.shape[0]),
        "action_dimension": int(total.shape[1]),
        "valid": bool(
            finite
            and reconstruction <= 1e-6
            and bound_violation <= 1e-6
        ),
        "endpoint_upper_power": float(upper_power),
        "endpoint_lower_power": float(lower_power),
        "endpoint_joint_feasible": bool(
            upper_power
            <= spec.UPPER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
            and lower_power
            <= spec.LOWER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
        ),
        "floor_ratio_mean": float(np.mean(ratios)),
        "floor_ratio_p95": float(np.quantile(ratios, 0.95)),
        "floor_ratio_maximum": float(np.max(ratios)),
        "floor_ratio_ema_maximum": float(np.max(ema)),
        "floor_power_excess_mean": float(np.mean(excess)),
        "forecast_joint_infeasible_rate": float(np.mean([
            not bool(row["joint_feasible_forecast"]) for row in projected
        ])),
        "prefix_upper_infeasible_rate": float(np.mean([
            not bool(row["prefix_upper_budget_feasible"])
            for row in projected
        ])),
        "prefix_unavoidable_upper_violation_rms_maximum": max(
            float(row["prefix_unavoidable_upper_violation_rms"])
            for row in projected
        ),
        "one_step_forecast_error_rms_mean": float(np.mean(
            forecast_errors[1:]
            if forecast_errors.size > 1 else forecast_errors
        )),
        "reconstruction_error_maximum": reconstruction,
        "bound_violation_maximum": bound_violation,
    }


def assess_score(
    candidate_id: str,
    score_field: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    floor = [row for row in rows if row["actor_floor"]]
    reference = [row for row in rows if row["reference_feasible"]]
    environment_reference = [
        row
        for row in reference
        if row["environment"] == spec.EXPECTED_ACTOR_FLOOR_ENVIRONMENT
    ]
    ranking = sorted(
        rows,
        key=lambda row: (
            -float(row[score_field]),
            str(row["environment"]),
            str(row["disturbance_mode"]),
            int(row["evaluation_seed"]),
        ),
    )
    unresolved_rank = next(
        index
        for index, row in enumerate(ranking, start=1)
        if _is_unresolved(row)
    )
    global_auc = _rank_auc(
        [float(row[score_field]) for row in floor],
        [float(row[score_field]) for row in reference],
    )
    environment_auc = _rank_auc(
        [float(row[score_field]) for row in floor],
        [float(row[score_field]) for row in environment_reference],
    )
    top_counts = {
        str(count): sum(row["actor_floor"] for row in ranking[:count])
        for count in (7, 14, 28)
    }
    eligible = bool(
        environment_auc >= 0.75
        and int(top_counts["14"]) >= 6
        and unresolved_rank <= 14
    )
    return {
        "candidate_id": str(candidate_id),
        "score_field": str(score_field),
        "global_rank_auc": float(global_auc),
        "actor_floor_environment_rank_auc": float(environment_auc),
        "top_k_actor_floor_count": top_counts,
        "unresolved_v17_14_path_rank": int(unresolved_rank),
        "actor_floor_score_mean": float(np.mean([
            float(row[score_field]) for row in floor
        ])),
        "reference_score_mean": float(np.mean([
            float(row[score_field]) for row in reference
        ])),
        "feedback_screen_eligible": eligible,
    }


def assessment_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(row["feedback_screen_eligible"]),
        -int(row["top_k_actor_floor_count"]["14"]),
        -float(row["actor_floor_environment_rank_auc"]),
        -float(row["global_rank_auc"]),
        int(row["unresolved_v17_14_path_rank"]),
        str(row["candidate_id"]),
        str(row["score_field"]),
    )


def run(reference_root: Path, workers: int) -> dict[str, Any]:
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=spec.FROZEN_CORE_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        require_verified=True,
    )
    panel = load_reused_panel(Path(reference_root))
    _validate_panel(panel)
    payloads = [
        (row, candidate_id)
        for candidate_id in spec.CANDIDATES
        for row in panel
    ]
    started = time.perf_counter()
    rows = _parallel_map(payloads, workers=workers)
    runtime = time.perf_counter() - started
    rows.sort(key=_row_key)
    assessments = [
        assess_score(
            candidate_id,
            score_field,
            [row for row in rows if row["candidate_id"] == candidate_id],
        )
        for candidate_id in spec.CANDIDATES
        for score_field in SCORE_FIELDS
    ]
    assessments.sort(key=assessment_key)
    selected = assessments[0]
    eligible = bool(selected["feedback_screen_eligible"])
    return {
        "status": (
            "actor_floor_signal_authorizes_frozen_debt_feedback_screen"
            if eligible
            else "actor_floor_signal_stops_debt_feedback_direction"
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
        "score_count": len(SCORE_FIELDS),
        "worker_count": int(workers),
        "runtime_seconds": float(runtime),
        "actor_correction_targets_accessed": False,
        "fresh_validation_paths_accessed": False,
        "fresh_path_access_allowed": False,
        "selected_signal": selected,
        "signal_assessments": assessments,
        "path_rows": rows,
        "feedback_screen_allowed": eligible,
        "support_gate": False,
        "diagnostic_contract": spec.DIAGNOSTIC_CONTRACT,
        "claim_boundary": spec.DIAGNOSTIC_CONTRACT["claim_boundary"],
    }


def _parallel_map(
    payloads: list[tuple[dict[str, Any], str]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if int(workers) == 1:
        rows = []
        for index, payload in enumerate(payloads, start=1):
            rows.append(evaluate_path(payload))
            _progress(index, len(payloads))
        return rows
    rows = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(evaluate_path, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            _progress(index, len(payloads))
    return rows


def _progress(completed: int, total: int) -> None:
    if completed % 20 == 0 or completed == total:
        print(
            f"PROGRESS v18.5 actor-floor signal {completed}/{total}",
            flush=True,
        )


def _validate_panel(panel: list[dict[str, Any]]) -> None:
    if len(panel) != spec.EXPECTED_PATH_COUNT:
        raise RuntimeError("v18.5 reused panel path count mismatch")
    if (
        sum(bool(row["oracle_joint_feasible"]) for row in panel)
        != spec.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
    ):
        raise RuntimeError("v18.5 reused panel feasibility count mismatch")
    keys = {_path_key(row) for row in panel}
    expected = {
        (str(environment), str(mode), int(seed))
        for environment in spec.ENVIRONMENTS
        for mode in spec.DISTURBANCE_MODES
        for seed in spec.REUSED_SELECTION_SEEDS
    }
    if keys != expected:
        raise RuntimeError("v18.5 reused panel cells do not match freeze")
    floor_by_seed = {
        int(seed): sum(
            int(row["evaluation_seed"]) == int(seed)
            and not bool(row["oracle_joint_feasible"])
            for row in panel
        )
        for seed in spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
    }
    if floor_by_seed != spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED:
        raise RuntimeError("v18.5 actor-floor seed counts do not match freeze")


def _rank_auc(positive: list[float], negative: list[float]) -> float:
    comparisons = [
        1.0 if left > right else 0.5 if left == right else 0.0
        for left in positive
        for right in negative
    ]
    return float(np.mean(np.asarray(comparisons, dtype=np.float64)))


def _causal_ema(values: np.ndarray, *, alpha: float) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    state = 0.0
    for index, value in enumerate(values):
        state += float(alpha) * (float(value) - state)
        output[index] = state
    return output


def _is_unresolved(row: dict[str, Any]) -> bool:
    expected = spec.UNRESOLVED_V17_14_PATH
    return bool(
        str(row["environment"]) == str(expected["environment"])
        and str(row["disturbance_mode"])
        == str(expected["disturbance_mode"])
        and int(row["evaluation_seed"])
        == int(expected["evaluation_seed"])
    )


def _path_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["environment"]),
        str(row["disturbance_mode"]),
        int(row["evaluation_seed"]),
    )


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["candidate_id"]), *_path_key(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if int(args.workers) < 1:
        raise SystemExit("v18.5 workers must be positive")
    summary = run(args.reference_root, int(args.workers))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "signal_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE v18.5 actor-floor signal status={summary['status']} "
        f"selected={summary['selected_signal']['candidate_id']}/"
        f"{summary['selected_signal']['score_field']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
