#!/usr/bin/env python3
"""Evaluate the frozen v17.7 causal MPC candidates on one reused path."""

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

from freq_hrl.core.receding_horizon_responsibility import (  # noqa: E402
    CausalRecedingHorizonResponsibilityPlanner,
)
from freq_hrl.experiments.mujoco.control_validation import (  # noqa: E402
    _model_parameter_sha256,
    load_paired_mujoco_checkpoint,
)
from freq_hrl.experiments.mujoco.full_horizon_responsibility_oracle import (  # noqa: E402
    responsibility_frequency_powers,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import (  # noqa: E402
    mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4,
)
from scripts import (  # noqa: E402
    mujoco_v17_7_causal_mpc_diagnostic_spec as spec,
)
from scripts.run_mujoco_v17_5_feasibility_diagnostic import (  # noqa: E402
    _legacy_candidate_rows,
    _model_from_summary,
    _read_csv,
    _rollout,
)
from scripts.run_mujoco_v17_6_full_horizon_oracle_path import (  # noqa: E402
    legacy_replay_audit,
)


def evaluate_candidate(
    total: np.ndarray,
    *,
    candidate_id: str,
) -> dict[str, Any]:
    config = dict(spec.CANDIDATES[str(candidate_id)])
    planner = CausalRecedingHorizonResponsibilityPlanner(
        upper_window=8,
        lower_window=32,
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
    upper_rows: list[np.ndarray] = []
    lower_rows: list[np.ndarray] = []
    reconstruction_max = 0.0
    bound_violation_max = 0.0
    prefix_feasible_count = 0
    prefix_unavoidable_max = 0.0
    prefix_projection_rms_values: list[float] = []
    actor_floor_power_excesses: list[float] = []
    actor_floor_ratio_excesses: list[float] = []
    forecast_errors: list[float] = []
    started = time.perf_counter()
    for total_row in np.asarray(total, dtype=np.float64):
        row = planner.split(
            total_row,
            upper_limit=spec.UPPER_ACTION_LIMIT,
            lower_limit=spec.LOWER_ACTION_LIMIT,
        )
        upper = np.asarray(row["upper"], dtype=np.float64)
        lower = np.asarray(row["lower"], dtype=np.float64)
        upper_rows.append(upper)
        lower_rows.append(lower)
        reconstruction_max = max(
            reconstruction_max,
            float(np.max(np.abs(upper + lower - total_row))),
        )
        bound_violation_max = max(
            bound_violation_max,
            float(np.max(np.maximum(
                np.abs(upper) - spec.UPPER_ACTION_LIMIT, 0.0
            ))),
            float(np.max(np.maximum(
                np.abs(lower) - spec.LOWER_ACTION_LIMIT, 0.0
            ))),
        )
        prefix_feasible_count += int(row["prefix_upper_budget_feasible"])
        prefix_unavoidable_max = max(
            prefix_unavoidable_max,
            float(row["prefix_unavoidable_upper_violation_rms"]),
        )
        prefix_projection_rms_values.append(float(
            row["prefix_upper_projection_rms"]
        ))
        actor_floor_power_excesses.append(float(
            row["actor_floor_power_excess"]
        ))
        actor_floor_ratio_excesses.append(float(
            row["actor_floor_ratio_excess_squared"]
        ))
        forecast_errors.append(float(row["forecast_error_rms"]))
    runtime = time.perf_counter() - started
    upper_trace = np.stack(upper_rows)
    lower_trace = np.stack(lower_rows)
    upper_power, lower_power = responsibility_frequency_powers(
        total, upper_trace
    )
    trace_reconstruction = float(np.max(np.abs(
        upper_trace + lower_trace - total
    )))
    return {
        "candidate_id": str(candidate_id),
        "config": config,
        "trajectory_length": int(total.shape[0]),
        "action_dimension": int(total.shape[1]),
        "upper_power": upper_power,
        "lower_power": lower_power,
        "upper_budget_pass": bool(
            upper_power
            <= spec.UPPER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
        ),
        "lower_budget_pass": bool(
            lower_power
            <= spec.LOWER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
        ),
        "joint_budget_pass": bool(
            upper_power
            <= spec.UPPER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
            and lower_power
            <= spec.LOWER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
        ),
        "prefix_upper_budget_feasible_rate": (
            prefix_feasible_count / total.shape[0]
        ),
        "prefix_unavoidable_upper_violation_rms_max": (
            prefix_unavoidable_max
        ),
        "prefix_projection_rms_mean": float(np.mean(
            prefix_projection_rms_values
        )),
        "actor_floor_power_excess_mean": float(np.mean(
            actor_floor_power_excesses
        )),
        "actor_floor_power_excess_max": float(np.max(
            actor_floor_power_excesses
        )),
        "actor_floor_positive_rate": float(np.mean(
            np.asarray(actor_floor_power_excesses) > 0.0
        )),
        "actor_floor_ratio_excess_squared_mean": float(np.mean(
            actor_floor_ratio_excesses
        )),
        "one_step_total_forecast_error_rms_mean": float(np.mean(
            forecast_errors[1:] if len(forecast_errors) > 1 else forecast_errors
        )),
        "reconstruction_error_max": max(
            reconstruction_max, trace_reconstruction
        ),
        "bound_violation_max": bound_violation_max,
        "runtime_seconds": float(runtime),
    }


def run_path(
    *,
    env_id: str,
    disturbance_mode: str,
    evaluation_seed: int,
    checkpoint_dir: Path,
    oracle_path: Path,
) -> dict[str, Any]:
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=spec.FROZEN_CORE_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        require_verified=True,
    )
    summary_path = Path(checkpoint_dir) / "cell_summary.json"
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
    legacy_rows_path = Path(checkpoint_dir) / "evaluation_rows.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = _model_from_summary(summary)
    checkpoint_metadata = load_paired_mujoco_checkpoint(
        model,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        env_id=str(env_id),
        optimizer_seed=spec.OPTIMIZER_SEED,
        expected_code_revision=v17_4.FROZEN_ALGORITHM_REVISION,
        expected_source_manifest_sha256=v17_4.FROZEN_SOURCE_MANIFEST_SHA256,
        expected_method="freq_hrl",
        expected_router_mode="causal_streaming_audit_projection",
        expected_router_strength=0.0,
        expected_router_observe_strength=False,
        expected_responsibility_mode="additive",
        expected_protocol_version=v17_4.FROZEN_CORE_PROTOCOL_VERSION,
    )
    parameter_sha256 = _model_parameter_sha256(model)
    legacy_rows = _legacy_candidate_rows(_read_csv(legacy_rows_path))
    path_key = (str(disturbance_mode), int(evaluation_seed))
    if path_key not in legacy_rows:
        raise ValueError("v17.7 path is outside the frozen v17.4 matrix")
    responsibility_trace: dict[str, Any] = {}
    replay = _rollout(
        model,
        env_id=str(env_id),
        disturbance_mode=str(disturbance_mode),
        seed=int(evaluation_seed),
        router_mode="causal_streaming_audit_projection",
        responsibility_trace_output=responsibility_trace,
    )
    replay_audit = legacy_replay_audit(legacy_rows[path_key], replay)
    if not replay_audit["exact"]:
        raise RuntimeError("v17.7 source path does not exactly replay v17.4")
    if parameter_sha256 != _model_parameter_sha256(model):
        raise RuntimeError("v17.7 path replay mutated the checkpoint")

    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    if (
        oracle.get("environment") != str(env_id)
        or oracle.get("disturbance_mode") != str(disturbance_mode)
        or int(oracle.get("evaluation_seed", -1)) != int(evaluation_seed)
        or oracle.get("frozen_oracle_revision")
        != "5a6efa2dccb441334b55cabd25556fc78b55ad3b"
        or oracle.get("frozen_source_manifest_sha256")
        != "8d001993b7da2913052ce9ee91ff329410592dede1c8b2aae48da7a1054bc0d1"
        or oracle.get("legacy_replay_audit", {}).get("exact") is not True
    ):
        raise ValueError("v17.7 oracle dependency does not match the path")

    total = np.asarray(
        responsibility_trace["total_action"], dtype=np.float64
    )
    baseline_upper = np.asarray(
        responsibility_trace["upper_action"], dtype=np.float64
    )
    baseline_upper_power, baseline_lower_power = (
        responsibility_frequency_powers(total, baseline_upper)
    )
    baseline_joint = bool(
        baseline_upper_power
        <= spec.UPPER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
        and baseline_lower_power
        <= spec.LOWER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
    )
    candidates = {
        candidate_id: evaluate_candidate(total, candidate_id=candidate_id)
        for candidate_id in spec.CANDIDATES
    }
    oracle_joint = bool(oracle["oracle"]["joint_feasible"])
    for candidate in candidates.values():
        candidate["recovers_oracle_recoverable_failure"] = bool(
            not baseline_joint
            and oracle_joint
            and candidate["joint_budget_pass"]
        )
        candidate["preserves_baseline_feasible_path"] = bool(
            baseline_joint and candidate["joint_budget_pass"]
        )
    return {
        "status": "causal_mpc_path_complete",
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": (
            spec.FROZEN_SOURCE_MANIFEST_SHA256
        ),
        "source_identity": source_identity,
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "evaluation_seed": int(evaluation_seed),
        "optimizer_seed": spec.OPTIMIZER_SEED,
        "source_checkpoint": checkpoint_metadata,
        "source_checkpoint_parameter_sha256": parameter_sha256,
        "legacy_replay_audit": replay_audit,
        "baseline": {
            "upper_power": baseline_upper_power,
            "lower_power": baseline_lower_power,
            "joint_feasible": baseline_joint,
        },
        "oracle": {
            "joint_feasible": oracle_joint,
            "upper_power": float(oracle["oracle"]["upper_power"]),
            "lower_power": float(oracle["oracle"]["lower_power"]),
            "status": str(oracle["oracle"]["status"]),
        },
        "candidates": candidates,
        "claim_boundary": (
            "causal candidate selection on reused rejected paths; not fresh "
            "training or publication evidence"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=spec.ENVIRONMENTS, required=True)
    parser.add_argument(
        "--disturbance-mode", choices=spec.DISTURBANCE_MODES, required=True
    )
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--oracle-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.evaluation_seed) not in spec.EVALUATION_SEEDS:
        raise SystemExit("evaluation seed is outside the v17.7 design")
    payload = run_path(
        env_id=args.env_id,
        disturbance_mode=args.disturbance_mode,
        evaluation_seed=args.evaluation_seed,
        checkpoint_dir=args.checkpoint_dir,
        oracle_path=args.oracle_path,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "causal_mpc_path.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    joint = sum(
        int(row["joint_budget_pass"])
        for row in payload["candidates"].values()
    )
    print(
        f"DONE v17.7 causal-mpc env={args.env_id} "
        f"mode={args.disturbance_mode} seed={args.evaluation_seed} "
        f"joint_candidates={joint}/{len(spec.CANDIDATES)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
