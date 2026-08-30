#!/usr/bin/env python3
"""Solve one v17.6 frozen-total-action full-horizon oracle path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.mujoco.control_validation import (  # noqa: E402
    _model_parameter_sha256,
    load_paired_mujoco_checkpoint,
)
from freq_hrl.experiments.mujoco.full_horizon_responsibility_oracle import (  # noqa: E402
    responsibility_frequency_powers,
    solve_full_horizon_responsibility_oracle,
)
from scripts import (  # noqa: E402
    mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4,
)
from scripts import mujoco_v17_6_full_horizon_oracle_spec as spec  # noqa: E402
from scripts.run_mujoco_v17_5_feasibility_diagnostic import (  # noqa: E402
    LEGACY_NUMERIC_KEYS,
    TRACE_KEYS,
    _legacy_candidate_rows,
    _model_from_summary,
    _read_csv,
    _rollout,
)


def legacy_replay_audit(
    legacy: dict[str, Any], replay: dict[str, Any]
) -> dict[str, Any]:
    trace_mismatches = {
        key: int(str(legacy[key]) != str(replay[key])) for key in TRACE_KEYS
    }
    numeric = {
        key: abs(float(legacy[key]) - float(replay[key]))
        for key in LEGACY_NUMERIC_KEYS
    }
    return {
        "trace_mismatches": trace_mismatches,
        "numeric_absolute_differences": numeric,
        "exact": bool(
            not any(trace_mismatches.values())
            and max(numeric.values(), default=0.0)
            <= spec.LEGACY_NUMERIC_TOLERANCE
        ),
    }


def run_path(
    *,
    env_id: str,
    disturbance_mode: str,
    evaluation_seed: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
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
    path = (str(disturbance_mode), int(evaluation_seed))
    if path not in legacy_rows:
        raise ValueError("v17.6 path is outside the frozen v17.4 matrix")
    responsibility_trace: dict[str, Any] = {}
    replay = _rollout(
        model,
        env_id=str(env_id),
        disturbance_mode=str(disturbance_mode),
        seed=int(evaluation_seed),
        router_mode="causal_streaming_audit_projection",
        responsibility_trace_output=responsibility_trace,
    )
    replay_audit = legacy_replay_audit(legacy_rows[path], replay)
    if not replay_audit["exact"]:
        raise RuntimeError("v17.6 source path does not exactly replay v17.4")
    if parameter_sha256 != _model_parameter_sha256(model):
        raise RuntimeError("v17.6 path replay mutated the checkpoint")

    total = responsibility_trace["total_action"]
    baseline_upper = responsibility_trace["upper_action"]
    baseline_upper_power, baseline_lower_power = (
        responsibility_frequency_powers(total, baseline_upper)
    )
    if (
        abs(baseline_upper_power - float(replay["UpperHFPowerAbs"]))
        > spec.LEGACY_NUMERIC_TOLERANCE
        or abs(baseline_lower_power - float(replay["LowerLFDriftAbs"]))
        > spec.LEGACY_NUMERIC_TOLERANCE
    ):
        raise RuntimeError("v17.6 oracle filters do not match v17.4 endpoints")
    oracle = solve_full_horizon_responsibility_oracle(
        total,
        upper_rms_budget=spec.UPPER_RMS_BUDGET,
        lower_rms_budget=spec.LOWER_RMS_BUDGET,
        solver_tolerance=spec.SOLVER_TOLERANCE,
        power_tolerance=spec.POWER_TOLERANCE,
        multiplier_bisection_steps=spec.MULTIPLIER_BISECTION_STEPS,
    )
    baseline_joint_feasible = bool(
        baseline_upper_power
        <= spec.UPPER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
        and baseline_lower_power
        <= spec.LOWER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
    )
    return {
        "status": "full_horizon_oracle_path_complete",
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_oracle_revision": spec.FROZEN_ORACLE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "evaluation_seed": int(evaluation_seed),
        "optimizer_seed": spec.OPTIMIZER_SEED,
        "source_checkpoint": checkpoint_metadata,
        "source_checkpoint_parameter_sha256": parameter_sha256,
        "legacy_replay_audit": replay_audit,
        "baseline": {
            "episode_return": float(replay["episode_return"]),
            "upper_power": baseline_upper_power,
            "lower_power": baseline_lower_power,
            "joint_feasible": baseline_joint_feasible,
        },
        "oracle": oracle.summary(),
        "recoverable_by_responsibility_split": bool(
            oracle.joint_feasible and not baseline_joint_feasible
        ),
        "claim_boundary": (
            "acausal development oracle on a reused rejected path; not an "
            "online policy and not publication evidence"
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.evaluation_seed) not in spec.EVALUATION_SEEDS:
        raise SystemExit("evaluation seed is outside the v17.6 oracle design")
    payload = run_path(
        env_id=args.env_id,
        disturbance_mode=args.disturbance_mode,
        evaluation_seed=args.evaluation_seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oracle_path.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE v17.6 oracle env={args.env_id} "
        f"mode={args.disturbance_mode} seed={args.evaluation_seed} "
        f"status={payload['oracle']['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
