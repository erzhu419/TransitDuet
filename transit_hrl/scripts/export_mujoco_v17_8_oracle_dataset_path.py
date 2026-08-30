#!/usr/bin/env python3
"""Export one reused v17.4 total/oracle pair for v17.8 on node003."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


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
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import (  # noqa: E402
    mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4,
)
from scripts import (  # noqa: E402
    mujoco_v17_8_causal_fir_distillation_spec as spec,
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


def export_reused_path(
    *,
    env_id: str,
    disturbance_mode: str,
    evaluation_seed: int,
    checkpoint_dir: Path,
    artifact_path: Path,
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
    path_key = (str(disturbance_mode), int(evaluation_seed))
    legacy_rows = _legacy_candidate_rows(_read_csv(legacy_rows_path))
    if path_key not in legacy_rows:
        raise ValueError("v17.8 reused path is outside the frozen v17.4 panel")
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
        raise RuntimeError("v17.8 dataset path does not exactly replay v17.4")
    if parameter_sha256 != _model_parameter_sha256(model):
        raise RuntimeError("v17.8 dataset replay mutated the checkpoint")

    total = np.asarray(responsibility_trace["total_action"], dtype=np.float64)
    baseline_upper = np.asarray(
        responsibility_trace["upper_action"], dtype=np.float64
    )
    baseline_upper_power, baseline_lower_power = (
        responsibility_frequency_powers(total, baseline_upper)
    )
    oracle = solve_full_horizon_responsibility_oracle(
        total,
        upper_rms_budget=spec.UPPER_RMS_BUDGET,
        lower_rms_budget=spec.LOWER_RMS_BUDGET,
        upper_action_limit=spec.UPPER_ACTION_LIMIT,
        lower_action_limit=spec.LOWER_ACTION_LIMIT,
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
    )
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        total_action=total,
        baseline_upper_action=baseline_upper,
        oracle_upper_action=oracle.upper,
    )
    baseline_joint = bool(
        baseline_upper_power
        <= spec.UPPER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
        and baseline_lower_power
        <= spec.LOWER_RMS_BUDGET ** 2 + spec.POWER_TOLERANCE
    )
    return {
        "status": "v17_8_reused_oracle_dataset_path_exported",
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "source_identity": source_identity,
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "evaluation_seed": int(evaluation_seed),
        "seed_role": "reused_grouped_selection",
        "trajectory_length": int(total.shape[0]),
        "action_dimension": int(total.shape[1]),
        "server_artifact_path": str(artifact),
        "legacy_replay_audit": replay_audit,
        "source_checkpoint": checkpoint_metadata,
        "source_checkpoint_parameter_sha256": parameter_sha256,
        "baseline": {
            "episode_return": float(replay["episode_return"]),
            "upper_power": baseline_upper_power,
            "lower_power": baseline_lower_power,
            "joint_feasible": baseline_joint,
        },
        "oracle": oracle.summary(),
        "claim_boundary": (
            "reused rejected path and acausal training label; server-only "
            "development input, not manuscript evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=spec.ENVIRONMENTS, required=True)
    parser.add_argument(
        "--disturbance-mode", choices=spec.DISTURBANCE_MODES, required=True
    )
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--marker-dir", type=Path, required=True)
    args = parser.parse_args()
    if int(args.evaluation_seed) not in spec.REUSED_SELECTION_SEEDS:
        raise SystemExit("seed is outside the v17.8 reused selection panel")
    payload = export_reused_path(
        env_id=args.env_id,
        disturbance_mode=args.disturbance_mode,
        evaluation_seed=args.evaluation_seed,
        checkpoint_dir=args.checkpoint_dir,
        artifact_path=args.artifact_path,
    )
    args.marker_dir.mkdir(parents=True, exist_ok=True)
    (args.marker_dir / "dataset_path.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE v17.8 dataset env={args.env_id} "
        f"mode={args.disturbance_mode} seed={args.evaluation_seed} "
        f"oracle={payload['oracle']['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
