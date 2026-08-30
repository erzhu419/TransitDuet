#!/usr/bin/env python3
"""Export one causal actor-state trace from a frozen v17.4 replay."""

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
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import (  # noqa: E402
    mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4,
)
from scripts import mujoco_v18_1_state_actor_dataset_spec as spec  # noqa: E402
from scripts.run_mujoco_v17_5_feasibility_diagnostic import (  # noqa: E402
    _legacy_candidate_rows,
    _model_from_summary,
    _read_csv,
    _rollout,
)
from scripts.run_mujoco_v17_6_full_horizon_oracle_path import (  # noqa: E402
    legacy_replay_audit,
)


def validate_state_trace(
    env_id: str,
    responsibility_trace: dict[str, Any],
    actor_trace: dict[str, Any],
) -> dict[str, int]:
    expected_responsibility = {
        "executed_action",
        "lower_action",
        "total_action",
        "upper_action",
    }
    expected_actor = {
        "disturbance",
        "episode_step",
        "latent_lower_action",
        "lower_policy_state",
        "observation",
        "upper_decision",
        "upper_policy_action",
    }
    if set(responsibility_trace) != expected_responsibility:
        raise ValueError("v18.1 responsibility trace has unexpected keys")
    if set(actor_trace) != expected_actor:
        raise ValueError("v18.1 actor trace has unexpected keys")
    arrays = {
        **{
            key: np.asarray(value)
            for key, value in responsibility_trace.items()
        },
        **{key: np.asarray(value) for key, value in actor_trace.items()},
    }
    total = np.asarray(arrays["total_action"], dtype=np.float64)
    if total.ndim != 2 or not total.shape[0]:
        raise ValueError("v18.1 total action trace must be a non-empty matrix")
    length = int(total.shape[0])
    action_dim = int(spec.EXPECTED_ACTION_DIMENSION[str(env_id)])
    observation_dim = int(spec.EXPECTED_OBSERVATION_DIMENSION[str(env_id)])
    state_dim = int(spec.EXPECTED_LOWER_STATE_DIMENSION[str(env_id)])
    matrix_shapes = {
        "total_action": (length, action_dim),
        "upper_action": (length, action_dim),
        "lower_action": (length, action_dim),
        "executed_action": (length, action_dim),
        "disturbance": (length, action_dim),
        "upper_policy_action": (length, action_dim),
        "latent_lower_action": (length, action_dim),
        "observation": (length, observation_dim),
        "lower_policy_state": (length, state_dim),
    }
    for key, shape in matrix_shapes.items():
        value = arrays[key]
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"v18.1 {key} trace is invalid")
    if arrays["upper_decision"].shape != (length,):
        raise ValueError("v18.1 upper-decision trace is misaligned")
    if arrays["episode_step"].shape != (length,):
        raise ValueError("v18.1 episode-step trace is misaligned")
    if not np.issubdtype(arrays["upper_decision"].dtype, np.bool_):
        raise ValueError("v18.1 upper-decision trace must be boolean")
    if not np.issubdtype(arrays["episode_step"].dtype, np.integer):
        raise ValueError("v18.1 episode-step trace must be integral")
    if int(arrays["episode_step"][0]) != 0 or not bool(
        arrays["upper_decision"][0]
    ):
        raise ValueError("v18.1 actor trace does not begin at a macro boundary")
    reconstruction = (
        np.asarray(arrays["upper_action"], dtype=np.float64)
        + np.asarray(arrays["lower_action"], dtype=np.float64)
        - total
    )
    if float(np.max(np.abs(reconstruction))) > 1e-7:
        raise ValueError("v18.1 responsibility trace does not reconstruct total")
    return {
        "trajectory_length": length,
        "action_dimension": action_dim,
        "observation_dimension": observation_dim,
        "lower_state_dimension": state_dim,
        "upper_decision_count": int(np.count_nonzero(
            arrays["upper_decision"]
        )),
    }


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
    rows_path = Path(checkpoint_dir) / "evaluation_rows.csv"
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
    legacy_rows = _legacy_candidate_rows(_read_csv(rows_path))
    if path_key not in legacy_rows:
        raise ValueError("v18.1 path is outside the frozen v17.4 panel")
    responsibility_trace: dict[str, Any] = {}
    actor_trace: dict[str, Any] = {}
    replay = _rollout(
        model,
        env_id=str(env_id),
        disturbance_mode=str(disturbance_mode),
        seed=int(evaluation_seed),
        router_mode="causal_streaming_audit_projection",
        responsibility_trace_output=responsibility_trace,
        actor_trace_output=actor_trace,
    )
    replay_audit = legacy_replay_audit(legacy_rows[path_key], replay)
    if not replay_audit["exact"]:
        raise RuntimeError("v18.1 state trace does not exactly replay v17.4")
    if parameter_sha256 != _model_parameter_sha256(model):
        raise RuntimeError("v18.1 state export mutated the checkpoint")
    trace_summary = validate_state_trace(
        str(env_id), responsibility_trace, actor_trace
    )

    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        total_action=np.asarray(
            responsibility_trace["total_action"], dtype=np.float32
        ),
        baseline_upper_action=np.asarray(
            responsibility_trace["upper_action"], dtype=np.float32
        ),
        baseline_lower_action=np.asarray(
            responsibility_trace["lower_action"], dtype=np.float32
        ),
        executed_action=np.asarray(
            responsibility_trace["executed_action"], dtype=np.float32
        ),
        observation=np.asarray(actor_trace["observation"], dtype=np.float32),
        lower_policy_state=np.asarray(
            actor_trace["lower_policy_state"], dtype=np.float32
        ),
        disturbance=np.asarray(
            actor_trace["disturbance"], dtype=np.float32
        ),
        upper_policy_action=np.asarray(
            actor_trace["upper_policy_action"], dtype=np.float32
        ),
        latent_lower_action=np.asarray(
            actor_trace["latent_lower_action"], dtype=np.float32
        ),
        upper_decision=np.asarray(
            actor_trace["upper_decision"], dtype=np.bool_
        ),
        episode_step=np.asarray(actor_trace["episode_step"], dtype=np.int64),
    )
    return {
        "status": "v18_1_causal_actor_state_path_exported",
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "source_identity": source_identity,
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "evaluation_seed": int(evaluation_seed),
        "seed_role": "reused_grouped_selection",
        "server_artifact_path": str(artifact),
        "trace_summary": trace_summary,
        "legacy_replay_audit": replay_audit,
        "source_checkpoint": checkpoint_metadata,
        "source_checkpoint_parameter_sha256": parameter_sha256,
        "target_labels_accessed": False,
        "dataset_contract": spec.DATASET_CONTRACT,
        "claim_boundary": spec.DATASET_CONTRACT["claim_boundary"],
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
        raise SystemExit("seed is outside the v18.1 reused selection panel")
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
        f"DONE v18.1 state dataset env={args.env_id} "
        f"mode={args.disturbance_mode} seed={args.evaluation_seed} "
        f"steps={payload['trace_summary']['trajectory_length']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
