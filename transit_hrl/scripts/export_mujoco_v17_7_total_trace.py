#!/usr/bin/env python3
"""Export one small frozen total-action trace for v17.7 mechanism diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def export_trace(
    *,
    env_id: str,
    disturbance_mode: str,
    evaluation_seed: int,
    checkpoint_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
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
        raise ValueError("trace path is outside the frozen v17.4 matrix")
    responsibility_trace: dict[str, object] = {}
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
        raise RuntimeError("trace export does not exactly replay v17.4")
    if parameter_sha256 != _model_parameter_sha256(model):
        raise RuntimeError("trace export mutated the checkpoint")
    total = np.asarray(responsibility_trace["total_action"], dtype=np.float64)
    baseline_upper = np.asarray(
        responsibility_trace["upper_action"], dtype=np.float64
    )
    upper_power, lower_power = responsibility_frequency_powers(
        total, baseline_upper
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    columns = ["step", *(
        f"total_action_{index}" for index in range(total.shape[1])
    )]
    np.savetxt(
        output / "total_action.csv",
        np.column_stack((np.arange(total.shape[0]), total)),
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt=["%d", *(["%.17g"] * total.shape[1])],
    )
    metadata = {
        "status": "single_path_total_trace_exported",
        "evidence_role": "development_mechanism_debug_not_claim_evidence",
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "source_identity": source_identity,
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "evaluation_seed": int(evaluation_seed),
        "trajectory_length": int(total.shape[0]),
        "action_dimension": int(total.shape[1]),
        "baseline_upper_power": upper_power,
        "baseline_lower_power": lower_power,
        "legacy_replay_audit": replay_audit,
        "source_checkpoint": checkpoint_metadata,
        "source_checkpoint_parameter_sha256": parameter_sha256,
        "claim_boundary": (
            "one reused path exported only to diagnose a failed development "
            "smoke; not evidence for an empirical claim"
        ),
    }
    (output / "trace_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=spec.ENVIRONMENTS, required=True)
    parser.add_argument(
        "--disturbance-mode", choices=spec.DISTURBANCE_MODES, required=True
    )
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = export_trace(
        env_id=args.env_id,
        disturbance_mode=args.disturbance_mode,
        evaluation_seed=args.evaluation_seed,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )
    print(
        f"DONE v17.7 trace env={metadata['environment']} "
        f"steps={metadata['trajectory_length']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
