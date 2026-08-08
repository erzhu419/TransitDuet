#!/usr/bin/env python3
"""Source-preserving runtime for frozen MuJoCo v13 behavioral confirmatory cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.mujoco import control_validation as validation  # noqa: E402
from scripts import mujoco_v13_behavioral_confirmatory_spec as spec  # noqa: E402


LAUNCHER_PATH = Path(__file__).with_name(
    "submit_mujoco_v13_behavioral_confirmatory_scheduleurm.py"
)
SPEC_PATH = Path(spec.__file__).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _metadata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--confirmatory-arm", required=True)
    parser.add_argument("--confirmatory-runtime-revision", required=True)
    parser.add_argument("--confirmatory-launcher-sha256", required=True)
    parser.add_argument("--confirmatory-runtime-sha256", required=True)
    parser.add_argument("--confirmatory-spec-sha256", required=True)
    return parser


def _validate_core_args(args: argparse.Namespace, arm: str) -> None:
    try:
        arm_spec = spec.ARMS[str(arm)]
    except KeyError as exc:
        raise ValueError(f"unknown frozen confirmatory arm: {arm}") from exc
    expected = {
        "method": arm_spec["method"],
        "env_id": None,
        "disturbance_mode": "standard",
        "training_disturbance_modes": list(spec.TRAINING_DISTURBANCE_MODES),
        "evaluation_disturbance_modes": list(spec.EVALUATION_DISTURBANCE_MODES),
        "train_seeds": list(spec.TRAIN_SEEDS),
        "selection_seeds": list(spec.CHECKPOINT_SELECTION_SEEDS),
        "safety_selection_seeds": list(spec.SAFETY_SELECTION_SEEDS),
        "eval_seeds": list(spec.HELDOUT_EVALUATION_SEEDS),
        "steps": spec.STEPS,
        "episode_horizon": spec.EPISODE_HORIZON,
        "iterations": spec.ITERATIONS,
        "upper_period": spec.UPPER_PERIOD,
        "hidden_dim": spec.HIDDEN_DIM,
        "learning_rate": spec.LEARNING_RATE,
        "lower_lf_rms_budget": spec.LOWER_LF_RMS_BUDGET,
        "upper_action_scale": spec.UPPER_ACTION_SCALE,
        "lower_action_scale": spec.LOWER_ACTION_SCALE,
        "responsibility_mode": arm_spec["responsibility_mode"],
        "lower_constraint_update_mode": spec.LOWER_CONSTRAINT_UPDATE_MODE,
        "checkpoint_smoothing_window": spec.CHECKPOINT_SMOOTHING_WINDOW,
        "checkpoint_min_delta": spec.CHECKPOINT_MIN_DELTA,
        "checkpoint_evaluation_interval": spec.CHECKPOINT_EVALUATION_INTERVAL,
        "code_revision": spec.FROZEN_ALGORITHM_REVISION,
        "source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
    }
    if str(args.env_id) not in spec.ENVIRONMENTS:
        raise ValueError("confirmatory environment registry drifted")
    if int(args.optimizer_seed) not in spec.OPTIMIZER_SEEDS:
        raise ValueError("confirmatory optimizer seed is not preregistered")
    for name, value in expected.items():
        if name == "env_id":
            continue
        if getattr(args, name) != value:
            raise ValueError(f"confirmatory core argument drifted: {name}")


def _validate_runtime_metadata(args: argparse.Namespace) -> dict[str, str]:
    revision = str(args.confirmatory_runtime_revision).strip().lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("confirmatory runtime revision must be a full Git SHA")
    observed = {
        "launcher_sha256": _sha256(LAUNCHER_PATH),
        "runtime_sha256": _sha256(Path(__file__)),
        "spec_sha256": _sha256(SPEC_PATH),
    }
    expected = {
        "launcher_sha256": str(args.confirmatory_launcher_sha256).lower(),
        "runtime_sha256": str(args.confirmatory_runtime_sha256).lower(),
        "spec_sha256": str(args.confirmatory_spec_sha256).lower(),
    }
    if observed != expected:
        raise ValueError(
            "confirmatory runtime file identity mismatch: "
            f"expected={expected}, observed={observed}"
        )
    return {"runtime_revision": revision, **observed}


def _run_core(args: argparse.Namespace) -> None:
    payload, rows, model = validation.train_mujoco_method(
        method=args.method,
        env_id=args.env_id,
        disturbance_mode=args.disturbance_mode,
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        safety_selection_seeds=args.safety_selection_seeds,
        eval_seeds=args.eval_seeds,
        steps=args.steps,
        iterations=args.iterations,
        optimizer_seed=args.optimizer_seed,
        episode_horizon=args.episode_horizon,
        upper_period=args.upper_period,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        lower_lf_rms_budget=args.lower_lf_rms_budget,
        checkpoint_smoothing_window=args.checkpoint_smoothing_window,
        checkpoint_min_delta=args.checkpoint_min_delta,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        training_disturbance_modes=args.training_disturbance_modes,
        evaluation_disturbance_modes=args.evaluation_disturbance_modes,
        upper_action_scale=args.upper_action_scale,
        lower_action_scale=args.lower_action_scale,
        responsibility_mode=args.responsibility_mode,
        lower_constraint_update_mode=args.lower_constraint_update_mode,
        code_revision=args.code_revision,
        expected_source_manifest_sha256=args.source_manifest_sha256,
    )
    validation.write_cell(args.output_dir, payload, rows, model)


def _annotate_summary(
    output_dir: Path,
    *,
    arm: str,
    runtime_identity: dict[str, str],
) -> None:
    path = Path(output_dir) / "cell_summary.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != spec.FROZEN_CORE_PROTOCOL_VERSION:
        raise ValueError("confirmatory cell core protocol drifted")
    if payload.get("code_revision") != spec.FROZEN_ALGORITHM_REVISION:
        raise ValueError("confirmatory cell algorithm revision drifted")
    if payload.get("source_manifest_sha256") != spec.FROZEN_SOURCE_MANIFEST_SHA256:
        raise ValueError("confirmatory cell source manifest drifted")
    payload.update({
        "confirmatory_protocol_version": spec.CONFIRMATORY_PROTOCOL_VERSION,
        "confirmatory_runtime_adapter_version": spec.RUNTIME_ADAPTER_VERSION,
        "confirmatory_arm": str(arm),
        "confirmatory_runtime_revision": runtime_identity["runtime_revision"],
        "confirmatory_launcher_sha256": runtime_identity["launcher_sha256"],
        "confirmatory_runtime_sha256": runtime_identity["runtime_sha256"],
        "confirmatory_spec_sha256": runtime_identity["spec_sha256"],
        "confirmatory_evidence_role": "fresh_seed_confirmatory_unanalyzed",
        "confirmatory_seed_namespace": "mujoco_v13_behavioral_confirmatory",
    })
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    metadata, remaining = _metadata_parser().parse_known_args()
    core_args = validation.build_parser().parse_args(remaining)
    _validate_core_args(core_args, metadata.confirmatory_arm)
    runtime_identity = _validate_runtime_metadata(metadata)
    _run_core(core_args)
    _annotate_summary(
        core_args.output_dir,
        arm=metadata.confirmatory_arm,
        runtime_identity=runtime_identity,
    )
    print(
        "mujoco_v13_behavioral_confirmatory_cell status=valid "
        f"arm={metadata.confirmatory_arm} env={core_args.env_id} "
        f"optimizer_seed={core_args.optimizer_seed} output={core_args.output_dir}"
    )


if __name__ == "__main__":
    main()
