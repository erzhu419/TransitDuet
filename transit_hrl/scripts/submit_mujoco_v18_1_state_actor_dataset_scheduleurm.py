#!/usr/bin/env python3
"""Submit v18.1 causal actor-state exports through scheduleurm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v18_1_state_actor_dataset_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    DEFAULT_LINUX_PYTHON,
    SCHEDULER,
    STAGE_EXCLUDES,
    execute,
    execute_bulk,
    parse_csv,
    source_identity,
)


WORKER = Path("scripts/export_mujoco_v18_1_state_actor_dataset_path.py")
SIGNATURE_VERSION = "mujoco-v18-1-state-actor-dataset-v1"
DATA_LOCAL_NODE = "node003"


def checkpoint_relative_dir(environment: str) -> Path:
    return (
        Path(".server_artifacts") / spec.SOURCE_RUN_NAME / "cells"
        / str(environment) / f"replicate_{spec.OPTIMIZER_SEED}"
    )


def marker_relative_dir(
    run_name: str, environment: str, mode: str, seed: int
) -> Path:
    return (
        Path("results") / str(run_name) / "markers" / str(environment)
        / str(mode) / f"seed_{int(seed)}"
    )


def artifact_relative_path(
    run_name: str, environment: str, mode: str, seed: int
) -> Path:
    return (
        Path(".server_artifacts") / str(run_name) / "paths"
        / str(environment) / str(mode) / f"seed_{int(seed)}"
        / "actor_state_path.npz"
    )


def selected_paths(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    return [
        (environment, mode, int(seed))
        for environment in args.environments
        for mode in args.disturbance_modes
        for seed in args.evaluation_seeds
    ]


def build_command(
    args: argparse.Namespace,
    environment: str,
    mode: str,
    seed: int,
) -> str:
    command = [
        str(args.python_executable),
        "-u",
        str(WORKER),
        "--env-id", str(environment),
        "--disturbance-mode", str(mode),
        "--evaluation-seed", str(seed),
        "--checkpoint-dir", str(checkpoint_relative_dir(environment)),
        "--artifact-path", str(artifact_relative_path(
            args.run_name, environment, mode, seed
        )),
        "--marker-dir", str(marker_relative_dir(
            args.run_name, environment, mode, seed
        )),
    ]
    environment_variables = [
        "MUJOCO_GL=egl",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=.",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "TORCH_NUM_THREADS=1",
        "CUDA_VISIBLE_DEVICES=",
    ]
    return " ".join([*environment_variables, shlex.join(command)])


def build_scheduler_spec(
    args: argparse.Namespace,
    environment: str,
    mode: str,
    seed: int,
) -> dict[str, object]:
    marker = ROOT / marker_relative_dir(
        args.run_name, environment, mode, seed
    )
    return {
        "project": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "description": (
            f"Freq-HRL v18.1 state dataset {environment} {mode} {seed}"
        ),
        "cmd": build_command(args, environment, mode, seed),
        "cwd": str(ROOT),
        "signature": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{args.run_name}/{environment}/"
            f"{mode}/seed-{int(seed)}"
        ),
        "resource_family": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{environment}/path"
        ),
        "vram": 0,
        "ram_mb": 1536,
        "cpu": 1,
        "priority": str(args.priority),
        "ckpt_dir": None,
        "result_dir": str(marker),
        "local_result_dir": str(marker),
        "wait_for_files": [],
        "allow_cpu_training": False,
        "reroute_on_node_down": False,
        "allowed_nodes": [DATA_LOCAL_NODE],
        "require_node": DATA_LOCAL_NODE,
        "stage_excludes": [*STAGE_EXCLUDES, ".server_artifacts"],
        "stage_input_paths": [
            str((ROOT / "scripts").resolve()),
            str((ROOT / "freq_hrl").resolve()),
        ],
        "skip_launch_staging": False,
        "allow_duplicate": False,
    }


def write_registration(args: argparse.Namespace) -> None:
    output = ROOT / "results" / str(args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "registered_before_v18_1_causal_state_export",
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "source_run_name": spec.SOURCE_RUN_NAME,
        "environments": list(args.environments),
        "disturbance_modes": list(args.disturbance_modes),
        "reused_selection_seeds": list(args.evaluation_seeds),
        "path_count": len(selected_paths(args)),
        "trace_keys": list(spec.TRACE_KEYS),
        "target_labels_accessed": False,
        "scheduler_contract": {
            "scheduler": "scheduleurm",
            "require_node": DATA_LOCAL_NODE,
            "binding_reason": (
                "source checkpoints and generated state arrays remain "
                "server-only on node003"
            ),
            "cpu_per_path": 1,
            "ram_mb_per_path": 1536,
            "local_sync": "per-path JSON marker only",
            "slurm_used": False,
        },
        "dataset_contract": spec.DATASET_CONTRACT,
        "claim_boundary": spec.DATASET_CONTRACT["claim_boundary"],
    }
    (output / "dataset_registration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--environments", default=",".join(spec.ENVIRONMENTS))
    parser.add_argument(
        "--disturbance-modes", default=",".join(spec.DISTURBANCE_MODES)
    )
    parser.add_argument(
        "--evaluation-seeds",
        default=",".join(map(str, spec.REUSED_SELECTION_SEEDS)),
    )
    parser.add_argument("--python-executable", default=DEFAULT_LINUX_PYTHON)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="normal"
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.environments = parse_csv(args.environments)
    args.disturbance_modes = parse_csv(args.disturbance_modes)
    args.evaluation_seeds = parse_csv(args.evaluation_seeds, int)
    if not args.environments or not set(args.environments).issubset(
        spec.ENVIRONMENTS
    ):
        raise SystemExit("invalid v18.1 environment subset")
    if not args.disturbance_modes or not set(args.disturbance_modes).issubset(
        spec.DISTURBANCE_MODES
    ):
        raise SystemExit("invalid v18.1 disturbance subset")
    if not args.evaluation_seeds or not set(args.evaluation_seeds).issubset(
        spec.REUSED_SELECTION_SEEDS
    ):
        raise SystemExit("invalid v18.1 reused-seed subset")
    revision, manifest = source_identity(spec.FROZEN_CORE_REVISION)
    if (
        revision != spec.FROZEN_CORE_REVISION
        or manifest != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise SystemExit("v18.1 frozen core identity mismatch")
    write_registration(args)
    paths = selected_paths(args)
    execute_bulk(
        [build_scheduler_spec(args, *path) for path in paths],
        dry_run=bool(args.dry_run),
        intent_label=(
            f"Freq-HRL {spec.DEVELOPMENT_PROTOCOL_VERSION} {args.run_name}"
        ),
    )
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)


if __name__ == "__main__":
    main()
