#!/usr/bin/env python3
"""Submit the v17.13 grouped causal actor-adapter selector to node003."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v17_13_causal_actor_adapter_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    DEFAULT_LINUX_PYTHON,
    SCHEDULER,
    STAGE_EXCLUDES,
    execute,
    execute_bulk,
    source_identity,
)


WORKER = Path("scripts/train_mujoco_v17_13_causal_actor_adapter.py")
SIGNATURE_VERSION = "mujoco-v17-13-causal-actor-adapter-v1"
DATA_LOCAL_NODE = "node003"
CPU_TRAINING_JUSTIFICATION = (
    "The v17.13 selector performs deterministic NumPy weighted ridge fitting "
    "and SciPy convex responsibility oracles on server-only node003 paths; it "
    "does not train a neural network."
)


def build_command(args: argparse.Namespace) -> str:
    command = [
        str(args.python_executable),
        "-u",
        str(WORKER),
        "--dataset-root",
        str(Path(".server_artifacts") / spec.SOURCE_DATASET_RUN / "paths"),
        "--target-root",
        str(
            Path(".server_artifacts")
            / spec.SOURCE_TARGET_RUN
            / "actor_floor_targets"
        ),
        "--output-dir",
        str(Path("results") / args.run_name),
        "--oracle-workers",
        str(int(args.workers)),
    ]
    environment_variables = [
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


def build_scheduler_spec(args: argparse.Namespace) -> dict[str, object]:
    output = ROOT / "results" / str(args.run_name)
    return {
        "project": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "description": "Freq-HRL v17.13 grouped causal actor adapter",
        "cmd": build_command(args),
        "cwd": str(ROOT),
        "signature": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{args.run_name}/selection"
        ),
        "resource_family": f"Freq-HRL/{SIGNATURE_VERSION}/selection",
        "vram": 0,
        "ram_mb": int(args.ram_mb),
        "cpu": int(args.cpu),
        "priority": str(args.priority),
        "ckpt_dir": None,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [],
        "allow_cpu_training": True,
        "cpu_training_justification": CPU_TRAINING_JUSTIFICATION,
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


def _write_registration(args: argparse.Namespace) -> None:
    output = ROOT / "results" / str(args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection_registration.json").write_text(
        json.dumps({
            "status": "registered_before_v17_13_target_access",
            "evidence_role": spec.EVIDENCE_ROLE,
            "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
            "frozen_core_revision": spec.FROZEN_CORE_REVISION,
            "frozen_source_manifest_sha256": (
                spec.FROZEN_SOURCE_MANIFEST_SHA256
            ),
            "source_dataset_run": spec.SOURCE_DATASET_RUN,
            "source_target_run": spec.SOURCE_TARGET_RUN,
            "candidate_count": (
                len(spec.FIR_WINDOWS)
                * len(spec.RIDGE_PENALTIES)
                * len(spec.ACTOR_FLOOR_PATH_WEIGHTS)
                * len(spec.OUTPUT_GAINS)
                * len(spec.CORRECTION_ABS_LIMITS)
            ),
            "selection_contract": spec.SELECTION_CONTRACT,
            "scheduler_contract": {
                "scheduler": "scheduleurm",
                "require_node": DATA_LOCAL_NODE,
                "binding_reason": (
                    "v17.8 paths and v17.12 targets remain server-only on node003"
                ),
                "cpu": int(args.cpu),
                "oracle_workers": int(args.workers),
                "ram_mb": int(args.ram_mb),
                "slurm_used": False,
            },
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--python-executable", default=DEFAULT_LINUX_PYTHON)
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="normal"
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.cpu < 1
        or args.workers < 1
        or args.workers > args.cpu
        or args.ram_mb < 2048
    ):
        raise SystemExit("v17.13 selection resources must be positive")
    revision, manifest = source_identity(spec.FROZEN_CORE_REVISION)
    if (
        revision != spec.FROZEN_CORE_REVISION
        or manifest != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise SystemExit("v17.13 frozen core identity mismatch")
    _write_registration(args)
    execute_bulk(
        [build_scheduler_spec(args)],
        dry_run=bool(args.dry_run),
        intent_label=(
            f"Freq-HRL {spec.DEVELOPMENT_PROTOCOL_VERSION} {args.run_name}"
        ),
    )
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)


if __name__ == "__main__":
    main()
