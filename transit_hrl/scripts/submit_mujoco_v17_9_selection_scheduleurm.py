#!/usr/bin/env python3
"""Submit one v17.9 prefix-HPF FIR selection task to node003."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v17_9_prefix_hpf_fir_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    DEFAULT_LINUX_PYTHON,
    SCHEDULER,
    STAGE_EXCLUDES,
    execute,
    execute_bulk,
    source_identity,
)


WORKER = Path("scripts/train_mujoco_v17_9_prefix_hpf_fir.py")
SIGNATURE_VERSION = "mujoco-v17-9-prefix-hpf-fir-selection-v1"
DATA_LOCAL_NODE = "node003"
CPU_JUSTIFICATION = (
    "The v17.9 selector performs deterministic NumPy ridge solves and causal "
    "array projections over server-local paths; it has no GPU workload."
)


def build_command(args: argparse.Namespace) -> str:
    command = [
        str(args.python_executable),
        "-u",
        str(WORKER),
        "--dataset-root",
        str(Path(".server_artifacts") / args.dataset_run_name / "paths"),
        "--output-dir",
        str(Path("results") / args.run_name),
    ]
    environment_variables = [
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=.",
        f"OMP_NUM_THREADS={int(args.cpu)}",
        f"MKL_NUM_THREADS={int(args.cpu)}",
        f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
        f"NUMEXPR_NUM_THREADS={int(args.cpu)}",
        "TORCH_NUM_THREADS=1",
        "CUDA_VISIBLE_DEVICES=",
    ]
    return " ".join([*environment_variables, shlex.join(command)])


def build_scheduler_spec(args: argparse.Namespace) -> dict[str, object]:
    output = ROOT / "results" / str(args.run_name)
    return {
        "project": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "description": "Freq-HRL v17.9 grouped prefix-HPF FIR selection",
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
        "cpu_training_justification": CPU_JUSTIFICATION,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--python-executable", default=DEFAULT_LINUX_PYTHON)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="normal"
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cpu < 1 or args.ram_mb < 1024:
        raise SystemExit("v17.9 selection resources must be positive")
    revision, manifest = source_identity(spec.FROZEN_CORE_REVISION)
    if (
        revision != spec.FROZEN_CORE_REVISION
        or manifest != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise SystemExit("v17.9 frozen core identity mismatch")
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
