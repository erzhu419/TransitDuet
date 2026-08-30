#!/usr/bin/env python3
"""Submit the v17.14 exhaustive actor-adapter oracle audit to node003."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v17_14_exhaustive_actor_oracle_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    DEFAULT_LINUX_PYTHON,
    SCHEDULER,
    STAGE_EXCLUDES,
    execute,
    execute_bulk,
    source_identity,
)


WORKER = Path("scripts/run_mujoco_v17_14_exhaustive_actor_oracle.py")
SIGNATURE_VERSION = "mujoco-v17-14-exhaustive-actor-oracle-v1"
DATA_LOCAL_NODE = "node003"
CPU_TRAINING_JUSTIFICATION = (
    "The v17.14 audit performs deterministic cached NumPy ridge reconstruction "
    "and exact SciPy convex responsibility oracles for the unexamined frozen "
    "grid; it does not train a neural network."
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
        "--v17-13-summary",
        str(
            Path("results")
            / spec.SOURCE_V17_13_SELECTION_RUN
            / "selection_summary.json"
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
        "description": "Freq-HRL v17.14 exhaustive actor oracle",
        "cmd": build_command(args),
        "cwd": str(ROOT),
        "signature": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{args.run_name}/exhaustive"
        ),
        "resource_family": f"Freq-HRL/{SIGNATURE_VERSION}/exhaustive",
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
    (output / "audit_registration.json").write_text(
        json.dumps({
            "status": "registered_before_v17_14_remainder_oracle_access",
            "evidence_role": spec.EVIDENCE_ROLE,
            "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
            "frozen_core_revision": spec.FROZEN_CORE_REVISION,
            "frozen_source_manifest_sha256": (
                spec.FROZEN_SOURCE_MANIFEST_SHA256
            ),
            "source_dataset_run": spec.SOURCE_DATASET_RUN,
            "source_target_run": spec.SOURCE_TARGET_RUN,
            "source_v17_13_selection_run": (
                spec.SOURCE_V17_13_SELECTION_RUN
            ),
            "remainder_candidate_count": (
                spec.EXPECTED_REMAINDER_CANDIDATE_COUNT
            ),
            "combined_candidate_count": (
                spec.EXPECTED_FULL_GRID_CANDIDATE_COUNT
            ),
            "selection_contract": spec.SELECTION_CONTRACT,
            "scheduler_contract": {
                "scheduler": "scheduleurm",
                "require_node": DATA_LOCAL_NODE,
                "binding_reason": (
                    "source paths and actor targets remain server-only on node003"
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
    parser.add_argument("--cpu", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--ram-mb", type=int, default=32768)
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
        or args.ram_mb < 4096
    ):
        raise SystemExit("v17.14 audit resources must be positive")
    revision, manifest = source_identity(spec.FROZEN_CORE_REVISION)
    if (
        revision != spec.FROZEN_CORE_REVISION
        or manifest != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise SystemExit("v17.14 frozen core identity mismatch")
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
