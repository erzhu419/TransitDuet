#!/usr/bin/env python3
"""Submit the server-local v17.5 feasibility diagnostic through scheduleurm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4,
)
from scripts.run_mujoco_v17_5_feasibility_diagnostic import (  # noqa: E402
    DIAGNOSTIC_PROTOCOL_VERSION,
    EVIDENCE_ROLE,
)
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    DEFAULT_LINUX_PYTHON,
    SCHEDULER,
    STAGE_EXCLUDES,
    execute,
    execute_bulk,
    source_identity,
)


WORKER = Path("scripts/run_mujoco_v17_5_feasibility_diagnostic.py")
SIGNATURE_VERSION = "mujoco-v17-5-feasibility-diagnostic-v1"
DATA_LOCAL_NODE = "node003"


def checkpoint_relative_dir(
    source_run_name: str,
    environment: str,
    optimizer_seed: int,
) -> Path:
    return (
        Path(".server_artifacts")
        / str(source_run_name)
        / "cells"
        / str(environment)
        / f"replicate_{int(optimizer_seed)}"
    )


def output_relative_dir(
    run_name: str,
    environment: str,
    optimizer_seed: int,
) -> Path:
    return (
        Path("results")
        / str(run_name)
        / "cells"
        / str(environment)
        / f"replicate_{int(optimizer_seed)}"
    )


def build_command(
    args: argparse.Namespace,
    environment: str,
    optimizer_seed: int,
) -> str:
    command = [
        str(args.python_executable),
        "-u",
        str(WORKER),
        "--env-id",
        str(environment),
        "--optimizer-seed",
        str(optimizer_seed),
        "--checkpoint-dir",
        str(checkpoint_relative_dir(
            args.source_run_name, environment, optimizer_seed
        )),
        "--output-dir",
        str(output_relative_dir(args.run_name, environment, optimizer_seed)),
        "--diagnostic-code-revision",
        str(args.code_revision),
        "--diagnostic-source-manifest-sha256",
        str(args.source_manifest_sha256),
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
    optimizer_seed: int,
) -> dict[str, object]:
    checkpoint = ROOT / checkpoint_relative_dir(
        args.source_run_name, environment, optimizer_seed
    )
    output = ROOT / output_relative_dir(
        args.run_name, environment, optimizer_seed
    )
    return {
        "project": DIAGNOSTIC_PROTOCOL_VERSION,
        "description": f"Freq-HRL v17.5 diagnostic {environment}",
        "cmd": build_command(args, environment, optimizer_seed),
        "cwd": str(ROOT),
        "signature": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{args.run_name}/{environment}/"
            f"rep-{int(optimizer_seed)}"
        ),
        "resource_family": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{environment}/replay"
        ),
        "vram": 0,
        "ram_mb": 1024,
        "cpu": 1,
        "priority": str(args.priority),
        "ckpt_dir": str(checkpoint),
        "ckpt_glob": "checkpoint.pt",
        "result_dir": str(output),
        "local_result_dir": str(output),
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


def _write_registration(args: argparse.Namespace) -> None:
    output = ROOT / "results" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "registered_before_diagnostic_replay",
        "evidence_role": EVIDENCE_ROLE,
        "diagnostic_protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "diagnostic_code_revision": args.code_revision,
        "diagnostic_source_manifest_sha256": args.source_manifest_sha256,
        "source_run_name": args.source_run_name,
        "source_checkpoint_protocol_version": (
            v17_4.FROZEN_CORE_PROTOCOL_VERSION
        ),
        "source_checkpoint_algorithm_revision": (
            v17_4.FROZEN_ALGORITHM_REVISION
        ),
        "environments": list(args.environments),
        "optimizer_seeds": list(v17_4.OPTIMIZER_SEEDS),
        "scheduler_contract": {
            "scheduler": "scheduleurm",
            "require_node": DATA_LOCAL_NODE,
            "binding_reason": (
                "the reused checkpoint exists only on node003 and remains "
                "server-only"
            ),
            "cpu_per_task": 1,
            "ram_mb_per_task": 1024,
            "slurm_used": False,
        },
        "artifact_contract": {
            "synced": ["diagnostic_summary.json", "diagnostic_rows.csv"],
            "server_only": ["checkpoint.pt", "training_history.json"],
        },
        "decision_rule": (
            "continue projection work only when avoidable budget regret exceeds "
            "1e-7; otherwise move to learned-policy feasibility constraints"
        ),
    }
    (output / "diagnostic_registration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--source-run-name",
        default="mujoco_v17_4_streaming_audit_projection_preflight_20260831_r1",
    )
    parser.add_argument(
        "--environments", default=",".join(v17_4.ENVIRONMENTS)
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
    args.environments = [
        value.strip() for value in args.environments.split(",") if value.strip()
    ]
    if not args.environments or not set(args.environments).issubset(
        v17_4.ENVIRONMENTS
    ):
        raise SystemExit("invalid v17.5 diagnostic environment subset")
    args.code_revision, args.source_manifest_sha256 = source_identity()
    _write_registration(args)
    seed = v17_4.OPTIMIZER_SEEDS[0]
    specs = [
        build_scheduler_spec(args, environment, seed)
        for environment in args.environments
    ]
    execute_bulk(
        specs,
        dry_run=bool(args.dry_run),
        intent_label=f"Freq-HRL {DIAGNOSTIC_PROTOCOL_VERSION} {args.run_name}",
    )
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)


if __name__ == "__main__":
    main()
