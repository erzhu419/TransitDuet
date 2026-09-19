#!/usr/bin/env python3
"""Submit independent multiscale goal-control cells to the Linux CPU pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import multiscale_goal_stage1_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    DEFAULT_LINUX_PYTHON,
    LINUX_CPU_NODES,
    SCHEDULER,
    STAGE_EXCLUDES,
    execute_bulk,
)


CPU_JUSTIFICATION = (
    "Small independent PPO tracking cells use single-thread CPU networks and "
    "are dynamically distributed across the physical-core Linux pool."
)


def cell_relative_dir(
    run_name: str,
    scenario: str,
    method: str,
    optimizer_seed: int,
) -> Path:
    return (
        Path("results")
        / run_name
        / "cells"
        / scenario
        / method
        / f"replicate_{int(optimizer_seed)}"
    )


def training_command(
    run_name: str,
    cell: tuple[str, str, int],
    *,
    preflight: bool,
) -> str:
    scenario, method, optimizer_seed = cell
    options = spec.cell_options(optimizer_seed, preflight=preflight)
    output = cell_relative_dir(
        run_name, scenario, method, optimizer_seed
    ) / "result.json"
    command = [
        DEFAULT_LINUX_PYTHON,
        "-u",
        "scripts/run_multiscale_goal_stage1.py",
        "--methods",
        method,
        "--scenarios",
        scenario,
        "--iterations",
        str(options["iterations"]),
        "--horizon",
        str(options["horizon"]),
        "--optimizer-seed",
        str(optimizer_seed),
        "--checkpoint-evaluation-interval",
        str(options["checkpoint_evaluation_interval"]),
        "--reference-hidden-dim",
        str(options["reference_hidden_dim"]),
        "--learning-rate",
        str(options["learning_rate"]),
        "--dt-seconds",
        str(options["dt_seconds"]),
        "--upper-period-seconds",
        str(options["upper_period_seconds"]),
        "--history-seconds",
        str(options["history_seconds"]),
        "--fast-period-seconds",
        str(options["fast_period_seconds"]),
        "--train-seeds",
        *map(str, options["train"]),
        "--selection-seeds",
        *map(str, options["selection"]),
        "--eval-seeds",
        *map(str, options["evaluation"]),
        "--output",
        str(output),
    ]
    environment = (
        "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "
        "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        "TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="
    )
    return (
        environment
        + " "
        + shlex.join(command)
        + " && printf '%s\\n' 'complete: result.json written'"
    )


def task_specification(
    run_name: str,
    cell: tuple[str, str, int],
    *,
    preflight: bool,
) -> dict[str, object]:
    scenario, method, optimizer_seed = cell
    relative = cell_relative_dir(
        run_name, scenario, method, optimizer_seed
    )
    phase = "preflight" if preflight else "development"
    return {
        "project": spec.PROTOCOL,
        "description": (
            f"Freq-HRL stage1 {phase} {scenario} {method} "
            f"root{optimizer_seed}"
        ),
        "cmd": training_command(run_name, cell, preflight=preflight),
        "cwd": str(ROOT),
        "signature": (
            f"Freq-HRL/{spec.PROTOCOL}/{run_name}/{scenario}/"
            f"{method}/{optimizer_seed}"
        ),
        "resource_family": f"Freq-HRL/{spec.PROTOCOL}/cell",
        "cpu": 1,
        "ram_mb": 1536,
        "vram": 0,
        "priority": "normal",
        "allowed_nodes": list(LINUX_CPU_NODES),
        "require_node": None,
        "allow_cpu_training": True,
        "cpu_training_justification": CPU_JUSTIFICATION,
        "allow_no_ckpt": True,
        "allow_no_resume": True,
        "result_dir": str(ROOT / relative),
        "local_result_dir": str(ROOT / relative),
        "stage_input_paths": [
            str(ROOT / "scripts"),
            str(ROOT / "freq_hrl"),
        ],
        "stage_excludes": list(STAGE_EXCLUDES),
        "allow_duplicate": False,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 300,
    }


def _inventory(run_name: str) -> list[dict[str, object]]:
    process = subprocess.run(
        [
            sys.executable,
            str(SCHEDULER),
            "results",
            "--signature",
            f"Freq-HRL/{spec.PROTOCOL}/{run_name}/*",
            "--status",
            "queued",
            "launching",
            "running",
            "done",
            "failed",
            "cancelled",
            "--limit",
            "0",
            "--include-empty",
            "--no-log-scan",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(process.stdout[process.stdout.index("{"):])
    return list(payload.get("results", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            spec.ALGORITHM_REVISION,
            "--",
            "freq_hrl",
            "scripts/run_multiscale_goal_stage1.py",
        ],
        cwd=ROOT,
        check=True,
    )
    if _inventory(args.run_name):
        raise SystemExit("run already registered; inspect it before resubmission")
    cells = spec.cells(preflight=args.preflight)
    run_directory = ROOT / "results" / args.run_name
    if any(run_directory.glob("cells/**/result.json")):
        raise SystemExit("run already contains completed cells")
    run_directory.mkdir(parents=True, exist_ok=True)
    registration = {
        "protocol": spec.PROTOCOL,
        "algorithm_revision": spec.ALGORITHM_REVISION,
        "preflight": bool(args.preflight),
        "methods": list(spec.METHODS),
        "scenarios": list(spec.SCENARIOS),
        "optimizer_seeds": (
            list(spec.OPTIMIZER_SEEDS[:1])
            if args.preflight else list(spec.OPTIMIZER_SEEDS)
        ),
        "cells": [list(cell) for cell in cells],
        "options": {
            str(seed): spec.cell_options(seed, preflight=args.preflight)
            for seed in (
                spec.OPTIMIZER_SEEDS[:1]
                if args.preflight else spec.OPTIMIZER_SEEDS
            )
        },
        "scheduler": {
            "nodes": list(LINUX_CPU_NODES),
            "require_node": None,
            "cpu_per_cell": 1,
            "ram_mb_per_cell": 1536,
        },
        "artifacts": {
            "synced": ["result.json"],
            "checkpoints": "disabled",
        },
    }
    registration_path = run_directory / "preregistration.json"
    registration_path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tasks = [
        task_specification(
            args.run_name,
            cell,
            preflight=args.preflight,
        )
        for cell in cells
    ]
    execute_bulk(
        tasks,
        dry_run=args.dry_run,
        intent_label=f"{spec.PROTOCOL}:{args.run_name}",
    )
    if not args.dry_run:
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
