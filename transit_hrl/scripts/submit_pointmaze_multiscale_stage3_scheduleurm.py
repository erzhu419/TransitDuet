#!/usr/bin/env python3
"""Submit PointMaze stage-3 factorial cells to the Linux CPU pool."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pointmaze_multiscale_stage3_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    DEFAULT_LINUX_PYTHON,
    LINUX_CPU_NODES,
    SCHEDULER,
    STAGE_EXCLUDES,
    execute_bulk,
)


CPU_JUSTIFICATION = (
    "Independent PointMaze factorial cells use single-thread CPU policies and "
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


def task_signature(
    run_name: str,
    scenario: str,
    method: str,
    optimizer_seed: int,
) -> str:
    return (
        f"Freq-HRL/{spec.PROTOCOL}/{run_name}/{scenario}/"
        f"{method}/{int(optimizer_seed)}"
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
        "scripts/run_pointmaze_multiscale_stage3.py",
        "--methods",
        method,
        "--scenarios",
        scenario,
        "--env-id",
        str(options["env_id"]),
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
        "--upper-period-seconds",
        str(options["upper_period_seconds"]),
        "--history-seconds",
        str(options["history_seconds"]),
        "--fast-period-seconds",
        str(options["fast_period_seconds"]),
        "--maximum-subgoal-delta",
        str(options["maximum_subgoal_delta"]),
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
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 "
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
            f"Freq-HRL PointMaze stage3 {phase} {scenario} {method} "
            f"root{optimizer_seed}"
        ),
        "cmd": training_command(run_name, cell, preflight=preflight),
        "cwd": str(ROOT),
        "signature": task_signature(
            run_name, scenario, method, optimizer_seed
        ),
        "resource_family": f"Freq-HRL/{spec.PROTOCOL}/cell",
        "cpu": 1,
        "ram_mb": 2560,
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


def sync_results(
    run_name: str,
    *,
    preflight: bool,
    workers: int,
) -> None:
    tasks = {
        str(task["signature"]): task
        for task in _inventory(run_name)
    }
    expected: list[tuple[str, Path, dict[str, object]]] = []
    for scenario, method, optimizer_seed in spec.cells(preflight=preflight):
        signature = task_signature(
            run_name, scenario, method, optimizer_seed
        )
        task = tasks.get(signature)
        if task is None:
            raise SystemExit(f"stage-3 sync task missing: {signature}")
        if task.get("status") != "done" or not task.get("node"):
            raise SystemExit(
                f"stage-3 task is not done: {task.get('id')} "
                f"status={task.get('status')}"
            )
        path = ROOT / cell_relative_dir(
            run_name, scenario, method, optimizer_seed
        )
        expected.append((signature, path, task))

    scheduler_dir = str(SCHEDULER.parent)
    if scheduler_dir not in sys.path:
        sys.path.insert(0, scheduler_dir)
    import scheduler as scheduler_runtime  # type: ignore  # noqa: E402

    pending = [
        item for item in expected
        if not (item[1] / "result.json").is_file()
    ]

    def sync_one(
        item: tuple[str, Path, dict[str, object]],
    ) -> tuple[str, bool, str]:
        signature, path, task = item
        ok, message = scheduler_runtime._sync_one_result({
            "node": task["node"],
            "result_dir": str(path),
            "local_result_dir": str(path),
        })
        return signature, bool(ok), str(message)

    errors: dict[str, str] = {}
    for attempt in range(1, 4):
        if not pending:
            break
        errors = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(workers))
        ) as executor:
            for signature, ok, message in executor.map(sync_one, pending):
                if not ok:
                    errors[signature] = message
        pending = [
            item for item in pending
            if not (item[1] / "result.json").is_file()
        ]
        if pending and attempt < 3:
            time.sleep(float(attempt))
    if pending:
        signature, path, _ = pending[0]
        raise SystemExit(
            f"stage-3 result sync incomplete: {len(pending)} cells; "
            f"first={signature}; path={path}; "
            f"error={errors.get(signature, 'missing result.json')}"
        )

    for signature, path, _ in expected:
        payload = json.loads(
            (path / "result.json").read_text(encoding="utf-8")
        )
        if (
            payload.get("status") != "complete"
            or payload.get("protocol", {}).get("protocol_version")
            != spec.PROTOCOL
            or len(payload.get("cells", [])) != 1
        ):
            raise SystemExit(f"invalid synced result: {signature}")

    manifest = {
        "run_name": str(run_name),
        "protocol": spec.PROTOCOL,
        "algorithm_revision": spec.ALGORITHM_REVISION,
        "cell_count": len(expected),
        "artifact_contract": "result_json_only_v1",
        "nodes": {
            node: sum(str(item[2]["node"]) == node for item in expected)
            for node in sorted({str(item[2]["node"]) for item in expected})
        },
        "tasks": {
            signature: {
                "task_id": task["id"],
                "node": task["node"],
            }
            for signature, _, task in expected
        },
    }
    output = ROOT / "results" / run_name
    (output / "run_scoped_result_sync.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"synced {len(expected)} PointMaze stage-3 result JSON files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-results", action="store_true")
    parser.add_argument("--sync-workers", type=int, default=4)
    args = parser.parse_args()

    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            spec.ALGORITHM_REVISION,
            "--",
            "freq_hrl",
            "scripts/run_pointmaze_multiscale_stage3.py",
        ],
        cwd=ROOT,
        check=True,
    )
    if args.sync_results:
        if args.dry_run:
            raise SystemExit("--sync-results cannot be combined with --dry-run")
        sync_results(
            args.run_name,
            preflight=args.preflight,
            workers=args.sync_workers,
        )
        return 0
    if _inventory(args.run_name):
        raise SystemExit("run already registered; inspect it before resubmission")
    cells = spec.cells(preflight=args.preflight)
    run_directory = ROOT / "results" / args.run_name
    if any(run_directory.glob("cells/**/result.json")):
        raise SystemExit("run already contains completed cells")
    run_directory.mkdir(parents=True, exist_ok=True)
    optimizer_seeds = (
        spec.OPTIMIZER_SEEDS[:1]
        if args.preflight else spec.OPTIMIZER_SEEDS
    )
    registration = {
        "protocol": spec.PROTOCOL,
        "algorithm_revision": spec.ALGORITHM_REVISION,
        "evidence_stage": "preflight" if args.preflight else "development",
        "preflight": bool(args.preflight),
        "methods": list(spec.METHODS),
        "scenarios": list(spec.SCENARIOS),
        "optimizer_seeds": list(optimizer_seeds),
        "runtime_expectations": dict(spec.RUNTIME_EXPECTATIONS),
        "cells": [list(cell) for cell in cells],
        "options": {
            str(seed): spec.cell_options(seed, preflight=args.preflight)
            for seed in optimizer_seeds
        },
        "claim_gate": {
            "primary_stress": "mixed_causal_stress",
            "primary_endpoint": "success",
            "clean_success_noninferiority_margin": 0.10,
            "requires_positive_factorial_interaction": True,
        },
        "scheduler": {
            "nodes": list(LINUX_CPU_NODES),
            "require_node": None,
            "cpu_per_cell": 1,
            "ram_mb_per_cell": 2560,
        },
        "artifacts": {
            "synced": ["result.json"],
            "checkpoints": "disabled",
        },
    }
    (run_directory / "preregistration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tasks = [
        task_specification(args.run_name, cell, preflight=args.preflight)
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
