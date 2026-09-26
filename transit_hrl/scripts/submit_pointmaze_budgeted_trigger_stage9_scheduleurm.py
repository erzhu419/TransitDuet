#!/usr/bin/env python3
"""Submit the frozen Stage-9 PointMaze trigger matrix to scheduleurm."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pointmaze_budgeted_trigger_stage9_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    DEFAULT_LINUX_PYTHON,
    LINUX_CPU_NODES,
    SCHEDULER,
    STAGE_EXCLUDES,
    execute_bulk,
)
from scripts.submit_pointmaze_multiscale_stage3_scheduleurm import (  # noqa: E402
    _inventory_by_signature,
)


def cell_relative_dir(run_name: str, root: int) -> Path:
    return Path("results") / run_name / "cells" / spec.POLICY / f"replicate_{root}"


def task_signature(run_name: str, root: int, *, confirmation: bool = False) -> str:
    protocol = (
        spec.CONFIRMATION_EXPERIMENT_PROTOCOL
        if confirmation else spec.EXPERIMENT_PROTOCOL
    )
    return f"Freq-HRL/{protocol}/{run_name}/{spec.POLICY}/{root}"


def training_command(run_name: str, root: int, *, preflight: bool) -> str:
    options = spec.cell_options(root, preflight=preflight)
    output = cell_relative_dir(run_name, root) / "result.json"
    command = [
        DEFAULT_LINUX_PYTHON,
        "-u",
        spec.RUNNER_SCRIPT,
        "--env-id", str(options["env_id"]),
        "--iterations", str(options["iterations"]),
        "--horizon", str(options["horizon"]),
        "--optimizer-seed", str(root),
        "--checkpoint-evaluation-interval", str(options["checkpoint_evaluation_interval"]),
        "--reference-hidden-dim", str(options["reference_hidden_dim"]),
        "--learning-rate", str(options["learning_rate"]),
        "--upper-period-seconds", str(options["upper_period_seconds"]),
        "--history-seconds", str(options["history_seconds"]),
        "--fast-period-seconds", str(options["fast_period_seconds"]),
        "--maximum-subgoal-delta", str(options["maximum_subgoal_delta"]),
        "--branch-window-seconds", str(options["branch_window_seconds"]),
        "--max-events-per-class", str(options["max_events_per_class"]),
        "--ridge-alpha-grid", *map(str, options["ridge_alpha_grid"]),
        "--threshold-quantile", str(options["threshold_quantile"]),
        "--max-offset-steps", str(options["max_offset_steps"]),
        "--check-stride-steps", str(options["check_stride_steps"]),
        "--regime-dwell-seconds", *map(str, options["regime_dwell_seconds"]),
        "--target-speed-modes", *map(str, options["target_speed_modes"]),
        "--force-pulse-amplitude", str(options["force_pulse_amplitude"]),
        "--force-pulse-duration-seconds", *map(str, options["force_pulse_duration_seconds"]),
        "--force-pulse-gap-seconds", *map(str, options["force_pulse_gap_seconds"]),
        "--distractor-amplitude", str(options["distractor_amplitude"]),
        "--distractor-dwell-seconds", *map(str, options["distractor_dwell_seconds"]),
        "--train-seeds", *map(str, options["train"]),
        "--selection-seeds", *map(str, options["selection"]),
        "--branch-fit-seeds", *map(str, options["branch_fit"]),
        "--trigger-eval-seeds", *map(str, options["trigger_eval"]),
        "--output", str(output),
    ]
    environment = (
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 "
        "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        "TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="
    )
    return environment + " " + shlex.join(command) + (
        " && printf '%s\\n' 'complete: result.json written'"
    )


def task_specification(
    run_name: str, root: int, *, preflight: bool, confirmation: bool = False
) -> dict[str, object]:
    relative = cell_relative_dir(run_name, root)
    phase = (
        "confirmation" if confirmation else
        "preflight" if preflight else "development"
    )
    protocol = (
        spec.CONFIRMATION_EXPERIMENT_PROTOCOL
        if confirmation else spec.EXPERIMENT_PROTOCOL
    )
    return {
        "project": protocol,
        "description": f"Freq-HRL PointMaze stage9 {phase} {spec.POLICY} root{root}",
        "cmd": training_command(run_name, root, preflight=preflight),
        "cwd": str(ROOT),
        "signature": task_signature(run_name, root, confirmation=confirmation),
        "resource_family": f"Freq-HRL/{protocol}/cell",
        "cpu": 1,
        "ram_mb": 1536,
        "vram": 0,
        "priority": "normal",
        "allowed_nodes": list(LINUX_CPU_NODES),
        "require_node": None,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Independent PointMaze controller roots and paired CPU branch replay "
            "are single-threaded tasks dynamically placed on the Linux pool."
        ),
        "allow_no_ckpt": True,
        "allow_no_resume": True,
        "result_dir": str(ROOT / relative),
        "local_result_dir": str(ROOT / relative),
        "stage_input_paths": [str(ROOT / "scripts"), str(ROOT / "freq_hrl")],
        "stage_excludes": list(STAGE_EXCLUDES),
        "allow_duplicate": False,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 300,
    }


def _inventory(run_name: str, *, confirmation: bool = False) -> list[dict[str, object]]:
    protocol = (
        spec.CONFIRMATION_EXPERIMENT_PROTOCOL
        if confirmation else spec.EXPERIMENT_PROTOCOL
    )
    process = subprocess.run(
        [
            sys.executable, str(SCHEDULER), "results", "--signature",
            f"Freq-HRL/{protocol}/{run_name}/*",
            "--status", "queued", "launching", "running", "done", "failed",
            "cancelled", "--limit", "0", "--include-empty", "--no-log-scan", "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(process.stdout[process.stdout.index("{"):])
    return list(payload.get("results", []))


def sync_results(
    run_name: str, *, preflight: bool, confirmation: bool = False, workers: int
) -> None:
    tasks = _inventory_by_signature(_inventory(run_name, confirmation=confirmation))
    expected = []
    for _, root in spec.cells(preflight=preflight, confirmation=confirmation):
        signature = task_signature(run_name, root, confirmation=confirmation)
        task = tasks.get(signature)
        if task is None or task.get("status") != "done" or not task.get("node"):
            raise SystemExit(f"Stage-9 task is not done: {signature}")
        expected.append((signature, ROOT / cell_relative_dir(run_name, root), task))
    scheduler_dir = str(SCHEDULER.parent)
    if scheduler_dir not in sys.path:
        sys.path.insert(0, scheduler_dir)
    import scheduler as scheduler_runtime  # type: ignore  # noqa: E402

    def sync_one(item: tuple[object, Path, dict[str, object]]) -> tuple[str, bool, str]:
        signature, path, task = item
        if (path / "result.json").is_file():
            return str(signature), True, "already present"
        ok, message = scheduler_runtime._sync_one_result({
            "node": task["node"],
            "result_dir": str(path),
            "local_result_dir": str(path),
        })
        return str(signature), bool(ok), str(message)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(sync_one, expected))
    failures = [item for item in results if not item[1]]
    if failures:
        raise SystemExit(f"Stage-9 compact result sync failed: {failures}")
    for signature, path, _ in expected:
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        if (
            result.get("status") != "complete"
            or result.get("protocol", {}).get("protocol_version") != spec.PROTOCOL
            or len(result.get("cells", [])) != 1
        ):
            raise SystemExit(f"invalid Stage-9 result: {signature}")
    print(f"synced {len(expected)} Stage-9 result JSON files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-results", action="store_true")
    parser.add_argument("--sync-workers", type=int, default=4)
    args = parser.parse_args()
    if args.preflight and args.confirmation:
        parser.error("preflight and confirmation are distinct phases")
    subprocess.run(
        ["git", "diff", "--exit-code", spec.ALGORITHM_REVISION, "--",
         "freq_hrl", spec.RUNNER_SCRIPT],
        cwd=ROOT,
        check=True,
    )
    if args.sync_results:
        sync_results(
            args.run_name, preflight=args.preflight,
            confirmation=args.confirmation, workers=args.sync_workers,
        )
        return 0
    if _inventory(args.run_name, confirmation=args.confirmation):
        raise SystemExit("Stage-9 run already registered")
    run_dir = ROOT / "results" / args.run_name
    if any(run_dir.glob("cells/**/result.json")):
        raise SystemExit("Stage-9 run already contains completed cells")
    run_dir.mkdir(parents=True, exist_ok=True)
    roots = (
        spec.CONFIRMATION_OPTIMIZER_SEEDS if args.confirmation else
        spec.PREFLIGHT_OPTIMIZER_SEEDS if args.preflight else spec.OPTIMIZER_SEEDS
    )
    protocol = (
        spec.CONFIRMATION_EXPERIMENT_PROTOCOL
        if args.confirmation else spec.EXPERIMENT_PROTOCOL
    )
    registration = {
        "protocol": protocol,
        "runtime_protocol": spec.PROTOCOL,
        "algorithm_revision": spec.ALGORITHM_REVISION,
        "preflight": bool(args.preflight),
        "confirmation": bool(args.confirmation),
        "optimizer_seeds": list(roots),
        "runtime_expectations": spec.RUNTIME_EXPECTATIONS,
        "cells": [list(cell) for cell in spec.cells(
            preflight=args.preflight, confirmation=args.confirmation,
        )],
        "options": {
            str(root): spec.cell_options(root, preflight=args.preflight)
            for root in roots
        },
        "claim_gate": (
            spec.CONFIRMATION_CLAIM_GATE if args.confirmation else spec.CLAIM_GATE
        ),
        "scheduler": {
            "nodes": list(LINUX_CPU_NODES), "require_node": None,
            "cpu_per_cell": 1, "ram_mb_per_cell": 1536,
        },
        "artifacts": {"synced": ["result.json"], "checkpoints": "disabled"},
    }
    (run_dir / "preregistration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    execute_bulk(
        [task_specification(
            args.run_name, root, preflight=args.preflight,
            confirmation=args.confirmation,
        ) for _, root in spec.cells(
            preflight=args.preflight, confirmation=args.confirmation,
        )],
        dry_run=args.dry_run,
        intent_label=f"{protocol}:{args.run_name}",
    )
    if not args.dry_run:
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
