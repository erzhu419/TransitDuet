#!/usr/bin/env python3
"""Submit equal-shape PointMaze routing-attribution cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pointmaze_routing_stage4_v2_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    LINUX_CPU_NODES,
    SCHEDULER,
    execute_bulk,
)
from scripts import submit_pointmaze_routing_stage4_scheduleurm as base  # noqa: E402


RUNNER_SCRIPT = "scripts/run_pointmaze_routing_stage4_v2.py"


def training_command(
    run_name: str,
    cell: tuple[str, str, int],
    *,
    preflight: bool,
) -> str:
    return base.training_command(
        run_name,
        cell,
        preflight=preflight,
        protocol_spec=spec,
        runner_script=RUNNER_SCRIPT,
    )


def task_specification(
    run_name: str,
    cell: tuple[str, str, int],
    *,
    preflight: bool,
) -> dict[str, object]:
    return base.task_specification(
        run_name,
        cell,
        preflight=preflight,
        protocol_spec=spec,
        runner_script=RUNNER_SCRIPT,
        stage_label="stage4-v2",
    )


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
            RUNNER_SCRIPT,
        ],
        cwd=ROOT,
        check=True,
    )
    if args.sync_results:
        if args.dry_run:
            raise SystemExit("--sync-results cannot be combined with --dry-run")
        base.sync_results(
            args.run_name,
            preflight=args.preflight,
            workers=args.sync_workers,
            protocol_spec=spec,
        )
        return 0
    if base._inventory(args.run_name, protocol=spec.PROTOCOL):
        raise SystemExit("run already registered; inspect it before resubmission")
    cells = spec.cells(preflight=args.preflight)
    run_directory = ROOT / "results" / args.run_name
    if any(run_directory.glob("cells/**/result.json")):
        raise SystemExit("run already contains completed cells")
    run_directory.mkdir(parents=True, exist_ok=True)
    optimizer_seeds = (
        spec.OPTIMIZER_SEEDS[:1] if args.preflight else spec.OPTIMIZER_SEEDS
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
        "routing_shape_contract": (
            "identical_upper_lower_state_shapes_parameters_and_"
            "initialization_per_root"
        ),
        "claim_gate": {
            "routed_vs_all_success": "positive_in_both_primary_stresses",
            "routed_vs_swapped_success": "positive_in_both_primary_stresses",
            "observation_routed_vs_filter_success": "positive",
            "clean_routed_vs_all_success": "noninferior_margin_0.10",
        },
        "scheduler": {
            "nodes": list(LINUX_CPU_NODES),
            "require_node": None,
            "cpu_per_cell": 1,
            "ram_mb_per_cell": 2560,
        },
        "artifacts": {"synced": ["result.json"], "checkpoints": "disabled"},
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
