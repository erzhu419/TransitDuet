#!/usr/bin/env python3
"""Dispatch the deployed-state paired timing diagnostic via scheduleurm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pointmaze_deployed_pair_diagnostic_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import (
    LINUX_CPU_NODES,
    SCHEDULER,
    execute_bulk,
)
from scripts.submit_pointmaze_budgeted_trigger_stage9_scheduleurm import (
    cell_relative_dir,
    task_specification as base_task_specification,
)
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import (
    inventory,
    sync_results,
    task_signature,
)


def task_specification(run_name: str, root: int, *, preflight: bool) -> dict:
    source = spec.source_result(root, preflight=preflight)
    if not source.is_file():
        raise FileNotFoundError(source)
    task = base_task_specification(
        run_name, root, preflight=preflight, confirmation=not preflight,
        runner_script=spec.RUNNER_SCRIPT,
        extra_args=(
            "--source-result", str(source),
            "--pairs-per-class", "1" if preflight else "2",
        ),
    )
    task.update({
        "project": spec.EXPERIMENT_PROTOCOL,
        "description": f"Freq-HRL deployed-state timing diagnostic root{root}",
        "signature": task_signature(run_name, root, protocol_spec=spec),
        "resource_family": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/cell",
        "stage_input_paths": [*task["stage_input_paths"], str(source.parent)],
    })
    return task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-results", action="store_true")
    args = parser.parse_args()
    if args.sync_results:
        sync_results(
            args.run_name, preflight=args.preflight, workers=2,
            protocol_spec=spec,
        )
        return 0
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Stage-13 run already registered")
    roots = spec.roots(preflight=args.preflight)
    tasks = [task_specification(args.run_name, root, preflight=args.preflight)
             for root in roots]
    run_dir = ROOT / "results" / args.run_name
    if any(run_dir.glob("cells/**/result.json")):
        raise SystemExit("Stage-13 run already has completed cells")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preregistration.json").write_text(
        json.dumps({
            "protocol": spec.EXPERIMENT_PROTOCOL,
            "evidence_role": "mechanism_diagnostic_only",
            "source_protocol": "pointmaze_timing_pair_stage12_v1_development",
            "preflight": args.preflight,
            "optimizer_roots": list(roots),
            "pairs_per_class": 1 if args.preflight else 2,
            "continuation": "static_factual_schedule_after_one_bin_flip",
            "scheduler": {
                "nodes": list(LINUX_CPU_NODES), "require_node": None,
                "cpu_per_cell": 1, "ram_mb_per_cell": 1536,
            },
            "artifacts": {"synced": ["result.json"], "checkpoints": "disabled"},
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    execute_bulk(
        tasks, dry_run=args.dry_run,
        intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}",
    )
    if not args.dry_run:
        subprocess.run(["python3", str(SCHEDULER), "dispatch"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
