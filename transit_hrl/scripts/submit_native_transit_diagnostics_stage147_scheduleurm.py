#!/usr/bin/env python3
"""Schedule frozen native diagnostics with shared server-side checkpoints."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_diagnostics_stage147 as spec
from scripts.submit_native_transit_routing_stage146_scheduleurm import task_specification as routing_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, method, root, *, preflight):
    task = routing_task(run_name, method, root, preflight=preflight)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_diagnostics_stage147.py",
        "--method", method, "--seed", str(root), "--output", str(Path(task["result_dir"]) / "result.json")]
    if preflight:
        command.append("--preflight")
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL frozen native diagnostics {method} seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{method}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'preflight' if preflight else 'diagnostics'}",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        ram_mb=2048, cpu_training_justification="Frozen native evaluation only; server-side final checkpoint reuse, no training or checkpoint downloads.")
    task["stage_input_paths"].append(str(ROOT / "results" / spec.SOURCE_RUN / "cells"))
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = json.loads((ROOT / "results" / spec.SOURCE_RUN / "summary.json").read_text())
    if not source["software_qualified"] or source["optimizer_roots"] != list(spec.ROOTS):
        raise SystemExit("Completed matched native source is required")
    if not args.preflight:
        if args.preflight_result is None:
            raise SystemExit("Frozen diagnostics require their source-reproducing preflight")
        result = json.loads(args.preflight_result.read_text())
        if not (result["software_qualified"] and result["baseline_reproduced"]
                and result["protocol"] == spec.EXPERIMENT_PROTOCOL
                and result["contract"] == spec.contract(True)):
            raise SystemExit("Passive native preflight did not qualify")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Native diagnostics run is already registered")
    pairs = [("correct", spec.ROOTS[0])] if args.preflight else [
        (method, root) for method in spec.METHODS for root in spec.ROOTS]
    tasks = [task_specification(args.run_name, method, root, preflight=args.preflight) for method, root in pairs]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(args.preflight),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cells": [{"method": method, "root": root} for method, root in pairs],
        "tasks": len(tasks), "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
            "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": 2048},
    })
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
