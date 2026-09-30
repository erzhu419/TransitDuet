#!/usr/bin/env python3
"""Dynamically place a 2-CPU offline diagnosis without pulling native traces."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import pointmaze_plan_error_stage53_spec as spec
from scripts.submit_pointmaze_plan_alignment_stage52_scheduleurm import task_specification as source_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name):
    task = source_task(run_name, 310001, preflight=True)
    task.update(project=spec.EXPERIMENT_PROTOCOL, stage_input_paths=[],
        description="Stage53 recorded native forecast/control diagnosis",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "
            + shlex.join([DEFAULT_LINUX_PYTHON, "-B", "scripts/analyze_pointmaze_plan_error_stage53.py", "--run-name", run_name]),
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/offline",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/analysis",
        cpu_training_justification="Offline NPZ decomposition and paired bootstrap; no environment or optimizer steps.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Stage53 run already registered")
    task = task_specification(args.run_name)
    write_json(spec.ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "budget": spec.budget(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=spec.ROOT, text=True).strip(),
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
                      "cpu": task["cpu"], "ram_mb": task["ram_mb"]}, "artifacts": "compact_json_only_raw_stays_remote"})
    execute_bulk([task], dry_run=args.dry_run, intent_label=args.run_name)
    if not args.dry_run:
        found = inventory(args.run_name, protocol_spec=spec)
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch", "--task-id", found[0]["id"]], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
