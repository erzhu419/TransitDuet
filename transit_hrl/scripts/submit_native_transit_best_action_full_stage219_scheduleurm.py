#!/usr/bin/env python3
"""Schedule full-cohort optimal-action sensitivity without new rollouts."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_native_transit_best_action_full_stage219 as spec
from scripts.submit_native_transit_best_action_stage218_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name):
    task = base_task(run_name)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_native_transit_best_action_full_stage219.py",
        "--output", str(ROOT/"results"/run_name/"report/result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL full240-state best-action expressivity sensitivity",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/report", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/diagnostic",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="One240-state view and two future partitions, at most8 single-thread linear programs; zero native episodes.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    if inventory(args.run_name, protocol_spec=spec) or (directory/"preregistration.json").exists():
        raise SystemExit("Full-cohort sensitivity already registered")
    prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != spec.source.EXPERIMENT_PROTOCOL or prior["contract"] != spec.source.contract() or not prior["linear_bounds_qualified"]:
        raise SystemExit("Complete qualified Stage218 before full-cohort sensitivity")
    task = task_specification(args.run_name)
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 1, "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
            "cpu_per_task": task["cpu"], "ram_mb_per_task": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
