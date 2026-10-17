#!/usr/bin/env python3
"""Schedule training-cache abstention versus matched random thinning."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_zero_abstention_stage291 as spec
from scripts.submit_native_transit_fleet_screen_stage290_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run):
    task = base_task(run)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_zero_abstention_stage291.py", "--run-name", run]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL zero-event abstention and count-matched thinning",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/analysis", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Reuse frozen Stage290 actions and Stage288 train futures; exact expected count-matched thinning and prefix inference, no fits, simulations or held access.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = spec.directory(args.run_name)
    if inventory(args.run_name, protocol_spec=spec) or (directory/"preregistration.json").exists():
        raise SystemExit("Stage291 already registered")
    spec.qualify_parent()
    task = task_specification(args.run_name)
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None, "cpu_per_task": 1, "ram_mb": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
