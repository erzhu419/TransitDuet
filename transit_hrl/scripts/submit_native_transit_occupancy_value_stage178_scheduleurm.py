#!/usr/bin/env python3
"""Four matched CPU fits; no new native episodes or local training jobs."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_occupancy_value_stage178 as spec
from scripts.submit_native_transit_action_value_stage176_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, method):
    task = base_task(run_name)
    directory = ROOT/"results"/run_name
    result = directory/"cells"/method
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_occupancy_value_stage178.py",
        "--method", method, "--data", str(directory/"inputs/training.json"),
        "--labels", str(directory/"inputs/labels.json"), "--output", str(result/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL occupancy value {method}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{method}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Small matched 96-16-2 full-batch value fits; no native actor or lower training.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    summary = json.loads((ROOT/"results"/spec.SOURCE_RUN/"summary.json").read_text())
    labels = json.loads((ROOT/"results"/spec.source.SOURCE_RUN/"summary.json").read_text())
    spec.training_rows(summary, labels)
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Occupancy value run is already registered")
    directory = ROOT/"results"/args.run_name
    write_json(directory/"inputs/training.json", summary)
    write_json(directory/"inputs/labels.json", labels)
    tasks = [task_specification(args.run_name, method) for method in spec.value.METHODS]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 4, "budget": {"fitted_models": 72, "critic_updates": 72*spec.value.STEPS,
            "new_native_episodes": 0, "actor_updates": 0},
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
