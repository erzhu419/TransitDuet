#!/usr/bin/env python3
"""One unpinned CPU task for the fixed offline action-value screen."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_action_value_stage176 as spec
from scripts.submit_native_transit_plan_credit_stage158_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name):
    task = native_task(run_name, 397, "low_noise")
    directory = ROOT/"results"/run_name
    result = directory/"fit"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_action_value_stage176.py",
        "--data", str(directory/"inputs/training.json"), "--output", str(result/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL paired value fit and grouped validation",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/fit",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        stage_input_paths=[*task["stage_input_paths"], str(directory/"inputs")],
        cpu_training_justification="Small fixed linear pairwise fit; zero native episodes or policy updates.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    summary = json.loads((ROOT/"results"/spec.SOURCE_RUN/"summary.json").read_text())
    spec.training_rows(summary)
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Action-value fit run is already registered")
    directory = ROOT/"results"/args.run_name
    write_json(directory/"inputs/training.json", summary)
    task = task_specification(args.run_name)
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 1, "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
