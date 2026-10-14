#!/usr/bin/env python3
"""Schedule one prefix/future calibration cell per unbound Linux CPU task."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_prefix_policy_stage222 as spec
from scripts.submit_native_transit_gap_calibration_stage221_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_mean_coverage_stage195_scheduleurm import require_completed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, view=None, fold=None, partition=None, direction=None):
    task = base_task(run_name, view, fold)
    output = ROOT/"results"/run_name/"report" if view is None else spec.result_path(run_name, view, fold, partition, direction).parent
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_prefix_policy_stage222.py",
        "--run-name", run_name, "--operation", "report" if view is None else "fit"]
    if view is not None:
        command.extend(["--view", view, "--fold", str(fold), "--partition", partition, "--direction", str(direction)])
    suffix = "report" if view is None else f"fit/{view}/{fold}/{partition}/{direction}"
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL prefix/future and selected-action calibration {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        result_dir=str(output), local_result_dir=str(output),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="One outer cell reusing Stage221 final heads,20 inner54-state cost/wait fits, single-thread CPU; no native/RL episodes.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("fits", "report"), required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    if args.phase == "fits":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage222 already registered")
        prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        if prior["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or prior["contract"] != spec.previous.contract() or not prior["coverage_qualified"]:
            raise SystemExit("Complete Stage221 before prefix calibration")
        tasks = [task_specification(args.run_name, *k) for k in spec.KEYS]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "fit_tasks": len(tasks), "report_tasks": 1, "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
                "require_node": None, "cpu_per_task": tasks[0]["cpu"], "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        registered = json.loads((directory/"preregistration.json").read_text())
        if registered["contract"] != spec.contract(): raise SystemExit("Changed prefix calibration protocol")
        require_completed(existing, {task_specification(args.run_name, *k)["signature"] for k in spec.KEYS})
        tasks = [task_specification(args.run_name)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
