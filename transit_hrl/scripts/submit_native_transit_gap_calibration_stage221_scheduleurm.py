#!/usr/bin/env python3
"""Schedule train-only future calibration without local fitting or raw pulls."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_gap_calibration_stage221 as spec
from scripts.submit_native_transit_best_action_crossfit_stage220_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_mean_coverage_stage195_scheduleurm import require_completed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, view=None, fold=None):
    task = base_task(run_name, view, fold)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_gap_calibration_stage221.py",
        "--run-name", run_name, "--operation", "report" if view is None else "fit"]
    if view is not None: command.extend(["--view", view, "--fold", str(fold)])
    suffix = "report" if view is None else f"fit/{view}/{fold}"
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL cost-gap and inner-future wait shrinkage {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Four60-state cells; full and2 inner wait fits per architecture at600 updates, no native episode or RL training.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("fits", "report"), required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    if args.phase == "fits":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage221 already registered")
        prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        if prior["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or prior["contract"] != spec.previous.contract() or not prior["coverage_qualified"]:
            raise SystemExit("Complete Stage220 before calibration")
        tasks = [task_specification(args.run_name, v, f) for v in spec.VIEWS for f in (0, 1)]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "fit_tasks": 8, "report_tasks": 1, "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
                "require_node": None, "cpu_per_task": tasks[0]["cpu"], "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        registered = json.loads((directory/"preregistration.json").read_text())
        if registered["contract"] != spec.contract(): raise SystemExit("Changed Stage221 registration")
        require_completed(existing, {task_specification(args.run_name, v, f)["signature"] for v in spec.VIEWS for f in (0, 1)})
        tasks = [task_specification(args.run_name)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
