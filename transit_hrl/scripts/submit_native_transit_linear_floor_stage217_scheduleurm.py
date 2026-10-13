#!/usr/bin/env python3
"""Schedule train-only fixed-feature projection and linear feasibility probes."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_native_transit_linear_floor_stage217 as spec
from scripts.submit_native_transit_fit_transfer_stage216_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name):
    task = base_task(run_name)
    result = ROOT/"results"/run_name/"report"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_native_transit_linear_floor_stage217.py", "--output", str(result/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL fixed-feature cost projection and stable order/wait linear bounds",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/report", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/diagnostic",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="47 weighted98-feature SVD projections and at most188 single-thread LP probes; no actor or native training.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    if inventory(args.run_name, protocol_spec=spec) or (directory/"preregistration.json").exists():
        raise SystemExit("Linear-class diagnosis already registered")
    prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or prior["contract"] != spec.previous.contract() or not prior["controls_reproduced"]:
        raise SystemExit("Complete corrected Stage216 before projection diagnosis")
    task = task_specification(args.run_name)
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 1, "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
            "cpu_per_task": task["cpu"], "ram_mb_per_task": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
