#!/usr/bin/env python3
"""Schedule one cached-result admission contrast; no training or raw pulls."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_stability_admission_stage236 as spec
from scripts.submit_native_transit_prefix_stability_stage235_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name):
    task = base_task(run_name)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_stability_admission_stage236.py", "--run-name", run_name]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL cached stability admission report",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/report", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Read frozen inner stability curves and outer actions; admission contrast only, zero training/rollouts.")
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    directory = ROOT/"results"/args.run_name
    if inventory(args.run_name, protocol_spec=spec) or (directory/"preregistration.json").exists():
        raise SystemExit("Stage236 already registered")
    prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or prior["contract"] != spec.previous.contract() or not prior["coverage_qualified"]:
        raise SystemExit("Complete Stage235 before admission contrasts")
    task = task_specification(args.run_name)
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
            "cpu_per_task": task["cpu"], "ram_mb_per_task": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__": main()
