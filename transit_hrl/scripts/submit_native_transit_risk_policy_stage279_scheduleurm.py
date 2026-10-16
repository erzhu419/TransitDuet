#!/usr/bin/env python3
"""Schedule zero-fit frozen risk-head policy evaluation on Linux CPU caches."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_risk_policy_stage279 as spec
from scripts.submit_native_transit_effective_risk_stage278_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run):
    task = base_task(run)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_risk_policy_stage279.py", "--run-name", run]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL frozen risk-head plug-in policy outcomes",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/analysis", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Frozen raw-reference inference, cached risk predictions and candidate outcomes only; no fitting, native simulation, label-based gate or actor update.")
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    directory = ROOT/"results"/a.run_name
    if inventory(a.run_name, protocol_spec=spec) or (directory/"preregistration.json").exists(): raise SystemExit("Stage279 already registered")
    parent = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
    receipt = json.loads((ROOT/"results"/spec.SOURCE_RUN/"completion_receipt.json").read_text())
    if (parent["protocol"] != spec.source.EXPERIMENT_PROTOCOL or parent["contract"] != spec.source.contract()
            or not parent["coverage_qualified"] or receipt["scheduler"]["status_counts"] != {"done": 1}
            or receipt["scheduler"]["exit_code_counts"] != {"0": 1}): raise SystemExit("Complete qualified Stage278 first")
    task = task_specification(a.run_name)
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None, "cpu_per_task": 1, "ram_mb": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{a.run_name}")


if __name__ == "__main__": main()
