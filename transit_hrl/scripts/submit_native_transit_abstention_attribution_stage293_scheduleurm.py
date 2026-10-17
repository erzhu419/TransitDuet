#!/usr/bin/env python3
"""Schedule cache-only post-confirmation attribution without new candidates."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_abstention_attribution_stage293 as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run):
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_abstention_attribution_stage293.py", "--run-name", run]
    return {"project": spec.EXPERIMENT_PROTOCOL, "description": "Freq-HRL fixed/rematched inference bridge and prefix attribution",
        "signature": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/analysis", "resource_family": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        "cwd": str(ROOT), "cmd": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        "cpu": 1, "ram_mb": 3072, "vram": 0, "priority": "normal", "require_node": None,
        "allowed_nodes": [f"node{i:03}" for i in range(1, 7)], "allow_cpu_training": True,
        "cpu_training_justification": "Reuse frozen old/new panels for scalar attribution; no fits, simulations, models or held access.",
        "reroute_on_node_down": True, "node_down_requeue_s": 300,
        "result_dir": str(spec.directory(run)/"report"), "local_result_dir": str(spec.directory(run)/"report"),
        "stage_excludes": ["results", "**/__pycache__"],
        "stage_input_paths": [str(ROOT/"scripts"), str(ROOT/"freq_hrl"), str(spec.directory(run)/"inputs")]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()
    registration = spec.directory(a.run_name)/"inputs/preregistration.json"
    if registration.exists() or inventory(a.run_name, protocol_spec=spec):
        p.error("Stage293 already registered")
    spec.qualify_parent()
    task = task_specification(a.run_name)
    write_json(registration, {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None, "cpu_per_task": 1, "ram_mb": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{a.run_name}")


if __name__ == "__main__":
    main()
