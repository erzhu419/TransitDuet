#!/usr/bin/env python3
"""Schedule one cached screening-transfer diagnostic, with no new policies."""

import argparse
import shlex
import subprocess
import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_screen_transfer_stage295 as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run):
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_screen_transfer_stage295.py", "--run-name", run]
    return {"project": spec.EXPERIMENT_PROTOCOL, "description": "Freq-HRL frozen S8 to independent E32 risk transfer audit",
        "signature": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/analysis",
        "resource_family": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline", "cwd": str(ROOT),
        "cmd": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        "cpu": 1, "ram_mb": 3072, "vram": 0, "priority": "normal", "require_node": None,
        "allowed_nodes": [f"node{i:03}" for i in range(1, 7)], "allow_cpu_training": True,
        "cpu_training_justification": "Frozen independent-confirmation diagnostic; no fits, new simulations or policy selection.",
        "reroute_on_node_down": True, "node_down_requeue_s": 300,
        "result_dir": str(spec.directory(run)/"report"), "local_result_dir": str(spec.directory(run)/"report"),
        "stage_excludes": ["results", "**/__pycache__"],
        "stage_input_paths": [str(ROOT/"scripts"), str(ROOT/"freq_hrl"), str(spec.directory(run)/"inputs")]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    run = parser.parse_args().run_name
    registration = spec.directory(run)/"inputs/preregistration.json"
    if registration.exists() or inventory(run, protocol_spec=spec):
        parser.error("Stage295 already registered")
    spec.qualify_parent()
    task = task_specification(run)
    write_json(registration, {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None, "cpu_per_task": 1, "ram_mb": 3072}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{run}")
