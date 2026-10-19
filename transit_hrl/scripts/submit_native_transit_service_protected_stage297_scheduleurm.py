#!/usr/bin/env python3
"""Schedule the frozen-source Stage297 development calibration, not new trials."""

import argparse
import shlex
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_protected_stage297 as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from freq_hrl.experiments.pointmaze_root_response import write_json


def task(run):
    cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_service_protected_stage297.py", "--run-name", run]
    return {"project": spec.PROTOCOL, "description": "Freq-HRL Stage297 service-protected development calibration",
        "signature": f"Freq-HRL/{spec.PROTOCOL}/{run}/calibrate", "resource_family": f"Freq-HRL/{spec.PROTOCOL}",
        "cwd": str(ROOT), "cmd": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(cmd),
        "cpu": 1, "ram_mb": 3072, "vram": 0, "priority": "normal", "require_node": None,
        "allowed_nodes": [f"node{i:03}" for i in range(1, 7)], "allow_cpu_training": True,
        "cpu_training_justification": "Frozen critic and raw/lower weights; cached fit/calibration only, no fits or simulation.",
        "reroute_on_node_down": True, "node_down_requeue_s": 300,
        "result_dir": str(spec.directory(run)/"report"), "local_result_dir": str(spec.directory(run)/"report"),
        "stage_excludes": ["results", "**/__pycache__"],
        "stage_input_paths": [str(ROOT/"scripts"), str(ROOT/"freq_hrl"), str(spec.directory(run)/"inputs")]}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    a = p.parse_args()
    parent = spec.parent.read(spec.parent.directory(spec.PARENT)/"completion_receipt.json")
    if parent["scheduler"]["report"]["status"] != "done" or parent["scheduler"]["report"]["exit_code"] != 0 or not parent["coverage_qualified"]:
        p.error("Complete the Stage296 independent evaluation first")
    sys.path.insert(0, str(SCHEDULER.parent)); import scheduler
    job = task(a.run_name)
    if any(t.get("signature") == job["signature"] for t in scheduler.load_state()["tasks"]):
        p.error("Stage297 already submitted")
    registration = spec.directory(a.run_name)/"inputs/preregistration.json"
    if registration.exists(): p.error("Stage297 already registered")
    write_json(registration, {"protocol": spec.PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "Stage296_E32_excluded_from_policy_family_and_calibration": True})
    execute_bulk([job], dry_run=False, intent_label=spec.PROTOCOL+":"+a.run_name)


if __name__ == "__main__": main()
