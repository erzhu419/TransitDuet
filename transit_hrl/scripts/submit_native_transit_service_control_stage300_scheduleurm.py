#!/usr/bin/env python3
"""Schedule old-cohort service-only calibration without reading Stage298 E32."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_control_stage300 as spec
from scripts.submit_native_transit_service_protected_stage297_scheduleurm import task as source_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from freq_hrl.experiments.pointmaze_root_response import write_json


def task(run):
    job = source_task(run)
    job.update(project=spec.PROTOCOL, description="Freq-HRL Stage300 service-only development control",
        signature=f"Freq-HRL/{spec.PROTOCOL}/{run}/calibrate", resource_family=spec.PROTOCOL,
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(
            [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_service_control_stage300.py", "--run-name", run]),
        cpu_training_justification="Old fit/calibration caches only; frozen critic, no training or native simulation.")
    return job


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    a = p.parse_args()
    sys.path.insert(0, str(SCHEDULER.parent)); import scheduler
    job = task(a.run_name)
    if any(t.get("signature") == job["signature"] for t in scheduler.load_state()["tasks"]): p.error("Stage300 already submitted")
    registration = spec.directory(a.run_name)/"inputs/preregistration.json"
    if registration.exists(): p.error("Stage300 already registered")
    write_json(registration, {"protocol": spec.PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    execute_bulk([job], dry_run=False, intent_label=spec.PROTOCOL+":"+a.run_name)


if __name__ == "__main__": main()
