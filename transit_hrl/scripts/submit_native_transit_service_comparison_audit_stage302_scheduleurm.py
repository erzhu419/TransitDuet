#!/usr/bin/env python3
"""Schedule read-only paired inference after the qualified Stage301 report."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_comparison_audit_stage302 as spec
from scripts.submit_native_transit_service_protected_stage297_scheduleurm import task as source_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from freq_hrl.experiments.pointmaze_root_response import write_json


def task(run):
    job = source_task(run)
    job.update(project=spec.PROTOCOL, description="Freq-HRL Stage302 frozen joint/service paired inference",
        signature=f"Freq-HRL/{spec.PROTOCOL}/{run}/audit", resource_family=spec.PROTOCOL,
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(
            [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_service_comparison_audit_stage302.py", "--run-name", run]),
        cpu_training_justification="Read-only qualified Stage301 E32 cache; no fitting or native simulation.")
    return job


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    a = p.parse_args()
    receipt = spec.source.read(spec.source.directory(spec.SOURCE)/"evaluation_completion_receipt.json")
    if receipt["report_task"]["status"] != "done" or receipt["report_task"]["exit_code"] != 0 or not receipt["coverage_qualified"]:
        p.error("Complete the Stage301 report first")
    sys.path.insert(0, str(SCHEDULER.parent)); import scheduler
    job = task(a.run_name)
    if any(t.get("signature") == job["signature"] for t in scheduler.load_state()["tasks"]): p.error("Stage302 already submitted")
    registration = spec.directory(a.run_name)/"inputs/preregistration.json"
    if registration.exists(): p.error("Stage302 already registered")
    write_json(registration, {"protocol": spec.PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_report_task": receipt["report_task"]["id"]})
    execute_bulk([job], dry_run=False, intent_label=spec.PROTOCOL+":"+a.run_name)


if __name__ == "__main__": main()
