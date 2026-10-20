#!/usr/bin/env python3
"""Schedule the two-step on-policy pilot, excluding its outcomes from claims."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_closed_loop_preflight_stage303 as spec
from scripts.submit_native_transit_service_protected_stage297_scheduleurm import task as source_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from freq_hrl.experiments.pointmaze_root_response import write_json


def task(run, root, scenario, arm):
    job = source_task(run)
    output = spec.directory(run)/"cells"/str(root)/scenario/arm
    job.update(project=spec.PROTOCOL, description=f"Freq-HRL two-step closed-loop pilot {root}/{scenario}/{arm}",
        signature=f"Freq-HRL/{spec.PROTOCOL}/{run}/{root}/{scenario}/{arm}", resource_family=spec.PROTOCOL,
        ram_mb=1536, result_dir=str(output), local_result_dir=str(output),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(
            [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_closed_loop_preflight_stage303.py",
             "--run-name", run, "--root", str(root), "--scenario", scenario, "--arm", arm]),
        cpu_training_justification="Two-decision on-policy software qualification with frozen controllers; no training.")
    return job


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True); a = p.parse_args()
    completion = spec.source.read(spec.source.directory(spec.SOURCE)/"evaluation_completion_receipt.json")
    report = spec.source.read(spec.source.directory(spec.SOURCE)/"report/result.json")
    if (not completion["coverage_qualified"] or completion["report_task"]["status"] != "done"
            or completion["report_task"]["exit_code"] != 0 or not report["joint_transfer_supported"]
            or not all(report["comparisons"]["service_only_vs_raw"][ci][f][1] < 0
                for ci in ("prefix_CI95", "conditional_MC_CI95") for f in ("cost", "wait_min"))):
        p.error("Qualified frozen joint and service-only transfer evidence required")
    sys.path.insert(0, str(SCHEDULER.parent)); import scheduler
    jobs = [task(a.run_name, r, s, arm) for r in spec.source.ROOTS for s in spec.source.SCENARIOS for arm in spec.ARMS]
    registration = spec.directory(a.run_name)/"inputs/preregistration.json"
    signatures = {t["signature"] for t in jobs}
    if registration.exists() or signatures & {t.get("signature") for t in scheduler.load_state()["tasks"]}: p.error("Pilot already registered")
    write_json(registration, {"protocol": spec.PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_report_task": completion["report_task"]["id"]})
    execute_bulk(jobs, dry_run=False, intent_label=spec.PROTOCOL+":"+a.run_name)


if __name__ == "__main__": main()
