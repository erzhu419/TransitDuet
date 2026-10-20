#!/usr/bin/env python3
"""Schedule frozen whole-episode closed-loop validation after native qualification."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_closed_loop_confirmation_stage304 as spec
from scripts.submit_native_transit_service_protected_stage297_scheduleurm import task as source_task
from scripts.submit_native_transit_paired_fleet_stage296_scheduleurm import require_completed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from freq_hrl.experiments.pointmaze_root_response import write_json


def task(run, phase, root=None, scenario=None, replica=None, arm=None):
    job = source_task(run); command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_closed_loop_confirmation_stage304.py", "--run-name", run, "--phase", phase]
    suffix = phase; output = spec.directory(run)/phase
    if phase == "evaluate":
        command += ["--root", str(root), "--scenario", scenario, "--replica", str(replica), "--arm", arm]
        suffix += f"/{root}/{scenario}/replica{replica}/{arm}"; output = spec.cell_path(run, root, scenario, replica, arm)
    job.update(project=spec.PROTOCOL, signature=f"Freq-HRL/{spec.PROTOCOL}/{run}/{suffix}", resource_family=spec.PROTOCOL+"/"+phase,
        description="Freq-HRL frozen two-step closed-loop "+suffix, ram_mb=1536 if phase == "evaluate" else 3072,
        result_dir=str(output), local_result_dir=str(output),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Frozen two-decision native controller performance evaluation; no training or reference fitting.")
    return job


def tasks(run, phase):
    if phase == "evaluate":
        return [task(run, phase, r, s, n, a) for r in spec.source.source.ROOTS for s in spec.source.source.SCENARIOS for n in spec.REPLICAS for a in spec.ARMS]
    return [task(run, phase)]


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True); p.add_argument("--phase", required=True, choices=spec.PHASES); a = p.parse_args()
    sys.path.insert(0, str(SCHEDULER.parent)); import scheduler
    prefix = f"Freq-HRL/{spec.PROTOCOL}/{a.run_name}/"
    existing = [t for t in scheduler.load_state()["tasks"] if str(t.get("signature", "")).startswith(prefix)]
    registration = spec.directory(a.run_name)/"inputs/preregistration.json"
    if a.phase == "freeze":
        receipt = spec.source.source.read(spec.directory(spec.AUDIT)/"execution_receipt.json")
        report = spec.source.source.read(spec.directory(spec.AUDIT)/"report/result.json")
        if not report["all_pilots_qualified"] or receipt["scheduler"]["status"] != "done" or receipt["scheduler"]["exit_code"] != 0: p.error("Native preflight audit must pass")
        if existing or registration.exists(): p.error("Formal closed-loop run already registered")
        write_json(registration, {"protocol": spec.PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "native_qualification_task": receipt["scheduler"]["id"], "pilot_metrics_used_for_selection": False})
    else:
        if spec.source.source.read(registration)["contract"] != spec.contract(): p.error("Changed formal registration")
        prior = tasks(a.run_name, spec.PHASES[spec.PHASES.index(a.phase)-1])
        if {t["signature"] for t in prior}-{t["signature"] for t in existing}:
            existing = [t for t in scheduler.load_archive_tasks() if str(t.get("signature", "")).startswith(prefix)]+existing
        require_completed(existing, prior)
    batch = tasks(a.run_name, a.phase)
    if {t["signature"] for t in batch}&{t["signature"] for t in existing}: p.error("Phase already submitted")
    execute_bulk(batch, dry_run=False, intent_label=spec.PROTOCOL+":"+a.run_name+":"+a.phase)


if __name__ == "__main__": main()
