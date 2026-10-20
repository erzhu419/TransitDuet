#!/usr/bin/env python3
"""Schedule the frozen joint/service-only comparison on fresh prefixes."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_comparison_stage301 as spec
from scripts import submit_native_transit_service_confirmation_stage298_scheduleurm as previous
from scripts.submit_hyperparameter_pilot_scheduleurm import SCHEDULER, execute_bulk
from freq_hrl.experiments.pointmaze_root_response import write_json


def task(run, phase, key=None, probe=None, bank=None, index=None, chunk=0):
    job = previous.task(run, phase, key, probe, bank, index, chunk)
    job.update(project=spec.PROTOCOL,
        signature=job["signature"].replace(spec.previous.PROTOCOL, spec.PROTOCOL),
        resource_family=spec.PROTOCOL+"/"+phase,
        description="Freq-HRL fresh frozen joint/service-only "+phase,
        cmd=job["cmd"].replace("run_native_transit_service_confirmation_stage298.py", "run_native_transit_service_comparison_stage301.py"))
    return job


def tasks(run, phase):
    if phase == "capture": return [task(run, phase, k) for k in spec.keys()]
    if phase == "menus": return [task(run, phase, k, p, b) for k in spec.keys() for p in spec.PROBES for b in ("proposal", "screen")]
    if phase == "evaluate":
        plan = spec.read(spec.directory(run)/"seal/result.json")
        if plan["contract"] != spec.contract(run) or not plan["actions_and_weights_frozen"]: raise ValueError("Changed evaluation seal")
        return [task(run, phase, index=i, chunk=c) for i in plan["active_indices"] for c in range(4)]
    return [task(run, phase)]


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True); p.add_argument("--phase", required=True, choices=spec.PHASES)
    a = p.parse_args(); sys.path.insert(0, str(SCHEDULER.parent)); import scheduler
    prefix = f"Freq-HRL/{spec.PROTOCOL}/{a.run_name}/"
    existing = [t for t in scheduler.load_state()["tasks"] if str(t.get("signature", "")).startswith(prefix)]
    registration = spec.directory(a.run_name)/"inputs/preregistration.json"
    if a.phase == "freeze":
        if existing or registration.exists(): p.error("Comparison already registered")
        completion = spec.read(spec.directory(spec.CONTROL)/"execution_receipt.json")
        calibrated = spec.read(spec.directory(spec.CONTROL)/"report/result.json")
        if completion["scheduler"]["status"] != "done" or completion["scheduler"]["exit_code"] != 0 or not calibrated["fresh_confirmation_admitted"]: p.error("Qualified service-only development selection required")
        selected = calibrated["selection"]
        joint = spec.read(spec.directory(spec.SOURCE)/"inputs/locked_policy.json")["joint"]
        write_json(spec.directory(a.run_name)/"inputs/locked_policy.json", {"joint": joint,
            "service": {k: selected[k] for k in ("family", "threshold", "index")}})
        write_json(registration, {"protocol": spec.PROTOCOL, "contract": spec.contract(a.run_name),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "service_calibration_task": completion["scheduler"]["id"]})
    else:
        if spec.read(registration)["contract"] != spec.contract(a.run_name): p.error("Changed prospective registration")
        before = dict(zip(spec.PHASES[1:], spec.PHASES[:-1]))[a.phase]
        required = tasks(a.run_name, before)
        if {t["signature"] for t in required}-{t["signature"] for t in existing}:
            existing = [t for t in scheduler.load_archive_tasks() if str(t.get("signature", "")).startswith(prefix)]+existing
        previous.gates.require_completed(existing, required)
    batch = tasks(a.run_name, a.phase)
    if {t["signature"] for t in batch}&{t["signature"] for t in existing}: p.error("Phase already submitted")
    execute_bulk(batch, dry_run=False, intent_label=spec.PROTOCOL+":"+a.run_name+":"+a.phase)


if __name__ == "__main__": main()
