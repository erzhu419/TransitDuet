#!/usr/bin/env python3
"""Schedule fresh confirmation only after the service-protected calibration gate."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_confirmation_stage298 as spec
from scripts import submit_native_transit_paired_fleet_stage296_scheduleurm as gates
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from freq_hrl.experiments.pointmaze_root_response import write_json


def task(run, phase, key=None, probe=None, bank=None, index=None, chunk=0):
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_service_confirmation_stage298.py",
        "--run-name", run, "--phase", phase]
    suffix = phase
    if key is not None:
        command += ["--role", key[0], "--root", str(key[1]), "--scenario", key[2], "--replica", str(key[3])]
        suffix += "/"+"/".join(map(str, key))
    if probe is not None:
        command += ["--probe", str(probe), "--bank", bank]
        suffix += f"/probe{probe}/{bank}"
    if index is not None:
        command += ["--state-index", str(index), "--chunk", str(chunk)]
        suffix += f"/state{index}/chunk{chunk}"
    native = phase in ("pilot", "capture", "menus", "evaluate")
    output = spec.receipt_path(run, phase, key, probe, bank, chunk, index) if native else spec.directory(run)/phase
    return {"project": spec.PROTOCOL, "description": "Freq-HRL fresh service-protected confirmation "+suffix,
        "signature": f"Freq-HRL/{spec.PROTOCOL}/{run}/{suffix}", "resource_family": f"Freq-HRL/{spec.PROTOCOL}/{phase}",
        "cwd": str(ROOT), "cmd": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        "cpu": 1, "ram_mb": 1536 if native else 3072, "vram": 0, "priority": "normal",
        "require_node": None, "allowed_nodes": [f"node{i:03}" for i in range(1, 7)],
        "allow_cpu_training": True, "cpu_training_justification": "Frozen critic/raw/lower and Stage297 cutoff; native new-prefix evaluation only, no fits.",
        "reroute_on_node_down": True, "node_down_requeue_s": 300,
        "result_dir": str(output), "local_result_dir": str(output), "stage_excludes": ["results", "**/__pycache__"],
        "stage_input_paths": [str(ROOT/"scripts"), str(ROOT/"freq_hrl"), str(spec.directory(run)/"inputs")]}


def tasks(run, phase):
    if phase in ("capture", "pilot"):
        return [task(run, phase, key) for key in spec.keys("test" if phase == "capture" else "pilot")]
    if phase == "menus":
        return [task(run, phase, key, probe, bank) for key in spec.keys() for probe in spec.PROBES for bank in ("proposal", "screen")]
    if phase == "evaluate":
        plan = spec.read(spec.directory(run)/"seal/result.json")
        if plan["contract"] != spec.contract(run) or not plan["actions_and_weights_frozen"]: raise ValueError("Changed fresh evaluation seal")
        return [task(run, phase, index=i, chunk=c) for i in plan["active_indices"] for c in range(4)]
    return [task(run, phase)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True); p.add_argument("--phase", required=True, choices=spec.PHASES)
    a = p.parse_args()
    sys.path.insert(0, str(SCHEDULER.parent)); import scheduler
    registration = spec.directory(a.run_name)/"inputs/preregistration.json"
    prefix = f"Freq-HRL/{spec.PROTOCOL}/{a.run_name}/"
    existing = [t for t in scheduler.load_state()["tasks"] if str(t.get("signature", "")).startswith(prefix)]
    if a.phase == "freeze":
        if existing or registration.exists(): p.error("Fresh confirmation already registered")
        parent = spec.read(spec.directory(spec.SOURCE)/"execution_receipt.json")
        result = spec.read(spec.directory(spec.SOURCE)/"report/result.json")
        if parent["scheduler"]["status"] != "done" or parent["scheduler"]["exit_code"] != 0 or not result["fresh_confirmation_admitted"]:
            p.error("A qualifying frozen Stage297 development selection is required")
        write_json(spec.directory(a.run_name)/"inputs/locked_policy.json", {
            name: {k: result["selection"][name][k] for k in ("family", "threshold", "index")} for name in ("joint", "risk")})
        write_json(registration, {"protocol": spec.PROTOCOL, "contract": spec.contract(a.run_name),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_calibration_task": parent["scheduler"]["id"], "policy_locked_before_fresh_scenes": True})
    else:
        if spec.read(registration)["contract"] != spec.contract(a.run_name): p.error("Changed fresh registration")
        previous = dict(zip(spec.PHASES[1:], spec.PHASES[:-1]))[a.phase]
        required = tasks(a.run_name, previous)
        if {t["signature"] for t in required}-{t["signature"] for t in existing}:
            existing = [t for t in scheduler.load_archive_tasks() if str(t.get("signature", "")).startswith(prefix)]+existing
        gates.require_completed(existing, required)
        if a.phase not in ("pilot", "pilot-report") and not spec.read(spec.directory(a.run_name)/"pilot-report/result.json")["software_qualified"]:
            p.error("Native pilots must qualify before formal sampling")
    batch = tasks(a.run_name, a.phase)
    if {t["signature"] for t in existing}&{t["signature"] for t in batch}: p.error("Phase already submitted")
    execute_bulk(batch, dry_run=False, intent_label=spec.PROTOCOL+":"+a.run_name+":"+a.phase)


if __name__ == "__main__": main()
