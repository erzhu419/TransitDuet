#!/usr/bin/env python3
"""Schedule frozen-model new-prefix validation without moving server caches."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_frozen_prefix_stage287 as spec
from scripts.submit_native_transit_service_level_stage285_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run, phase, key=None, chunk=None):
    task = base_task(run)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_frozen_prefix_stage287.py",
        "--run-name", run, "--phase", phase]
    suffix = phase
    if key is not None:
        role, root, scenario, replica = key
        command += ["--role", role, "--seed", str(root), "--scenario", scenario, "--replica", str(replica)]
        suffix += "/"+"/".join(map(str, key))
    if chunk is not None:
        command += ["--chunk", str(chunk)]
        suffix += "/"+str(chunk)
    if phase == "collect":
        suffix += "/rng_repaired"
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL frozen five-selector independent-prefix validation "+suffix,
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{phase}",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Frozen native CPU inference and paired continuations only; zero raw-reference or calibration refits, no actor updates.")
    if phase in ("prefix", "replay-pilot", "collect"):
        task["ram_mb"] = 1536
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("freeze", "pilot", "pilot-report", "prefix", "seal-main", "replay-pilot", "replay-pilot-report", "collect", "report"), required=True)
    a = p.parse_args()
    directory = spec.directory(a.run_name)
    registration = directory/"preregistration.json"
    if a.phase == "freeze":
        if registration.exists():
            p.error("Stage287 already registered")
        task = task_specification(a.run_name, "freeze")
        write_json(registration, {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
                "cpu_per_task": task["cpu"], "ram_mb": task["ram_mb"]}})
    elif spec.read(registration)["contract"] != spec.contract():
        p.error("Registered experiment contract changed")
    gates = {"pilot": "freeze/result.json", "replay-pilot": "pilot/result.json",
        "prefix": "pilot/result.json", "seal-main": "pilot/result.json", "collect": "main_plan.json"}
    if a.phase in gates:
        gate = spec.read(directory/gates[a.phase])
        if gate["contract"] != spec.contract() or not gate.get("software_qualified", a.phase == "collect"):
            p.error("Required previous phase did not qualify")
    if a.phase in ("pilot", "replay-pilot"):
        tasks = [task_specification(a.run_name, a.phase, k) for k in spec.prefixes(pilot=True)]
    elif a.phase == "prefix":
        tasks = [task_specification(a.run_name, a.phase, k) for k in spec.prefixes() if k[0] == "new"]
    elif a.phase == "collect":
        if not spec.read(directory/"replay-pilot/result.json")["software_qualified"]:
            p.error("Cross-worker replay must qualify before resubmitting main collection")
        tasks = [task_specification(a.run_name, a.phase, tuple(t["key"]), t["chunk"]) for t in gate["tasks"]]
    else:
        tasks = [task_specification(a.run_name, a.phase)]
    existing = inventory(a.run_name, protocol_spec=spec)
    if any(t["signature"] in {s["signature"] for s in tasks} for t in existing):
        p.error("This phase is already scheduled")
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{a.run_name}:{a.phase}")


if __name__ == "__main__":
    main()
