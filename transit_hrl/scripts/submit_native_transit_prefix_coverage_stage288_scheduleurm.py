#!/usr/bin/env python3
"""Stage matched prefix-coverage experiments on unpinned scheduler CPU nodes."""

import argparse
import shlex
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_prefix_coverage_stage288 as spec
from scripts.submit_native_transit_service_level_stage285_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_pair_critic_stage246_scheduleurm import require_completed_attempts
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run, phase, key=None, probe=None, chunk=None):
    task = base_task(run)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_prefix_coverage_stage288.py", "--run-name", run, "--phase", phase]
    suffix = phase
    if key is not None:
        role, root, scenario, replica = key
        command += ["--role", role, "--seed", str(root), "--scenario", scenario, "--replica", str(replica)]
        suffix += "/"+"/".join(map(str, key))
    if probe is not None:
        command += ["--probe", str(probe), "--chunk", str(chunk)]
        suffix += f"/{probe}/{chunk}"
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL matched independent prefix coverage "+suffix,
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{phase}",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Fresh matched prefix information and fixed-geometry residual solves; never refit raw references or use Stage287 confirmation labels.")
    if phase in ("capture-pilot", "pilot-collect", "capture", "collect"):
        task["ram_mb"] = 1536
    return task


def tasks(run, phase):
    if phase in ("capture-pilot", "pilot-collect"):
        return [task_specification(run, phase, k) for k in spec.keys("pilot")]
    if phase == "capture":
        return [task_specification(run, phase, k) for k in spec.keys()]
    if phase == "collect":
        return [task_specification(run, phase, k, probe, c) for k in spec.keys() for probe in spec.PROBES for c in spec.chunks(k[0])]
    return [task_specification(run, phase)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=spec.PHASES, required=True)
    a = p.parse_args()
    directory = spec.directory(a.run_name)
    registration = directory/"preregistration.json"
    existing = inventory(a.run_name, protocol_spec=spec)
    if a.phase == "freeze":
        if existing or registration.exists():
            p.error("Stage288 already registered")
        receipt = spec.read(spec.native.directory(spec.SOURCE_RUN)/"completion_receipt.json")
        if receipt["scheduler"]["collection_status_counts"] != {"done": 1360} or not receipt["complex_level_branch_stopped"]:
            p.error("Close Stage287 before the new coverage experiment")
        write_json(registration, {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": [f"node{i:03}" for i in range(1, 7)], "require_node": None,
                "cpu_per_task": 1, "native_ram_mb": 1536, "offline_ram_mb": 3072}})
    else:
        if spec.read(registration)["contract"] != spec.contract():
            p.error("Changed registered coverage comparison")
        preceding = {"capture-pilot": "freeze", "pilot-collect": "capture-pilot", "pilot-report": "pilot-collect",
            "capture": "pilot-report", "collect": "capture", "fit": "collect", "report": "fit"}[a.phase]
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        records = {t["id"]: t for t in scheduler.load_state()["tasks"]}
        missing = {t["id"] for t in existing}-records.keys()
        if missing:
            records.update({t["id"]: t for t in scheduler.load_archive_tasks() if t["id"] in missing})
        require_completed_attempts(existing, tasks(a.run_name, preceding), records)
        if a.phase in ("capture", "collect", "fit", "report") and not spec.read(directory/"pilot-report/result.json")["software_qualified"]:
            p.error("Native/solver preflight must qualify before main")
    batch = tasks(a.run_name, a.phase)
    if {t["signature"] for t in existing} & {t["signature"] for t in batch}:
        p.error("Phase already dispatched")
    execute_bulk(batch, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{a.run_name}:{a.phase}")


if __name__ == "__main__":
    main()
