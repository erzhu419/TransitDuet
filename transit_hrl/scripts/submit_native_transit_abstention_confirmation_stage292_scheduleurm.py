#!/usr/bin/env python3
"""Schedule frozen-policy fresh-future confirmation on unpinned CPU nodes."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_abstention_confirmation_stage292 as spec
from scripts.submit_native_transit_zero_abstention_stage291_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_pair_critic_stage246_scheduleurm import require_completed_attempts
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run, phase, index=None, chunk=None):
    task = base_task(run)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_abstention_confirmation_stage292.py", "--run-name", run, "--phase", phase]
    suffix = phase
    if index is not None:
        command += ["--state-index", str(index)]
        suffix += f"/state{index}"
    if chunk is not None:
        command += ["--chunk", str(chunk)]
        suffix += f"/chunk{chunk}"
    output = (f"pilot/state{index}" if phase == "pilot" else f"receipts/state{index}/chunk{chunk}" if phase == "collect" else phase)
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL frozen abstention confirmation "+suffix,
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/{suffix}", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{phase}",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        result_dir=str(spec.directory(run)/output), local_result_dir=str(spec.directory(run)/output),
        ram_mb=1536 if phase in ("pilot", "collect") else 3072,
        cpu_training_justification="Frozen Stage291 actions on new continuation futures from archived Stage288 prefixes; no fits, model inference for selection, actor updates or held-prefix access.")
    if phase != "freeze":
        # Output-directory staging mirrors the local copy and removes remote-only ledgers.
        task["stage_input_paths"] = [p for p in task["stage_input_paths"] if p != str(spec.directory(run))]
    return task


def tasks(run, phase):
    if phase in ("pilot", "collect"):
        plan = spec.read(spec.directory(run)/"freeze/result.json")
        if plan["contract"] != spec.contract() or not plan["software_qualified"]:
            raise ValueError("Changed frozen future plan")
        if phase == "pilot":
            return [task_specification(run, phase, i) for i in plan["pilot_indices"]]
        return [task_specification(run, phase, i, c) for i in plan["active_indices"] for c in range(16)]
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
            p.error("Stage292 already registered")
        spec.qualify_parent()
        write_json(registration, {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": [f"node{i:03}" for i in range(1, 7)], "require_node": None,
                "cpu_per_task": 1, "native_ram_mb": 1536, "offline_ram_mb": 3072}})
    else:
        if spec.read(registration)["contract"] != spec.contract():
            p.error("Changed registered confirmation")
        preceding = {"pilot": "freeze", "pilot-report": "pilot", "collect": "pilot-report", "report": "collect"}[a.phase]
        required = tasks(a.run_name, preceding)
        signatures = {t["signature"] for t in required}
        attempts = [t for t in existing if t["signature"] in signatures]
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        records = {t["id"]: t for t in scheduler.load_state()["tasks"]}
        missing = {t["id"] for t in attempts}-records.keys()
        if missing:
            records.update({t["id"]: t for t in scheduler.load_archive_tasks() if t["id"] in missing})
        require_completed_attempts(attempts, required, records)
        if a.phase in ("collect", "report") and not spec.read(directory/"pilot-report/result.json")["software_qualified"]:
            p.error("Native replay preflight must qualify first")
    batch = tasks(a.run_name, a.phase)
    if {t["signature"] for t in existing}&{t["signature"] for t in batch}:
        p.error("Phase already dispatched")
    execute_bulk(batch, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{a.run_name}:{a.phase}")


if __name__ == "__main__":
    main()
