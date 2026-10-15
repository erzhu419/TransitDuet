#!/usr/bin/env python3
"""Schedule paired-margin critic regressions from shared remote caches."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_pair_critic_stage246 as spec
from scripts.submit_native_transit_feasible_critic_stage245_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_frozen_future_stage241_scheduleurm import require_completed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, key=None):
    task = base_task(run_name, key)
    suffix = f"fit/{key[0]}/{key[1]}/{key[2]}/{key[3]}" if key else "report"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_pair_critic_stage246.py",
        "--run-name", run_name, "--operation", "fit" if key else "report"]
    if key: command.extend(["--view", key[0], "--fold", str(key[1]), "--partition", key[2], "--direction", str(key[3])])
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL paired-margin critic {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Small cached quadratic paired regressions; no raw-reference refits, actor training or native replay.")
    return task


def require_completed_attempts(existing, tasks, records):
    parents = {records[t["id"]].get("parent_id") for t in existing}
    leaves = [t for t in existing if t["id"] not in parents]
    if len(leaves) != len(tasks): raise SystemExit("Need exactly one terminal attempt per preceding task")
    require_completed(leaves, tasks, records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("preflight", "remaining", "report"), required=True)
    args = p.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    first = task_specification(args.run_name, spec.PREFLIGHT)
    if args.phase == "preflight":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage246 already registered")
        old = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        if old["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or old["contract"] != spec.previous.contract() or not old["coverage_qualified"]:
            raise SystemExit("Complete Stage245 before paired regression")
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": first["allowed_nodes"], "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": first["ram_mb"]}})
        tasks = [first]
    else:
        if json.loads((directory/"preregistration.json").read_text())["contract"] != spec.contract(): raise SystemExit("Changed registered paired-regression protocol")
        preceding = [first] if args.phase == "remaining" else [task_specification(args.run_name, k) for k in spec.KEYS]
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        records = {t["id"]: t for t in scheduler.load_state()["tasks"]}
        missing = {t["id"] for t in existing}-records.keys()
        if missing: records.update({t["id"]: t for t in scheduler.load_archive_tasks() if t["id"] in missing})
        require_completed_attempts(existing, preceding, records)
        tasks = [task_specification(args.run_name, k) for k in spec.KEYS if k != spec.PREFLIGHT] if args.phase == "remaining" else [task_specification(args.run_name)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__": main()
