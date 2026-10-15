#!/usr/bin/env python3
"""Schedule server-only cached future/prefix transfer without new fitting."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_fresh_transfer_stage242 as spec
from scripts.submit_native_transit_refit_consistency_stage240_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_frozen_future_stage241_scheduleurm import require_completed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, key=None, operation="prepare"):
    task = base_task(run_name, *(key if key else (None, None, None, None)))
    result = spec.result_path(run_name, *key).parent if key else ROOT/"results"/run_name/operation
    suffix = f"evaluate/{key[0]}/{key[1]}/{key[2]}/{key[3]}" if key else operation
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_fresh_transfer_stage242.py", "--run-name", run_name,
        "--operation", "evaluate" if key else operation]
    if key: command.extend(["--view", key[0], "--fold", str(key[1]), "--partition", key[2], "--direction", str(key[3])])
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL cached fresh-prefix transfer {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Cached model inference and matched future/prefix diagnostics only; no new native replay or raw/residual/actor training.")
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("prepare", "evaluate", "report"), required=True)
    args = p.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    prepare_task = task_specification(args.run_name)
    if args.phase == "prepare":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage242 already registered")
        source = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        if source["protocol"] != spec.source.EXPERIMENT_PROTOCOL or source["contract"] != spec.source.contract() or not source["coverage_qualified"]:
            raise SystemExit("Complete Stage241 before transfer diagnostics")
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": prepare_task["allowed_nodes"], "require_node": None,
                "cpu_per_task": 1, "ram_mb_per_task": prepare_task["ram_mb"]}})
        tasks = [prepare_task]
    else:
        if json.loads((directory/"preregistration.json").read_text())["contract"] != spec.contract(): raise SystemExit("Changed cached transfer protocol")
        preceding = [prepare_task]
        if args.phase == "report": preceding.extend(task_specification(args.run_name, k) for k in spec.KEYS)
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        records = {t["id"]: t for t in scheduler.load_state()["tasks"]}
        missing = {t["id"] for t in existing}-records.keys()
        if missing: records.update({t["id"]: t for t in scheduler.load_archive_tasks() if t["id"] in missing})
        require_completed(existing, preceding, records)
        tasks = [task_specification(args.run_name, k) for k in spec.KEYS] if args.phase == "evaluate" else [task_specification(args.run_name, operation="report")]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__": main()
