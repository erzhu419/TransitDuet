#!/usr/bin/env python3
"""Seal frozen actions before qualifying and expanding independent futures."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_frozen_future_stage241 as spec
from scripts.submit_native_transit_mean_coverage_stage195_scheduleurm import task_specification as native_task
from scripts.submit_native_transit_objective_oof_stage239_scheduleurm import task_specification as offline_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, *, key=None, operation="seal"):
    task = offline_task(run_name) if key is None else native_task(run_name, key=key)
    if key is None:
        suffix, result = operation, ROOT/"results"/run_name/operation
        args = ["--operation", operation]
    else:
        root, scenario, replica, probe, draw = key
        suffix, result = f"draw/{scenario}/{root}/{replica}/{probe}/{draw}", spec.draw_path(run_name, *key).parent
        args = ["--operation", "collect", "--seed", str(root), "--scenario", scenario,
            "--replica", str(replica), "--probe", str(probe), "--draw", str(draw)]
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_frozen_future_stage241.py", "--run-name", run_name, *args]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL frozen-action future replication {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'native' if key else 'offline'}",
        stage_input_paths=offline_task(run_name)["stage_input_paths"],
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Cached model inference or eight matched native frozen replays; no raw reference, residual or RL training.")
    return task


def require_completed(existing, tasks):
    if {t["signature"] for t in existing} != {t["signature"] for t in tasks} or any(
            t["status"] != "done" or t.get("exit_code") != 0 for t in existing):
        raise SystemExit("Complete exactly the preceding phase with exit zero before expanding")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("seal", "preflight", "remaining", "report"), required=True)
    args = p.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    seal_task = task_specification(args.run_name)
    first = [task_specification(args.run_name, key=k) for k in spec.cells(preflight=True)]
    if args.phase == "seal":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage241 already registered")
        previous = json.loads((ROOT/"results"/spec.PREVIOUS_RUN/"report/result.json").read_text())
        if previous["protocol"] != spec.prior.EXPERIMENT_PROTOCOL or previous["contract"] != spec.prior.contract() or not previous["coverage_qualified"]:
            raise SystemExit("Complete Stage240 before independent future replication")
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": seal_task["allowed_nodes"], "require_node": None, "cpu_per_task": 1,
                "seal_ram_mb": seal_task["ram_mb"], "draw_ram_mb": first[0]["ram_mb"]}})
        tasks = [seal_task]
    else:
        if json.loads((directory/"preregistration.json").read_text())["contract"] != spec.contract():
            raise SystemExit("Changed registered future replication")
        preceding = [seal_task]
        if args.phase == "remaining": preceding.extend(first)
        elif args.phase == "report": preceding.extend(task_specification(args.run_name, key=k) for k in spec.cells())
        require_completed(existing, preceding)
        if args.phase == "preflight": tasks = first
        elif args.phase == "remaining":
            initial = set(spec.cells(preflight=True))
            tasks = [task_specification(args.run_name, key=k) for k in spec.cells() if k not in initial]
        else: tasks = [task_specification(args.run_name, operation="report")]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__": main()
