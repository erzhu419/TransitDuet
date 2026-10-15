#!/usr/bin/env python3
"""Seal frozen policies, qualify sparse replay, then collect independent futures."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_frozen_confirmation_stage248 as spec
from scripts.submit_native_transit_frozen_future_stage241_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_pair_critic_stage246_scheduleurm import require_completed_attempts
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, *, key=None, operation="seal"):
    task = base_task(run_name, key=key, operation=operation)
    if key is None:
        suffix, result = operation, ROOT/"results"/run_name/operation
        args = ["--operation", operation]
    else:
        r, s, i, p, d = key
        suffix, result = f"draw/{s}/{r}/{i}/{p}/{d}", spec.draw_path(run_name, *key).parent
        args = ["--operation", "collect", "--seed", str(r), "--scenario", s,
            "--replica", str(i), "--probe", str(p), "--draw", str(d)]
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_frozen_confirmation_stage248.py", "--run-name", run_name, *args]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL frozen64 confirmation {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'native' if key else 'offline'}",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Frozen policies only: exact source replay, same-tape identity and sealed divergent actions; no fitting or policy selection.")
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("seal", "preflight", "preflight_report", "remaining", "report"), required=True)
    args = p.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    first = task_specification(args.run_name)
    if args.phase == "seal":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage248 already registered")
        prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        if prior["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or prior["contract"] != spec.previous.contract() or not prior["coverage_qualified"]:
            raise SystemExit("Complete Stage247 before independent-future confirmation")
        native = task_specification(args.run_name, key=(*spec.coverage.states()[0], 0))
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": first["allowed_nodes"], "require_node": None, "cpu_per_task": 1,
                "seal_ram_mb": first["ram_mb"], "draw_ram_mb": native["ram_mb"]}})
        tasks = [first]
    else:
        if json.loads((directory/"preregistration.json").read_text())["contract"] != spec.contract(): raise SystemExit("Changed registered future confirmation")
        preceding = [first]
        registered = json.loads((directory/"seal/result.json").read_text())
        if registered["contract"] != spec.contract() or not registered["coverage_qualified"]: raise SystemExit("Qualify and sync the action seal first")
        initial = spec.cells(registered, preflight=True)
        if args.phase in ("preflight_report", "remaining", "report"):
            preceding.extend(task_specification(args.run_name, key=k) for k in
                (spec.cells(registered) if args.phase == "report" else initial))
        if args.phase in ("remaining", "report"): preceding.append(task_specification(args.run_name, operation="preflight_report"))
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        records = {t["id"]: t for t in scheduler.load_state()["tasks"]}
        missing = {t["id"] for t in existing}-records.keys()
        if missing: records.update({t["id"]: t for t in scheduler.load_archive_tasks() if t["id"] in missing})
        require_completed_attempts(existing, preceding, records)
        if args.phase == "preflight": tasks = [task_specification(args.run_name, key=k) for k in initial]
        elif args.phase == "remaining":
            preflight = json.loads((directory/"preflight_report/result.json").read_text())
            if preflight["contract"] != spec.contract() or not preflight["coverage_qualified"]: raise SystemExit("Qualify and sync native preflight before expansion")
            tasks = [task_specification(args.run_name, key=k) for k in spec.cells(registered) if k not in set(initial)]
            # The exact phase gate already excludes every remaining signature.
            # Avoid the scheduler's quadratic identity scan for this trusted batch.
            for task in tasks: task["allow_duplicate"] = True
        else: tasks = [task_specification(args.run_name, operation=args.phase)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__": main()
