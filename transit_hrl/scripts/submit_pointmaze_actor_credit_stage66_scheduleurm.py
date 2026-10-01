#!/usr/bin/env python3
"""Schedule frozen pre-update credit diagnosis and dependent aggregation."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import pointmaze_actor_credit_stage66_spec as spec
from scripts.submit_pointmaze_normalized_update_stage65_scheduleurm import task_specification as previous_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight)
    output = ROOT / "results" / run_name / "cells" / f"replicate_{root}" / "result.json"
    command = [DEFAULT_LINUX_PYTHON, "-u", spec.RUNNER_SCRIPT, "--optimizer-seed", str(root), "--output", str(output)]
    if preflight:
        command.append("--preflight")
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL Stage66 read-only actor credit root{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{spec.POLICY}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/native_{'preflight' if preflight else 'full'}",
        cpu=3, ram_mb=3072,
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Two archive reconstruction workers and one read-only gradient learner; no native rollouts or optimizer steps.")
    return task


def qualification_task(run_name, *, preflight):
    task = task_specification(run_name, spec.roots(preflight=preflight)[0], preflight=preflight)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_pointmaze_actor_credit_stage66.py", "--run-name", run_name]
    if preflight:
        command.append("--preflight")
    task.update(description="Freq-HRL Stage66 complete-roster credit qualification",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/qualification",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/qualification", cpu=1, ram_mb=1024,
        result_dir=None, local_result_dir=None,
        wait_for_files=[str(ROOT / "results" / run_name / "cells" / f"replicate_{r}" / "completion" / "ready.json") for r in spec.roots(preflight=preflight)],
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command))
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Stage66 run already registered")
    tasks = [task_specification(args.run_name, r, preflight=args.preflight) for r in spec.roots(preflight=args.preflight)]
    tasks.append(qualification_task(args.run_name, preflight=args.preflight))
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "preflight": args.preflight,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "optimizer_roots": list(spec.roots(preflight=args.preflight)), "budget_per_root": spec.budget(preflight=args.preflight),
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None, "cpu_per_task": 3, "ram_mb_per_task": 3072},
        "artifacts": {"pull": "completion_markers_and_compact_json_only", "traces_and_weights": "server_only"}})
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
