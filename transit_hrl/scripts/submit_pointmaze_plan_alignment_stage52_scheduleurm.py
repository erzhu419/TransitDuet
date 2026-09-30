#!/usr/bin/env python3
"""Place fixed-budget native plan controls dynamically on node001-node006."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import pointmaze_plan_alignment_stage52_spec as spec
from scripts.submit_pointmaze_lower_learnability_stage51_scheduleurm import task_specification as previous_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight)
    output = ROOT / "results" / run_name / "cells" / f"replicate_{root}" / "result.json"
    command = [DEFAULT_LINUX_PYTHON, "-u", spec.RUNNER_SCRIPT, "--optimizer-seed", str(root), "--output", str(output)]
    if preflight:
        command.append("--preflight")
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL stage52 plan alignment root{root}",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "
            + shlex.join(command) + " && printf '%s\\n' 'Training complete: result.json written'",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{spec.POLICY}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/evaluation",
        cpu_training_justification="Eight persistent native MuJoCo workers plus parent; matched upper/lower calls and causal held/phase reference controls, no optimizer updates.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Stage52 run already registered")
    tasks = [task_specification(args.run_name, root, preflight=args.preflight) for root in spec.roots(preflight=args.preflight)]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "preflight": args.preflight,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_run": spec.SOURCE_PREFLIGHT_RUN if args.preflight else spec.SOURCE_FULL_RUN,
        "optimizer_roots": list(spec.roots(preflight=args.preflight)), "options": spec.options(preflight=args.preflight),
        "seed_roles": {str(r): spec.seed_roles(r, preflight=args.preflight) for r in spec.roots(preflight=args.preflight)},
        "budget_per_root": spec.budget(preflight=args.preflight),
        "total_method_steps": len(tasks) * spec.budget(preflight=args.preflight)["total_primitive_steps"],
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                      "cpu_per_task": tasks[0]["cpu"], "ram_mb_per_task": tasks[0]["ram_mb"]},
        "artifacts": {"pull": "compact_json_only", "raw_and_weights": "server_only"}})
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")
    if not args.dry_run:
        tasks = inventory(args.run_name, protocol_spec=spec)
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch", *[arg for task in tasks for arg in ("--task-id", str(task["id"]))]], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
