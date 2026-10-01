#!/usr/bin/env python3
"""Dispatch fixed-budget joint mean probes dynamically across node001-006."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import pointmaze_joint_mean_stage82_spec as spec
from scripts.submit_pointmaze_actor_parts_stage81_scheduleurm import task_specification as previous_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight)
    output = ROOT / "results" / run_name / "cells" / f"replicate_{root}" / "result.json"
    command = [DEFAULT_LINUX_PYTHON, "-u", spec.RUNNER_SCRIPT, "--optimizer-seed", str(root), "--output", str(output)]
    if preflight:command.append("--preflight")
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL Stage82 joint mean directions root{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{spec.POLICY}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/joint_mean/{'preflight' if preflight else 'full'}",
        cpu_training_justification="Native rollout workers plus fixed-budget joint mean scoring; source policies frozen.",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command))
    return task


def qualification_task(run_name, *, preflight):
    task = task_specification(run_name, spec.roots(preflight=preflight)[0], preflight=preflight)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_pointmaze_joint_mean_stage82.py", "--run-name", run_name]
    if preflight:command.append("--preflight")
    task.update(description="Freq-HRL Stage82 all-root joint mean qualification",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/qualification",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/qualification", cpu=1, ram_mb=2048,
        result_dir=None, local_result_dir=None,
        wait_for_files=[str(ROOT / "results" / run_name / "cells" / f"replicate_{r}" / "completion" / "ready.json") for r in spec.roots(preflight=preflight)],
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command))
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if inventory(a.run_name, protocol_spec=spec):raise SystemExit("Stage82 run already registered")
    tasks = [task_specification(a.run_name, r, preflight=a.preflight) for r in spec.roots(preflight=a.preflight)]
    tasks.append(qualification_task(a.run_name, preflight=a.preflight))
    write_json(ROOT / "results" / a.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "preflight": a.preflight,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "optimizer_roots": list(spec.roots(preflight=a.preflight)), "options": spec.options(preflight=a.preflight),
        "seed_roles": {str(r): spec.seed_roles(r, preflight=a.preflight) for r in spec.roots(preflight=a.preflight)},
        "budget_per_root": spec.budget(preflight=a.preflight), "endpoints": list(spec.ENDPOINTS),
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": tasks[0]["cpu"], "ram_mb_per_task": tasks[0]["ram_mb"]},
        "artifacts": {"pull": "completion_markers_and_compact_JSON_only", "native_trace_or_checkpoint_writes": 0}})
    execute_bulk(tasks, dry_run=a.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{a.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
