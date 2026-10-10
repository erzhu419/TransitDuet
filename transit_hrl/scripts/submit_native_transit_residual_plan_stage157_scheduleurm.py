#!/usr/bin/env python3
"""Schedule bounded learned native upper-plan tests without staging raw data."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_residual_plan_stage157 as spec
from scripts.submit_native_transit_critic_regularization_stage153_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root):
    task = native_task(run_name, spec.METHOD, root)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_residual_plan_stage157.py",
        "--seed", str(root), "--output", str(Path(task["result_dir"]) / "result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL learned native service residual seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{spec.METHOD}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/development",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Upper-only shared SAC with frozen native lower; one CPU per independent upper seed.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    completed = json.loads((ROOT / "results" / spec.SOURCE_RUN / "summary.json").read_text())
    if not (completed["software_qualified"] and completed["protocol"] == spec.source.EXPERIMENT_PROTOCOL
            and completed["contract"] == spec.source.contract()):
        raise SystemExit("Learned residual plans require completed service-allocation qualification")
    for root in spec.ROOTS:
        spec.load_source(root)
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Learned residual-plan run is already registered")
    tasks = [task_specification(args.run_name, root) for root in spec.ROOTS]
    contract = spec.contract()
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "budget": {"training_episodes": 240, "frozen_episodes": 160,
            "native_ticks": 400 * 61380, "worker_preflight_ticks": len(tasks) * (3 * 61380 + 2 * 5400),
            "upper_updates_per_cell": 5500, "native_upper_lower_updates": 0},
        "tasks": len(tasks), "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
            "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
