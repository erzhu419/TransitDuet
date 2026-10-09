#!/usr/bin/env python3
"""Schedule the bounded native critic-coordinate factorial on node001-node006."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_critic_units_stage152 as spec
from scripts import run_native_transit_dispatch_train_stage151 as source
from scripts.submit_native_transit_dispatch_train_stage151_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, method, root):
    task = native_task(run_name, method, root)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_critic_units_stage152.py",
        "--method", method, "--seed", str(root), "--output", str(Path(task["result_dir"]) / "result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL native critic coordinates {method} seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{method}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/development",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command))
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    completed = json.loads((ROOT / "results/native_transit_dispatch_training_stage151_development_20261009_r1/summary.json").read_text())
    if not (completed["software_qualified"] and completed["protocol"] == source.EXPERIMENT_PROTOCOL
            and completed["contract"] == source.contract(False)):
        raise SystemExit("Critic units require completed matched native dispatch evidence")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Native critic-coordinate run is already registered")
    tasks = [task_specification(args.run_name, method, root) for method in spec.METHODS for root in spec.ROOTS]
    contract = spec.contract(False)
    evaluations = len(spec.CONDITIONS) * len(contract["scenarios"]) * contract["evaluation_episodes_per_scenario"]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
        "worker_preflight_contract": spec.contract(True),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "budget": {"training_episodes": len(tasks) * contract["train_episodes"],
            "frozen_episodes": len(tasks) * evaluations,
            "native_ticks": len(tasks) * (contract["train_episodes"] + evaluations) * contract["training_clock_s"],
            "worker_preflight_episodes": len(tasks) * 5, "worker_preflight_ticks": len(tasks) * 5 * 5400},
        "tasks": len(tasks), "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
            "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
