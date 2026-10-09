#!/usr/bin/env python3
"""Schedule the learned native dispatch factorial on node001-node006."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_dispatch_train_stage151 as spec
from scripts import run_native_transit_dispatch_stage150 as qualification
from scripts.submit_native_transit_routing_stage146_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, method, root):
    task = native_task(run_name, method, root, preflight=False)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_dispatch_train_stage151.py",
        "--method", method, "--seed", str(root), "--output", str(Path(task["result_dir"]) / "result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL matched learned native dispatch {method} seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{method}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/development",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Matched fresh two-level native RE-SAC learning; one CPU per method/root; server-only final checkpoints.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = json.loads((ROOT / "results/native_transit_dispatch_stage150_qualification_20261009_r1/summary.json").read_text())
    if not (source["software_qualified"] and source["protocol"] == qualification.EXPERIMENT_PROTOCOL
            and source["contract"] == qualification.contract()):
        raise SystemExit("Learned dispatch requires qualified causal signed execution")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Learned native dispatch run is already registered")
    tasks = [task_specification(args.run_name, method, root) for method in spec.METHODS for root in spec.ROOTS]
    contract = spec.contract(False)
    eval_episodes = len(spec.CONDITIONS) * len(contract["scenarios"]) * contract["evaluation_episodes_per_scenario"]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
        "worker_preflight_contract": spec.contract(True),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "budget": {"training_episodes": len(tasks) * contract["train_episodes"],
            "frozen_episodes": len(tasks) * eval_episodes,
            "native_ticks": len(tasks) * (contract["train_episodes"] + eval_episodes) * contract["training_clock_s"],
            "worker_preflight_episodes": len(tasks) * 5, "worker_preflight_ticks": len(tasks) * 5 * 5400,
            "upper_updates_per_cell": spec.expected_updates(False)["upper"],
            "lower_updates_per_cell": spec.expected_updates(False)["lower"]},
        "tasks": len(tasks), "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
            "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
