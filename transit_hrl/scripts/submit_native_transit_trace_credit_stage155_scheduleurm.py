#!/usr/bin/env python3
"""Schedule four matched fresh native delayed-credit cells."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_trace_credit_stage155 as spec
from scripts import run_native_transit_phase_adaptation_stage154 as qualification
from scripts.submit_native_transit_critic_regularization_stage153_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, method, root):
    task = native_task(run_name, method, root)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_trace_credit_stage155.py",
        "--method", method, "--seed", str(root), "--output", str(Path(task["result_dir"]) / "result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL delayed upper credit {method} seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{method}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/development",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command))
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    completed = json.loads((ROOT / "results/native_transit_phase_adaptation_stage154_frozen_20261010_r1/summary.json").read_text())
    if not (completed["software_qualified"] and completed["protocol"] == qualification.EXPERIMENT_PROTOCOL
            and completed["contract"] == qualification.contract()):
        raise SystemExit("Trace-credit development requires closed constant-phase diagnosis")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Trace-credit run is already registered")
    tasks = [task_specification(args.run_name, method, root) for method in spec.METHODS for root in spec.ROOTS]
    contract, preflight = spec.contract(False), spec.contract(True)
    episodes = lambda c: len(spec.CONDITIONS) * len(c["scenarios"]) * c["evaluation_episodes_per_scenario"]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract, "worker_preflight_contract": preflight,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "budget": {"training_episodes": len(tasks) * contract["train_episodes"],
            "frozen_episodes": len(tasks) * episodes(contract),
            "native_ticks": len(tasks) * (contract["train_episodes"] + episodes(contract)) * contract["training_clock_s"],
            "worker_preflight_ticks": len(tasks) * (preflight["train_episodes"] + episodes(preflight)) * preflight["training_clock_s"],
            "upper_updates_per_cell": spec.expected_updates(False)["upper"],
            "lower_updates_per_cell": spec.expected_updates(False)["lower"],
            "critic_target_horizon": spec.HORIZONS},
        "tasks": len(tasks), "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
            "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
