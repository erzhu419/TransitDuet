#!/usr/bin/env python3
"""Schedule the native authority repair factorial on the shared CPU pool."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_authority_stage148 as spec
from scripts import run_native_transit_diagnostics_stage147 as diagnosis
from scripts.submit_native_transit_routing_stage146_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, method, root, *, preflight):
    task = native_task(run_name, method, root, preflight=preflight)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_authority_stage148.py",
        "--method", method, "--seed", str(root), "--output", str(Path(task["result_dir"]) / "result.json")]
    if preflight:
        command.append("--preflight")
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL native authority {method} seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{method}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'preflight' if preflight else 'mechanism_development'}",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command))
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-summary", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = json.loads((ROOT / "results/native_transit_diagnostics_stage147_full_20261009_r1/summary.json").read_text())
    if not (source["software_qualified"] and source["protocol"] == diagnosis.EXPERIMENT_PROTOCOL):
        raise SystemExit("Native authority repair requires the completed frozen diagnosis")
    if not args.preflight:
        if args.preflight_summary is None:
            raise SystemExit("Authority development requires its four-method preflight")
        result = json.loads(args.preflight_summary.read_text())
        if not (result["software_qualified"] and result["preflight"]
                and result["protocol"] == spec.EXPERIMENT_PROTOCOL and result["contract"] == spec.contract(True)):
            raise SystemExit("Authority preflight did not qualify")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Native authority run is already registered")
    contract = spec.contract(args.preflight)
    tasks = [task_specification(args.run_name, method, root, preflight=args.preflight)
             for method in spec.METHODS for root in contract["roots"]]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "budget": {"training_episodes": len(tasks) * contract["train_episodes"],
            "frozen_episodes": len(tasks) * len(contract["conditions"]) * len(contract["scenarios"])},
        "tasks": len(tasks), "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
            "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
