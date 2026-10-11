#!/usr/bin/env python3
"""Schedule two small critics; both decision ablations share every fitted model."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_floor_value_stage181 as spec
from scripts.submit_native_transit_occupancy_value_stage178_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, method):
    task = base_task(run_name, method)
    directory = ROOT/"results"/run_name
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_floor_value_stage181.py", "--method", method]
    for name in ("first", "labels", "second", "baselines"):
        command.extend([f"--{name}", str(directory/"inputs"/f"{name}.json")])
    command.extend(["--output", str(Path(task["result_dir"])/"result.json")])
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL anchored fleet value {method}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{method}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Matched 96-16-3 value fits on existing labels; selectors reuse weights, no new native episodes.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    prior = ROOT/"results/native_transit_value_balance_stage180_fit_20261011_r1"
    completed = json.loads((prior/"summary.json").read_text())
    if not (completed["protocol"] == spec.source.EXPERIMENT_PROTOCOL and completed["contract"] == spec.source.contract()
            and completed["software_qualified"] and completed["critic_updates"] == 84000):
        raise SystemExit("Complete the Stage180 matched comparison before registering Stage181")
    inputs = {name: json.loads((prior/"inputs"/f"{name}.json").read_text()) for name in ("first", "labels", "second")}
    inputs["baselines"] = spec.source_baselines()
    rows = spec.training_rows(*(inputs[n] for n in ("first", "labels", "second", "baselines")))
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Anchored fleet run is already registered")
    directory = ROOT/"results"/args.run_name
    for name, data in inputs.items(): write_json(directory/"inputs"/f"{name}.json", data)
    tasks = [task_specification(args.run_name, m) for m in spec.value.METHODS]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 2, "decision_variants": 4, "training_groups": [list(g) for g in sorted({spec.paired.scene_group(r) for r in rows})],
        "budget": {"fitted_models": 56, "critic_updates": 56*spec.occupancy.STEPS, "new_native_episodes": 0, "actor_updates": 0},
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
