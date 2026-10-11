#!/usr/bin/env python3
"""Schedule one frozen-budget full timetable encoder experiment."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_sequence_value_stage186 as spec
from scripts.submit_native_transit_tail_value_stage184_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name):
    task = base_task(run_name)
    directory = ROOT/"results"/run_name
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_sequence_value_stage186.py"]
    for name in ("data", "original", "control"):
        command.extend([f"--{name}", str(directory/"inputs"/f"{name}.json")])
    command.extend(["--output", str(Path(task["result_dir"])/"result.json")])
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL learned complete timetable encoder",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/timetable_sequence",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="28 same-step 2147-parameter sequence value fits; no new control, actor or simulator runs.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    source = ROOT/"results"/spec.SOURCE_RUN
    inputs = {"data": json.loads((source/"summary.json").read_text()),
        "original": json.loads((source/"inputs/training.json").read_text()),
        "control": json.loads((ROOT/"results"/spec.CONTROL_RUN/"fit/result.json").read_text())}
    spec.training_rows(inputs["data"], inputs["original"])
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Sequence value experiment is already registered")
    directory = ROOT/"results"/args.run_name
    for name, data in inputs.items(): write_json(directory/"inputs"/f"{name}.json", data)
    task = task_specification(args.run_name)
    steps = [r["steps"] for r in inputs["control"]["fit_ledger"]]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 1, "frozen_selected_steps": steps, "budget": {"new_fits": 28, "new_critic_updates": sum(steps),
            "new_control_updates": 0, "new_inner_updates": 0, "new_native_episodes": 0, "actor_updates": 0},
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
