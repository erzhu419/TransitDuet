#!/usr/bin/env python3
"""Schedule nested scene fitting; reuse the completed fixed-step control."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_scene_stop_stage182 as spec
from scripts.submit_native_transit_occupancy_value_stage178_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name):
    task = base_task(run_name, "scene_stop")
    directory = ROOT/"results"/run_name
    result = directory/"fit"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_scene_stop_stage182.py"]
    for name in ("first", "labels", "second", "baselines"):
        command.extend([f"--{name}", str(directory/"inputs"/f"{name}.json")])
    command.extend(["--output", str(result/"result.json")])
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL nested scene checkpoint and refit",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/scene_stop",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Small nested 96-16-3 critic fits on existing labels; no control retraining or new native episodes.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    prior = ROOT/"results"/spec.SOURCE_RUN
    completed = json.loads((prior/"summary.json").read_text())
    if not (completed["protocol"] == spec.source.EXPERIMENT_PROTOCOL and completed["contract"] == spec.source.contract()
            and completed["software_qualified"] and completed["critic_updates"] == 84000):
        raise SystemExit("Complete the Stage181 matched analysis before registering Stage182")
    inputs = {n: json.loads((prior/"inputs"/f"{n}.json").read_text()) for n in ("first", "labels", "second", "baselines")}
    inputs["control"] = json.loads((prior/"cells/floor_absolute/result.json").read_text())
    rows = spec.source.training_rows(*(inputs[n] for n in ("first", "labels", "second", "baselines")))
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Nested scene run is already registered")
    directory = ROOT/"results"/args.run_name
    for name, data in inputs.items(): write_json(directory/"inputs"/f"{name}.json", data)
    task = task_specification(args.run_name)
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 1, "training_groups": [list(g) for g in sorted({spec.paired.scene_group(r) for r in rows})],
        "budget": {"new_fits": 56, "max_new_critic_updates": 84000, "historical_control_updates": 42000,
            "new_native_episodes": 0, "actor_updates": 0},
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
