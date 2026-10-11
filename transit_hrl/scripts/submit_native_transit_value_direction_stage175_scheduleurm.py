#!/usr/bin/env python3
"""Schedule the missing thirty opposite-direction native training states."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_direction_stage175 as spec
from scripts.submit_native_transit_value_data_stage174_scheduleurm import task_specification as original_task
from scripts.submit_hyperparameter_pilot_scheduleurm import execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, scenario, index):
    task = original_task(run_name, root, scenario, index)
    command = shlex.split(task["cmd"])
    command[command.index("scripts/run_native_transit_value_data_stage174.py")] = "scripts/run_native_transit_value_direction_stage175.py"
    command += ["--paired-data", str(ROOT/"results"/run_name/"inputs/paired_source.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL opposite-direction value {scenario} seed{root} probe{index}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{scenario}/{root}/{index}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/frozen", cmd=shlex.join(command))
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    source_dir = ROOT/"results"/spec.SOURCE_RUN
    previous = json.loads((source_dir/"summary.json").read_text())
    for root in spec.ROOTS:
        for scenario in spec.SCENARIOS:
            registered = spec.source_record(root, scenario)
            for index in spec.PROBES:
                paired = spec.paired_reference(root, scenario, index, previous)
                spec.validate_pair({"seed": root, "scenario": scenario, "probe_index": index,
                    "observer": {"scene_seed": registered["scene_seed"],
                        "probe": {**registered["supply"]["decisions"][index], "index": index}}}, paired)
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Opposite-direction value run is already registered")
    directory = ROOT/"results"/args.run_name
    write_json(directory/"inputs/models.json", json.loads((source_dir/"inputs/models.json").read_text()))
    write_json(directory/"inputs/paired_source.json", previous)
    tasks = [task_specification(args.run_name, r, s, i) for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": len(tasks), "budget": {"new_frozen_episodes": 210, "new_native_ticks": 210*61380,
            "new_source_control_reproductions": 60, "new_paired_interventions": 180,
            "cumulative_training_states": 60, "cumulative_frozen_episodes": 420, "training_updates": 0},
        "model_inputs": ["inputs/models.json", "inputs/paired_source.json"],
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
