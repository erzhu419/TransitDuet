#!/usr/bin/env python3
"""Collect thirty fixed training states through the six-node CPU scheduler."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_data_stage174 as spec
from scripts.submit_native_transit_plan_credit_stage158_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, scenario, index):
    task = native_task(run_name, root, scenario)
    inputs = ROOT/"results"/run_name/"inputs"
    task["result_dir"] = str(Path(task["result_dir"])/f"probe_{index}")
    task["local_result_dir"] = task["result_dir"]
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_value_data_stage174.py",
        "--seed", str(root), "--scenario", scenario, "--probe", str(index), "--models", str(inputs/"models.json"),
        "--output", str(Path(task["result_dir"])/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL value-data {scenario} seed{root} probe{index}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{scenario}/{root}/{index}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/frozen",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        stage_input_paths=[*task["stage_input_paths"], str(inputs)],
        cpu_training_justification="Frozen counterfactual native training-label collection; zero learned policy updates.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    completed = json.loads((ROOT/"results/native_transit_candidate_value_stage173_frozen_20261011_r1/summary.json").read_text())
    if not (completed["software_qualified"] and completed["protocol"] == spec.shared.EXPERIMENT_PROTOCOL
            and completed["contract"] == spec.shared.contract() and completed["source_control_reproductions"] == 20):
        raise SystemExit("Native value labels require completed candidate-value diagnosis")
    scenes = [{"root": r, "scenario": s, "scene_seed": spec.source_record(r, s)["scene_seed"]}
        for r in spec.ROOTS for s in spec.SCENARIOS]
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Native value training data run is already registered")
    directory = ROOT/"results"/args.run_name
    write_json(directory/"inputs/models.json", json.loads((ROOT/"results"/spec.SOURCE_RUN/"inputs/duration_models.json").read_text()))
    tasks = [task_specification(args.run_name, r, s, i) for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_input": "inputs/models.json", "tasks": len(tasks), "training_scenes": scenes,
        "budget": {"training_episodes": 0, "frozen_episodes": 210, "native_ticks": 210*61380,
            "source_control_reproductions": 60, "paired_interventions": 180, "window_ledger_qualified_episodes": 210},
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
