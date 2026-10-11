#!/usr/bin/env python3
"""Schedule a matched frozen score-horizon factorial, not additional training."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_commitment_scoring_stage168 as spec
from scripts.submit_native_transit_plan_credit_stage158_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, scenario):
    task = native_task(run_name, root, scenario)
    inputs = ROOT/"results"/run_name/"inputs"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_commitment_scoring_stage168.py",
        "--seed", str(root), "--scenario", scenario, "--duration-model", str(inputs/"duration_models.json"),
        "--output", str(Path(task["result_dir"])/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL commitment scoring {scenario} seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{scenario}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/frozen",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        stage_input_paths=[*task["stage_input_paths"], str(inputs)],
        cpu_training_justification="Frozen score-horizon factorial only; no native/actor training.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    completed = json.loads((ROOT/"results"/spec.SOURCE_RUN/"summary.json").read_text())
    if not (completed["software_qualified"] and completed["protocol"] == spec.source.EXPERIMENT_PROTOCOL
            and completed["contract"] == spec.source.contract() and completed["source_control_reproductions"] == 40):
        raise SystemExit("Commitment scoring requires the completed Stage167 audit")
    for root in spec.ROOTS:
        for scenario in spec.SCENARIOS:
            spec.source_cell(root, scenario)
            spec.source.source_cell(root, scenario)
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Commitment scoring run is already registered")
    directory = ROOT/"results"/args.run_name
    models = json.loads((ROOT/"results"/spec.SOURCE_RUN/"inputs/duration_models.json").read_text())
    write_json(directory/"inputs/duration_models.json", models)
    tasks = [task_specification(args.run_name, r, s) for r in spec.ROOTS for s in spec.SCENARIOS]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "duration_model_input": "inputs/duration_models.json", "tasks": len(tasks),
        "budget": {"training_episodes": 0, "frozen_episodes": 200, "native_ticks": 200*61380,
            "source_control_reproductions": 80, "window_ledger_qualified_episodes": 160},
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
