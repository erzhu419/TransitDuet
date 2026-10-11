#!/usr/bin/env python3
"""Ten passive native replays; keep branch labels and checkpoints on servers."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_context_stage177 as spec
from scripts.submit_native_transit_plan_credit_stage158_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, scenario):
    task = native_task(run_name, root, scenario)
    inputs = ROOT/"results"/run_name/"inputs"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_value_context_stage177.py",
        "--seed", str(root), "--scenario", scenario, "--models", str(inputs/"models.json"),
        "--training-data", str(inputs/"training.json"), "--output", str(Path(task["result_dir"])/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL passive value context {scenario} root{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{scenario}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/audit",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        stage_input_paths=[*task["stage_input_paths"], str(inputs)],
        cpu_training_justification="One unchanged native source replay; passive preaction context, no policy updates.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    source = ROOT/"results"/spec.SOURCE_RUN
    summary = json.loads((source/"summary.json").read_text())
    spec.prior.training_rows(summary)
    fitted = json.loads((ROOT/"results/native_transit_action_value_stage176_fit_20261011_r1/fit/result.json").read_text())
    if not (fitted["software_qualified"] and fitted["protocol"] == spec.prior.EXPERIMENT_PROTOCOL
            and fitted["contract"] == spec.prior.contract()):
        raise SystemExit("Passive value context requires completed fixed Stage176 validation")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Passive value-context run is already registered")
    directory = ROOT/"results"/args.run_name
    write_json(directory/"inputs/training.json", summary)
    write_json(directory/"inputs/models.json", json.loads((source/"inputs/models.json").read_text()))
    tasks = [task_specification(args.run_name, r, s) for r in spec.ROOTS for s in spec.SCENARIOS]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 10, "budget": {"new_native_episodes": 10, "new_native_steps": 613800,
            "paired_labels_reused": 360, "policy_updates": 0},
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
