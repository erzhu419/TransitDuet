#!/usr/bin/env python3
"""Schedule ten unpinned native single-candidate diagnostic cells."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_candidate_value_stage173 as spec
from scripts.submit_native_transit_plan_credit_stage158_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, scenario):
    task = native_task(run_name, root, scenario)
    inputs = ROOT/"results"/run_name/"inputs"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_candidate_value_stage173.py",
        "--seed", str(root), "--scenario", scenario, "--models", str(inputs/"models.json"),
        "--output", str(Path(task["result_dir"])/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL candidate-value {scenario} seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{scenario}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/frozen",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        stage_input_paths=[*task["stage_input_paths"], str(inputs)],
        cpu_training_justification="Frozen native one-action continuation diagnostics; no policy training.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    source = ROOT/"results"/spec.SOURCE_RUN
    completed = json.loads((source/"summary.json").read_text())
    if not (completed["software_qualified"] and completed["protocol"] == spec.source.EXPERIMENT_PROTOCOL
            and completed["contract"] == spec.source.contract() and completed["source_control_reproductions"] == 20
            and completed["window_ledger_qualified_episodes"] == 30):
        raise SystemExit("Candidate values require completed qualified Stage172 evidence")
    probes = [{"root": r, "scenario": s, "index": spec.probe_index(spec.source_cell(r, s))}
        for r in spec.ROOTS for s in spec.SCENARIOS]
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Native candidate-value run is already registered")
    directory = ROOT/"results"/args.run_name
    write_json(directory/"inputs/models.json", json.loads((source/"inputs/models.json").read_text()))
    tasks = [task_specification(args.run_name, r, s) for r in spec.ROOTS for s in spec.SCENARIOS]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_input": "inputs/models.json", "tasks": len(tasks), "probes": probes,
        "budget": {"training_episodes": 0, "frozen_episodes": 70, "native_ticks": 70*61380,
            "source_control_reproductions": 20, "paired_interventions": 60, "window_ledger_qualified_episodes": 70},
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
