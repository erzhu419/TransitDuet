#!/usr/bin/env python3
"""Schedule a matched two-feature tail-summary ablation on existing labels."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_tail_value_stage184 as spec
from scripts.submit_native_transit_proper_value_stage183_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name):
    task = base_task(run_name)
    directory = ROOT/"results"/run_name
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_tail_value_stage184.py"]
    for name in ("first", "labels", "second", "baselines", "control"):
        command.extend([f"--{name}", str(directory/"inputs"/f"{name}.json")])
    command.extend(["--output", str(Path(task["result_dir"])/"result.json")])
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL causal full-tail summaries",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/tail_summary",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="28 tiny 98-16-3 same-step critic fits; no new control, actor or native episodes.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    prior = ROOT/"results"/spec.SOURCE_RUN
    summary = json.loads((prior/"summary.json").read_text())
    if not (summary["protocol"] == spec.source.EXPERIMENT_PROTOCOL and summary["contract"] == spec.source.contract()
            and summary["software_qualified"]):
        raise SystemExit("Complete Stage183 proper-value qualification first")
    inputs = {n: json.loads((prior/"inputs"/f"{n}.json").read_text()) for n in ("first", "labels", "second", "baselines")}
    inputs["control"] = json.loads((prior/"fit/result.json").read_text())
    rows = spec.source.source.source.training_rows(*(inputs[n] for n in ("first", "labels", "second", "baselines")))
    diagnostics = spec.alias_diagnostics(rows)
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Tail-summary ablation is already registered")
    directory = ROOT/"results"/args.run_name
    for name, data in inputs.items(): write_json(directory/"inputs"/f"{name}.json", data)
    task = task_specification(args.run_name)
    steps = [r["steps"] for r in inputs["control"]["fit_ledger"]]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 1, "frozen_selected_steps": steps, "representation_diagnostics": diagnostics,
        "budget": {"new_fits": 28, "new_critic_updates": sum(steps), "new_inner_updates": 0,
            "new_control_updates": 0, "new_native_episodes": 0, "actor_updates": 0},
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
