#!/usr/bin/env python3
"""Schedule capacity-matched fleet-credit fits, then a compact node-side report."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_fleet_credit_stage194 as spec
from scripts.submit_native_transit_fleet_residual_stage193_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, *, fold=None):
    task = base_task(run_name)
    result = ROOT/"results"/run_name/"report" if fold is None else spec.result_path(run_name, fold).parent
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_native_transit_fleet_credit_stage194.py",
        "--run-name", run_name, "--output", str(result/"result.json")] if fold is None else [
        DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_fleet_credit_stage194.py", "--run-name", run_name,
        "--split", fold["split"], "--fold-index", str(fold["index"])]
    suffix = "report" if fold is None else f"{fold['split']}/{fold['index']}"
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL separated fleet credit {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline", result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Three100-step1601-parameter critics or compact aggregation; frozen controls, no simulator/actor updates.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("fits", "report"), required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    if args.phase == "fits":
        if existing or (directory/"preregistration.json").exists():
            raise SystemExit("Fleet-credit experiment is already registered")
        completed = json.loads((ROOT/"results"/spec.PREVIOUS_RUN/"report/result.json").read_text())
        if not (completed["protocol"] == spec.previous.EXPERIMENT_PROTOCOL and completed["contract"] == spec.previous.contract()
                and completed["software_qualified"] and completed["new_fits"] == 34 and completed["reused_fits"] == 17):
            raise SystemExit("Complete Stage193 before changing fleet-credit training")
        registered = spec.manifest(json.loads((ROOT/"results"/spec.PREVIOUS_RUN/"inputs/manifest.json").read_text()))
        write_json(directory/"inputs/manifest.json", registered)
        tasks = [task_specification(args.run_name, fold=f) for f in registered["folds"]]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tasks": 18, "budget": spec.contract()["budget"],
            "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        registered = json.loads((directory/"inputs/manifest.json").read_text())
        expected = {task_specification(args.run_name, fold=f)["signature"] for f in registered["folds"]}
        if {t["signature"] for t in existing} != expected or any(t["status"] != "done" for t in existing):
            raise SystemExit("Complete all seventeen fleet-credit folds before aggregation")
        tasks = [task_specification(args.run_name)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
