#!/usr/bin/env python3
"""Prepare mean targets remotely, fit34 grouped values, then sync a small report."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_expected_value_stage192 as spec
from scripts.submit_native_transit_calibrated_value_stage190_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, *, mode=None, fold=None, report=False):
    task = base_task(run_name)
    if report:
        result = ROOT/"results"/run_name/"report"
        command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_native_transit_expected_value_stage192.py",
            "--run-name", run_name, "--output", str(result/"result.json")]
        suffix = "report"
    else:
        result = ROOT/"results"/run_name/"prepare" if mode is None else spec.result_path(run_name, mode, fold).parent
        command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_expected_value_stage192.py",
            "--run-name", run_name, "--operation", "prepare" if mode is None else "fit"]
        suffix = "prepare" if mode is None else f"{mode}/{fold['split']}/{fold['index']}"
        if mode is not None:
            command.extend(["--mode", mode, "--split", fold["split"], "--fold-index", str(fold["index"])])
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL expected-label comparison {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline", result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Mean-label preparation, one100-step tiny critic fit or compact aggregation; no native or actor training.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("prepare", "fits", "report"), required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    if args.phase == "prepare":
        if existing or (directory/"preregistration.json").exists():
            raise SystemExit("Expected-label experiment is already registered")
        completed = json.loads((ROOT/"results"/spec.NOISE_RUN/"report/result.json").read_text())
        if not (completed["software_qualified"] and completed["protocol"] == spec.noise.EXPERIMENT_PROTOCOL
                and completed["contract"] == spec.noise.contract() and completed["native_episodes"] == 1920):
            raise SystemExit("Complete Stage191 diagnosis before changing target labels")
        registered = spec.manifest(json.loads((ROOT/"results"/spec.SOURCE_RUN/"inputs/manifest.json").read_text()))
        write_json(directory/"inputs/manifest.json", registered)
        tasks = [task_specification(args.run_name)]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tasks": 36, "budget": spec.contract()["budget"], "fits_per_mode": 17,
            "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        registered = json.loads((directory/"inputs/manifest.json").read_text())
        expected = {task_specification(args.run_name)["signature"]}
        if args.phase == "report":
            expected.update(task_specification(args.run_name, mode=m, fold=f)["signature"] for m in spec.MODES for f in registered["folds"])
        if {t["signature"] for t in existing} != expected or any(t["status"] != "done" for t in existing):
            raise SystemExit("Complete the registered preceding phase first")
        tasks = [task_specification(args.run_name, report=True)] if args.phase == "report" else [
            task_specification(args.run_name, mode=m, fold=f) for m in spec.MODES for f in registered["folds"]]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
