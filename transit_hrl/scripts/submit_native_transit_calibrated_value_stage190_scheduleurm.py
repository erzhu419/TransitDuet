#!/usr/bin/env python3
"""Schedule nested calibration on the existing server bank, then a small report."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_calibrated_value_stage190 as spec
from scripts.submit_native_transit_coverage_value_stage188_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, *, mode=None, fold=None):
    task = base_task(run_name)
    if mode is None:
        result = ROOT/"results"/run_name/"report"
        command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_native_transit_calibrated_value_stage190.py",
            "--run-name", run_name, "--output", str(result/"result.json")]
        suffix = "report"
    else:
        result = spec.result_path(run_name, mode, fold).parent
        command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_calibrated_value_stage190.py",
            "--run-name", run_name, "--mode", mode, "--split", fold["split"], "--fold-index", str(fold["index"])]
        suffix = f"{mode}/{fold['split']}/{fold['index']}"
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL nested cost/wait calibration {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Four tiny training-only subset critics plus scene calibration, or compact aggregation; no simulator or actor updates.")
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
            raise SystemExit("Nested calibration is already registered")
        source_dir = ROOT/"results"/spec.SOURCE_RUN
        receipt = json.loads((source_dir/"report/result.json").read_text())
        if not (receipt["protocol"] == spec.source.EXPERIMENT_PROTOCOL and receipt["contract"] == spec.source.contract()
                and receipt["software_qualified"] and receipt["fits"] == 216):
            raise SystemExit("Complete Stage188 qualification first")
        registered = spec.manifest(json.loads((source_dir/"inputs/manifest.json").read_text()))
        write_json(directory/"inputs/manifest.json", registered)
        tasks = [task_specification(args.run_name, mode=m, fold=f) for m in spec.MODES for f in registered["folds"]]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tasks": len(tasks), "budget": spec.contract()["budget"], "report_tasks": 1,
            "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None, "cpu_per_task": 1,
                "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        registered = json.loads((directory/"inputs/manifest.json").read_text())
        expected = {task_specification(args.run_name, mode=m, fold=f)["signature"] for m in spec.MODES for f in registered["folds"]}
        if {t["signature"] for t in existing} != expected:
            raise SystemExit("Register all nested folds before node-side report")
        tasks = [task_specification(args.run_name)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
