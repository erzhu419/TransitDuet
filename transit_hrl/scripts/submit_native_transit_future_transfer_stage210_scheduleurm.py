#!/usr/bin/env python3
"""Schedule remote half-label preparation, paired fits and a compact report."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_future_transfer_stage210 as spec
from scripts.submit_native_transit_wait_feasibility_stage206_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_mean_coverage_stage195_scheduleurm import require_completed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, fold=None, *, operation="fit"):
    task = base_task(run_name, fold, operation=operation)
    result = ROOT/"results"/run_name/operation if fold is None else spec.result_path(run_name, fold).parent
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_future_transfer_stage210.py", "--run-name", run_name,
        "--operation", operation]
    if fold is not None:
        command.extend(["--split", fold["split"], "--fold-index", str(fold["index"])])
    suffix = operation if fold is None else f"fit/{fold['split']}/{fold['index']}"
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL paired future/prefix wait transfer {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Four half-trained GLM/neural pairs per fold with diagnostic checkpoints; no actor or native training.")
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
            raise SystemExit("Future-transfer experiment is already registered")
        old = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        if not (old["protocol"] == spec.previous.EXPERIMENT_PROTOCOL and old["contract"] == spec.previous.contract() and old["software_qualified"]):
            raise SystemExit("Complete Stage209 before paired future transfer")
        registered = spec.manifest(json.loads((ROOT/"results"/spec.SOURCE_RUN/"inputs/manifest.json").read_text()))
        write_json(directory/"inputs/manifest.json", registered)
        tasks = [task_specification(args.run_name, operation="prepare")]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "prepare_tasks": 1, "fit_tasks": 47, "report_tasks": 1,
            "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                "cpu_per_task": tasks[0]["cpu"], "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        registered = json.loads((directory/"inputs/manifest.json").read_text())
        if registered["contract"] != spec.contract():
            raise SystemExit("Changed the registered future-transfer protocol")
        prepare = {task_specification(args.run_name, operation="prepare")["signature"]}
        if args.phase == "fits":
            require_completed(existing, prepare)
            tasks = [task_specification(args.run_name, f) for f in registered["folds"]]
        else:
            require_completed(existing, prepare | {task_specification(args.run_name, f)["signature"] for f in registered["folds"]})
            tasks = [task_specification(args.run_name, operation="report")]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
