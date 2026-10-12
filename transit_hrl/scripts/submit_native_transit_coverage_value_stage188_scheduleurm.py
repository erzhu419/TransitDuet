#!/usr/bin/env python3
"""Prepare the bank remotely, then dispatch 216 matched fold fits."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_coverage_value_stage188 as spec
from scripts.submit_native_transit_sequence_value_stage186_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def result_path(directory, mode, fold):
    return directory/"cells"/mode/fold["split"]/f"fold_{fold['index']:03}"/"result.json"


def task_specification(run_name, *, mode=None, fold=None):
    task = base_task(run_name)
    directory = ROOT/"results"/run_name
    operation = "prepare" if mode is None else "fit"
    result = directory/"prepare" if mode is None else result_path(directory, mode, fold).parent
    suffix = "prepare" if mode is None else f"{mode}/{fold['split']}/{fold['index']}"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_coverage_value_stage188.py",
        "--run-name", run_name, "--operation", operation, "--output", str(result/"result.json")]
    if mode is not None:
        command.extend(["--mode", mode, "--split", fold["split"], "--fold-index", str(fold["index"])])
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL expanded coverage {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{operation}",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Qualified server-only bank assembly or one 100-step tiny grouped value fit; no native or actor updates.")
    return task


def report_task_specification(run_name):
    task = task_specification(run_name)
    result = ROOT/"results"/run_name/"report"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_native_transit_coverage_value_stage188.py",
        "--run-name", run_name, "--output", str(result/"result.json")]
    task.update(description="Freq-HRL compact expanded coverage comparison",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/report",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/report",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Aggregate completed matched folds into compact summary only; no model, actor or native updates.")
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
            raise SystemExit("Coverage value run is already registered")
        old = ROOT/"results"/spec.OLD_RUN
        old_rows = spec.prior.training_rows(json.loads((old/"inputs/data.json").read_text()),
            json.loads((old/"inputs/original.json").read_text()))
        for path in (old/"fit/result.json", old/"inputs/control.json"):
            cell = json.loads(path.read_text())
            if not (cell["software_qualified"] and len(cell["fit_ledger"]) == 28
                    and all(f["steps"] == spec.STEPS for f in cell["fit_ledger"])):
                raise SystemExit("Coverage comparison requires the frozen 100-step historical references")
        source_tasks = inventory(spec.COLLECTION_RUN, protocol_spec=spec.collection)
        if not (len({t["signature"] for t in source_tasks if t["status"] == "done"}) == 480
                and all(t["status"] == "done" for t in source_tasks)):
            raise SystemExit("Complete the fixed 480-cell development collection first")
        registered = spec.manifest(old_rows)
        write_json(directory/"inputs/manifest.json", registered)
        task = task_specification(args.run_name)
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tasks": 217, "folds_per_mode": len(registered["folds"]), "budget": spec.contract()["budget"],
            "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
                "cpu_per_task": 1, "ram_mb_per_task": task["ram_mb"]}})
        tasks = [task]
    elif args.phase == "fits":
        receipt = json.loads((directory/"prepare/result.json").read_text())
        if not (receipt["protocol"] == spec.EXPERIMENT_PROTOCOL and receipt["contract"] == spec.contract()
                and receipt["software_qualified"] and receipt["training_states"] == 600
                and receipt["paired_labels"] == 3600 and receipt["independent_scene_groups"] == 100
                and {t["signature"] for t in existing} == {task_specification(args.run_name)["signature"]}):
            raise SystemExit("Coverage fits require the prepared complete server-only bank")
        registered = json.loads((directory/"inputs/manifest.json").read_text())
        tasks = [task_specification(args.run_name, mode=m, fold=f) for m in spec.MODES for f in registered["folds"]]
    else:
        registered = json.loads((directory/"inputs/manifest.json").read_text())
        expected = {task_specification(args.run_name)["signature"],
            *[task_specification(args.run_name, mode=m, fold=f)["signature"] for m in spec.MODES for f in registered["folds"]]}
        if {t["signature"] for t in existing} != expected:
            raise SystemExit("Register every coverage fold before node-side aggregation")
        # Completion is verified from exported folds on the shared server; status can lag.
        tasks = [report_task_specification(args.run_name)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
