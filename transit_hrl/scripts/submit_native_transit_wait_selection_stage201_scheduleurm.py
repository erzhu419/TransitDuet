#!/usr/bin/env python3
"""Schedule the read-only diagnosis over frozen Stage200 selections."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_native_transit_wait_selection_stage201 as spec
from scripts.submit_native_transit_wait_margin_stage200_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name):
    task = base_task(run_name)
    result = ROOT/"results"/run_name/"report"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_native_transit_wait_selection_stage201.py", "--output", str(result/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description="Freq-HRL frozen wait selection decomposition",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/report",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Read47 frozen prediction files and mean8 labels; no fitting, model inference or simulator updates.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    if inventory(args.run_name, protocol_spec=spec) or (directory/"preregistration.json").exists():
        raise SystemExit("Wait-selection diagnostic is already registered")
    previous = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
    if not (previous["protocol"] == spec.source.EXPERIMENT_PROTOCOL and previous["contract"] == spec.source.contract()
            and previous["software_qualified"] and previous["new_closed_form_fits"] == 188):
        raise SystemExit("Complete Stage200 before decomposing wait selection")
    write_json(directory/"inputs/source.json", {"source_run": spec.SOURCE_RUN, "contract": spec.contract()})
    task = task_specification(args.run_name)
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "tasks": 1,
        "scheduler": {"allowed_nodes": task["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": task["ram_mb"]}})
    execute_bulk([task], dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
