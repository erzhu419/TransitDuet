#!/usr/bin/env python3
"""Schedule one conditional-curvature analysis of saved prefix models."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_native_transit_curvature_stage261 as spec
from scripts.submit_native_transit_prefix_influence_stage259_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON,execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run):
    task = base_task(run)
    command = [DEFAULT_LINUX_PYTHON,"-u","scripts/run_native_transit_curvature_stage261.py","--run-name",run]
    task.update(project=spec.EXPERIMENT_PROTOCOL,description="Freq-HRL cached curvature and prefix leverage",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/analysis",resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Small cached weighted-design decompositions and inference only; no fit, selection, evaluation panels or native simulation.")
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name",required=True)
    a = p.parse_args()
    directory = ROOT/"results"/a.run_name
    if inventory(a.run_name,protocol_spec=spec) or (directory/"preregistration.json").exists(): raise SystemExit("Stage261 already registered")
    report = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
    receipt = json.loads((ROOT/"results"/spec.SOURCE_RUN/"completion_receipt.json").read_text())
    if (report["protocol"]!=spec.source.EXPERIMENT_PROTOCOL or report["contract"]!=spec.source.contract()
            or not report["coverage_qualified"] or receipt["scheduler"]["status_counts"]!={"done":33}
            or receipt["scheduler"]["exit_code_counts"]!={"0":33}): raise SystemExit("Complete qualified Stage260 before curvature analysis")
    task = task_specification(a.run_name)
    write_json(directory/"preregistration.json",{"protocol":spec.EXPERIMENT_PROTOCOL,"contract":spec.contract(),
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "scheduler":{"allowed_nodes":task["allowed_nodes"],"require_node":None,"cpu_per_task":1,"ram_mb":task["ram_mb"]}})
    execute_bulk([task],dry_run=False,intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{a.run_name}")


if __name__=="__main__":main()
