#!/usr/bin/env python3
"""Schedule paired cached refit consistency diagnostics on Linux CPU nodes."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_refit_consistency_stage240 as spec
from scripts.submit_native_transit_objective_oof_stage239_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_mean_coverage_stage195_scheduleurm import require_completed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, view=None, fold=None, partition=None, direction=None):
    task = base_task(run_name, view, fold, partition, direction)
    cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_refit_consistency_stage240.py", "--run-name", run_name,
        "--operation", "report" if view is None else "fit"]
    if view is not None: cmd.extend(["--view", view, "--fold", str(fold), "--partition", partition, "--direction", str(direction)])
    suffix = "report" if view is None else f"fit/{view}/{fold}/{partition}/{direction}"
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL cached refit consistency {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(cmd),
        cpu_training_justification="Cached references and fixed selections; only small residual refits and paired diagnostics, no raw training or native rollout.")
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("fit", "report"), required=True)
    args = p.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    if args.phase == "fit":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage240 already registered")
        prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        if prior["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or prior["contract"] != spec.previous.contract() or not prior["coverage_qualified"]:
            raise SystemExit("Complete Stage239 before consistency diagnostics")
        tasks = [task_specification(args.run_name, *k) for k in spec.KEYS]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                "cpu_per_task": tasks[0]["cpu"], "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        if json.loads((directory/"preregistration.json").read_text())["contract"] != spec.contract(): raise SystemExit("Changed consistency protocol")
        require_completed(existing, {task_specification(args.run_name, *k)["signature"] for k in spec.KEYS})
        tasks = [task_specification(args.run_name)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__": main()
