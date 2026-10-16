#!/usr/bin/env python3
"""Schedule frozen-calibration prefix fits after a successful complete preflight."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_calibration_ablation_stage260 as spec
from scripts.submit_native_transit_training_count_stage258_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_pair_critic_stage246_scheduleurm import require_completed_attempts
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run, key=None):
    task = base_task(run, key)
    suffix = f"fit/{key[0]}/{key[1]}/{key[2]}/{key[3]}" if key else "report"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_calibration_ablation_stage260.py",
        "--run-name", run, "--operation", "fit" if key else "report"]
    if key: command += ["--view", key[0], "--fold", str(key[1]), "--partition", key[2], "--direction", str(key[3])]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL frozen calibration {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/{suffix}", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/offline",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Small prefix54 GLS solves with frozen full60 calibration and cached endpoints; no raw/full critic training, selection or native replay.")
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("preflight", "remaining", "report"), required=True)
    a = p.parse_args()
    run, directory = a.run_name, ROOT/"results"/a.run_name
    existing, first = inventory(run, protocol_spec=spec), task_specification(run, spec.PREFLIGHT)
    if a.phase == "preflight":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage260 already registered")
        parent = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        receipt = json.loads((ROOT/"results"/spec.SOURCE_RUN/"completion_receipt.json").read_text())
        if (parent["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or parent["contract"] != spec.previous.contract()
                or not parent["coverage_qualified"] or receipt["scheduler"]["status_counts"] != {"done": 1}
                or receipt["scheduler"]["exit_code_counts"] != {"0": 1}): raise SystemExit("Complete qualified Stage259 before calibration ablation")
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": first["allowed_nodes"], "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": first["ram_mb"]}})
        tasks = [first]
    else:
        if json.loads((directory/"preregistration.json").read_text())["contract"] != spec.contract(): raise SystemExit("Changed registered calibration comparison")
        preceding = [first] if a.phase == "remaining" else [task_specification(run, key) for key in spec.KEYS]
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        records = {t["id"]: t for t in scheduler.load_state()["tasks"]}
        missing = {t["id"] for t in existing}-records.keys()
        if missing: records.update({t["id"]: t for t in scheduler.load_archive_tasks() if t["id"] in missing})
        require_completed_attempts(existing, preceding, records)
        tasks = [task_specification(run, key) for key in spec.KEYS if key != spec.PREFLIGHT] if a.phase == "remaining" else [task_specification(run)]
        for task in tasks: task["allow_duplicate"] = True
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{run}")


if __name__ == "__main__": main()
