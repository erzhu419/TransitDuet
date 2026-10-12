#!/usr/bin/env python3
"""Dispatch software preflight, remaining repeats, or a server-side report."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_continuation_noise_stage191 as spec
from scripts.submit_native_transit_value_expansion_stage187_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_calibrated_value_stage190_scheduleurm import task_specification as report_base
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, scenario, probe, draw):
    task = base_task(run_name, root, scenario, 0, probe)
    result = spec.result_path(ROOT/"results"/run_name, root, scenario, probe, draw).parent
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_continuation_noise_stage191.py",
        "--seed", str(root), "--scenario", scenario, "--probe", str(probe), "--draw", str(draw),
        "--models", str(ROOT/"results"/run_name/"inputs/models.json"), "--output", str(result/"receipt.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL same-state continuation noise {scenario} root{root} probe{probe} draw{draw}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{scenario}/{root}/{probe}/{draw}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/data", result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Eight frozen native replays on an existing state; vary only post-probe innovations, no training.")
    return task


def report_task_specification(run_name):
    task = report_base(run_name)
    result = ROOT/"results"/run_name/"report"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/analyze_native_transit_continuation_noise_stage191.py",
        "--run-name", run_name, "--output", str(result/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/report",
        description="Freq-HRL conditional innovation variance report", resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/report",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Aggregate existing conditional-draw labels into compact diagnostics; no simulation or training.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("preflight", "remaining", "report"), required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    first = spec.cells(preflight=True)
    if args.phase == "preflight":
        if existing or (directory/"preregistration.json").exists():
            raise SystemExit("Continuation diagnostic is already registered")
        previous = json.loads((ROOT/"results"/spec.CALIBRATION_RUN/"report/result.json").read_text())
        if not (previous["software_qualified"] and previous["protocol"] == spec.calibration_source.EXPERIMENT_PROTOCOL
                and previous["contract"] == spec.calibration_source.contract() and previous["fits"] == 856):
            raise SystemExit("Complete Stage190 before changing the evidence target")
        write_json(directory/"inputs/models.json", json.loads((ROOT/"results"/spec.COLLECTION_RUN/"inputs/models.json").read_text()))
        tasks = [task_specification(args.run_name, *k) for k in first]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "states": [{"root": r, "scenario": s, "scene_seed": spec.source.scene_seed(r, s, 0), "probe_index": p}
                for r in spec.ROOTS for s in spec.SCENARIOS for p in spec.PROBES],
            "tasks": 240, "preflight_tasks": 10, "report_tasks": 1, "budget": spec.contract()["budget"],
            "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    elif args.phase == "remaining":
        if {t["signature"] for t in existing} != {task_specification(args.run_name, *k)["signature"] for k in first}:
            raise SystemExit("Remaining draws require exactly the registered ten preflight tasks")
        # Workers qualify full labels before exiting; no receipt download is needed.
        if any(t["status"] != "done" or t.get("exit_code") != 0 for t in existing):
            raise SystemExit("Complete software qualification on both roots and all regimes first")
        tasks = [task_specification(args.run_name, *k) for k in spec.cells() if k not in first]
    else:
        if {t["signature"] for t in existing} != {task_specification(args.run_name, *k)["signature"] for k in spec.cells()}:
            raise SystemExit("Register all conditional draws before node-side aggregation")
        tasks = [report_task_specification(args.run_name)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
