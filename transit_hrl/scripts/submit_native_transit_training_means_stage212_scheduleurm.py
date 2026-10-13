#!/usr/bin/env python3
"""Schedule singleton-target replication through the shared CPU scheduler."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_training_means_stage212 as spec
from scripts.submit_native_transit_value_context_stage177_scheduleurm import task_specification as native_task
from scripts.submit_native_transit_expected_value_stage192_scheduleurm import task_specification as offline_task
from scripts.submit_native_transit_mean_coverage_stage195_scheduleurm import require_completed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, key=None, operation="prepare"):
    if key is None:
        task = offline_task(run_name)
        suffix, arguments = operation, ["--operation", operation]
        result = ROOT/"results"/run_name/operation
    else:
        root, scenario, origin, index, probe, draw = key
        task = native_task(run_name, root, scenario)
        suffix = f"draw/{origin}/{scenario}/{root}/{index}/{probe}/{draw}"
        arguments = ["--operation", "collect", "--seed", str(root), "--scenario", scenario,
            "--origin", origin, "--scene-index", str(index), "--probe", str(probe), "--draw", str(draw)]
        result = spec.draw_path(run_name, key).parent
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_training_means_stage212.py", "--run-name", run_name, *arguments]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL conditional training means {suffix}",
        ram_mb=1536 if key is not None else task["ram_mb"],
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'native' if key is not None else 'offline'}",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Eight frozen matched native episodes per draw; unchanged inputs, separate target precision and sample weights, remote raw storage.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("preflight", "preflight_report", "remaining", "prepare"), required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    if args.phase == "preflight":
        if existing or (directory/"preregistration.json").exists():
            raise SystemExit("Training-target experiment is already registered")
        report = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        if report["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or report["contract"] != spec.previous.contract() or not report["software_qualified"]:
            raise SystemExit("Complete the bounded-head experiment before changing target precision")
        registered = spec.manifest(json.loads((ROOT/"results"/spec.SOURCE_RUN/"inputs/manifest.json").read_text()))
        write_json(directory/"inputs/manifest.json", registered)
        write_json(directory/"inputs/models.json", json.loads((ROOT/"results"/spec.MODEL_RUN/"inputs/models.json").read_text()))
        tasks = [task_specification(args.run_name, k) for k in spec.cells(registered, preflight=True)]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "preflight_tasks": 20, "collection_tasks": 2880, "preflight_report_tasks": 1, "prepare_tasks": 1,
            "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        registered = json.loads((directory/"inputs/manifest.json").read_text())
        if registered["contract"] != spec.contract():
            raise SystemExit("Changed the registered target-precision protocol")
        first = spec.cells(registered, preflight=True)
        keys = spec.cells(registered) if args.phase == "prepare" else first
        expected = {task_specification(args.run_name, k)["signature"] for k in keys}
        if args.phase != "preflight_report":
            expected.add(task_specification(args.run_name, operation="preflight_report")["signature"])
        require_completed(existing, expected)
        if args.phase == "remaining":
            report = json.loads((directory/"preflight_report/receipt.json").read_text())
            if not (report["protocol"] == spec.EXPERIMENT_PROTOCOL and report["contract"] == spec.contract()
                    and report["software_qualified"] and report["draws"] == 20 and report["native_episodes"] == 160
                    and report["by_origin"] == {"existing": 10, "fresh": 10}):
                raise SystemExit("Qualify both replay origins before the remaining 2860 draws")
            tasks = [task_specification(args.run_name, k) for k in spec.cells(registered) if k not in set(first)]
        else:
            tasks = [task_specification(args.run_name, operation=args.phase)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
