#!/usr/bin/env python3
"""Schedule signed native dispatch qualification on dynamic CPU nodes."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_dispatch_stage150 as spec
from scripts.submit_native_transit_preservation_stage145_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root):
    task = native_task(run_name, root, preflight=False)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_dispatch_stage150.py",
        "--seed", str(root), "--output", str(Path(task["result_dir"]) / "result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL causal signed native dispatch qualification seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/qualification",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Frozen native signed dispatch execution; zero optimizer updates; server-only checkpoint reuse.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    frontier = spec.frontier
    source = json.loads((ROOT / "results/native_transit_frontier_stage149_full_20261009_r1/summary.json").read_text())
    if not (source["software_qualified"] and source["protocol"] == frontier.EXPERIMENT_PROTOCOL
            and source["contract"] == frontier.contract(False)):
        raise SystemExit("Signed dispatch follows completed Stage149 opportunity diagnosis")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Signed dispatch qualification run is already registered")
    tasks = [task_specification(args.run_name, root) for root in spec.ROOTS]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "budget": {"training_updates": 0, "episodes": len(tasks) * len(spec.CONDITIONS),
                   "native_ticks": len(tasks) * len(spec.CONDITIONS) * 61380},
        "tasks": len(tasks), "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
            "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
