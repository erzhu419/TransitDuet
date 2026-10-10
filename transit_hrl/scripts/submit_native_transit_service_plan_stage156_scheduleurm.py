#!/usr/bin/env python3
"""Schedule service-plan qualification using server-resident source checkpoints."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_plan_stage156 as spec
from scripts.submit_native_transit_critic_regularization_stage153_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root):
    task = native_task(run_name, spec.METHOD, root)
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_service_plan_stage156.py",
        "--seed", str(root), "--output", str(Path(task["result_dir"]) / "result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL frozen native service-allocation authority seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{spec.METHOD}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/frozen_authority",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Frozen native lower and server-resident checkpoints; no training.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    source = json.loads((ROOT / "results" / spec.SOURCE_RUN / "summary.json").read_text())
    if not (source["software_qualified"] and source["protocol"] == spec.source_spec.EXPERIMENT_PROTOCOL
            and source["contract"] == spec.source_spec.contract(False)):
        raise SystemExit("Service-allocation qualification requires closed Stage155")
    for root in spec.ROOTS:
        spec.load_source(root)
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Service-allocation qualification is already registered")
    tasks = [task_specification(args.run_name, root) for root in spec.ROOTS]
    contract = spec.contract()
    episodes = len(tasks) * contract["episodes_per_root"]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "budget": {"training_updates": 0, "frozen_episodes": episodes,
                   "native_ticks": episodes * contract["training_clock_s"]},
        "tasks": len(tasks), "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"],
            "require_node": None, "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
