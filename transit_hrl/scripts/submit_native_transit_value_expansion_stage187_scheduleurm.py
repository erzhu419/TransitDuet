#!/usr/bin/env python3
"""Register fresh development labels, qualify ten cells, then schedule the rest."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_expansion_stage187 as spec
from scripts.analyze_native_transit_value_expansion_stage187 import load_summary, result_path
from scripts.submit_native_transit_value_context_stage177_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, scenario, replica, probe):
    task = base_task(run_name, root, scenario)
    inputs = ROOT/"results"/run_name/"inputs"
    result = result_path(ROOT/"results"/run_name, root, scenario, replica, probe).parent
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_value_expansion_stage187.py",
        "--seed", str(root), "--scenario", scenario, "--replica", str(replica), "--probe", str(probe),
        "--models", str(inputs/"models.json"), "--output", str(result/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL fresh development {scenario} root{root} replica{replica} probe{probe}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{scenario}/{root}/{replica}/{probe}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/data",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Seven frozen native rollouts for six paired one-action development labels; no actor or critic updates.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("preflight", "remaining"), required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    first = spec.cells(preflight=True)
    if args.phase == "preflight":
        if existing or (directory/"preregistration.json").exists():
            raise SystemExit("Fresh development run is already registered")
        prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"summary.json").read_text())
        if not (prior["software_qualified"] and prior["protocol"] == spec.prior.EXPERIMENT_PROTOCOL
                and prior["contract"] == spec.prior.contract() and not prior["third_scene_screen_eligible"]):
            raise SystemExit("Development expansion requires the completed unsuccessful Stage186 screen")
        reserved = {spec.labels.source_record(r, s, scene_index=i)["scene_seed"]
            for r in spec.ROOTS for s in spec.SCENARIOS for i in range(4)}
        reserved.update(spec.shared.source.source.check_scene(r, s) for r in spec.ROOTS for s in spec.SCENARIOS)
        spec.validate_scene_roles(reserved)
        models = json.loads((ROOT/"results"/spec.context_source.MODEL_SOURCE_RUN/"inputs/models.json").read_text())
        write_json(directory/"inputs/models.json", models)
        tasks = [task_specification(args.run_name, *k) for k in first]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scene_manifest": [{"root": r, "scenario": s, "replica": i, "scene_seed": spec.scene_seed(r, s, i)}
                for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.REPLICAS],
            "reserved_scene_seeds": sorted(reserved), "tasks": 480, "preflight_tasks": 10,
            "budget": {"new_native_episodes": 3360, "new_native_steps": 3360*61380,
                "new_development_scene_groups": 80, "new_states": 480, "new_labels": 2880, "policy_updates": 0},
            "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        registered = json.loads((directory/"preregistration.json").read_text())
        if not (registered["protocol"] == spec.EXPERIMENT_PROTOCOL and registered["contract"] == spec.contract()
                and {t["signature"] for t in existing} ==
                {task_specification(args.run_name, *k)["signature"] for k in first}):
            raise SystemExit("Remaining cohort requires exactly the ten preregistered preflight tasks")
        summary = load_summary(directory, preflight=True)
        write_json(directory/"preflight_summary.json", summary)
        tasks = [task_specification(args.run_name, *k) for k in spec.cells() if k not in first]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
