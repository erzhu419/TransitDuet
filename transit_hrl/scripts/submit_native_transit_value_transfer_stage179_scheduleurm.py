#!/usr/bin/env python3
"""Schedule sixty unbound frozen transfer cells, keeping raw data on servers."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_transfer_stage179 as spec
from scripts.submit_native_transit_value_data_stage174_scheduleurm import task_specification as label_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, scenario, index):
    task = label_task(run_name, root, scenario, index)
    inputs = ROOT/"results"/run_name/"inputs"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_value_transfer_stage179.py",
        "--seed", str(root), "--scenario", scenario, "--probe", str(index),
        "--models", str(inputs/"models.json"), "--value-model", str(inputs/"value_model.json"),
        "--output", str(Path(task["result_dir"])/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL frozen transfer {scenario} seed{root} probe{index}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{scenario}/{root}/{index}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/frozen",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Frozen critic transfer with matched one-action continuations; zero learned updates.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    primary = json.loads((ROOT/"results"/spec.SOURCE_RUN/"cells"/spec.fitted.PRIMARY/"result.json").read_text())
    pack = spec.model_input(primary)
    scenes = [{"root": r, "scenario": s, "scene_seed": spec.source_record(r, s)["scene_seed"]}
        for r in spec.ROOTS for s in spec.SCENARIOS]
    if any([s["root"], s["scene_seed"]] in pack["model"]["training_groups"] for s in scenes):
        raise SystemExit("Transfer overlaps frozen critic fitting scenes")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Frozen value transfer run is already registered")
    write_json(directory/"inputs/value_model.json", pack)
    write_json(directory/"inputs/models.json", json.loads((ROOT/"results"/spec.labels.SOURCE_RUN/"inputs/duration_models.json").read_text()))
    tasks = [task_specification(args.run_name, r, s, i)
        for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_source_revision": json.loads((ROOT/"results"/spec.SOURCE_RUN/"preregistration.json").read_text())["source_revision"],
        "tasks": len(tasks), "evaluation_scenes": scenes, "critic_training_groups": pack["model"]["training_groups"],
        "budget": {"training_episodes": 0, "critic_updates": 0, "actor_updates": 0,
            "frozen_episodes": 420, "native_ticks": 420*61380, "evaluation_states": 60,
            "source_control_reproductions": 120, "paired_labels": 360, "window_ledger_qualified_episodes": 420},
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
