#!/usr/bin/env python3
"""Schedule twenty passive full-timetable replays, without branch reruns."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_timetable_context_stage185 as spec
from scripts.submit_native_transit_value_context_stage177_scheduleurm import task_specification as base_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, scenario, scene_index):
    task = base_task(run_name, root, scenario)
    inputs = ROOT/"results"/run_name/"inputs"
    result = Path(task["result_dir"])/f"scene_{scene_index}"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_timetable_context_stage185.py",
        "--seed", str(root), "--scenario", scenario, "--scene-index", str(scene_index),
        "--models", str(inputs/"models.json"), "--training-data", str(inputs/"training.json"),
        "--output", str(result/"result.json")]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL full timetable context {scenario} root{root} scene{scene_index}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{scenario}/{root}/{scene_index}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/audit",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="One unchanged native replay to recover complete preaction timetables; no branch or learning updates.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    source = ROOT/"results"/spec.SOURCE_RUN
    summary = json.loads((source/"summary.json").read_text())
    if not (summary["software_qualified"] and summary["protocol"] == spec.prior.EXPERIMENT_PROTOCOL
            and summary["contract"] == spec.prior.contract()):
        raise SystemExit("Complete Stage184 representation diagnosis first")
    rows = spec.prior.source.source.source.training_rows(*(json.loads((source/"inputs"/f"{n}.json").read_text())
        for n in ("first", "labels", "second", "baselines")))
    models = json.loads((ROOT/"results"/spec.MODEL_SOURCE_RUN/"inputs/models.json").read_text())
    transfer_models = json.loads((ROOT/"results/native_transit_value_transfer_stage179_frozen_20261011_r1/inputs/models.json").read_text())
    if models != transfer_models:
        raise SystemExit("The two labelled banks used different duration inputs")
    for root in spec.ROOTS:
        for scenario in spec.SCENARIOS:
            for i in (0, 1):
                ref = spec.labels.source_record(root, scenario, scene_index=i)
                if sum((r["root"], r["scenario"], r["scene_seed"]) == (root, scenario, ref["scene_seed"]) for r in rows) != 6:
                    raise SystemExit("Full timetable replay did not match an existing development scene")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Full timetable context run is already registered")
    directory = ROOT/"results"/args.run_name
    write_json(directory/"inputs/training.json", rows)
    write_json(directory/"inputs/models.json", models)
    tasks = [task_specification(args.run_name, r, s, i) for r in spec.ROOTS for s in spec.SCENARIOS for i in (0, 1)]
    write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": 20, "budget": {"new_native_episodes": 20, "new_native_steps": 20*61380,
            "paired_labels_reused": 720, "new_counterfactual_episodes": 0, "policy_updates": 0},
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
