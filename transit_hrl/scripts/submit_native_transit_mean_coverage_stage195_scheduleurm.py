#!/usr/bin/env python3
"""Schedule qualified continuations, mean preparation, grouped fits and report."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_mean_coverage_stage195 as spec
from scripts.submit_native_transit_value_expansion_stage187_scheduleurm import task_specification as native_task
from scripts.submit_native_transit_expected_value_stage192_scheduleurm import task_specification as offline_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, *, key=None, fold=None, operation="prepare"):
    if key is not None:
        root, scenario, replica, probe, draw = key
        task = native_task(run_name, root, scenario, replica, probe)
        result = spec.draw_path(run_name, *key).parent
        suffix = f"draw/{scenario}/{root}/{replica}/{probe}/{draw}"
        arguments = ["--operation", "collect", "--seed", str(root), "--scenario", scenario,
            "--replica", str(replica), "--probe", str(probe), "--draw", str(draw)]
    else:
        task = offline_task(run_name)
        suffix = operation if fold is None else f"fit/{fold['split']}/{fold['index']}"
        result = ROOT/"results"/run_name/operation if fold is None else spec.result_path(run_name, "tail_summary", fold).parent
        arguments = ["--operation", operation] if fold is None else ["--operation", "fit",
            "--split", fold["split"], "--fold-index", str(fold["index"])]
    script = "analyze_native_transit_mean_coverage_stage195.py" if operation == "report" else "run_native_transit_mean_coverage_stage195.py"
    command = [DEFAULT_LINUX_PYTHON, "-u", f"scripts/{script}", "--run-name", run_name]
    command.extend(["--output", str(result/"result.json")] if operation == "report" else arguments)
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL complete-group mean coverage {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'native' if key is not None else 'offline'}",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Eight frozen matched native replays, server-only target averaging, one100-step shared three-head fit, or compact report.")
    return task


def require_completed(existing, expected):
    if {t["signature"] for t in existing} != expected or any(t["status"] != "done" for t in existing):
        raise SystemExit("Complete exactly the registered preceding phase before dispatching more work")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("preflight", "remaining", "prepare", "fits", "report"), required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    existing = inventory(args.run_name, protocol_spec=spec)
    if args.phase == "preflight":
        if existing or (directory/"preregistration.json").exists():
            raise SystemExit("Mean-coverage experiment is already registered")
        previous = json.loads((ROOT/"results"/spec.PREVIOUS_RUN/"report/result.json").read_text())
        if not (previous["software_qualified"] and previous["new_fits"] == 51
                and previous["protocol"] == "native_transit_fleet_credit_stage194_v1"):
            raise SystemExit("Complete Stage194 before expanding target coverage")
        registered = spec.manifest(json.loads((ROOT/"results"/spec.SOURCE_RUN/"inputs/manifest.json").read_text()))
        models = json.loads((ROOT/"results"/spec.noise.COLLECTION_RUN/"inputs/models.json").read_text())
        write_json(directory/"inputs/manifest.json", registered)
        write_json(directory/"inputs/models.json", models)
        tasks = [task_specification(args.run_name, key=k) for k in spec.cells(preflight=True)]
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "preflight_tasks": 10, "collection_tasks": 1680, "prepare_tasks": 1, "fit_tasks": 47, "report_tasks": 1,
            "budget": spec.contract()["budget"], "mean_states": registered["mean_states"],
            "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]}})
    else:
        registered = json.loads((directory/"inputs/manifest.json").read_text())
        preregistered = json.loads((directory/"preregistration.json").read_text())
        if registered["contract"] != spec.contract() or preregistered["contract"] != spec.contract():
            raise SystemExit("Changed registered coverage protocol")
        keys = spec.cells(preflight=True) if args.phase == "remaining" else spec.cells()
        preceding = {task_specification(args.run_name, key=k)["signature"] for k in keys}
        if args.phase in ("fits", "report"):
            preceding.add(task_specification(args.run_name)["signature"])
        if args.phase == "report":
            preceding.update(task_specification(args.run_name, fold=f)["signature"] for f in registered["folds"])
        require_completed(existing, preceding)
        if args.phase == "remaining":
            first = set(spec.cells(preflight=True))
            tasks = [task_specification(args.run_name, key=k) for k in spec.cells() if k not in first]
        elif args.phase == "fits":
            tasks = [task_specification(args.run_name, fold=f) for f in registered["folds"]]
        else:
            tasks = [task_specification(args.run_name, operation=args.phase)]
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
