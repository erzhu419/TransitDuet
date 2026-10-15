#!/usr/bin/env python3
"""Seal Stage252 actions, qualify native replay, then dispatch fresh futures."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_selector_confirmation_stage253 as spec
from scripts.submit_native_transit_frozen_future_stage241_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_pair_critic_stage246_scheduleurm import require_completed_attempts
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run, *, operation="seal", key=None):
    task = base_task(run, key=key)
    args = ["--operation", operation]
    if key is None:
        suffix, result = operation, ROOT/"results"/run/operation
    else:
        r, s, i, p, d = key
        suffix = f"{operation}/{s}/{r}/{i}/{p}/{d}"
        result = spec.draw_path(run, *key).parent if operation == "preflight" else spec.chunk_path(run, key).parent
        args += ["--seed", str(r), "--scenario", s, "--replica", str(i), "--probe", str(p),
            "--draw" if operation == "preflight" else "--chunk", str(d)]
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_selector_confirmation_stage253.py", "--run-name", run, *args]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL frozen selector confirmation {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'native' if key else 'offline'}",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="New paired frozen-action futures with native source and identity controls; no fitting, inference or policy selection.")
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("seal", "preflight", "preflight_report", "remaining", "report"), required=True)
    args = p.parse_args()
    run, phase = args.run_name, args.phase
    directory = ROOT/"results"/run
    existing = inventory(run, protocol_spec=spec)
    first = task_specification(run)
    if phase == "seal":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage253 already registered")
        prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        receipt = json.loads((ROOT/"results"/spec.SOURCE_RUN/"completion_receipt.json").read_text())
        if (prior["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or prior["contract"] != spec.previous.contract()
                or not prior["coverage_qualified"] or prior["reproduction"]["mismatches"]
                or receipt["scheduler"]["status_counts"] != {"done": 33} or receipt["scheduler"]["exit_code_counts"] != {"0": 33}):
            raise SystemExit("Complete qualified Stage252 before fresh confirmation")
        native = task_specification(run, operation="preflight", key=(*spec.coverage.states()[0], 0))
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": first["allowed_nodes"], "require_node": None, "cpu_per_task": 1,
                "seal_ram_mb": first["ram_mb"], "native_ram_mb": native["ram_mb"]}})
        tasks = [first]
    else:
        if json.loads((directory/"preregistration.json").read_text())["contract"] != spec.contract(): raise SystemExit("Changed registered confirmation")
        sealed = json.loads((directory/"seal/result.json").read_text())
        if sealed["contract"] != spec.contract() or not sealed["coverage_qualified"]: raise SystemExit("Qualify and sync action seal first")
        initial = [task_specification(run, operation="preflight", key=k) for k in spec.preflight_keys(sealed)]
        preceding = [first]
        if phase != "preflight": preceding += initial
        if phase in ("remaining", "report"): preceding.append(task_specification(run, operation="preflight_report"))
        if phase == "report": preceding.extend(task_specification(run, operation="collect", key=k) for k in spec.chunk_keys(sealed))
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        records = {t["id"]: t for t in scheduler.load_state()["tasks"]}
        missing = {t["id"] for t in existing}-records.keys()
        if missing: records.update({t["id"]: t for t in scheduler.load_archive_tasks() if t["id"] in missing})
        require_completed_attempts(existing, preceding, records)
        if phase == "preflight": tasks = initial
        elif phase == "remaining":
            receipt = json.loads((directory/"preflight_report/result.json").read_text())
            if receipt["contract"] != spec.contract() or not receipt["coverage_qualified"]: raise SystemExit("Qualify native preflight before expansion")
            tasks = [task_specification(run, operation="collect", key=k) for k in spec.chunk_keys(sealed)]
        else: tasks = [task_specification(run, operation=phase)]
        for task in tasks: task["allow_duplicate"] = True
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{run}")


if __name__ == "__main__": main()
