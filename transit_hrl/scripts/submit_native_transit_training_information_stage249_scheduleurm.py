#!/usr/bin/env python3
"""Schedule independent training-future collection, then fixed-structure fits."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_training_information_stage249 as spec
from scripts.submit_native_transit_frozen_future_stage241_scheduleurm import task_specification as base_task
from scripts.submit_native_transit_pair_critic_stage246_scheduleurm import require_completed_attempts
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run, *, operation, key=None):
    native = operation in ("preflight", "collect")
    task = base_task(run, key=key if native else None)
    args = ["--operation", operation]
    if native:
        r, s, i, p, c = key
        suffix = f"{operation}/{s}/{r}/{i}/{p}/{c}"
        result = spec.draw_path(run, *key).parent if operation == "preflight" else spec.chunk_path(run, key).parent
        args += ["--seed", str(r), "--scenario", s, "--replica", str(i), "--probe", str(p)]
        if operation == "collect": args += ["--chunk", str(c)]
    elif operation == "fit":
        suffix, result = f"fit/{key[0]}/{key[1]}/{key[2]}/{key[3]}", spec.result_path(run, *key).parent
        args += ["--view", key[0], "--fold", str(key[1]), "--partition", key[2], "--direction", str(key[3])]
    else: suffix, result = operation, ROOT/"results"/run/operation
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_training_information_stage249.py", "--run-name", run, *args]
    task.update(project=spec.EXPERIMENT_PROTOCOL, description=f"Freq-HRL training information {suffix}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run}/{suffix}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'native' if native else 'offline'}",
        result_dir=str(result), local_result_dir=str(result),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= "+shlex.join(command),
        cpu_training_justification="Independent same-prefix frozen native replays or fixed-capacity residual fits; never retrain raw reference or consume Stage248 labels.")
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("preflight", "preflight_report", "remaining", "prepare", "fit_preflight", "fits", "report"), required=True)
    args = p.parse_args()
    run, phase = args.run_name, args.phase
    directory = ROOT/"results"/run
    existing = inventory(run, protocol_spec=spec)
    first = [task_specification(run, operation="preflight", key=k) for k in spec.preflight_keys()]
    if phase == "preflight":
        if existing or (directory/"preregistration.json").exists(): raise SystemExit("Stage249 already registered")
        prior = json.loads((ROOT/"results"/spec.SOURCE_RUN/"report/result.json").read_text())
        if prior["protocol"] != spec.previous.EXPERIMENT_PROTOCOL or prior["contract"] != spec.previous.contract() or not prior["coverage_qualified"]:
            raise SystemExit("Complete Stage247 before information collection")
        write_json(directory/"preregistration.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "scheduler": {"allowed_nodes": first[0]["allowed_nodes"], "require_node": None,
                "cpu_per_task": 1, "native_ram_mb": first[0]["ram_mb"]}})
        tasks = first
    else:
        if json.loads((directory/"preregistration.json").read_text())["contract"] != spec.contract(): raise SystemExit("Changed registered information comparison")
        preceding = list(first)
        if phase != "preflight_report": preceding.append(task_specification(run, operation="preflight_report"))
        if phase in ("prepare", "fit_preflight", "fits", "report"):
            preceding.extend(task_specification(run, operation="collect", key=k) for k in spec.chunk_keys())
        if phase in ("fit_preflight", "fits", "report"): preceding.append(task_specification(run, operation="prepare"))
        if phase in ("fits", "report"):
            fit_keys = spec.KEYS if phase == "report" else (spec.previous.PREFLIGHT,)
            preceding.extend(task_specification(run, operation="fit", key=k) for k in fit_keys)
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        records = {t["id"]: t for t in scheduler.load_state()["tasks"]}
        missing = {t["id"] for t in existing}-records.keys()
        if missing: records.update({t["id"]: t for t in scheduler.load_archive_tasks() if t["id"] in missing})
        require_completed_attempts(existing, preceding, records)
        if phase == "remaining":
            receipt = json.loads((directory/"preflight_report/result.json").read_text())
            if receipt["contract"] != spec.contract() or not receipt["coverage_qualified"]: raise SystemExit("Qualify native preflight before expanding")
            tasks = [task_specification(run, operation="collect", key=k) for k in spec.chunk_keys()]
        elif phase in ("fit_preflight", "fits"):
            keys = (spec.previous.PREFLIGHT,) if phase == "fit_preflight" else [k for k in spec.KEYS if k != spec.previous.PREFLIGHT]
            tasks = [task_specification(run, operation="fit", key=k) for k in keys]
        else: tasks = [task_specification(run, operation=phase)]
        # Exact phase coverage above excludes all new signatures in this trusted batch.
        for task in tasks: task["allow_duplicate"] = True
    execute_bulk(tasks, dry_run=False, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{run}")


if __name__ == "__main__": main()
