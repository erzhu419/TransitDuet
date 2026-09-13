#!/usr/bin/env python3
"""Freeze and dispatch upper-only sample-consistency cells on the CPU pool."""

import argparse
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import mujoco_v25_sample_consistent_upper_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import (
    DEFAULT_LINUX_PYTHON, LINUX_CPU_NODES, SCHEDULER, STAGE_EXCLUDES, execute_bulk,
)


def cells(preflight=False):
    seeds = spec.PREFLIGHT_SEEDS["optimizer"] if preflight else spec.OPTIMIZER_SEEDS
    return [(env, arm, seed) for seed, env, arm in itertools.product(seeds, spec.ENVIRONMENTS, spec.ARMS)]


def cell_dir(run_name, env, arm, seed):
    return Path("results") / run_name / "cells" / env / arm / f"replicate_{seed}"


def training_command(run_name, cell, *, preflight=False):
    env, arm, seed = cell
    small = cell_dir(run_name, *cell)
    full = Path(".server_artifacts") / run_name / "cells" / env / arm / f"replicate_{seed}"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_mujoco_cell_small_export.py",
               "--full-output-dir", str(full), "--server-full-output-dir", str(full),
               "--export-output-dir", str(small), "--", "--env-id", env, "--optimizer-seed", str(seed)]
    for key, value in spec.options(arm, preflight=preflight).items():
        if value is True:
            command.append("--" + key)
        else:
            command.extend(["--" + key, *map(str, value if isinstance(value, (tuple, list)) else [value])])
    return "MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command)


def task_spec(run_name, cell, *, preflight=False):
    env, arm, seed = cell
    relative = cell_dir(run_name, *cell)
    return {
        "project": spec.PROTOCOL, "description": f"Freq-HRL v25 {run_name} {env} {arm} seed{seed}",
        "cmd": training_command(run_name, cell, preflight=preflight), "cwd": str(ROOT),
        "signature": f"Freq-HRL/{spec.PROTOCOL}/{run_name}/{env}/{arm}/{seed}",
        "resource_family": f"Freq-HRL/{spec.PROTOCOL}/cell", "cpu": 1, "ram_mb": 1536, "vram": 0,
        "priority": "normal", "allowed_nodes": list(LINUX_CPU_NODES), "require_node": None,
        "allow_cpu_training": True, "cpu_training_justification": "Single-thread CPU MuJoCo with small PPO networks.",
        "result_dir": str(ROOT / relative), "local_result_dir": str(ROOT / relative),
        "stage_input_paths": [str(ROOT / "scripts"), str(ROOT / "freq_hrl")],
        "stage_excludes": [*STAGE_EXCLUDES, ".server_artifacts"], "allow_duplicate": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if spec.ALGORITHM_REVISION == "pending_source_commit":
        raise SystemExit("freeze algorithm source before dispatch")
    subprocess.run(["git", "diff", "--exit-code", spec.ALGORITHM_REVISION, "--", "freq_hrl",
                    "scripts/run_mujoco_cell_small_export.py"], cwd=ROOT, check=True)
    tasks = [task_spec(args.run_name, cell, preflight=args.preflight) for cell in cells(args.preflight)]
    # Inventory the run before submission, including archived scheduler results.
    result = subprocess.run([sys.executable, str(SCHEDULER), "results", "--signature",
                             f"Freq-HRL/{spec.PROTOCOL}/{args.run_name}/*", "--status",
                             "queued", "launching", "running", "done", "failed", "cancelled",
                             "--limit", "0", "--include-empty", "--no-log-scan", "--json"],
                            capture_output=True, text=True, check=True)
    previous = json.loads(result.stdout[result.stdout.index("{"):]).get("results", [])
    if previous:
        raise SystemExit("run already registered: inspect existing tasks before any resubmission")
    directory = ROOT / "results" / args.run_name
    if any(directory.glob("cells/**/cell_summary.json")):
        raise SystemExit("run already contains completed cells")
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(protocol=spec.PROTOCOL, algorithm_revision=spec.ALGORITHM_REVISION,
                   preflight=args.preflight, cells=cells(args.preflight), contract=spec.CONTRACT,
                   options={arm: spec.options(arm, preflight=args.preflight) for arm in spec.ARMS},
                   scheduler=dict(nodes=LINUX_CPU_NODES, require_node=None, cpu=1, ram_mb=1536),
                   artifacts=dict(synced=["cell_summary.json", "evaluation_rows.csv", "server_artifact_location.json"],
                                  server_only=["checkpoint.pt", "training_history.json"]))
    path = directory / "preregistration.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(payload)):
        raise SystemExit("existing registration differs")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=spec.PROTOCOL)
    if not args.dry_run:
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch"], check=True)


if __name__ == "__main__":
    main()
