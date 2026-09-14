#!/usr/bin/env python3
"""Submit one non-evidentiary v25 cProfile task through scheduleurm."""

from __future__ import annotations

import argparse
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

PROFILE_PROTOCOL = "mujoco_v25_action_sample_cprofile_v1"
PROFILE_ROOTS = {
    "optimizer": 25009501,
    "train": (25009511, 25009512, 25009513, 25009514),
    "selection": (25009521, 25009522, 25009523, 25009524),
    "evaluation": (25009531,),
}


def command(run_name: str) -> str:
    full = Path(".server_artifacts") / run_name
    export = Path("results") / run_name
    options = spec.options(spec.CANDIDATE, preflight=False)
    options.update({
        "train-seeds": PROFILE_ROOTS["train"],
        "selection-seeds": PROFILE_ROOTS["selection"],
        "eval-seeds": PROFILE_ROOTS["evaluation"],
        "iterations": 8,
        "checkpoint-minimum-iteration": 3,
        "checkpoint-evaluation-interval": 8,
    })
    argv = [
        DEFAULT_LINUX_PYTHON, "-u", "scripts/run_mujoco_cprofile_small_export.py",
        "--full-output-dir", str(full), "--export-output-dir", str(export), "--",
        "--env-id", "HalfCheetah-v5", "--optimizer-seed", str(PROFILE_ROOTS["optimizer"]),
    ]
    for key, value in options.items():
        if value is True:
            argv.append("--" + key)
        else:
            values = value if isinstance(value, (tuple, list)) else (value,)
            argv.extend(("--" + key, *map(str, values)))
    return (
        "MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "
        + shlex.join(argv)
    )


def task(run_name: str) -> dict[str, object]:
    result = ROOT / "results" / run_name
    return {
        "project": PROFILE_PROTOCOL,
        "description": "Freq-HRL v25 action-sample cProfile efficiency diagnostic",
        "cmd": command(run_name), "cwd": str(ROOT),
        "signature": f"Freq-HRL/{PROFILE_PROTOCOL}/{run_name}",
        "resource_family": f"Freq-HRL/{PROFILE_PROTOCOL}/cell",
        "cpu": 1, "ram_mb": 2048, "vram": 0, "priority": "normal",
        "allowed_nodes": list(LINUX_CPU_NODES), "require_node": None,
        "allow_cpu_training": True,
        "cpu_training_justification": "Single representative CPU profile of the active MuJoCo training path.",
        "result_dir": str(result), "local_result_dir": str(result),
        "stage_input_paths": [str(ROOT / "scripts"), str(ROOT / "freq_hrl")],
        "stage_excludes": [*STAGE_EXCLUDES, ".server_artifacts"],
        "allow_duplicate": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    signature = f"Freq-HRL/{PROFILE_PROTOCOL}/{args.run_name}"
    query = subprocess.run([
        sys.executable, str(SCHEDULER), "results", "--signature", signature,
        "--status", "queued", "launching", "running", "done", "failed", "cancelled",
        "--limit", "0", "--include-empty", "--no-log-scan", "--json",
    ], check=True, capture_output=True, text=True)
    if json.loads(query.stdout[query.stdout.index("{"):]).get("results"):
        raise SystemExit("profile run already registered")
    result = ROOT / "results" / args.run_name
    result.mkdir(parents=True, exist_ok=True)
    (result / "preregistration.json").write_text(json.dumps({
        "protocol": PROFILE_PROTOCOL,
        "evidence_role": "efficiency_diagnostic_only_not_algorithm_evidence",
        "algorithm_revision": spec.ALGORITHM_REVISION,
        "environment": "HalfCheetah-v5", "arm": spec.CANDIDATE,
        "steps": 512, "iterations": 8, "roots": PROFILE_ROOTS,
        "profile_binary": "server_only", "synced": ["profile_summary.json", "profile_top.txt", "cell_summary.json"],
    }, indent=2) + "\n")
    execute_bulk([task(args.run_name)], dry_run=args.dry_run, intent_label=PROFILE_PROTOCOL)
    if not args.dry_run:
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch"], check=True)


if __name__ == "__main__":
    main()
