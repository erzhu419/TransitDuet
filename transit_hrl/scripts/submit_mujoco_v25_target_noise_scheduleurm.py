#!/usr/bin/env python3
"""Dispatch the fixed v25 same-state diagnostic through scheduleurm."""

import argparse
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_mujoco_v25_projection_target_noise as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import (
    DEFAULT_LINUX_PYTHON, LINUX_CPU_NODES, SCHEDULER, STAGE_EXCLUDES, execute_bulk,
)


def cells():
    return list(itertools.product(spec.ROOTS, spec.DIMENSIONS, spec.MEAN_AMPLITUDES, spec.STANDARD_DEVIATIONS))


def cell_name(seed, dimension, amplitude, std):
    return f"d{dimension}_mean{amplitude:g}_std{std:g}_seed{seed}"


def task_spec(run_name, revision, cell):
    seed, dimension, amplitude, std = cell
    name = cell_name(*cell)
    relative = Path("results") / run_name / "cells" / name
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/diagnose_mujoco_v25_projection_target_noise.py",
               "--dimension", str(dimension), "--amplitude", str(amplitude),
               "--std", str(std), "--seed", str(seed), "--code-revision", revision,
               "--output-dir", str(relative)]
    return {
        "project": spec.PROTOCOL,
        "description": f"Freq-HRL v25 target noise {name}",
        "cmd": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 " + shlex.join(command),
        "cwd": str(ROOT),
        "signature": f"Freq-HRL/{spec.PROTOCOL}/{run_name}/{name}",
        "resource_family": f"Freq-HRL/{spec.PROTOCOL}/cell",
        "cpu": 1, "ram_mb": 768, "vram": 0, "priority": "normal",
        "allowed_nodes": list(LINUX_CPU_NODES), "require_node": None,
        "allow_cpu_training": True,
        "cpu_training_justification": "Single-threaded NumPy projector diagnostic; no neural training.",
        "result_dir": str(ROOT / relative), "local_result_dir": str(ROOT / relative),
        "stage_input_paths": [str(ROOT / "scripts"), str(ROOT / "freq_hrl")],
        "stage_excludes": [*STAGE_EXCLUDES, ".server_artifacts"],
        "allow_duplicate": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    selected = cells()[:1] if args.preflight_only else cells()
    directory = ROOT / "results" / args.run_name
    directory.mkdir(parents=True, exist_ok=True)
    registration = directory / "preregistration.json"
    if not registration.exists():
        registration.write_text(json.dumps({
            "protocol": spec.PROTOCOL, "code_revision": revision,
            "cells": cells(), "upper_draws": spec.UPPER_DRAWS,
            "lower_draws": spec.LOWER_DRAWS, "prefix_steps": spec.PREFIX_STEPS,
            "evidence_role": "controlled_mechanism_only_no_performance_selection",
        }, indent=2) + "\n")
    else:
        frozen = json.loads(registration.read_text())
        if frozen["code_revision"] != revision:
            raise SystemExit("existing diagnostic run belongs to another revision")
    execute_bulk([task_spec(args.run_name, revision, cell) for cell in selected],
                 dry_run=args.dry_run, intent_label=spec.PROTOCOL)
    if not args.dry_run:
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch"], check=True)


if __name__ == "__main__":
    main()
