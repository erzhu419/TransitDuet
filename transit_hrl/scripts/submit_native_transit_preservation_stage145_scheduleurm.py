#!/usr/bin/env python3
"""Schedule native extraction qualification dynamically on node001-node006."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_preservation_stage145 as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, LINUX_CPU_NODES, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, seed, *, preflight):
    relative = Path("results") / run_name / f"seed_{seed}"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_preservation_stage145.py",
               "--seed", str(seed), "--output", str(ROOT / relative / "result.json")]
    if preflight:
        command.append("--preflight")
    return {
        "project": spec.EXPERIMENT_PROTOCOL,
        "description": f"Freq-HRL native Transit preservation seed{seed}",
        "cmd": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
               "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        "cwd": str(ROOT),
        "signature": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{seed}",
        "resource_family": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'preflight' if preflight else 'qualification'}",
        "cpu": 1, "ram_mb": 2048 if preflight else 3072, "vram": 0,
        "priority": "normal", "allowed_nodes": list(LINUX_CPU_NODES), "require_node": None,
        "allow_cpu_training": True,
        "cpu_training_justification": "Single-thread paired native RE-SAC learning and action/network equivalence.",
        "allow_no_ckpt": True, "allow_no_resume": True,
        "result_dir": str(ROOT / relative), "local_result_dir": str(ROOT / relative),
        "stage_input_paths": [str(ROOT / name) for name in ("scripts", "freq_hrl", "native_freqduet")],
        "stage_excludes": ["results", "**/__pycache__"],
        "allow_duplicate": False, "reroute_on_node_down": True, "node_down_requeue_s": 300,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.preflight:
        if args.preflight_result is None:
            raise SystemExit("Full native qualification requires a completed preflight JSON")
        result = json.loads(args.preflight_result.read_text())
        if not (result.get("passed") and result.get("preflight")
                and result.get("protocol") == spec.EXPERIMENT_PROTOCOL
                and result.get("contract") == spec.contract(True)):
            raise SystemExit("Preflight did not qualify this native extraction protocol")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Native preservation run is already registered")
    seeds = spec.ROOTS[:1] if args.preflight else spec.ROOTS
    tasks = [task_specification(args.run_name, seed, preflight=args.preflight) for seed in seeds]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(args.preflight),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seeds": list(seeds), "scheduler": {
            "allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]},
    })
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
