#!/usr/bin/env python3
"""Schedule native routing development dynamically on node001-node006."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_routing_stage146 as spec
from scripts import run_native_transit_preservation_stage145 as preservation
from scripts.submit_native_transit_preservation_stage145_scheduleurm import task_specification as native_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, method, root, *, preflight):
    task = native_task(run_name, root, preflight=preflight)
    relative = Path("results") / run_name / "cells" / method / f"seed_{root}"
    command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_native_transit_routing_stage146.py",
               "--method", method, "--seed", str(root), "--output", str(ROOT / relative / "result.json")]
    if preflight:
        command.append("--preflight")
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL native routing {method} seed{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{method}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'preflight' if preflight else 'development'}",
        result_dir=str(ROOT / relative), local_result_dir=str(ROOT / relative),
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command),
        cpu_training_justification="Matched full native upper/lower RE-SAC learning; one CPU per method/root; final checkpoints remain server-only.")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preservation-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-summary", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for root in preservation.ROOTS:
        result = json.loads((args.preservation_dir / f"seed_{root}" / "result.json").read_text())
        if not (result["passed"] and not result["preflight"] and result["contract"] == preservation.contract(False)):
            raise SystemExit("Full native core extraction has not qualified")
    if not args.preflight:
        if args.preflight_summary is None:
            raise SystemExit("Native routing development requires its completed three-control preflight")
        summary = json.loads(args.preflight_summary.read_text())
        if not (summary["software_qualified"] and summary["preflight"]
                and summary["protocol"] == spec.EXPERIMENT_PROTOCOL):
            raise SystemExit("Native routing preflight did not qualify")
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Native routing run is already registered")
    roots = spec.ROOTS[:1] if args.preflight else spec.ROOTS
    tasks = [task_specification(args.run_name, method, root, preflight=args.preflight)
             for method in spec.METHODS for root in roots]
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(args.preflight),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "optimizer_roots": list(roots), "tasks": len(tasks),
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
                      "cpu_per_task": 1, "ram_mb_per_task": tasks[0]["ram_mb"]},
    })
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
