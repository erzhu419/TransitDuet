#!/usr/bin/env python3
"""Register the full equal-budget joint-training matrix on node001-node006."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import pointmaze_joint_renewal_stage35_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import (
    DEFAULT_LINUX_PYTHON, LINUX_CPU_NODES, SCHEDULER, STAGE_EXCLUDES, execute_bulk,
)
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, method, *, preflight):
    output = ROOT / "results" / run_name / "cells" / method / f"replicate_{root}" / "result.json"
    command = [DEFAULT_LINUX_PYTHON, "-u", spec.RUNNER_SCRIPT, "--optimizer-seed", str(root),
               "--method", method, "--output", str(output)]
    if preflight:
        command.append("--preflight")
    workers = spec.options(preflight=preflight)["workers"]
    return {"project": spec.EXPERIMENT_PROTOCOL, "description": f"Freq-HRL stage35 {method} root{root}",
            "cmd": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                   "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "
                   + shlex.join(command) + " && printf '%s\\n' 'Training complete: result.json written'",
            "cwd": str(ROOT), "signature": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{method}/{root}",
            "resource_family": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/training", "cpu": workers + 1,
            "ram_mb": 4096 if preflight else 12288, "vram": 0, "priority": "normal",
            "allowed_nodes": list(LINUX_CPU_NODES), "require_node": None, "allow_cpu_training": True,
            "cpu_training_justification": "Native CPU MuJoCo and on-policy PPO, 8 persistent rollout workers plus one learner.",
            "allow_no_ckpt": True, "allow_no_resume": True, "result_dir": str(output.parent),
            "local_result_dir": str(output.parent),
            "stage_input_paths": [str(ROOT / "scripts"), str(ROOT / "freq_hrl")],
            "stage_excludes": list(STAGE_EXCLUDES), "allow_duplicate": False,
            "reroute_on_node_down": True, "node_down_requeue_s": 300}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Stage-35 run already registered")
    tasks = [task_specification(args.run_name, root, method, preflight=args.preflight)
             for root in spec.roots(preflight=args.preflight) for method in spec.METHODS]
    write_json(ROOT / "results" / args.run_name / "preregistration.json",
               {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "preflight": args.preflight,
                "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "optimizer_roots": list(spec.roots(preflight=args.preflight)), "methods": list(spec.METHODS),
                "options": spec.options(preflight=args.preflight), "budget_per_cell": spec.budget(preflight=args.preflight),
                "seed_roles": {str(root): spec.seed_roles(root, preflight=args.preflight) for root in spec.roots(preflight=args.preflight)},
                "scheduler": {"allowed_nodes": list(LINUX_CPU_NODES), "require_node": None,
                              "cpu_per_task": tasks[0]["cpu"], "ram_mb_per_task": tasks[0]["ram_mb"]},
                "artifacts": {"pull": "compact_json_only", "raw_and_weights": "server_only"}})
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")
    if not args.dry_run:
        tasks = inventory(args.run_name, protocol_spec=spec)
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch",
                        *[arg for task in tasks for arg in ("--task-id", str(task["id"]))]], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
