#!/usr/bin/env python3
"""Allocate independent Stage-33 training and branch-replay resources."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import pointmaze_root_response_stage33_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import (
    DEFAULT_LINUX_PYTHON, LINUX_CPU_NODES, SCHEDULER, STAGE_EXCLUDES, execute_bulk,
)
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory, sync_results
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_root_response import root_aggregate, write_json


def cell_dir(run_name, root):
    return ROOT / "results" / run_name / "cells" / spec.POLICY / f"replicate_{root}"


def task_specification(run_name, root, *, preflight, phase, controller_run=None):
    output = cell_dir(run_name, root) / "result.json"
    command = [DEFAULT_LINUX_PYTHON, "-u", spec.RUNNER_SCRIPT, "--optimizer-seed", str(root),
               "--phase", phase, "--output", str(output)]
    inputs = [str(ROOT / "scripts"), str(ROOT / "freq_hrl")]
    if preflight:
        command.append("--preflight")
    if phase == "response":
        source = cell_dir(controller_run, root) / "result.json"
        result = json.loads(source.read_text())
        if (result["status"] != "complete" or result["protocol"]["phase"] != "train"
                or result["protocol"]["protocol_version"] != spec.EXPERIMENT_PROTOCOL
                or result["protocol"]["optimizer_seed"] != root
                or result["protocol"]["options"] != _json_ready(spec.options(root, preflight=preflight))):
            raise ValueError("qualification requires this root's completed frozen controller")
        command.extend(["--controller-result", str(source)])
        inputs.append(str(source.parent))
    cpu, ram = (1, 3072) if phase == "train" else ((2, 3072) if preflight else (17, 24576))
    return {"project": spec.EXPERIMENT_PROTOCOL, "description": f"Freq-HRL stage33 {phase} root{root}",
            "cmd": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                   "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "
                   + shlex.join(command) + " && printf '%s\\n' 'Training complete: result.json written'",
            "cwd": str(ROOT), "signature": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{spec.POLICY}/{root}",
            "resource_family": f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{phase}",
            "cpu": cpu, "ram_mb": ram, "vram": 0, "priority": "normal",
            "allowed_nodes": list(LINUX_CPU_NODES), "require_node": None,
            "allow_cpu_training": True, "cpu_training_justification": "Serial PPO roots; separately allocated CPU branch workers.",
            "allow_no_ckpt": True, "allow_no_resume": True, "result_dir": str(output.parent),
            "local_result_dir": str(output.parent), "stage_input_paths": inputs,
            "stage_excludes": list(STAGE_EXCLUDES), "allow_duplicate": False,
            "reroute_on_node_down": True, "node_down_requeue_s": 300}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", choices=("train", "response", "pipeline"), required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--controller-run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-results", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.phase == "pipeline" and not args.preflight:
        parser.error("pipeline is reserved for the separate small preflight root")
    if args.phase == "response" and not args.controller_run and not (args.sync_results or args.aggregate):
        parser.error("response requires --controller-run")
    if args.sync_results:
        sync_results(args.run_name, preflight=args.preflight, workers=2, protocol_spec=spec)
        return 0
    if args.aggregate:
        if args.preflight or args.phase != "response":
            parser.error("only the complete full response roster can be aggregated")
        results = [json.loads((cell_dir(args.run_name, root) / "result.json").read_text()) for root in spec.OPTIMIZER_ROOTS]
        write_json(ROOT / "results" / args.run_name / "root_qualification.json", root_aggregate(results))
        return 0
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Stage-33 run already registered")
    if any((ROOT / "results" / args.run_name).glob("cells/**/result.json")):
        raise SystemExit("Stage-33 run already contains completed outcomes")
    tasks = [task_specification(args.run_name, root, preflight=args.preflight, phase=args.phase,
                                controller_run=args.controller_run) for root in spec.roots(preflight=args.preflight)]
    registration = {"protocol": spec.EXPERIMENT_PROTOCOL,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "phase": args.phase, "preflight": args.preflight, "controller_run": args.controller_run,
        "optimizer_roots": list(spec.roots(preflight=args.preflight)),
        "full_qualification_roots": list(spec.OPTIMIZER_ROOTS), "qualification": spec.qualification_contract(),
        "options": {str(root): spec.options(root, preflight=args.preflight) for root in spec.roots(preflight=args.preflight)},
        "budgets": {str(root): spec.budget(root, preflight=args.preflight) for root in spec.roots(preflight=args.preflight)},
        "scheduler": {"nodes": list(LINUX_CPU_NODES), "require_node": None,
                      "cpu_per_task": tasks[0]["cpu"], "ram_mb_per_task": tasks[0]["ram_mb"]},
        "artifacts": {"synced": ["result.json", "controller.json"], "weights_and_raw": "server_only"}}
    write_json(ROOT / "results" / args.run_name / "preregistration.json", registration)
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")
    if not args.dry_run:
        registered = inventory(args.run_name, protocol_spec=spec)
        ids = [str(task["id"]) for task in registered]
        subprocess.run([sys.executable, str(SCHEDULER), "dispatch",
                        *[arg for task_id in ids for arg in ("--task-id", task_id)]], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
