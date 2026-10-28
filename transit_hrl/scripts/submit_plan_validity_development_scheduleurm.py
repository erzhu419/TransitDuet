"""Prepare scheduler specs by default; submit preflight/pilot only explicitly."""

import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_development import (
    PHASES, PROTOCOL, cell_arguments, cell_directory, development_cells, expected_runs,
    load_plan, require_preflight_gate,
)
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.submit_hyperparameter_pilot_scheduleurm import (
    DEFAULT_LINUX_PYTHON, LINUX_CPU_NODES, STAGE_EXCLUDES, execute_bulk,
)


def build_bundle(args, plan):
    preregistration = args.preregistration.resolve().relative_to(ROOT)
    gate = args.preflight_gate.resolve().relative_to(ROOT) if args.preflight_gate else None
    if args.stage == "cost_pilot" and gate is None:
        raise ValueError("declare the future qualified --preflight-gate even when preparing pilot specs")
    if args.ram_mb < 1:
        raise ValueError("declare a positive per-cell RAM request")
    jobs, resolved_cells = [], []
    for cell in development_cells(plan, args.stage):
        resolved = cell_arguments(plan, args.run_name, cell, root=ROOT)
        resolved_cells.append({"task": cell["task"], "method": cell["method"], "optimizer_seed": cell["optimizer_seed"],
            "configuration": _json_ready({k: v for k, v in vars(resolved).items() if k not in ("output", "dry_run")})})
        directory = ROOT / cell_directory(args.run_name, cell)
        command = [args.python_executable, "-u", "scripts/run_plan_validity_development_cell.py",
            "--preregistration", str(preregistration), "--run-name", args.run_name,
            "--stage", args.stage, "--task", cell["task"], "--optimizer-seed", str(cell["optimizer_seed"]),
            "--method", cell["method"]]
        if gate is not None:
            command.extend(["--preflight-gate", str(gate)])
        env = ["PYTHONPATH=.", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
               "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", "TORCH_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=", "MUJOCO_GL=egl"]
        jobs.append({"project": "Freq-HRL-PlanValidity-Development",
            "description": f"Freq-HRL native {args.stage} {cell['task']} {cell['method']} root {cell['optimizer_seed']}",
            "signature": f"Freq-HRL/{PROTOCOL}/{args.run_name}/{args.stage}/{cell['task']}/{cell['method']}/{cell['optimizer_seed']}",
            "resource_family": f"Freq-HRL/{PROTOCOL}/{args.stage}/{cell['method']}",
            "cmd": " ".join(env) + " " + shlex.join(command), "cwd": str(ROOT),
            "cpu": 1, "vram": 0, "ram_mb": args.ram_mb, "priority": "normal",
            "allowed_nodes": list(LINUX_CPU_NODES), "require_node": None, "preferred_node": None,
            "allow_cpu_training": True,
            "cpu_training_justification": "Registered single-thread headless PPO development cell; no GPU use.",
            "no_auto_requeue": True, "reroute_on_node_down": False, "skip_resume_scan": True,
            "result_dir": str(directory / "public"), "local_result_dir": str(directory / "public"),
            "stage_excludes": list(STAGE_EXCLUDES),
            "stage_input_paths": [str(args.preregistration.resolve()), *([str(args.preflight_gate.resolve())] if gate else [])],
            "wait_for_files": [str(args.preflight_gate.resolve())] if gate else [],
            "skip_launch_staging": False, "allow_duplicate": False})
    return {"protocol": {"protocol_version": PROTOCOL, "expected_runs": expected_runs(plan, args.run_name, args.stage)},
        "stage": args.stage, "run_name": args.run_name, "submission_status": "prepared_not_submitted",
        "runtime_status": "candidate_interpreter_not_verified_on_execution_nodes",
        "python_executable": args.python_executable, "preregistration": str(preregistration),
        "preflight_gate": str(gate) if gate else None, "cells": resolved_cells, "jobs": jobs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=PHASES, required=True)
    parser.add_argument("--preflight-gate", type=Path)
    parser.add_argument("--python-executable", default=DEFAULT_LINUX_PYTHON)
    parser.add_argument("--ram-mb", type=int, default=2048)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_plan(args.preregistration)
    bundle = build_bundle(args, plan)
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if old["submission_status"] != "prepared_not_submitted" or (args.submit and old != bundle):
            raise FileExistsError("preserve submitted/changed campaign specs; choose a separate attempt manifest")
    if args.submit and args.stage == "cost_pilot":
        require_preflight_gate(plan, args.run_name, json.loads(args.preflight_gate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.submit:
        bundle["submission_status"] = "submitting"
        args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        execute_bulk(bundle["jobs"], dry_run=False, intent_label=f"{PROTOCOL}/{args.run_name}/{args.stage}")
        bundle["submission_status"] = "submitted"
    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
