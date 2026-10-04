#!/usr/bin/env python3
"""Schedule Stage117 plan probes dynamically on node001-node006."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import pointmaze_local_plan_gain_stage117_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import (
    task_specification as previous_task, qualification_task as previous_qualification,
)
from scripts.submit_hyperparameter_pilot_scheduleurm import execute_bulk
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import inventory
from freq_hrl.experiments.pointmaze_root_response import write_json


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight, protocol_spec=spec)
    task.update(cpu=3 if preflight else 5, ram_mb=4096,
        description=f"Freq-HRL Stage117 local plan gain root{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/local_plan/{'preflight' if preflight else 'full'}",
        cpu_training_justification="One-option native plan interventions; final Stage112 lower fixed; no policy training or checkpoint pull.")
    return task


def qualification_task(run_name, *, preflight):
    return previous_qualification(run_name, preflight=preflight, protocol_spec=spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if inventory(args.run_name, protocol_spec=spec):
        raise SystemExit("Stage117 run already registered")
    tasks = [task_specification(args.run_name, root, preflight=args.preflight)
        for root in spec.roots(preflight=args.preflight)]
    tasks.append(qualification_task(args.run_name, preflight=args.preflight))
    write_json(ROOT / "results" / args.run_name / "preregistration.json", {
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "preflight": args.preflight,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "optimizer_roots": list(spec.roots(preflight=args.preflight)), "options": spec.options(preflight=args.preflight),
        "seed_roles": {str(root): spec.seed_roles(root, preflight=args.preflight) for root in spec.roots(preflight=args.preflight)},
        "budget_per_root": spec.budget(preflight=args.preflight), "endpoints": list(spec.ENDPOINTS),
        "scheduler": {"allowed_nodes": tasks[0]["allowed_nodes"], "require_node": None,
            "cpu_per_task": tasks[0]["cpu"], "ram_mb_per_task": tasks[0]["ram_mb"]},
        "artifacts": {"pull": "completion_markers_and_compact_JSON_only", "checkpoint_or_trace_writes": 0}})
    execute_bulk(tasks, dry_run=args.dry_run, intent_label=f"{spec.EXPERIMENT_PROTOCOL}:{args.run_name}")


if __name__ == "__main__":
    main()
