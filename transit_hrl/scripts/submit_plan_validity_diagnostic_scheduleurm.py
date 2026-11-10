"""Freeze four pilot-root diagnostic cells; submit only with --submit."""

import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_development import load_plan
from freq_hrl.experiments.plan_validity_preregistration import seed_cells
from scripts.submit_hyperparameter_pilot_scheduleurm import (
    DEFAULT_LINUX_PYTHON, LINUX_CPU_NODES, STAGE_EXCLUDES, execute_bulk,
)

METHODS = ("history", "threshold", "termination_history")


def diagnostic_cells(summary, plan):
    if summary["status"] != "complete":
        raise ValueError("diagnostics require the completed logical pilot cohort")
    grouped = {}
    for source in summary["source_files"]:
        public = json.loads(Path(source).read_text())
        record = public["records"][0]
        grouped.setdefault((record["task"], record["optimizer_seed"]), {})[record["method"]] = (public, record)
    cells = []
    for reserved in seed_cells(plan):
        if reserved["phase"] != "cost_pilot":
            continue
        task, root = reserved["task"], reserved["optimizer_seed"]
        group = grouped[task, root]
        if set(group) != set(METHODS):
            raise ValueError("retain all three frozen methods for each pilot root")
        sources = {}
        for method in METHODS:
            public, record = group[method]
            if record["status"] != "validation_complete":
                raise ValueError("source measurement is not complete")
            for role, seeds in reserved["seed_roles"].items():
                if record["configuration"][role] != seeds:
                    raise ValueError("source seed roles differ from the reservation")
            sources[method] = {"report": public["private_source_report"], "run_id": record["run_id"],
                "validation_ise": record["validation_ise"], "selected_rule_summary": record["selected_rule_summary"]}
        evaluation = reserved["seed_roles"]["trigger_eval_seeds"]
        half = len(evaluation) // 2
        cells.append({"task": task, "optimizer_seed": root, "sources": sources,
            "expected_configuration": group["history"][1]["configuration"],
            "evaluation_seeds": evaluation, "probe_seeds": evaluation[:2] + evaluation[half:half + 2]})
    if len(cells) != 4:
        raise ValueError("the bounded diagnostic requires exactly the four reserved pilot roots")
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    cells = diagnostic_cells(json.loads(args.pilot_summary.read_text()), load_plan(args.preregistration))
    manifest_path = args.manifest.resolve().relative_to(ROOT)
    if manifest_path.parts[0] != "protocols":
        raise ValueError("place the small diagnostic manifest in protocols for source staging")
    jobs = []
    for index, cell in enumerate(cells):
        directory = ROOT / "results" / args.run_name / cell["task"] / str(cell["optimizer_seed"])
        cell["output_directory"] = str(directory.relative_to(ROOT))
        command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_development_diagnostic.py",
                   "--manifest", str(manifest_path), "--cell-index", str(index)]
        jobs.append({"project": "Freq-HRL-PlanValidity-Diagnostic",
            "description": f"Frozen pilot local-to-episode diagnostic {cell['task']} root {cell['optimizer_seed']}",
            "signature": f"Freq-HRL/pilot-local-episode/{args.run_name}/{cell['task']}/{cell['optimizer_seed']}",
            "cmd": "PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl " + shlex.join(command),
            "cwd": str(ROOT), "cpu": 1, "vram": 0, "ram_mb": 2048, "priority": "high",
            "protect_from_auto_eviction": True, "allowed_nodes": list(LINUX_CPU_NODES),
            "require_node": None, "preferred_node": None, "no_auto_requeue": True,
            "reroute_on_node_down": False, "skip_resume_scan": True, "skip_launch_staging": False,
            "stage_excludes": list(STAGE_EXCLUDES), "stage_input_paths": [],
            "result_dir": str(directory / "public"), "local_result_dir": str(directory / "public"),
            "wait_for_files": [str(args.manifest.resolve())], "allow_duplicate": False})
    manifest = {"protocol_version": "frozen_pilot_local_episode_diagnostic_v1", "run_name": args.run_name,
        "source_summary": str(args.pilot_summary), "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "scope": "reserved_pilot_evaluation_paths_not_confirmation", "profiles": ["in_distribution"],
        "probes_per_episode": 3, "conditional_replicates": 8,
        "future_seed_recipe": "SeedSequence([optimizer_seed,path_seed,probe_step,replica_index,403])",
        "risk_spec": {"distance": 1.0, "duration_steps": 25}, "risk_role": "telemetry_fixture_not_operational_calibration",
        "source_hardware_profile": json.loads(args.pilot_summary.read_text())["hardware_profile"],
        "controller_training": "none_load_frozen_artifacts", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False,
        "cells": cells, "jobs": jobs}
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen diagnostic manifest; choose a new run before resubmitting")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.submit:
        execute_bulk(jobs, dry_run=False, intent_label=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
