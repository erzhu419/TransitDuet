"""Freeze a matched-target development repair and submit native prechecks first."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_development import load_plan
from freq_hrl.experiments.plan_validity_preregistration import seed_cells
from freq_hrl.experiments.pointmaze_goal_validation import _training_seed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk


def prepare(source, plan, run_name, manifest_path):
    cells = deepcopy(source["cells"])
    reserved = {v for c in seed_cells(plan) for seeds in c["seed_roles"].values() for v in seeds}
    effective = {_training_seed(optimizer_seed=c["optimizer_seed"], rollout_root=s, iteration=i)
        for c in seed_cells(plan) for s in c["seed_roles"]["train_seeds"]
        for i in range(2 if c["phase"] == "preflight" else plan["contract"]["primary_iterations"])}
    jobs = {"precheck": [], "run": []}
    for index, cell in enumerate(cells):
        cell["training_prefix_seeds"] = cell["expected_configuration"]["branch_fit_seeds"]
        cell["held_prefix_seeds"] = list(range(cell["optimizer_seed"] + 6000, cell["optimizer_seed"] + 6008))
        if (set(cell["held_prefix_seeds"]) & (reserved | effective)):
            raise ValueError("new development holdout overlaps reserved or effective training seeds")
        cell["previous_evaluation_seeds"] = cell.pop("evaluation_seeds")
        cell.pop("probe_seeds")
        cell["output_directory"] = f"results/{run_name}/{cell['task']}/{cell['optimizer_seed']}"
        for phase in jobs:
            job = deepcopy(source["jobs"][index])
            directory = ROOT / cell["output_directory"] / phase
            command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_target_credit.py",
                       "--manifest", str(manifest_path), "--cell-index", str(index), "--phase", phase]
            job.update(project="Freq-HRL-TargetCredit", description=f"Matched L/W/G {phase} {cell['task']} {cell['optimizer_seed']}",
                signature=f"Freq-HRL/target-credit/{run_name}/{phase}/{cell['task']}/{cell['optimizer_seed']}",
                cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command),
                result_dir=str(directory / "public"), local_result_dir=str(directory / "public"),
                wait_for_files=[str(ROOT / manifest_path)])
            jobs[phase].append(job)
    return {"protocol_version": "matched_target_credit_development_v1", "run_name": run_name,
        "source_hardware_profile": source["source_hardware_profile"], "cells": cells, "jobs": jobs,
        "probes_per_path": 3, "conditional_replicates": 8, "fit_future_indices": [0, 1, 2, 3],
        "independent_future_indices": [4, 5, 6, 7], "effective_df": 8.0, "threshold_quantile": 0.75,
        "feature_view": "causal_validity_interactions_39", "targets": ["L", "W", "G"],
        "state_seed_recipe": "SeedSequence([optimizer_seed,path_seed,617])",
        "future_seed_recipe": "SeedSequence([optimizer_seed,path_seed,step,replica_index,618])",
        "behavior_and_continuation": "original_frozen_history_policy_for_every_target",
        "fit_role": "old_branch_fit_prefixes_block_A_only", "new_holdout_role": "root_plus_6000_through_6007",
        "seed_check": "disjoint_from_all_preregistered_roles_and_resolved_primary_training_seeds",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False,
        "no_selection_on_new_holdout": True, "controller_training": "none"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit-precheck", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen repair manifest")
    relative = args.manifest.resolve().relative_to(ROOT)
    result = prepare(json.loads(args.source_manifest.read_text()), load_plan(args.preregistration), args.run_name, relative)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit_precheck:
        execute_bulk(result["jobs"]["precheck"], dry_run=False, intent_label=args.run_name + "_precheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
