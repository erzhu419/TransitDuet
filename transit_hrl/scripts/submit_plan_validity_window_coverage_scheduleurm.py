"""Freeze an equal-future-budget W coverage/complexity development factorial."""

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
    registered = list(seed_cells(plan))
    forbidden = {s for c in registered for seeds in c["seed_roles"].values() for s in seeds}
    forbidden.update(s for c in cells for s in c["held_prefix_seeds"])
    forbidden.update(_training_seed(optimizer_seed=c["optimizer_seed"], rollout_root=s, iteration=i)
        for c in registered for s in c["seed_roles"]["train_seeds"]
        for i in range(2 if c["phase"] == "preflight" else plan["contract"]["primary_iterations"]))
    jobs = {"precheck": [], "run": []}
    for index, cell in enumerate(cells):
        root = cell["optimizer_seed"]
        cell["source_cache_directory"] = cell["output_directory"] + "/run/private"
        cell["source_public_result"] = cell["output_directory"] + "/run/public/result.json"
        cell["previous_held_prefix_seeds"] = cell.pop("held_prefix_seeds")
        cell["held_prefix_seeds"] = list(range(root + 7000, root + 7008))
        cell["new_training_prefix_seeds"] = list(range(root + 8000, root + 8024))
        fresh = cell["held_prefix_seeds"] + cell["new_training_prefix_seeds"]
        if set(fresh) & forbidden or len(fresh) != len(set(fresh)):
            raise ValueError("new training or held prefixes overlap exposed/reserved/effective seeds")
        forbidden.update(fresh)
        cell["output_directory"] = f"results/{run_name}/{cell['task']}/{root}"
        for phase in jobs:
            job = deepcopy(source["jobs"][phase][index])
            directory = ROOT / cell["output_directory"] / phase
            command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_window_coverage.py",
                "--manifest", str(manifest_path), "--cell-index", str(index), "--phase", phase]
            job.update(project="Freq-HRL-WindowCoverage", description=f"W coverage/complexity {phase} {cell['task']} {root}",
                signature=f"Freq-HRL/window-coverage/{run_name}/{phase}/{cell['task']}/{root}",
                cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command),
                result_dir=str(directory / "public"), local_result_dir=str(directory / "public"),
                wait_for_files=[str(ROOT / manifest_path)])
            jobs[phase].append(job)
    return {"protocol_version": "window_coverage_complexity_development_v1", "run_name": run_name,
        "source_hardware_profile": source["source_hardware_profile"], "cells": cells, "jobs": jobs,
        "probes_per_path": 3, "effective_df_values": [2, 8], "threshold_quantile": 0.75,
        "dense_prefixes": 8, "dense_fitting_futures_per_state": 4,
        "wide_prefixes": 32, "wide_fitting_futures_per_state": 1,
        "fitting_state_future_pairs_per_arm": 96, "independent_B_futures_per_state": 4,
        "held_A_futures_per_state": 4, "held_B_futures_per_state": 4,
        "shared_feature_view": "causal_validity_interactions_39", "shared_target": "W",
        "state_seed_recipe": "SeedSequence([optimizer_seed,path_seed,617])",
        "new_future_seed_recipe": "SeedSequence([optimizer_seed,path_seed,step,replica_index,728])",
        "old_bank_selection": "dense_A0:4_wide_A0:1_B4:8_no_outcome_selection",
        "continuation": "original_frozen_history", "matching_scope": "fitting_future_count_not_equal_total_cpu",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit-precheck", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen coverage manifest")
    result = prepare(json.loads(args.source_manifest.read_text()), load_plan(args.preregistration), args.run_name,
                     args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit_precheck:
        execute_bulk(result["jobs"]["precheck"], dry_run=False, intent_label=args.run_name + "_precheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
