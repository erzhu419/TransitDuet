"""Freeze the W sampling-support by clock-information factorial."""

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
    for cell in cells:
        forbidden.update(cell["held_prefix_seeds"] + cell["previous_held_prefix_seeds"] + cell["new_training_prefix_seeds"])
    forbidden.update(_training_seed(optimizer_seed=c["optimizer_seed"], rollout_root=s, iteration=i)
        for c in registered for s in c["seed_roles"]["train_seeds"]
        for i in range(2 if c["phase"] == "preflight" else plan["contract"]["primary_iterations"]))
    jobs = {"precheck": [], "run": []}
    for index, cell in enumerate(cells):
        root = cell["optimizer_seed"]
        cell["target_cache_directory"] = cell["source_cache_directory"]
        cell["coverage_cache_directory"] = cell["output_directory"] + "/run/private"
        cell["coverage_public_result"] = cell["output_directory"] + "/run/public/result.json"
        cell["fit_prefix_seeds"] = cell["training_prefix_seeds"] + cell["new_training_prefix_seeds"]
        cell["exposed_held_prefix_seeds"] = cell["previous_held_prefix_seeds"] + cell["held_prefix_seeds"]
        cell["held_prefix_seeds"] = list(range(root + 9000, root + 9008))
        if set(cell["held_prefix_seeds"]) & forbidden:
            raise ValueError("new phase-support holdout overlaps exposed/reserved/effective seeds")
        forbidden.update(cell["held_prefix_seeds"])
        cell["output_directory"] = f"results/{run_name}/{cell['task']}/{root}"
        for phase in jobs:
            job = deepcopy(source["jobs"][phase][index])
            directory = ROOT / cell["output_directory"] / phase
            command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_phase_support.py",
                "--manifest", str(manifest_path), "--cell-index", str(index), "--phase", phase]
            job.update(project="Freq-HRL-PhaseSupport", description=f"W phase support {phase} {cell['task']} {root}",
                signature=f"Freq-HRL/phase-support/{run_name}/{phase}/{cell['task']}/{root}",
                cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command),
                result_dir=str(directory / "public"), local_result_dir=str(directory / "public"),
                wait_for_files=[str(ROOT / manifest_path)])
            jobs[phase].append(job)
    return {"protocol_version": "window_phase_support_development_v1", "run_name": run_name,
        "source_hardware_profile": source["source_hardware_profile"], "cells": cells, "jobs": jobs,
        "arms": ["start_plain", "start_clock", "checks_plain", "checks_clock"],
        "probes_per_path": 3, "effective_df": 8, "threshold_quantile": 0.75,
        "fitting_futures_per_state": 1, "independent_B_futures_per_state": 4,
        "fitting_state_future_pairs_per_arm": 96, "held_A_futures_per_state": 4, "held_B_futures_per_state": 4,
        "clock_features": ["offset_fraction", "remaining_fraction"], "plain_features": 39, "clock_feature_count": 41,
        "sampling": "same_617_bins_then_uniform_reference_visited_nondeadline_check_per_bin_839",
        "future_seed_recipe": "SeedSequence([optimizer_seed,path_seed,step,replica_index,840])",
        "reference_and_continuation": "original_frozen_history", "controller_training": "none",
        "matching_scope": "same_fitting_count_and_effective_df_not_equal_total_cpu",
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
        raise FileExistsError("preserve the frozen phase-support manifest")
    result = prepare(json.loads(args.source_manifest.read_text()), load_plan(args.preregistration), args.run_name,
                     args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit_precheck:
        execute_bulk(result["jobs"]["precheck"], dry_run=False, intent_label=args.run_name + "_precheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
