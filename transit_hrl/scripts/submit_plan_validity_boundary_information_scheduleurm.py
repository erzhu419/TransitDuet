"""Freeze full-replay-budget dense/wide cohorts; release prechecks only."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_boundary_information import dense_extra_windows
from freq_hrl.experiments.plan_validity_development import load_plan
from freq_hrl.experiments.plan_validity_preregistration import seed_cells
from freq_hrl.experiments.pointmaze_goal_validation import _training_seed
from scripts.submit_plan_validity_independent_calibration_scheduleurm import literal_seeds
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk


def prepare(source, plan, forbidden, run_name, manifest_path):
    forbidden = set(forbidden) | literal_seeds(source)
    for c in seed_cells(plan):
        forbidden.update(literal_seeds(c))
        forbidden.update(_training_seed(optimizer_seed=c["optimizer_seed"], rollout_root=s, iteration=i)
            for s in c["seed_roles"]["train_seeds"]
            for i in range(2 if c["phase"] == "preflight" else plan["contract"]["primary_iterations"]))
    cells, jobs = deepcopy(source["cells"]), []
    for i, cell in enumerate(cells):
        root = cell["optimizer_seed"]
        cell.update(precheck_seed=root + 66000,
            wide_training_prefix_seeds=list(range(root + 67000, root + 67024)),
            readout_prefix_seeds=list(range(root + 68000, root + 68008)),
            frozen_probe_directory=cell["output_directory"],
            dense_extra_windows=dense_extra_windows(cell["development_prefix_seeds"]),
            output_directory=f"results/{run_name}/{cell['task']}/{root}")
        new = [cell["precheck_seed"], *cell["wide_training_prefix_seeds"], *cell["readout_prefix_seeds"]]
        if len(set(new)) != len(new) or set(new) & forbidden:
            raise ValueError("new roles overlap exposed, registered or effective training seeds")
        forbidden.update(new)
        job = deepcopy(source["jobs"][i])
        directory = ROOT / cell["output_directory"] / "precheck"
        cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_boundary_information_precheck.py",
            "--manifest", str(manifest_path), "--cell-index", str(i)]
        job.update(project="Freq-HRL-BoundaryInformation", description=f"Information precheck {cell['task']} {root}",
            signature=f"Freq-HRL/boundary-information/{run_name}/precheck/{cell['task']}/{root}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(directory / "public"), local_result_dir=str(directory / "public"),
            wait_for_files=[str(ROOT / manifest_path)])
        jobs.append(job)
    return {"protocol_version": "full_replay_budget_boundary_information_v1", "run_name": run_name,
        "source_manifest": "protocols/plan_validity_boundary_parent_scale_2026-11-11.json",
        "source_variance_manifest": "protocols/plan_validity_boundary_variance_2026-11-11.json",
        "source_hardware_profile": source["source_hardware_profile"], "cells": cells, "jobs": jobs,
        "horizon": 1200, "period": 50, "checks_per_prefix": 3, "windows_per_state": 4,
        "precheck_suffix_indices": [1000, 1001], "precheck_steps_per_root": 22800, "precheck_total_steps": 91200,
        "precheck_budget": "1200*(8_cached_arms+1_factual+2_original_arms+8_fresh_arms)",
        "dense": {"prefixes": 8, "cached_A_indices": [0, 1], "new_A_indices": [4, 5, 6, 7, 8, 9],
            "extra_A_index": 10, "extra_windows_per_root": 12,
            "state_index_order": "ascending_opportunity_step_within_each_prefix", "training_pairs": 780,
            "new_pairs": 588, "new_steps_per_root": 1411200, "planned_shard_steps": [710400, 700800]},
        "wide": {"prefixes": 32, "cached_prefixes": 8, "new_prefixes": 24, "new_A_indices": [4, 5],
            "training_pairs": 768, "new_pairs": 576, "new_steps_per_root": 1411200,
            "new_prefix_steps": 58800, "planned_shard_steps": [235200] * 6},
        "readout": {"new_prefixes_per_root": 8, "suffix_indices": list(range(20, 28)),
            "new_steps_per_root": 1852800, "halves": "first4_last4_suffixes_and_first2_last2_windows"},
        "formal_training_total_steps": 11289600, "formal_readout_total_steps": 7411200,
        "formal_total_steps": 18700800, "total_with_prechecks": 18792000,
        "fixed_model": "parent_scaled_120_columns_zero_intercept_potential_DF3",
        "planned_fits": "one_dense_and_one_wide_per_root_8_ridge_fits_total_after_sampling_no_hyperparameter_search",
        "baselines": "reuse_frozen_parent_probe_full_models_plus_zero_no_raw_reference_training",
        "cached_B_indices_never_fit": [2, 3], "fresh_window_seed_recipe": "original_11401",
        "new_suffix_seed_recipe": "original_11402_with_reserved_role_indices",
        "budget_scope": "matched_incremental_native_steps_including_factual_replays_not_equal_labels_or_CPU",
        "sampling": "new_prefixes_use_observe_617_839_3_checks_no_outcome_selection_no_redundant_original_arms",
        "fit_gate": "all_sample_shards_complete_then_freeze_all_8_models_before_any_readout_prefix_access",
        "readout_statistics": "paired_MSE_contrasts_vs_zero_frozen_base_and_dense_vs_wide_per_prefix_and_conditional_MC_SE_separate_roots_halves_report_both_no_winner_selection",
        "current_release": "four_mechanical_prechecks_only_no_formal_sampling_or_fitting",
        "followup": "explicit_next_stage_decision_after_prechecks_no_policy_or_original_confirmation",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit-precheck", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen information design")
    forbidden = set().union(*(literal_seeds(json.loads(p.read_text())) for p in (ROOT / "protocols").glob("plan_validity*.json")))
    result = prepare(json.loads(args.source_manifest.read_text()), load_plan(args.preregistration), forbidden,
        args.run_name, args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit_precheck:
        execute_bulk(result["jobs"], dry_run=False, intent_label=args.run_name + "_precheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
