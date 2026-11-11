"""Freeze and optionally release four cache-only boundary-tail development jobs."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk


def prepare(source, profile, run_name, manifest_path):
    cells, jobs = [], []
    for index, old in enumerate(source["cells"]):
        cell = {**old, "cache_directory": old["output_directory"],
            "output_directory": f"results/{run_name}/{old['task']}/{old['optimizer_seed']}"}
        job = deepcopy(source["jobs"][2 * index])
        output = ROOT / cell["output_directory"]
        cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_boundary_tail.py", "--manifest", str(manifest_path), "--cell-index", str(index)]
        job.update(project="Freq-HRL-BoundaryTail", description=f"Cache-only tail probe {cell['task']} {cell['optimizer_seed']}",
            signature=f"Freq-HRL/boundary-tail/{run_name}/{cell['task']}/{cell['optimizer_seed']}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(output / "public"), local_result_dir=str(output / "public"),
            wait_for_files=[str(ROOT / manifest_path)], cpu=1, ram_mb=2048, vram=0)
        cells.append(cell)
        jobs.append(job)
    return {"protocol_version": "cache_only_matched_capacity_boundary_tail_v1", "run_name": run_name,
        "source_run_name": source["run_name"], "source_hardware_profile": profile, "horizon": source["horizon"],
        "period": source["window_steps"], "cells": cells, "jobs": jobs,
        "methods": ["zero", "constant", "causal_t", "boundary_delta"], "total_effective_df": 3,
        "features": "39_existing_causal_interactions_plus_clock_remaining_spent_plus_39_times_remaining_plus_39_times_spent",
        "boundary_model": "shared_linear_potential_wait_minus_now_sign_augmented_no_intercept",
        "capacity": "causal_t_1_intercept_plus_2_slope_DF_boundary_0_intercept_plus_3_slope_DF",
        "fit_suffix_indices": [0, 1], "readout_suffix_indices": [2, 3],
        "folds": "leave_one_whole_prefix_out_8_folds_no_hyperparameter_search",
        "ridge_fits_per_root": 18, "constant_fits_per_root": 9, "total_ridge_fits": 72, "total_constant_fits": 36,
        "new_environment_steps": 0, "new_controller_updates": 0, "E8_reused": False,
        "qualifying_rule": "boundary_prefix_OOF_B_MSE_below_zero_constant_causal_t_overall_and_both_window_halves_each_root",
        "statistics": "training_A_no_noise_correction_B_and_prefix_OOF_correct_conditional_suffix_noise_negative_estimates_retained",
        "data_scope": "previously_exposed_development_prefixes_not_new_confirmation_no_policy_decision",
        "followup": "none_no_upstream_distillation_simulation_or_policy_release_automatically",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen cache-only probe manifest")
    source = json.loads(args.source_manifest.read_text())
    previous = json.loads((ROOT / source["source_manifest"]).read_text())
    manifest = prepare(source, previous["source_hardware_profile"], args.run_name, args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.submit:
        execute_bulk(manifest["jobs"], dry_run=False, intent_label=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
