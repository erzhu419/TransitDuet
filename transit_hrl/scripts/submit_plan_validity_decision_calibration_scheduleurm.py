"""Freeze four cache-only calibration jobs; no native or revealed-prefix inputs."""

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


def prepare(source, source_path, run_name, manifest_path):
    cells, jobs = [], []
    for index, old in enumerate(source["cells"]):
        cell = {k: deepcopy(old[k]) for k in ("task", "optimizer_seed", "fit_prefix_seeds")}
        cell.update(sample_directory=old["output_directory"] + "/sample", fit_directory=old["output_directory"] + "/fit",
            output_directory=f"results/{run_name}/{old['task']}/{old['optimizer_seed']}")
        cells.append(cell)
        directory = ROOT / cell["output_directory"] / "public"
        job = deepcopy(source["jobs"]["fit"][index])
        command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_decision_calibration.py",
            "--manifest", str(manifest_path), "--cell-index", str(index)]
        job.update(project="Freq-HRL-DecisionCalibration", description=f"Cache-only A-OOF utility calibration {cell['task']} {cell['optimizer_seed']}",
            signature=f"Freq-HRL/decision-calibration/{run_name}/{cell['task']}/{cell['optimizer_seed']}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command), result_dir=str(directory), local_result_dir=str(directory),
            wait_for_files=[str(ROOT / manifest_path)])
        jobs.append(job)
    return {"protocol_version": "training_A_oof_G_calibration_with_quantile_W_floor_v1", "run_name": run_name,
        "source_manifest": str(source_path), "source_run_name": source["run_name"],
        "source_hardware_profile": source["source_hardware_profile"], "cells": cells, "jobs": jobs,
        "effective_df": source["effective_df"], "original_quantile": source["threshold_quantile"],
        "selection_target": "common_C0_training_A_full_G_max_subject_W_not_below_original_quantile",
        "candidate_cuts": "original_cut_plus_unique_OOF_midpoints_and_finite_outer_endpoints",
        "ties": "maximize_G_then_minimize_changed_A_actions_then_distance_from_original_cut_then_declared_order",
        "B_readout": "B1_first_two_B2_last_two_and_all_four_futures_never_select_cuts_or_heads",
        "views": ["oof_scores", "frozen_full_scores"], "fold_reconstruction_ridge_calls": 256,
        "new_environment_steps": 0, "new_frozen_full_model_fits": 0,
        "statistical_scope": "known_training_prefix_future_readout_B_previously_revealed_development_data",
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
        raise FileExistsError("preserve the frozen decision-calibration manifest")
    result = prepare(json.loads(args.source_manifest.read_text()), args.source_manifest.resolve().relative_to(ROOT),
        args.run_name, args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit:
        execute_bulk(result["jobs"], dry_run=False, intent_label=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
