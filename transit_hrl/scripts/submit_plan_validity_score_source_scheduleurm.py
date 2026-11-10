"""Freeze four saved-score controls, with no repeated folds or native access."""

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
        cell = deepcopy(old)
        cell.update(calibration_directory=old["output_directory"],
            output_directory=f"results/{run_name}/{old['task']}/{old['optimizer_seed']}")
        cells.append(cell)
        directory = ROOT / cell["output_directory"] / "public"
        job = deepcopy(source["jobs"][index])
        command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_score_source.py",
            "--manifest", str(manifest_path), "--cell-index", str(index)]
        job.update(project="Freq-HRL-ScoreSource", description=f"Saved-score A-only calibration control {cell['task']} {cell['optimizer_seed']}",
            signature=f"Freq-HRL/score-source/{run_name}/{cell['task']}/{cell['optimizer_seed']}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command), result_dir=str(directory), local_result_dir=str(directory),
            wait_for_files=[str(ROOT / manifest_path)])
        jobs.append(job)
    return {"protocol_version": "frozen_OOF_vs_full_score_A_only_calibration_v1", "run_name": run_name,
        "source_manifest": str(source_path), "source_run_name": source["run_name"], "label_run_name": source["source_run_name"],
        "source_hardware_profile": source["source_hardware_profile"], "cells": cells, "jobs": jobs,
        "original_quantile": source["original_quantile"], "selection_target": source["selection_target"],
        "candidate_cuts": "original_cut_plus_unique_calibration_source_midpoints_and_finite_outer_endpoints",
        "ties": source["ties"], "calibration_sources": source["views"], "readout_views": source["views"],
        "B_readout": source["B_readout"], "primary_contrast": "full_selected_minus_oof_selected_on_same_frozen_full_scores",
        "ridge_fit_calls": 0, "new_environment_steps": 0, "new_frozen_full_model_fits": 0,
        "statistical_scope": "previously_revealed_known_training_prefix_B_full_score_A_calibration_is_in_sample",
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
        raise FileExistsError("preserve the frozen score-source manifest")
    result = prepare(json.loads(args.source_manifest.read_text()), args.source_manifest.resolve().relative_to(ROOT),
        args.run_name, args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit:
        execute_bulk(result["jobs"], dry_run=False, intent_label=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
