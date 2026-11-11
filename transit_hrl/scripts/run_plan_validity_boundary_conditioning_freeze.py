"""Freeze all eight full-A gate potentials and retain unchanged old controls."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, count_work, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_conditioning import METHODS, fit_conditioned_targets
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_wide_transfer import load_wide_inputs


def load_freeze_inputs(manifest):
    source = json.loads((ROOT / manifest["source_manifest"]).read_text())
    previous = json.loads((ROOT / source["output_directory"] / "public/result.json").read_text())
    if (previous["run_name"], previous["status"], previous["stage"], previous["new_ridge_fits"], previous["baseline_refits"],
        previous["new_environment_steps"], previous["evaluation_prefixes_accessed"], previous["frozen_models_modified"]) != (
        source["run_name"], "complete", "boundary_conditioning_A_complete", 512, 0, 0, False, False):
        raise ValueError("matching completed A-only conditioning comparison is required")
    primary, = [c for c in previous["cells"] if (c["task"], c["optimizer_seed"]) == (
        source["primary_root"]["task"], source["primary_root"]["optimizer_seed"])]
    if not primary["training_A_followup_rule_passed"]:
        raise ValueError("primary predeclared A-only followup rule did not pass")
    wide = json.loads((ROOT / source["source_manifest"]).read_text())
    design, prepared = load_wide_inputs(wide)
    original = json.loads((ROOT / wide["fit_manifest"]).read_text())
    bundle = json.loads((ROOT / original["output_directory"] / "private/frozen_models.json").read_text())
    if bundle["run_name"] != design["run_name"] or len(bundle["cells"]) != 4 or len(prepared) != 4:
        raise ValueError("matching complete original control bundle is required")
    for item, saved in zip(prepared, bundle["cells"]):
        cell = item["cell"]
        if (cell["task"], cell["optimizer_seed"]) != (saved["task"], saved["optimizer_seed"]):
            raise ValueError("old control root identity differs")
        if saved["models"]["wide"] != item["frozen_model"]:
            raise ValueError("uniform Wide control differs from its validated source")
        item["controls"] = {k: deepcopy(saved["models"][k]) for k in ("base", "wide", "constant", "causal_t")}
    return design, prepared


def freeze_models(manifest, directory, publish):
    with measured_phase("verify_all_training_A_and_unchanged_controls"):
        design, prepared = load_freeze_inputs(manifest)
    publish({"stage": "all_training_A_and_controls_verified_before_fits"})
    bundle = {"run_name": manifest["run_name"], "source_run_name": design["run_name"], "cells": [],
        "evaluation_prefixes_accessed": False}
    metadata = []
    with measured_phase("eight_full_A_mean_fits_and_global_freeze"):
        for item in prepared:
            cell, data = item["cell"], item["data"]
            models = deepcopy(item["controls"])
            for method in METHODS:
                predictions, models[method] = fit_conditioned_targets(data["now"], data["wait_one_check"], data["training_mean"],
                    np.ones(len(data["training_mean"]), dtype=bool), total_df=manifest["total_effective_df"], gate_kind=method)
                count_work("full_A_gate_window_predictions", len(predictions))
            bundle["cells"].append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "models": models,
                "training_roster": {"window_keys": data["window_keys"], "label_counts": data["label_counts"]}})
            metadata.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
                "prefixes": len(data["prefixes"]), "windows": len(data["training_mean"]), "A_labels": sum(data["label_counts"]),
                "model_metadata": {m: {k: models[m][k] for k in ("feature_columns", "total_effective_df", "gate_center", "gate_scale")} for m in METHODS}})
        (directory / "private/frozen_models.json").write_text(json.dumps(_json_ready(bundle)) + "\n")
    publish({"stage": "all_eight_full_A_models_and_controls_frozen", "all_models_frozen": True})
    return {"stage": "boundary_conditioning_full_A_freeze_complete", "cells": metadata, "all_models_frozen": True,
        "new_ridge_fits": manifest["ridge_fit_calls"], "baseline_refits": 0, "evaluation_prefixes_accessed": False,
        "original_models_modified": False, "training_metric_selection_performed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    directory = ROOT / manifest["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve full-A model freeze attempts")
    for part in ("public", "private"):
        (directory / part).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(result.get("stage", result["status"]), flush=True)

    try:
        design = json.loads((ROOT / manifest["design_manifest"]).read_text())
        with ledger.recording("boundary_conditioning_full_A_freeze"), single_thread_runtime() as profile:
            if profile != design["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen source")
            result["hardware_profile"] = profile
            summary = freeze_models(manifest, directory, publish)
        if ledger.summary()["counts"] != manifest["expected_counts"]:
            raise RuntimeError("freeze exceeded its eight-fit and zero-simulation budget")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), new_environment_steps=ledger.summary()["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    print(result.get("stage", result["status"]), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
