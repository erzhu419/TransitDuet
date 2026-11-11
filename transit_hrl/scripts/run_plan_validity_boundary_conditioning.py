"""A-only fixed-capacity conditioning, with existing uniform folds reused."""

import argparse
from functools import partial
import json
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_conditioning import METHODS, compare_views, fit_conditioned_targets, followup_rule
from freq_hrl.experiments.plan_validity_boundary_wide_transfer import diagnose, errors
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_wide_transfer import load_wide_inputs


def load_sources(manifest):
    source = json.loads((ROOT / manifest["source_manifest"]).read_text())
    design, prepared = load_wide_inputs(source)
    directory = ROOT / source["output_directory"]
    old = json.loads((directory / "public/result.json").read_text())
    if (old["run_name"], old["status"], old["stage"], old["new_ridge_fits"], old["new_environment_steps"],
        old["evaluation_prefixes_accessed"], old["frozen_models_modified"]) != (
        source["run_name"], "complete", "boundary_Wide_A_transfer_complete", 256, 0, False, False):
        raise ValueError("matching completed uniform diagnostic is required")
    for item, prior in zip(prepared, old["cells"]):
        cell, data = item["cell"], item["data"]
        if (cell["task"], cell["optimizer_seed"]) != (prior["task"], prior["optimizer_seed"]):
            raise ValueError("cached uniform root differs")
        saved = json.loads((directory / "private" / f"{cell['task']}_{cell['optimizer_seed']}.json").read_text())
        labels = np.asarray(data["training_replicas"])
        for r, view in enumerate(saved):
            if view["fitting_A_column"] != r:
                raise ValueError("cached uniform A view differs")
            reproduced = errors(np.asarray(view["matched_prediction_mean"]), np.asarray(view["matched_prediction_square_mean"]),
                np.asarray(view["prefix_held_predictions"]), labels[:, r], labels[:, 1 - r], np.ones(384, dtype=bool))
            if any(not np.isclose(value, prior["diagnostic"]["views"][r]["overall"][k], rtol=1e-10, atol=1e-12)
                for k, value in reproduced.items()):
                raise ValueError("cached uniform moments do not reproduce source losses")
        item.update(baseline_public=prior["diagnostic"], baseline_private=saved)
    return design, prepared


def run_comparison(manifest, directory, publish):
    with measured_phase("verify_all_A_banks_and_reuse_uniform_moments"):
        design, prepared = load_sources(manifest)
    publish({"stage": "all_A_banks_and_uniform_moments_verified_before_new_fits"})
    cells = []
    with measured_phase("fixed_DF_conditioned_folds_and_paired_readout"):
        for item in prepared:
            cell, data = item["cell"], item["data"]
            public, private = {}, {"uniform": item["baseline_private"]}
            for method in METHODS:
                fitter = partial(fit_conditioned_targets, gate_kind=method)
                public[method], private[method] = diagnose(data, cell["development_prefix_seeds"], total_df=manifest["total_effective_df"], fit_targets=fitter)
                (directory / "private" / f"{cell['task']}_{cell['optimizer_seed']}_{method}.json").write_text(json.dumps(_json_ready(private[method])) + "\n")
            contrasts = {candidate + "_minus_" + reference: compare_views(private[candidate], private[reference], data)
                for candidate, reference in (("tracking_distance", "uniform"), ("state_rms", "uniform"), ("tracking_distance", "state_rms"))}
            cells.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
                "baseline_mean": item["baseline_public"]["mean_across_views_descriptive"], "methods": public,
                "contrasts": contrasts, "training_A_followup_rule_passed": followup_rule(public, contrasts)})
            publish({"stage": "conditioning_roots", "completed_roots": len(cells)})
    return {"stage": "boundary_conditioning_A_complete", "source_run_name": design["run_name"], "cells": cells,
        "new_ridge_fits": manifest["ridge_fit_calls"], "baseline_refits": 0,
        "evaluation_prefixes_accessed": False, "frozen_models_modified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    directory = ROOT / manifest["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve completed conditioning attempts")
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
        with ledger.recording("boundary_conditioning_A"), single_thread_runtime() as profile:
            if profile != design["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen source")
            result["hardware_profile"] = profile
            summary = run_comparison(manifest, directory, publish)
        if ledger.summary()["counts"] != manifest["expected_counts"]:
            raise RuntimeError("conditioning exceeded its fixed fit/score and zero-simulation budget")
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
