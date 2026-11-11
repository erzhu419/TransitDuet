"""A-only diagnostic fits; leave frozen Wide and all E readouts untouched."""

import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, count_work, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_information_fit import assemble
from freq_hrl.experiments.plan_validity_boundary_parent_scale import predict_parent
from freq_hrl.experiments.plan_validity_boundary_wide_transfer import diagnose
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_tail import load_cache


def load_wide_inputs(manifest, *, cell_indices=(0, 1, 2, 3)):
    design = json.loads((ROOT / manifest["design_manifest"]).read_text())
    sample = json.loads((ROOT / manifest["sample_manifest"]).read_text())
    fit = json.loads((ROOT / manifest["fit_manifest"]).read_text())
    directory = ROOT / fit["output_directory"]
    report = json.loads((directory / "public/result.json").read_text())
    saved = json.loads((directory / "private/frozen_models.json").read_text())
    if (report["run_name"], report["status"], report["stage"], report["all_models_frozen"], report["new_ridge_fits"],
        report["new_environment_steps"], saved["run_name"], len(saved["cells"]), len(report["cells"])) != (
        design["run_name"], "complete", "boundary_information_fit_complete", True, 8, 0, design["run_name"], 4, 4):
        raise ValueError("matching completed frozen Wide source is required")
    source = json.loads((ROOT / design["source_manifest"]).read_text())
    prepared = []
    for i in cell_indices:
        cell, frozen, prior = design["cells"][i], saved["cells"][i], report["cells"][i]
        identity = (cell["task"], cell["optimizer_seed"])
        if any(identity != (c["task"], c["optimizer_seed"]) for c in (frozen, prior)):
            raise ValueError("source root identities differ")
        added = [r for shard in sample["shards"] if shard["cell_index"] == i and shard["method"] == "wide"
            for r in json.loads((ROOT / shard["output_directory"] / "private/rows.json").read_text())]
        data = assemble(cell, design, load_cache(cell, source), added, "wide")
        roster = frozen["training_rosters"]["wide"]
        if data["window_keys"] != roster["window_keys"] or data["label_counts"] != roster["label_counts"]:
            raise ValueError("Wide A bank differs from the frozen source roster")
        predictions = predict_parent(frozen["models"]["wide"], data["now"], data["wait_one_check"])
        count_work("frozen_model_window_predictions", len(predictions))
        mse = float(np.mean((predictions - data["training_mean"]) ** 2))
        expected = prior["training_readout"]["wide"]["MSE_to_own_training_A_mean"]["wide"]
        if not np.isclose(mse, expected, rtol=1e-10, atol=1e-12):
            raise ValueError("training A bank does not reproduce frozen Wide loss")
        prepared.append({"cell": cell, "data": data, "frozen_mean_A_MSE": mse,
            "frozen_model": frozen["models"]["wide"]})
    return design, prepared


def run_diagnostic(manifest, directory, publish):
    with measured_phase("assemble_only_Wide_A_and_reproduce_frozen_losses"):
        design, prepared = load_wide_inputs(manifest)
    publish({"stage": "all_Wide_A_banks_verified_before_fitting"})
    result = []
    with measured_phase("fixed_DF_matched_size_future_and_prefix_crossfit"):
        for item in prepared:
            cell = item["cell"]
            public, private = diagnose(item["data"], cell["development_prefix_seeds"], total_df=manifest["total_effective_df"])
            (directory / "private" / f"{cell['task']}_{cell['optimizer_seed']}.json").write_text(json.dumps(_json_ready(private)) + "\n")
            result.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
                "source_frozen_mean_A_MSE": item["frozen_mean_A_MSE"], "diagnostic": public})
            publish({"stage": "Wide_A_diagnostic_roots", "completed_roots": len(result)})
    return {"stage": "boundary_Wide_A_transfer_complete", "source_run_name": design["run_name"],
        "cells": result, "evaluation_prefixes_accessed": False, "frozen_models_modified": False,
        "new_ridge_fits": manifest["ridge_fit_calls"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    design = json.loads((ROOT / manifest["design_manifest"]).read_text())
    directory = ROOT / manifest["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve completed A-only diagnostic attempts")
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
        with ledger.recording("boundary_Wide_A_transfer"), single_thread_runtime() as profile:
            if profile != design["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen acquisition profile")
            result["hardware_profile"] = profile
            summary = run_diagnostic(manifest, directory, publish)
        if ledger.summary()["counts"] != manifest["expected_counts"]:
            raise RuntimeError("A diagnostic exceeded its fixed fit/score and zero-simulation budget")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), new_environment_steps=ledger.summary()["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
