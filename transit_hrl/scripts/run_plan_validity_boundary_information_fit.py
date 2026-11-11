"""Close all acquisition ledgers, then freeze eight fixed-capacity potentials."""

from copy import deepcopy
import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_information_fit import audit_public, assemble, row_map, training_readout
from freq_hrl.experiments.plan_validity_boundary_parent_scale import fit_parent_targets
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_tail import load_cache


def load_training_inputs(design, sample_release, *, total_df):
    records, raw = [], []
    for shard in sample_release["shards"]:
        directory = ROOT / shard["output_directory"]
        records.append(json.loads((directory / "public/result.json").read_text()))
    costs = audit_public(design, sample_release, records)
    for shard in sample_release["shards"]:
        rows = json.loads((ROOT / shard["output_directory"] / "private/rows.json").read_text())
        row_map(rows, shard["prefix_seeds"])
        raw.append(rows)
    source = json.loads((ROOT / design["source_manifest"]).read_text())
    prepared, bank_rosters = [], []
    for i, cell in enumerate(design["cells"]):
        cached = load_cache(cell, source)
        banks = {method: assemble(cell, design, cached,
            [r for shard, rows in zip(sample_release["shards"], raw)
                if shard["cell_index"] == i and shard["method"] == method for r in rows], method)
            for method in ("dense", "wide")}
        directory = ROOT / cell["frozen_probe_directory"]
        saved = json.loads((directory / "public/result.json").read_text())
        if (saved["run_name"], saved["task"], saved["optimizer_seed"], saved["status"], saved["stage"], saved["matched_total_effective_df"]) != (
            source["run_name"], cell["task"], cell["optimizer_seed"], "complete", "boundary_parent_scale_complete", total_df):
            raise ValueError("frozen baseline differs from the declared parent-scaled source")
        old = json.loads((directory / "private/models_and_predictions.json").read_text())["full_models"]
        controls = {"base": old["boundary_delta"], "constant": old["constant"], "causal_t": old["causal_t"]}
        prepared.append({"cell": cell, "banks": banks, "controls": controls})
        bank_rosters.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
            "strategies": {m: {"prefixes": len(b["prefixes"]), "windows": len(b["training_mean"]),
                "A_pairs": sum(b["label_counts"])} for m, b in banks.items()}})
    return {"source_run_name": design["run_name"], "shards": len(records), "environment_steps": sum(c["environment_steps"] for c in costs.values()),
        "new_pairs": sum(c["new_pairs"] for c in costs.values()), "costs_by_root_method": costs,
        "training_rosters": bank_rosters, "B_or_E_labels_used_for_fitting": False}, prepared


def run_fit(design, sample_release, manifest, directory, publish):
    with measured_phase("audit_all_public_and_private_training_banks"):
        audit, prepared = load_training_inputs(design, sample_release, total_df=manifest["total_effective_df"])
        (directory / "private/sampling_audit.json").write_text(json.dumps(_json_ready(audit), indent=2) + "\n")
    publish({"stage": "all_training_banks_audited_before_fitting", "sampling_audit": audit})
    bundle = {"run_name": design["run_name"], "cells": [], "evaluation_prefixes_accessed": False}
    with measured_phase("fixed_DF_A_only_potential_fits"):
        for item in prepared:
            models = deepcopy(item["controls"])
            for method, data in item["banks"].items():
                _, models[method] = fit_parent_targets(data["now"], data["wait_one_check"], data["training_mean"],
                    np.ones(len(data["training_mean"]), dtype=bool), total_df=manifest["total_effective_df"])
            cell = item["cell"]
            bundle["cells"].append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "models": models,
                "training_rosters": {m: {"window_keys": b["window_keys"], "label_counts": b["label_counts"]} for m, b in item["banks"].items()}})
        (directory / "private/frozen_models.json").write_text(json.dumps(_json_ready(bundle)) + "\n")
    publish({"stage": "all_eight_models_frozen_before_training_readout", "all_models_frozen": True})
    with measured_phase("training_only_descriptive_readout"):
        report = [{"task": c["task"], "optimizer_seed": c["optimizer_seed"],
            "actual_effective_df": {m: c["models"][m]["total_effective_df"] for m in ("dense", "wide")},
            "training_readout": {m: training_readout(b, c["models"]) for m, b in item["banks"].items()}}
            for c, item in zip(bundle["cells"], prepared)]
    return {"stage": "boundary_information_fit_complete", "sampling_audit": audit, "cells": report,
        "all_models_frozen": True, "new_ridge_fits": manifest["ridge_fit_calls"], "evaluation_prefixes_accessed": False,
        "scope": "A_only_training_fit_not_independent_evaluation_or_policy_effect"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    design = json.loads((ROOT / manifest["design_manifest"]).read_text())
    sample_release = json.loads((ROOT / manifest["sample_manifest"]).read_text())
    directory = ROOT / manifest["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve paid fit attempts and frozen model bundles")
    for name in ("public", "private"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": design["run_name"], "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(result.get("stage", result["status"]), flush=True)

    try:
        with ledger.recording("boundary_information_fit"), single_thread_runtime() as profile:
            if profile != design["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen acquisition profile")
            result["hardware_profile"] = profile
            summary = run_fit(design, sample_release, manifest, directory, publish)
        if ledger.summary()["counts"] != {"ridge_fit_calls": manifest["ridge_fit_calls"]}:
            raise RuntimeError("fitting exceeded the eight-model no-simulation budget")
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
