#!/usr/bin/env python3
"""Merge the native conditioning/credit factorial without a two-root CI claim."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_authority_stage148 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, *, preflight):
    contract = spec.contract(preflight)
    roots = contract["roots"]
    for root in roots:
        required = {(condition, scenario, spec.scene_seed(root, scenario))
                    for condition in spec.CONDITIONS for scenario in contract["scenarios"]}
        reference = cells["legacy", root]
        reference_rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in reference["evaluation"]}
        for method in spec.METHODS:
            cell = cells[method, root]
            if not (cell["software_qualified"] and cell["method"] == method and cell["seed"] == root
                    and cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == contract):
                raise ValueError(f"Incorrect native authority cell: {method}/{root}")
            expected_updates = {"upper": (contract["train_episodes"] - contract["upper_warmup"]) * (2 if preflight else 10),
                                "lower": contract["train_episodes"] * (2 if preflight else 30)}
            if cell["updates"] != expected_updates or not all(v > 0 for v in cell["actor_change_max_abs"].values()):
                raise ValueError(f"Missing native learning: {method}/{root}")
            if cell["actor_dims"] != contract["actor_dims"]:
                raise ValueError("Changed native actor geometry")
            for key in ("parameter_counts", "training_demand_counts"):
                if cell[key] != reference[key]:
                    raise ValueError(f"Unmatched {key}: {method}/{root}")
            rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in cell["evaluation"]}
            if len(rows) != len(cell["evaluation"]) or rows.keys() != required:
                raise ValueError(f"Incomplete or duplicate intervention scenes: {method}/{root}")
            if cell["native_steps"] != (contract["train_episodes"] + len(rows)) * contract["training_clock_s"]:
                raise ValueError("Unexpected native tick budget")
            for key, row in rows.items():
                for field in ("simulation_end_time_s", "passengers_generated", "N_fleet"):
                    if row[field] != reference_rows[key][field]:
                        raise ValueError(f"Unpaired evaluation {field}: {method}/{root}/{key}")
                if row["simulation_end_time_s"] != contract["training_clock_s"] or row["N_fleet"] != 12:
                    raise ValueError("Incorrect physical evaluation clock/fleet")
    result = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract, "preflight": preflight,
        "software_qualified": True, "optimizer_roots": roots,
        "native_steps": sum(cells[method, root]["native_steps"] for method in spec.METHODS for root in roots),
        "stage": "software_preflight" if preflight else "two_root_descriptive_mechanism_development"}
    if preflight:
        return result

    metrics = (*spec.routing.METRICS, "lower_action_mean", "upper_delta_mean")
    means = {(method, root, condition): {metric: float(np.mean([row[metric] for row in
        cells[method, root]["evaluation"] if row["condition"] == condition])) for metric in metrics}
        for method in spec.METHODS for root in roots for condition in spec.CONDITIONS}
    result["baseline_minus_legacy_root_deltas"] = {method: [{"root": root, **{metric:
        means[method, root, "baseline"][metric] - means["legacy", root, "baseline"][metric]
        for metric in metrics}} for root in roots] for method in spec.METHODS[1:]}
    result["physical_intervention_minus_baseline_root_deltas"] = {method: {condition: [
        {"root": root, **{metric: means[method, root, condition][metric] - means[method, root, "baseline"][metric]
                         for metric in metrics}} for root in roots] for condition in spec.CONDITIONS[1:]}
        for method in spec.METHODS}
    result["lower_band_action_effect_s_by_root"] = {method: [{"root": root, "mean_abs": float(np.mean([
        row["actors"]["lower"]["zero_input_effect_s"]["dynamic_band"]["mean_abs"]
        for row in cells[method, root]["evaluation"] if row["condition"] == "baseline"]))}
        for root in roots] for method in spec.METHODS}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {(method, root): json.loads((directory / "cells" / method / f"seed_{root}" / "result.json").read_text())
             for method in spec.METHODS for root in spec.contract(args.preflight)["roots"]}
    result = summarize(cells, preflight=args.preflight)
    write_json(directory / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
