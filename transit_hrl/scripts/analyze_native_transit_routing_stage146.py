#!/usr/bin/env python3
"""Merge matched native routing cells, clustering comparisons by optimizer root."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_routing_stage146 as spec
from freq_hrl.experiments.statistics import bootstrap_mean_ci
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, *, preflight):
    roots = spec.ROOTS[:1] if preflight else spec.ROOTS
    for method in spec.METHODS:
        for root in roots:
            cell = cells[method, root]
            if not (cell["passed"] and cell["method"] == method and cell["seed"] == root
                    and cell["protocol"] == spec.EXPERIMENT_PROTOCOL
                    and cell["contract"] == spec.contract(preflight)):
                raise ValueError(f"Incorrect native routing cell: {method}/{root}")
    for root in roots:
        correct = cells["correct", root]
        reference_scenes = {(r["scenario"], r["scene_seed"]): r for r in correct["evaluation"]}
        for method in spec.METHODS:
            cell = cells[method, root]
            for key in ("actor_dims", "parameter_counts", "updates", "native_steps", "training_demand_counts"):
                if cell[key] != correct[key]:
                    raise ValueError(f"Unmatched {key}: {method}/{root}")
            scenes = {(r["scenario"], r["scene_seed"]): r for r in cell["evaluation"]}
            if scenes.keys() != reference_scenes.keys():
                raise ValueError(f"Unpaired evaluation scenes: {method}/{root}")
            for key, reference in reference_scenes.items():
                for field in ("simulation_end_time_s", "passengers_generated", "N_fleet"):
                    if scenes[key][field] != reference[field]:
                        raise ValueError(f"Unmatched evaluation {field}: {method}/{root}/{key}")
    result = {
        "protocol": spec.EXPERIMENT_PROTOCOL, "preflight": preflight,
        "software_qualified": True, "optimizer_roots": list(roots),
        "native_steps": sum(c["native_steps"] for c in cells.values()),
        "parameter_counts": cells["correct", roots[0]]["parameter_counts"],
        "stage": spec.contract(preflight)["stage"],
    }
    if preflight:
        return result

    comparisons = {}
    for control in spec.METHODS[1:]:
        rows = []
        for root in roots:
            row = {"root": root}
            for metric in spec.METRICS:
                treatment = np.mean([r[metric] for r in cells["correct", root]["evaluation"]])
                baseline = np.mean([r[metric] for r in cells[control, root]["evaluation"]])
                row[metric] = float(treatment - baseline)
            rows.append(row)
        values = [r["service_cost_restricted"] for r in rows]
        ci = bootstrap_mean_ci(values, n_boot=10000, seed=146, alpha=0.025)
        comparisons[control] = {
            "primary_mean_delta": float(np.mean(values)), "family_adjusted_primary_ci": list(ci),
            "primary_direction": "negative_is_better",
            "primary_status": "development_supported" if ci[1] < 0 else "negative" if ci[0] > 0 else "inconclusive",
            "root_deltas": rows,
            "secondary_mean_deltas": {m: float(np.mean([r[m] for r in rows])) for m in spec.METRICS[1:]},
            "scenario_primary_deltas": {
                scenario: float(np.mean([
                    np.mean([r["service_cost_restricted"] for r in cells["correct", root]["evaluation"] if r["scenario"] == scenario])
                    - np.mean([r["service_cost_restricted"] for r in cells[control, root]["evaluation"] if r["scenario"] == scenario])
                    for root in roots])) for scenario in spec.SCENARIOS},
        }
    result["comparisons"] = comparisons
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    roots = spec.ROOTS[:1] if args.preflight else spec.ROOTS
    cells = {(method, root): json.loads((directory / "cells" / method / f"seed_{root}" / "result.json").read_text())
             for method in spec.METHODS for root in roots}
    summary = summarize(cells, preflight=args.preflight)
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
