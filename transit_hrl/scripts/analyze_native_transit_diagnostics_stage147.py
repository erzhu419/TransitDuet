#!/usr/bin/env python3
"""Summarize frozen actor diagnostics and paired actuator interventions."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_diagnostics_stage147 as spec
from scripts import run_native_transit_routing_stage146 as routing
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells):
    for method in spec.METHODS:
        for root in spec.ROOTS:
            cell = cells[method, root]
            if not (cell["software_qualified"] and cell["baseline_reproduced"]
                    and cell["method"] == method and cell["seed"] == root
                    and cell["protocol"] == spec.EXPERIMENT_PROTOCOL
                    and cell["contract"] == spec.contract(False)
                    and cell["training_updates"] == 0):
                raise ValueError(f"Unqualified frozen diagnostic: {method}/{root}")
            expected = {(condition, scenario, routing.evaluation_seeds(root, scenario, preflight=False)[0])
                        for condition in spec.conditions(method, preflight=False)
                        for scenario in spec.contract(False)["scenarios"]}
            rows = cell["evaluation"]
            actual = {(r["condition"], r["scenario"], r["scene_seed"]) for r in rows}
            if expected != actual or len(rows) != len(expected):
                raise ValueError(f"Incomplete frozen diagnostic scenes: {method}/{root}")
    actors = {}
    for method in spec.METHODS:
        rows = [r for root in spec.ROOTS for r in cells[method, root]["evaluation"]
                if r["condition"] == "baseline"]
        actors[method] = {}
        for level in ("upper", "lower"):
            values = [row["actors"][level] for row in rows]
            actors[method][level] = {
                key: float(np.mean([value[key] for value in values]))
                for key in ("proposal_mean_s", "proposal_std_s", "proposal_edge_fraction_1pct",
                            "latent_mean_abs_p95")}
            actors[method][level]["input_abs_mean"] = np.mean(
                [value["input_abs_mean"] for value in values], axis=0).tolist()
            actors[method][level]["zero_input_mean_abs_effect_s"] = {
                block: float(np.mean([value["zero_input_effect_s"][block]["mean_abs"] for value in values]))
                for block in values[0]["zero_input_effect_s"]}
    interventions = {}
    for condition in ("neutral_upper", "zero_holding"):
        deltas = []
        for root in spec.ROOTS:
            rows = cells["correct", root]["evaluation"]
            deltas.append({"root": root, **{
                metric: float(np.mean([r[metric] for r in rows if r["condition"] == condition])
                              - np.mean([r[metric] for r in rows if r["condition"] == "baseline"]))
                for metric in routing.METRICS}})
        interventions[condition] = {
            "direction": "intervention_minus_learned_baseline",
            "root_deltas": deltas,
            "mean_deltas": {metric: float(np.mean([r[metric] for r in deltas])) for metric in routing.METRICS}}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "software_qualified": True,
            "stage": "post_result_descriptive_diagnostics_not_independent_confirmation",
            "native_steps": sum(cell["native_steps"] for cell in cells.values()),
            "baseline_actor_diagnostics": actors, "physical_interventions": interventions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {(method, root): json.loads((directory / "cells" / method / f"seed_{root}" / "result.json").read_text())
             for method in spec.METHODS for root in spec.ROOTS}
    result = summarize(cells)
    write_json(directory / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
