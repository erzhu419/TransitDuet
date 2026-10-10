#!/usr/bin/env python3
"""Report learned-minus-constant phase effects without selecting an oracle."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_phase_adaptation_stage154 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, sources):
    if set(cells) != set(spec.ROOTS) or set(sources) != set(spec.ROOTS):
        raise ValueError("Incomplete phase-adaptation root matrix")
    contract, means, panels = spec.contract(), {}, []
    for root in spec.ROOTS:
        cell, source = cells[root], sources[root]
        spec.validate_source(source, root)
        constants = spec.constant_actions(source)
        if not (cell["software_qualified"] and cell["baseline_reproduced"] and cell["training_updates"] == 0
                and cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == contract
                and cell["method"] == spec.METHOD and cell["seed"] == root
                and cell["constant_actions_s"] == constants):
            raise ValueError("Unqualified frozen phase-adaptation cell")
        rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in cell["evaluation"]}
        required = {(condition, scenario, scene) for condition in spec.CONDITIONS
            for scenario in contract["scenarios"] for scene in spec.source_spec.scene_seeds(root, scenario, preflight=False)}
        if set(rows) != required or len(rows) != len(cell["evaluation"]):
            raise ValueError("Missing or duplicate frozen phase scenes")
        if cell["native_steps"] != len(rows) * contract["training_clock_s"]:
            raise ValueError("Incorrect frozen phase tick budget")
        original = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in source["evaluation"]}
        for key, row in rows.items():
            condition, scenario, scene = key
            spec.check_row(row, original["baseline", scenario, scene], condition)
            if row["fixed_action_s"] != constants.get(condition):
                raise ValueError("Changed registered constant command")
        for scenario in contract["scenarios"]:
            for condition in (*spec.CONDITIONS, "neutral_upper"):
                selected = [r for r in (original if condition == "neutral_upper" else rows).values()
                            if r["scenario"] == scenario and r["condition"] == condition]
                metrics = {key: float(np.mean([r[key] for r in selected])) for key in spec.METRICS}
                metrics["fleet_cost_component"] = float(np.mean([
                    max(0, r["peak_fleet"] - r["N_fleet"]) ** 2 / r["N_fleet"] for r in selected]))
                means[root, scenario, condition] = metrics
                panels.append({"root": root, "scenario": scenario, "condition": condition, **metrics})
    metrics = (*spec.METRICS, "fleet_cost_component")

    def contrasts(a, b):
        return [{"root": root, **{key: float(np.mean([
            means[root, scenario, a][key] - means[root, scenario, b][key]
            for scenario in contract["scenarios"]])) for key in metrics}} for root in spec.ROOTS]

    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract, "software_qualified": True,
        "stage": "post_result_phase_vs_adaptation_diagnosis", "training_updates": 0,
        "native_steps": sum(c["native_steps"] for c in cells.values()),
        "learned_minus_constant": {name: contrasts("baseline", name) for name in spec.CONDITIONS[1:]},
        "constant_minus_zero": {name: contrasts(name, "neutral_upper") for name in spec.CONDITIONS[1:]},
        "learned_minus_zero": contrasts("baseline", "neutral_upper"), "regime_root_means": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {root: json.loads((directory / "cells" / spec.METHOD / f"seed_{root}/result.json").read_text())
             for root in spec.ROOTS}
    sources = {root: spec.load_source(root)[1] for root in spec.ROOTS}
    result = summarize(cells, sources)
    write_json(directory / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "regime_root_means"}, indent=2))


if __name__ == "__main__":
    main()
