#!/usr/bin/env python3
"""Keep regularizer, critic identifiability and physical upper effects separate."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_critic_regularization_stage153 as spec
from scripts.analyze_native_transit_dispatch_train_stage151 import matched_means
from scripts.analyze_native_transit_critic_units_stage152 import qualified_critic_diagnostics
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells):
    contract, metrics, means, regimes = matched_means(cells, spec)
    diagnostics = qualified_critic_diagnostics(cells, spec, means)
    for row in diagnostics:
        method, root = row["method"], row["root"]
        cell = cells[method, root]
        if cell["weight_reg_mode"] != contract["weight_reg_mode"][method]:
            raise ValueError("Unmatched registered critic regularization")
        curves = [r["critic"]["action_curve"] for r in cell["evaluation"] if r["condition"] == "baseline"]
        row["episode_mean_q_action_range"] = float(np.mean([
            max(v["q_mean"] for v in curve.values()) - min(v["q_mean"] for v in curve.values())
            for curve in curves]))
        row["paired_q_difference_to_zero_abs_mean"] = {goal: float(np.mean([
            curve[goal]["q_difference_to_zero_abs_mean"] for curve in curves])) for goal in curves[0]}

    def difference(a, b, root, condition_a="baseline", condition_b="baseline"):
        return {"root": root, **{key: means[a, root, condition_a][key]
            - means[b, root, condition_b][key] for key in metrics}}

    comparisons = {"unit_minus_seconds": ("unit_sum", "seconds_sum"),
        "physical_minus_unit": ("unit_physical", "unit_sum"),
        "mean_minus_unit": ("unit_mean", "unit_sum"),
        "physical_minus_seconds": ("unit_physical", "seconds_sum")}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract, "software_qualified": True,
        "stage": "two_root_descriptive_regularization_not_confirmation",
        "native_steps": sum(c["native_steps"] for c in cells.values()),
        "worker_preflight_native_steps": sum(c["worker_preflight_native_steps"] for c in cells.values()),
        "root_contrasts": {name: [difference(a, b, root) for root in spec.ROOTS]
            for name, (a, b) in comparisons.items()},
        "learned_minus_neutral_upper": {method: [difference(method, method, root,
            condition_b="neutral_upper") for root in spec.ROOTS] for method in spec.METHODS},
        "zero_holding_minus_learned": {method: [difference(method, method, root,
            condition_a="zero_holding") for root in spec.ROOTS] for method in spec.METHODS},
        "critic_diagnostics": diagnostics,
        "regime_root_means": [{"method": method, "root": root, "condition": condition,
            "scenario": scenario, **values} for (method, root, condition, scenario), values in regimes.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {(method, root): json.loads((directory / "cells" / method / f"seed_{root}/result.json").read_text())
             for method in spec.METHODS for root in spec.ROOTS}
    result = summarize(cells)
    write_json(directory / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"regime_root_means", "critic_diagnostics"}}, indent=2))


if __name__ == "__main__":
    main()
