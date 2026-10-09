#!/usr/bin/env python3
"""Analyze paired critic units separately from learned upper authority."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_critic_units_stage152 as spec
from scripts.analyze_native_transit_dispatch_train_stage151 import matched_means
from freq_hrl.experiments.pointmaze_root_response import write_json


def qualified_critic_diagnostics(cells, spec, means):
    contract = spec.contract(False)
    diagnostic_rows = []
    for (method, root), cell in cells.items():
        units = contract["critic_action_units"][method]
        if cell["critic_action_units"] != units:
            raise ValueError("Unmatched critic action coordinates")
        learning = cell["training_curve"][-1]["upper_learning"]
        if not learning or not all(np.isfinite(v) for v in learning.values()):
            raise ValueError("Missing finite upper learning diagnostics")
        if not 0 <= learning["upper_target_clip_fraction"] <= 1:
            raise ValueError("Invalid Bellman target clip rate")
        average = cell["upper_learning_mean"]
        if cell["upper_learning_updates"] != spec.expected_updates(False)["upper"] or not average:
            raise ValueError("Incomplete successful upper update statistics")
        if not all(np.isfinite(value) for value in average.values()) or not 0 <= average["upper_target_clip_fraction"] <= 1:
            raise ValueError("Invalid full-training upper learning statistics")
        rows = [r for r in cell["evaluation"] if r["condition"] == "baseline"]
        for row in rows:
            scale = row["critic"]["input_scale"]
            if scale["critic_action_units"] != units or not all(np.isfinite(scale[key]) for key in
                    ("state_contribution_abs_mean", "action_contribution_abs_mean")):
                raise ValueError("Missing finite critic input contribution diagnostics")
        diagnostic_rows.append({"method": method, "root": root, "last_update": learning,
            "all_update_mean": average,
            "state_contribution_abs_mean": float(np.mean([
                r["critic"]["input_scale"]["state_contribution_abs_mean"] for r in rows])),
            "action_contribution_abs_mean": float(np.mean([
                r["critic"]["input_scale"]["action_contribution_abs_mean"] for r in rows])),
            "advance_count": sum(r["dispatch"]["advance_count"] for r in rows),
            "delay_count": sum(r["dispatch"]["delay_count"] for r in rows),
            "command_abs_mean_s": means[method, root, "baseline"]["command_abs_mean_s"],
            "subsecond_command_fraction": means[method, root, "baseline"]["subsecond_command_fraction"]})
    return diagnostic_rows


def summarize(cells):
    contract, metrics, means, regimes = matched_means(cells, spec)
    diagnostic_rows = qualified_critic_diagnostics(cells, spec, means)

    def difference(a, b, root, condition_a="baseline", condition_b="baseline"):
        return {"root": root, **{key: means[a, root, condition_a][key]
            - means[b, root, condition_b][key] for key in metrics}}

    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract, "software_qualified": True,
        "stage": "two_root_descriptive_critic_coordinates_not_confirmation",
        "native_steps": sum(c["native_steps"] for c in cells.values()),
        "worker_preflight_native_steps": sum(c["worker_preflight_native_steps"] for c in cells.values()),
        "unit_minus_seconds": {base: [difference(base + "_unit", base, root) for root in spec.ROOTS]
            for base in ("dispatch", "dispatch_service_credit")},
        "learned_minus_neutral_upper": {method: [difference(method, method, root,
            condition_b="neutral_upper") for root in spec.ROOTS] for method in spec.METHODS},
        "zero_holding_minus_learned": {method: [difference(method, method, root,
            condition_a="zero_holding") for root in spec.ROOTS] for method in spec.METHODS},
        "critic_diagnostics": diagnostic_rows,
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
    print(json.dumps({key: value for key, value in result.items()
                     if key not in {"regime_root_means", "critic_diagnostics"}}, indent=2))


if __name__ == "__main__":
    main()
