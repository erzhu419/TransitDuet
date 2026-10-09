#!/usr/bin/env python3
"""Analyze matched learned dispatch/credit effects without a four-root CI claim."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_dispatch_train_stage151 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def matched_means(cells, experiment=spec):
    spec = experiment
    contract = spec.contract(False)
    if set(cells) != {(method, root) for method in spec.METHODS for root in spec.ROOTS}:
        raise ValueError("Incomplete native dispatch training matrix")
    means, regime_means = {}, {}
    metrics = (*spec.authority.routing.METRICS, "lower_action_mean", "upper_delta_mean",
               "command_abs_mean_s", "subsecond_command_fraction")
    for root in spec.ROOTS:
        required = {(condition, scenario, scene) for condition in spec.CONDITIONS
            for scenario in contract["scenarios"] for scene in spec.scene_seeds(root, scenario, preflight=False)}
        reference = cells[next(iter(spec.METHODS)), root]
        reference_rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in reference["evaluation"]}
        for method in spec.METHODS:
            cell = cells[method, root]
            if not (cell["software_qualified"] and cell["worker_preflight_passed"]
                    and cell["worker_preflight_native_steps"] == 5 * 5400
                    and cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == contract
                    and cell["method"] == method and cell["seed"] == root
                    and cell["updates"] == spec.expected_updates(False)
                    and cell["actor_dims"] == contract["actor_dims"]
                    and set(cell["actor_change_max_abs"]) == {"upper", "lower"}
                    and all(v > 0 for v in cell["actor_change_max_abs"].values())):
                raise ValueError(f"Unqualified native dispatch training: {method}/{root}")
            for key in ("parameter_counts", "training_demand_counts", "training_fleets"):
                if cell[key] != reference[key]:
                    raise ValueError(f"Unmatched {key}: {method}/{root}")
            if any(len(cell[key]) != contract["train_episodes"] for key in ("training_demand_counts", "training_fleets")):
                raise ValueError("Incomplete training episode ledger")
            rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in cell["evaluation"]}
            if set(rows) != required or len(rows) != len(cell["evaluation"]):
                raise ValueError(f"Missing or duplicate dispatch scenes: {method}/{root}")
            if cell["native_steps"] != (contract["train_episodes"] + len(rows)) * contract["training_clock_s"]:
                raise ValueError("Unmatched native training/evaluation tick budget")
            for key, row in rows.items():
                for field in ("simulation_end_time_s", "passengers_generated", "N_fleet"):
                    if row[field] != reference_rows[key][field]:
                        raise ValueError(f"Unpaired evaluation {field}: {method}/{root}")
                if row["simulation_end_time_s"] != contract["training_clock_s"] or row["N_fleet"] != 12:
                    raise ValueError("Wrong physical evaluation clock/fleet")
                if not all(np.isfinite(row[metric]) for metric in metrics):
                    raise ValueError("Nonfinite native dispatch metric")
            for condition in spec.CONDITIONS:
                for scenario in contract["scenarios"]:
                    regime_means[method, root, condition, scenario] = {metric: float(np.mean([
                        row[metric] for row in rows.values() if row["condition"] == condition and row["scenario"] == scenario]))
                        for metric in metrics}
                means[method, root, condition] = {metric: float(np.mean([
                    regime_means[method, root, condition, scenario][metric] for scenario in contract["scenarios"]]))
                    for metric in metrics}

    return contract, metrics, means, regime_means


def summarize(cells):
    contract, metrics, means, regime_means = matched_means(cells)

    def delta(a, b, root, condition="baseline"):
        return {metric: means[a, root, condition][metric] - means[b, root, condition][metric] for metric in metrics}

    contrasts = {"dispatch_minus_hiro": ("dispatch", "hiro"),
        "dispatch_credit_minus_hiro_credit": ("dispatch_service_credit", "hiro_service_credit"),
        "credit_effect_in_dispatch": ("dispatch_service_credit", "dispatch")}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract, "software_qualified": True,
        "stage": "four_root_descriptive_learned_native_development",
        "native_steps": sum(cell["native_steps"] for cell in cells.values()),
        "worker_preflight_native_steps": sum(cell["worker_preflight_native_steps"] for cell in cells.values()),
        "root_contrasts": {name: [{"root": root, **delta(a, b, root)} for root in spec.ROOTS]
                           for name, (a, b) in contrasts.items()},
        "credit_by_action_mode_interaction": [{"root": root, **{metric:
            delta("dispatch_service_credit", "hiro_service_credit", root)[metric]
            - delta("dispatch", "hiro", root)[metric] for metric in metrics}} for root in spec.ROOTS],
        "learned_minus_neutral_upper": {method: [{"root": root, **{metric:
            means[method, root, "baseline"][metric] - means[method, root, "neutral_upper"][metric]
            for metric in metrics}} for root in spec.ROOTS] for method in spec.METHODS},
        "zero_holding_minus_learned": {method: [{"root": root, **{metric:
            means[method, root, "zero_holding"][metric] - means[method, root, "baseline"][metric]
            for metric in metrics}} for root in spec.ROOTS] for method in spec.METHODS},
        "regime_root_means": [{"method": method, "root": root, "condition": condition,
            "scenario": scenario, **values} for (method, root, condition, scenario), values in regime_means.items()],
        "execution_by_root": [{"method": method, "root": root,
            "advance_count": sum(row["dispatch"]["advance_count"] for row in cells[method, root]["evaluation"] if row["condition"] == "baseline"),
            "delay_count": sum(row["dispatch"]["delay_count"] for row in cells[method, root]["evaluation"] if row["condition"] == "baseline"),
            "baseline_mean_command_abs_s": means[method, root, "baseline"]["command_abs_mean_s"],
            "baseline_subsecond_command_fraction": means[method, root, "baseline"]["subsecond_command_fraction"]}
            for method in spec.METHODS for root in spec.ROOTS]}


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
                     if key not in {"regime_root_means", "execution_by_root"}}, indent=2))


if __name__ == "__main__":
    main()
