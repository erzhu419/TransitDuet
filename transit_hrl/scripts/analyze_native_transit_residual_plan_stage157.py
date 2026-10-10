#!/usr/bin/env python3
"""Require own forecast/constant controls before attributing learned plan value."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_residual_plan_stage157 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def validate_credit(credit, row, clock):
    if not (credit["decisions"] > 0 and credit["terminal_transitions"] == 1
            and credit["duration_s"] == clock
            and np.isclose(credit["final_cost"], row["service_cost_restricted"], rtol=0, atol=1e-6)
            and np.isclose(credit["reward_sum"], 100 * (credit["initial_cost"] - credit["final_cost"]), rtol=1e-10, atol=1e-8)):
        raise ValueError("Upper credit no longer represents the common terminal cost")


def summarize(cells):
    contract = spec.contract()
    if set(cells) != set(spec.ROOTS):
        raise ValueError("Incomplete learned residual roots")
    means, panels = {}, []
    metrics = spec.source.source_spec.authority.routing.METRICS
    expected = (contract["train_episodes"] - contract["warmup"]) * contract["updates_per_episode"]
    for root, cell in cells.items():
        training, short = cell["training"], cell["preflight_learning"]
        required = {(c, s, scene) for c in spec.CONDITIONS for s in contract["scenarios"]
            for scene in spec.source.source_spec.scene_seeds(spec.LOWER_ROOT[root], s, preflight=False)}
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == contract
                and cell["seed"] == root and cell["lower_root"] == spec.LOWER_ROOT[root]
                and cell["software_qualified"] and cell["native_training_updates"] == 0
                and cell["native_steps"] == (contract["train_episodes"] + len(required)) * 61380
                and cell["worker_preflight_native_steps"] == 3 * 61380 + 2 * 5400
                and {r["condition"] for r in cell["qualification"]} == {"source_baseline", "nominal_plan", "causal_forecast"}
                and len(cell["qualification"]) == 3 and all(r["reproduced"] for r in cell["qualification"])
                and training["updates"] == expected and training["actor_change_max_abs"] > 0
                and short["updates"] == 2 and short["actor_change_max_abs"] > 0):
            raise ValueError("Unqualified upper-only native training")
        seeds = training["training_scene_seeds"]
        if seeds != [700000000 + root * 1000 + ep for ep in range(contract["train_episodes"])]:
            raise ValueError("Changed/disjoint training scene contract")
        if set(seeds).intersection(scene for _, _, scene in required):
            raise ValueError("Training scenes overlap frozen evaluation")
        for run, clock in ((training, 61380), (short, 5400)):
            if not all(np.isfinite(v) for v in run["learning_mean"].values()):
                raise ValueError("Nonfinite upper learning")
            for r in run["training_curve"]:
                validate_credit(r["credit"], r, clock)
        constant = np.asarray(training["constant_action"])
        if constant.shape != (2,) or not np.all(np.isfinite(constant)) or np.any(np.abs(constant) > 1):
            raise ValueError("Invalid pre-evaluation training-state constant control")
        rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in cell["evaluation"]}
        if set(rows) != required or len(rows) != len(cell["evaluation"]):
            raise ValueError("Incomplete or duplicate frozen residual scenes")
        for (condition, scenario, scene), r in rows.items():
            ref = rows["forecast", scenario, scene]
            if (any(r[k] != ref[k] for k in ("simulation_end_time_s", "N_fleet", "passengers_generated"))
                    or r["simulation_end_time_s"] != 61380 or r["N_fleet"] != 12
                    or not all(np.isfinite(r[k]) for k in metrics)
                    or r["execution"]["endpoint_error_max_s"] != 0):
                raise ValueError("Unpaired physical residual-plan outcomes")
            if condition in {"nominal", "forecast"} and not r["source_control_reproduced"]:
                raise ValueError("Source controls failed exact reproduction")
            if condition != "nominal":
                validate_credit(r["credit"], r, 61380)
            if condition == "constant_residual" and not np.allclose(r["residual_action_mean"], constant, rtol=0, atol=1e-6):
                raise ValueError("Constant control changed during evaluation")
        for condition in spec.CONDITIONS:
            for scenario in contract["scenarios"]:
                values = {k: float(np.mean([r[k] for (c, s, _), r in rows.items()
                    if c == condition and s == scenario])) for k in metrics}
                panels.append({"root": root, "condition": condition, "scenario": scenario, **values})
            means[root, condition] = {k: float(np.mean([r[k] for r in panels
                if r["root"] == root and r["condition"] == condition])) for k in metrics}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract, "software_qualified": True,
        "stage": "two_root_learned_upper_frozen_lower_not_full_HRL_confirmation",
        "native_steps": sum(c["native_steps"] for c in cells.values()),
        "learned_minus_control": {c: [{"root": root, **{k: means[root, "learned"][k] - means[root, c][k]
            for k in metrics}} for root in spec.ROOTS] for c in spec.CONDITIONS if c != "learned"},
        "regime_root_means": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {root: json.loads((directory / "cells" / spec.METHOD / f"seed_{root}/result.json").read_text()) for root in spec.ROOTS}
    result = summarize(cells)
    write_json(directory / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_means"}}, indent=2))


if __name__ == "__main__":
    main()
