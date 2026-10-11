#!/usr/bin/env python3
"""Qualify the fixed absolute-fleet/wait primary and its matched ablations."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_floor_value_stage181 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells):
    if set(cells) != set(spec.value.METHODS):
        raise ValueError("Missing matched absolute fleet fits")
    for method, cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["method"] == cell["model"]["method"] == method
                and cell["model"]["training_states"] == 120 and len(cell["fit_ledger"]) == 28
                and cell["critic_updates"] == 28*spec.occupancy.STEPS
                and cell["actor_updates"] == cell["new_native_steps"] == 0
                and all(f["steps"] == spec.occupancy.STEPS and f["parameter_count"] == 1603 for f in cell["fit_ledger"])
                and set(cell["validation"]) == set(spec.value.POLICIES)):
            raise ValueError("Changed anchored fleet experiment or fit budget")
        for policy, validation in cell["validation"].items():
            cost = all(v["mean_actual_cost_delta"] < 0 for s in validation.values() for v in s["by_root"].values())
            wait = all(v["mean_wait_delta_min"] <= 0 for s in validation.values() for v in s["by_root"].values())
            if cell["gates"][policy] != {"cost": cost, "wait": wait}:
                raise ValueError("Changed registered cost/wait gate")
            for split, data in validation.items():
                if split not in ("scene", "regime", "lower_root") or len(data["predictions"]) != 120:
                    raise ValueError("Changed grouped held-out states")
                for r in data["predictions"]:
                    original = r["original_choice"]
                    heads, waits = np.asarray(r["predicted_heads"]), np.asarray(r["predicted_wait_delta_min"])
                    absolute = np.asarray(r["absolute_fleet_cost"])
                    cost = heads.sum(axis=1)
                    if policy == "wait_feasible": cost = np.where(waits <= 0, cost, np.inf)
                    if (r["selected"] != spec.paired.choice_from_cost(cost, original)
                            or np.any(heads[original] != 0) or waits[original] != 0
                            or not np.allclose(absolute-absolute[original], heads[:, 0], rtol=0, atol=1e-12)
                            or (method == "floor_absolute" and np.any(absolute < r["causal_fleet_floor"]))):
                        raise ValueError("Changed physical fleet floor, identity or learned wait selection")
        eligible = method == "floor_absolute" and all(cell["gates"]["wait_feasible"].values())
        if cell["third_scene_screen_eligible"] != eligible or cell["native_deployment_eligible"]:
            raise ValueError("Control swapped into the primary or premature deployment")
    variants = {f"{m}:{p}": {"validation": cells[m]["validation"][p], "gates": cells[m]["gates"][p],
        "apparent_training": cells[m]["apparent_training"][p]} for m in spec.value.METHODS for p in spec.value.POLICIES}
    for split in ("scene", "regime", "lower_root"):
        reference = variants[spec.PRIMARY]["validation"][split]
        keys = lambda v: [(r["root"], r["scene_seed"], r["probe_index"]) for r in v["predictions"]]
        folds = lambda v: [(f["held_out"], f["training_groups"], f["training_states"]) for f in v["folds"]]
        for variant in variants.values():
            v = variant["validation"][split]
            if keys(v) != keys(reference) or folds(v) != folds(reference):
                raise ValueError("Ablation changed held-out scenes or training groups")
        for cell in cells.values():
            a, b = (cell["validation"][p][split]["predictions"] for p in spec.value.POLICIES)
            if any(x[k] != y[k] for x, y in zip(a, b) for k in
                    ("predicted_heads", "predicted_wait_delta_min", "absolute_fleet_cost")):
                raise ValueError("Decision ablation changed shared fitted values")
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "critic_updates": sum(c["critic_updates"] for c in cells.values()), "actor_updates": 0, "new_native_steps": 0,
        "third_scene_screen_eligible": cells["floor_absolute"]["third_scene_screen_eligible"],
        "native_deployment_eligible": False, "variants": variants}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    result = summarize({m: json.loads((directory/"cells"/m/"result.json").read_text()) for m in spec.value.METHODS})
    write_json(directory/"summary.json", result)
    print(json.dumps({"software_qualified": True, "critic_updates": result["critic_updates"],
        "third_scene_screen_eligible": result["third_scene_screen_eligible"],
        "variants": {k: v["gates"] for k, v in result["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()
