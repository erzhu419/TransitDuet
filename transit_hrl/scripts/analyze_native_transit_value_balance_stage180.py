#!/usr/bin/env python3
"""Summarize matched expanded-bank value fits with the primary held fixed."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_balance_stage180 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells):
    if set(cells) != set(spec.NORMALIZATIONS):
        raise ValueError("Missing matched native value normalization")
    for normalization, cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["normalization"] == normalization
                and cell["model"]["normalization"] == normalization and cell["model"]["training_states"] == 120
                and len(cell["fit_ledger"]) == 28 and cell["critic_updates"] == 28*spec.value.STEPS
                and cell["actor_updates"] == cell["new_native_steps"] == 0
                and all(f["steps"] == spec.value.STEPS and f["parameter_count"] == 1586 for f in cell["fit_ledger"])):
            raise ValueError("Changed normalization experiment or matched fit budget")
        cost = all(v["mean_actual_cost_delta"] < 0 for s in cell["validation"].values() for v in s["by_root"].values())
        wait = all(v["mean_wait_delta_min"] <= 0 for s in cell["validation"].values() for v in s["by_root"].values())
        if (cell["cost_screen_passed"], cell["wait_screen_passed"], cell["third_scene_screen_eligible"],
                cell["native_deployment_eligible"]) != (cost, wait, normalization == spec.PRIMARY and cost and wait, False):
            raise ValueError("Changed registered primary gate")
    control, primary = cells["joint_native"], cells[spec.PRIMARY]
    for split in ("scene", "regime", "lower_root"):
        a, b = control["validation"][split], primary["validation"][split]
        keys = lambda s: [(r["root"], r["scene_seed"], r["probe_index"]) for r in s["predictions"]]
        folds = lambda s: [(f["held_out"], f["training_groups"], f["training_states"]) for f in s["folds"]]
        if len(a["predictions"]) != 120 or keys(a) != keys(b) or folds(a) != folds(b):
            raise ValueError("Normalization comparison changed held-out states or training groups")
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "critic_updates": sum(c["critic_updates"] for c in cells.values()), "actor_updates": 0, "new_native_steps": 0,
        "third_scene_screen_eligible": primary["third_scene_screen_eligible"], "native_deployment_eligible": False,
        "normalizations": {n: {k: c[k] for k in ("validation", "apparent_training", "cost_screen_passed", "wait_screen_passed")}
            for n, c in cells.items()},
        "matched_effect": {s: {k: primary["validation"][s]["overall"][k]-control["validation"][s]["overall"][k]
            for k in ("mean_actual_cost_delta", "mean_fleet_cost_delta", "mean_nonfleet_cost_delta", "mean_wait_delta_min")}
            for s in ("scene", "regime", "lower_root")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    result = summarize({n: json.loads((directory/"cells"/n/"result.json").read_text()) for n in spec.NORMALIZATIONS})
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "normalizations"}}, indent=2))


if __name__ == "__main__":
    main()
