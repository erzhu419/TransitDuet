#!/usr/bin/env python3
"""Compare proper cost-value fitting with the frozen same-step regret control."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_proper_value_stage183 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cell, control):
    if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["training_mode"] == cell["model"]["objective"] == "proper_mse"
            and cell["model"]["parameter_count"] == 1603 and cell["model"]["training_states"] == 120
            and len(cell["fit_ledger"]) == len(control["selection_ledger"]) == 28
            and control["protocol"] == spec.source.EXPERIMENT_PROTOCOL and control["contract"] == spec.source.contract()
            and control["software_qualified"] and control["training_mode"] == "scene_stop"
            and cell["critic_updates"] == control["refit_critic_updates"] == sum(cell["frozen_selected_steps"])
            and cell["actor_updates"] == cell["new_native_steps"] == cell["new_inner_updates"] == cell["new_control_updates"] == 0):
        raise ValueError("Changed proper-value objective experiment or control budget")
    for fit, old, record in zip(cell["fit_ledger"], control["fit_ledger"], control["selection_ledger"]):
        if not (fit["steps"] == old["steps"] == record["selected_steps"]
                and all(fit[k] == old[k] for k in ("training_groups", "training_states", "parameter_count", "target_scale", "regret_cost_scale"))):
            raise ValueError("Objective comparison changed fitting groups, scales, architecture or update count")
    for policy in spec.value.POLICIES:
        v = cell["validation"][policy]
        gates = {"cost": all(r["mean_actual_cost_delta"] < 0 for s in v.values() for r in s["by_root"].values()),
            "wait": all(r["mean_wait_delta_min"] <= 0 for s in v.values() for r in s["by_root"].values())}
        if cell["gates"][policy] != gates:
            raise ValueError("Changed fixed primary cost/wait screen")
        for split, data in v.items():
            keys = lambda d: [(r["root"], r["scene_seed"], r["probe_index"]) for r in d["predictions"]]
            folds = lambda d: [(r["held_out"], r["training_groups"], r["training_states"]) for r in d["folds"]]
            old = control["validation"][policy][split]
            if len(data["predictions"]) != 120 or keys(data) != keys(old) or folds(data) != folds(old):
                raise ValueError("Objective comparison changed outer held-out states")
    eligible = all(cell["gates"]["wait_feasible"].values())
    if cell["third_scene_screen_eligible"] != eligible or cell["native_deployment_eligible"]:
        raise ValueError("Changed registered primary or premature deployment")
    fields = ("mean_actual_cost_delta", "mean_fleet_cost_delta", "mean_nonfleet_cost_delta", "mean_wait_delta_min")
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "critic_updates": cell["critic_updates"], "new_control_updates": 0, "new_native_steps": 0,
        "third_scene_screen_eligible": eligible, "native_deployment_eligible": False,
        "variants": {f"{mode}:{p}": {"validation": c["validation"][p], "gates": c["gates"][p],
            "apparent_training": c["apparent_training"][p]}
            for mode, c in (("decision_regret", control), ("proper_mse", cell)) for p in spec.value.POLICIES},
        "matched_effect": {p: {s: {k: cell["validation"][p][s]["overall"][k]-control["validation"][p][s]["overall"][k]
            for k in fields} for s in ("scene", "regime", "lower_root")} for p in spec.value.POLICIES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    result = summarize(json.loads((directory/"fit/result.json").read_text()), json.loads((directory/"inputs/control.json").read_text()))
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("contract", "variants")}, indent=2))


if __name__ == "__main__":
    main()
