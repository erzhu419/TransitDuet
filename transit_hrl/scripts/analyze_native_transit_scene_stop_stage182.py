#!/usr/bin/env python3
"""Compare nested scene-selected fits with the completed fixed-step control."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_scene_stop_stage182 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cell, control):
    if not (control["protocol"] == spec.source.EXPERIMENT_PROTOCOL and control["contract"] == spec.source.contract()
            and control["software_qualified"] and control["method"] == "floor_absolute"
            and control["critic_updates"] == 42000 and control["model"]["parameter_count"] == 1603
            and cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["method"] == "floor_absolute" and cell["training_mode"] == "scene_stop"
            and len(cell["fit_ledger"]) == len(cell["selection_ledger"]) == 28
            and cell["model"]["training_states"] == 120 and cell["model"]["parameter_count"] == 1603
            and cell["inner_critic_updates"] == 42000
            and cell["refit_critic_updates"] == sum(f["steps"] for f in cell["fit_ledger"])
            and cell["critic_updates"] == cell["inner_critic_updates"]+cell["refit_critic_updates"]
            and cell["actor_updates"] == cell["new_native_steps"] == 0):
        raise ValueError("Changed nested scene experiment or source control")
    rows = [r for r in cell["validation"]["cost_only"]["scene"]["predictions"]]
    scenarios = {(r["root"], r["scene_seed"]): r["scenario"] for r in rows}
    for record, fit in zip(cell["selection_ledger"], cell["fit_ledger"]):
        groups = set(map(tuple, record["outer_training_groups"]))
        ordered = sorted(groups, key=lambda g: (scenarios[g], *g))
        held = set(ordered[::5])
        checkpoints = record["checkpoint_ledger"]
        if not (set(map(tuple, record["inner_validation_groups"])) == held
                and set(map(tuple, record["inner_training_groups"])) == groups-held
                and record["inner_validation_states"] == 6*len(held)
                and record["inner_training_states"] == 6*len(groups-held)
                and record["outer_training_states"] == fit["training_states"] == 6*len(groups)
                and record["inner_steps"] == spec.occupancy.STEPS
                and [c["step"] for c in checkpoints] == list(spec.value.CHECKPOINT_STEPS)
                and record["selected_steps"] == fit["steps"] == spec.selected_checkpoint(checkpoints)
                and fit["parameter_count"] == 1603 and fit["training_groups"] == record["outer_training_groups"]):
            raise ValueError("Checkpoint selection crossed scene roles or changed steps/budgets")
    if cell["model"]["checkpoint_selection"] != cell["selection_ledger"][-1]:
        raise ValueError("Full model lost its nested checkpoint lineage")
    for policy in spec.value.POLICIES:
        validation = cell["validation"][policy]
        gate = {"cost": all(v["mean_actual_cost_delta"] < 0 for s in validation.values() for v in s["by_root"].values()),
            "wait": all(v["mean_wait_delta_min"] <= 0 for s in validation.values() for v in s["by_root"].values())}
        if cell["gates"][policy] != gate:
            raise ValueError("Changed fixed primary cost/wait gate")
        for split, data in validation.items():
            other = control["validation"][policy][split]
            keys = lambda v: [(r["root"], r["scene_seed"], r["probe_index"]) for r in v["predictions"]]
            folds = lambda v: [(f["held_out"], f["training_groups"], f["training_states"]) for f in v["folds"]]
            if len(data["predictions"]) != 120 or keys(data) != keys(other) or folds(data) != folds(other):
                raise ValueError("Fixed-step comparison changed outer test states or fitting groups")
            for r in data["predictions"]:
                heads, wait, absolute = (np.asarray(r[k]) for k in ("predicted_heads", "predicted_wait_delta_min", "absolute_fleet_cost"))
                costs, original = heads.sum(axis=1), r["original_choice"]
                if policy == "wait_feasible": costs = np.where(wait <= 0, costs, np.inf)
                if (np.any(absolute < r["causal_fleet_floor"]) or np.any(heads[original] != 0) or wait[original] != 0
                        or r["selected"] != spec.paired.choice_from_cost(costs, original)):
                    raise ValueError("Changed physical floor, identity or wait selector")
    eligible = all(cell["gates"]["wait_feasible"].values())
    if cell["third_scene_screen_eligible"] != eligible or cell["native_deployment_eligible"]:
        raise ValueError("Changed fixed primary or premature native deployment")
    fields = ("mean_actual_cost_delta", "mean_fleet_cost_delta", "mean_nonfleet_cost_delta", "mean_wait_delta_min")
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "critic_updates": cell["critic_updates"], "new_control_updates": 0, "actor_updates": 0, "new_native_steps": 0,
        "third_scene_screen_eligible": eligible, "native_deployment_eligible": False,
        "selected_steps": [r["selected_steps"] for r in cell["selection_ledger"]],
        "variants": {f"{mode}:{p}": {"validation": c["validation"][p], "gates": c["gates"][p],
            "apparent_training": c["apparent_training"][p]}
            for mode, c in (("fixed1500", control), ("scene_stop", cell)) for p in spec.value.POLICIES},
        "matched_effect": {p: {s: {k: cell["validation"][p][s]["overall"][k]-control["validation"][p][s]["overall"][k]
            for k in fields} for s in ("scene", "regime", "lower_root")} for p in spec.value.POLICIES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    result = summarize(json.loads((directory/"fit/result.json").read_text()),
        json.loads((directory/"inputs/control.json").read_text()))
    write_json(directory/"summary.json", result)
    print(json.dumps({"software_qualified": True, "critic_updates": result["critic_updates"],
        "third_scene_screen_eligible": result["third_scene_screen_eligible"],
        "selected_steps": result["selected_steps"], "matched_effect": result["matched_effect"]}, indent=2))


if __name__ == "__main__":
    main()
