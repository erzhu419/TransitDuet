#!/usr/bin/env python3
"""Compare tail summaries with the frozen same-loss/same-step proper control."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_tail_value_stage184 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def candidate_ceiling(rows):
    result = {"role": "posthoc_outcome_oracle_diagnostic_not_available_to_controller", "by_root": {}}
    for root in (397, 401):
        group = [r for r in rows if r["root"] == root]
        selected = []
        for row in group:
            labels = {l["candidate"]: l for l in row["labels"]}
            costs = [labels[i]["actual_cost_delta"] if labels[i]["wait_delta_min"] <= 0 else float("inf")
                for i in range(6)]
            selected.append(labels[spec.paired.choice_from_cost(costs, row["original_choice"])])
        result["by_root"][str(root)] = {"states": len(group),
            "joint_improvable_states": sum(l["actual_cost_delta"] < 0 for l in selected),
            "oracle_mean_cost_delta": sum(l["actual_cost_delta"] for l in selected)/len(group),
            "oracle_mean_wait_delta_min": sum(l["wait_delta_min"] for l in selected)/len(group)}
    return result


def selected_wait_errors(validation):
    result = {}
    for split, data in validation.items():
        changed = [r for r in data["predictions"] if r["selected"] != r["original_choice"]]
        result[split] = {"changed_choices": len(changed),
            "false_feasible_choices": sum(r["predicted_wait_delta_min"][r["selected"]] <= 0
                and r["wait_delta_min"] > 0 for r in changed),
            "mean_selected_wait_residual_min": sum(r["wait_delta_min"]-r["predicted_wait_delta_min"][r["selected"]]
                for r in changed)/max(len(changed), 1)}
    return result


def summarize(cell, control):
    if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["training_mode"] == cell["model"]["representation"] == "tail_summary"
            and cell["model"]["objective"] == "proper_mse" and cell["model"]["parameter_count"] == 1635
            and cell["model"]["training_states"] == 120 and len(cell["fit_ledger"]) == len(control["fit_ledger"]) == 28
            and control["protocol"] == spec.source.EXPERIMENT_PROTOCOL and control["contract"] == spec.source.contract()
            and control["software_qualified"] and control["training_mode"] == "proper_mse"
            and cell["critic_updates"] == control["critic_updates"] == sum(cell["frozen_selected_steps"])
            and cell["actor_updates"] == cell["new_native_steps"] == cell["new_inner_updates"] == cell["new_control_updates"] == 0):
        raise ValueError("Changed tail-summary experiment or control budget")
    for fit, old in zip(cell["fit_ledger"], control["fit_ledger"]):
        if not (fit["parameter_count"] == 1635 and old["parameter_count"] == 1603
                and all(fit[k] == old[k] for k in ("training_groups", "training_states", "steps", "target_scale", "regret_cost_scale"))):
            raise ValueError("Tail comparison changed groups, targets, scales or steps")
    for policy in spec.value.POLICIES:
        v = cell["validation"][policy]
        gates = {"cost": all(r["mean_actual_cost_delta"] < 0 for s in v.values() for r in s["by_root"].values()),
            "wait": all(r["mean_wait_delta_min"] <= 0 for s in v.values() for r in s["by_root"].values())}
        if cell["gates"][policy] != gates:
            raise ValueError("Changed fixed primary cost/wait gate")
        for split, data in v.items():
            keys = lambda d: [(r["root"], r["scene_seed"], r["probe_index"]) for r in d["predictions"]]
            folds = lambda d: [(r["held_out"], r["training_groups"], r["training_states"]) for r in d["folds"]]
            old = control["validation"][policy][split]
            if len(data["predictions"]) != 120 or keys(data) != keys(old) or folds(data) != folds(old):
                raise ValueError("Tail comparison changed outer held-out states")
    eligible = all(cell["gates"]["wait_feasible"].values())
    if cell["third_scene_screen_eligible"] != eligible or cell["native_deployment_eligible"]:
        raise ValueError("Changed registered primary or premature deployment")
    fields = ("mean_actual_cost_delta", "mean_fleet_cost_delta", "mean_nonfleet_cost_delta", "mean_wait_delta_min")
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "critic_updates": cell["critic_updates"], "new_control_updates": 0, "new_native_steps": 0,
        "third_scene_screen_eligible": eligible, "native_deployment_eligible": False,
        "representation_diagnostics": cell["representation_diagnostics"],
        "selected_wait_errors": selected_wait_errors(cell["validation"]["wait_feasible"]),
        "variants": {f"{mode}:{p}": {"validation": c["validation"][p], "gates": c["gates"][p],
            "apparent_training": c["apparent_training"][p]}
            for mode, c in (("occupancy", control), ("tail_summary", cell)) for p in spec.value.POLICIES},
        "matched_effect": {p: {s: {k: cell["validation"][p][s]["overall"][k]-control["validation"][p][s]["overall"][k]
            for k in fields} for s in ("scene", "regime", "lower_root")} for p in spec.value.POLICIES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    result = summarize(json.loads((directory/"fit/result.json").read_text()), json.loads((directory/"inputs/control.json").read_text()))
    rows = spec.source.source.source.training_rows(*(json.loads((directory/"inputs"/f"{n}.json").read_text())
        for n in ("first", "labels", "second", "baselines")))
    result["candidate_ceiling"] = candidate_ceiling(rows)
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("contract", "variants")}, indent=2))


if __name__ == "__main__":
    main()
