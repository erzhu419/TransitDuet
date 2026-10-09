#!/usr/bin/env python3
"""Compare frozen physical goal outcomes and native critic rankings."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_frontier_stage149 as spec
from scripts.run_native_transit_diagnostics_stage147 import check_episode
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, sources, *, preflight):
    contract = spec.contract(preflight)
    pairs = [(spec.METHODS[0], spec.ROOTS[0])] if preflight else [
        (method, root) for method in spec.METHODS for root in spec.ROOTS]
    panels = []
    for method, root in pairs:
        cell = cells[method, root]
        if not (cell["software_qualified"] and cell["baseline_reproduced"] and cell["training_updates"] == 0
                and cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == contract
                and cell["method"] == method and cell["seed"] == root):
            raise ValueError("Incorrect native frontier cell")
        rows = {(r["scenario"], r["condition"]): r for r in cell["evaluation"]}
        required = {(scenario, condition) for scenario in contract["scenarios"] for condition in contract["new_conditions"]}
        if rows.keys() != required or len(rows) != len(cell["evaluation"]):
            raise ValueError("Incomplete or duplicate native frontier scenes")
        if cell["native_steps"] != len(rows) * 61380:
            raise ValueError("Unexpected frozen frontier tick budget")
        source = sources[method, root]
        original = {(r["scenario"], r["condition"]): r for r in source["evaluation"]}
        for scenario in contract["scenarios"]:
            baseline = rows[scenario, "baseline"]
            for condition in contract["new_conditions"]:
                row = rows[scenario, condition]
                check_episode(row, original[scenario, "baseline"], baseline=condition == "baseline")
                if row["scene_seed"] != spec.source_spec.scene_seed(root, scenario):
                    raise ValueError("Changed registered frontier scene seed")
            if set(baseline["critic_curve"]) != {str(goal) for goal in spec.GOALS}:
                raise ValueError("Incomplete native critic grid")
            if preflight:
                continue
            # Reuse only the already evaluated endpoints and neutral goal.
            goals = {goal: original[scenario, condition] for goal, condition in spec.CACHED_GOALS.items()}
            goals.update({goal: rows[scenario, goal] for goal in spec.NEW_GOALS})
            zero = goals[0]
            best_goal = min(spec.GOALS, key=lambda goal: (goals[goal]["service_cost_restricted"], abs(goal)))
            critic_goal = max(spec.GOALS, key=lambda goal: baseline["critic_curve"][str(goal)]["lcb_mean"])
            panels.append({"method": method, "root": root, "scenario": scenario,
                "oracle_grid_goal_s": best_goal, "mean_state_critic_lcb_goal_s": critic_goal,
                "oracle_cost_gain_against_zero": zero["service_cost_restricted"] - goals[best_goal]["service_cost_restricted"],
                "goals": [{"goal_s": goal, **{metric: goals[goal][metric] - zero[metric] for metric in
                    (*spec.source_spec.routing.METRICS, "lower_action_mean")}} for goal in spec.GOALS],
                "critic_curve": baseline["critic_curve"], "credit_ledger": baseline["credit_ledger"]})
    result = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract,
        "software_qualified": True, "preflight": preflight,
        "native_steps": sum(cells[pair]["native_steps"] for pair in pairs),
        "training_updates": 0, "stage": "software_preflight" if preflight else "post_result_opportunity_diagnosis"}
    if not preflight:
        result["panels"] = panels
        result["root_opportunity"] = [{"method": method, "root": root,
            "mean_oracle_cost_gain_against_zero": float(np.mean([
                panel["oracle_cost_gain_against_zero"] for panel in panels if (panel["method"], panel["root"]) == (method, root)])),
            "best_fixed_grid_goal_s": min(spec.GOALS, key=lambda goal: (
                np.mean([next(row["service_cost_restricted"] for row in panel["goals"] if row["goal_s"] == goal)
                    for panel in panels if (panel["method"], panel["root"]) == (method, root)]), abs(goal)))}
            for method, root in pairs]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    pairs = [(spec.METHODS[0], spec.ROOTS[0])] if args.preflight else [
        (method, root) for method in spec.METHODS for root in spec.ROOTS]
    cells = {pair: json.loads((directory / "cells" / pair[0] / f"seed_{pair[1]}" / "result.json").read_text()) for pair in pairs}
    sources = {pair: spec.load_source(*pair)[1] for pair in pairs}
    result = summarize(cells, sources, preflight=args.preflight)
    write_json(directory / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "panels"}, indent=2))


if __name__ == "__main__":
    main()
