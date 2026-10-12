#!/usr/bin/env python3
"""Compare cached single-label and mean-target values on the same test states."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_expected_value_stage192 as spec
from scripts.analyze_native_transit_coverage_value_stage188 import group_key, validation_summary
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json


def qualify(cell, mode, fold, registered):
    groups = [g for g in registered["groups"] if group_key(g, fold["split"]) != fold["held_out"]]
    expected = [[g["root"], g["scene_seed"]] for g in groups]
    test = {spec.state_key(r) for r in registered["mean_states"] if group_key(r, fold["split"]) == fold["held_out"]}
    if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["mode"] == mode and cell["fold"] == fold
            and cell["critic_updates"] == spec.STEPS and cell["actor_updates"] == cell["new_native_steps"] == 0
            and cell["training_mean_states"] == 30-len(test) and cell["evaluation_mean_states"] == len(test)
            and set(cell["fits"]) == set(cell["predictions"]) == set(spec.VARIANTS)):
        raise ValueError("Changed expected-value coverage, fold or update budget")
    for fit in cell["fits"].values():
        if not (fit["training_groups"] == expected and fit["training_states"] == 6*len(groups)
                and fit["steps"] == spec.STEPS and fit["parameter_count"] == spec.source.contract()["parameters"][mode]
                and fit["initialization_seed"] == spec.source.floor.occupancy.SEED
                and fit["objective"] == "proper_mse" and fit["representation"] == mode):
            raise ValueError("Expected-label fit leaked test groups or changed its matched method")
        if mode == "timetable_sequence" and not all(fit[k] > 0 for k in
                ("encoder_parameter_update_l2", "sequence_projection_update_l2", "encoder_gradient_l2_max")):
            raise ValueError("Expected-label sequence encoder did not learn")
    fits = list(cell["fits"].values())
    if any(fits[0][k] != fits[1][k] for k in ("center", "scale")):
        raise ValueError("Label replacement changed causal input normalization")
    for by_label in cell["predictions"].values():
        if set(by_label) != {"original", "mean8"}:
            raise ValueError("Missing same-state original/expected evaluation")
        for label, by_policy in by_label.items():
            if set(by_policy) != set(spec.source.value.POLICIES):
                raise ValueError("Missing shared-fit selector")
            for policy, predictions in by_policy.items():
                if len(predictions) != len(test) or {spec.state_key(r) for r in predictions} != test:
                    raise ValueError("Expected-value comparison changed its test states")
                for row in predictions:
                    heads, waits, original = np.asarray(row["predicted_heads"]), np.asarray(row["predicted_wait_delta_min"]), row["original_choice"]
                    costs = heads.sum(axis=1)
                    if policy == "wait_feasible": costs = np.where(waits <= 0, costs, np.inf)
                    if not (row["selected"] == spec.source.paired.choice_from_cost(costs, original)
                            and np.all(heads[original] == 0) and waits[original] == 0
                            and np.all(np.asarray(row["absolute_fleet_cost"]) >= row["causal_fleet_floor"])
                            and np.isclose(row["actual_cost_delta"], row["fleet_cost_delta"]+row["nonfleet_cost_delta"], rtol=0, atol=1e-12)):
                        raise ValueError("Changed identity, physical floor or expected-value selector")
        for policy in spec.source.value.POLICIES:
            paired = [{spec.state_key(r): r for r in by_label[label][policy]} for label in ("original", "mean8")]
            if any(paired[0][k][f] != paired[1][k][f] for k in test for f in
                    ("selected", "predicted_heads", "predicted_wait_delta_min", "absolute_fleet_cost")):
                raise ValueError("Evaluation labels changed fitted predictions or action choices")


def summarize(cells, registered, prepared):
    if set(cells) != {(m, f["split"], f["index"]) for m in spec.MODES for f in registered["folds"]}:
        raise ValueError("Missing or mixed expected-label folds")
    for mode in spec.MODES:
        for fold in registered["folds"]:
            qualify(cells[mode, fold["split"], fold["index"]], mode, fold, registered)
    variants = {}
    for mode in spec.MODES:
        for variant in spec.VARIANTS:
            for policy in spec.source.value.POLICIES:
                evaluations = {}
                for label in ("original", "mean8"):
                    evaluations[label] = {}
                    for split in spec.SPLITS:
                        rows = [r for fold in registered["folds"] if fold["split"] == split
                            for r in cells[mode, split, fold["index"]]["predictions"][variant][label][policy]]
                        if len(rows) != 30 or len({spec.state_key(r) for r in rows}) != 30:
                            raise ValueError("Expected-value aggregate lost a test state")
                        evaluations[label][split] = validation_summary(rows)
                gates = {"cost": all(v["mean_actual_cost_delta"] < 0 for s in evaluations["mean8"].values() for v in s["by_root"].values()),
                    "wait": all(v["mean_wait_delta_min"] <= 0 for s in evaluations["mean8"].values() for v in s["by_root"].values())}
                variants[f"{mode}:{variant}:{policy}"] = {"evaluation": evaluations, "expected_label_gates": gates}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "new_fits": len(cells), "reused_fits": len(cells), "critic_updates": len(cells)*spec.STEPS,
        "actor_updates": 0, "new_native_steps": 0, "evaluation_states": 30, "evaluation_prefix_scene_groups": 10,
        "variance_decomposition": prepared["variance_decomposition"], "variants": variants,
        "fit_wall_s_sum": {m: sum(c["fit_wall_s"] for (mode, _, _), c in cells.items() if mode == m) for m in spec.MODES},
        "expected_target_development_gate": all(variants[spec.contract()["primary"]]["expected_label_gates"].values()),
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    prepared = json.loads((directory/"prepare/result.json").read_text())
    cells = {(m, f["split"], f["index"]): json.loads((raw_directory(spec.result_path(args.run_name, m, f))/"result.json").read_text())
        for m in spec.MODES for f in registered["folds"]}
    result = summarize(cells, registered, prepared)
    write_json(args.output, result)
    print(json.dumps({"new_fits": result["new_fits"], "gates": {k: v["expected_label_gates"] for k, v in result["variants"].items()}}, indent=2))
    print("NATIVE_EXPECTED_VALUE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
