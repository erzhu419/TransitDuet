#!/usr/bin/env python3
"""Compare expanded grouped values and the matched original-state subset."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_coverage_value_stage188 as spec
from scripts.submit_native_transit_coverage_value_stage188_scheduleurm import result_path
from freq_hrl.experiments.pointmaze_root_response import write_json

FIELDS = ("mean_actual_cost_delta", "mean_fleet_cost_delta", "mean_nonfleet_cost_delta", "mean_wait_delta_min")


def state_key(row):
    return row["root"], row["scene_seed"], row["probe_index"]


def group_key(group, split):
    return {"scene": lambda: [group["root"], group["scene_seed"]], "regime": lambda: group["scenario"],
        "lower_root": lambda: group["root"]}[split]()


def qualify_fold(cell, mode, fold, registered):
    groups = registered["groups"]
    test = groups if fold["split"] == "full" else [g for g in groups if group_key(g, fold["split"]) == fold["held_out"]]
    train = groups if fold["split"] == "full" else [g for g in groups if group_key(g, fold["split"]) != fold["held_out"]]
    expected_groups = [[g["root"], g["scene_seed"]] for g in train]
    fit = cell["fit"]
    if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["mode"] == mode and cell["fold"] == fold
            and fit["training_groups"] == expected_groups and fit["training_states"] == 6*len(train)
            and fit["steps"] == cell["critic_updates"] == spec.STEPS
            and fit["parameter_count"] == spec.contract()["parameters"][mode]
            and fit["representation"] == mode and fit["objective"] == "proper_mse"
            and fit["initialization_seed"] == spec.floor.occupancy.SEED
            and cell["actor_updates"] == cell["new_native_steps"] == 0
            and set(cell["predictions"]) == set(spec.value.POLICIES)):
        raise ValueError("Changed coverage fold, training membership or frozen fitting budget")
    if mode == "timetable_sequence" and not all(fit[k] > 0 for k in
            ("encoder_parameter_update_l2", "sequence_projection_update_l2", "encoder_gradient_l2_max")):
        raise ValueError("Coverage sequence encoder received no learning gradient")
    expected_states = {(g["root"], g["scene_seed"], p) for g in test for p in spec.collection.PROBES}
    for policy, predictions in cell["predictions"].items():
        if len(predictions) != len(expected_states) or {state_key(r) for r in predictions} != expected_states:
            raise ValueError("Missing or repeated held-out preaction states")
        for row in predictions:
            heads, waits = np.asarray(row["predicted_heads"]), np.asarray(row["predicted_wait_delta_min"])
            absolute, original = np.asarray(row["absolute_fleet_cost"]), row["original_choice"]
            cost = heads.sum(axis=1)
            if policy == "wait_feasible": cost = np.where(waits <= 0, cost, np.inf)
            if not (row["selected"] == spec.paired.choice_from_cost(cost, original)
                    and np.all(heads[original] == 0) and waits[original] == 0
                    and np.all(absolute >= row["causal_fleet_floor"])
                    and np.allclose(absolute-absolute[original], heads[:, 0], rtol=0, atol=1e-12)
                    and np.isclose(row["actual_cost_delta"], row["fleet_cost_delta"]+row["nonfleet_cost_delta"], rtol=0, atol=1e-12)):
                raise ValueError("Changed physical units, fleet floor, identity or wait selector")
    by_policy = [{state_key(r): r for r in cell["predictions"][p]} for p in spec.value.POLICIES]
    for key in expected_states:
        if any(by_policy[0][key][k] != by_policy[1][key][k] for k in
                ("predicted_heads", "predicted_wait_delta_min", "absolute_fleet_cost")):
            raise ValueError("Selectors did not share the same fitted critic")


def validation_summary(rows):
    changed = [r for r in rows if r["selected"] != r["original_choice"]]
    return {"overall": spec.floor.summary(rows),
        "by_root": {str(root): spec.floor.summary([r for r in rows if r["root"] == root]) for root in spec.collection.ROOTS},
        "changed_choices": len(changed), "false_feasible_choices": sum(r["predicted_wait_delta_min"][r["selected"]] <= 0
            and r["wait_delta_min"] > 0 for r in changed),
        "mean_selected_wait_residual_min": sum(r["wait_delta_min"]-r["predicted_wait_delta_min"][r["selected"]]
            for r in changed)/max(len(changed), 1)}


def summarize(cells, registered, old_cells):
    folds = registered["folds"]
    required = {(m, f["split"], f["index"]) for m in spec.MODES for f in folds}
    if set(cells) != required:
        raise ValueError("Missing or mixed coverage value fits")
    for fold in folds:
        pair = [cells[m, fold["split"], fold["index"]] for m in spec.MODES]
        for mode, cell in zip(spec.MODES, pair): qualify_fold(cell, mode, fold, registered)
        if any(pair[0]["fit"][k] != pair[1]["fit"][k] for k in
                ("training_groups", "training_states", "steps", "center", "scale", "target_scale", "regret_cost_scale")):
            raise ValueError("Representation comparison changed groups, base or target scaling")
    old_keys = {(g["root"], g["scene_seed"], p) for g in registered["groups"] if g["origin"] == "existing"
        for p in spec.collection.PROBES}
    variants, coverage_effect = {}, {}
    for mode in spec.MODES:
        for policy in spec.value.POLICIES:
            validation, old_subset, effects = {}, {}, {}
            for split in spec.SPLITS:
                prediction = [r for f in folds if f["split"] == split
                    for r in cells[mode, split, f["index"]]["predictions"][policy]]
                subset = [r for r in prediction if state_key(r) in old_keys]
                old = old_cells[mode]["validation"][policy][split]
                if {state_key(r) for r in old["predictions"]} != {state_key(r) for r in subset}:
                    raise ValueError("Coverage comparison changed its original held-out states")
                validation[split], old_subset[split] = validation_summary(prediction), validation_summary(subset)
                effects[split] = {scope: {k: new[k]-reference[k] for k in FIELDS}
                    for scope, new, reference in [("overall", old_subset[split]["overall"], old["overall"]),
                        *[(str(root), old_subset[split]["by_root"][str(root)], old["by_root"][str(root)]) for root in spec.collection.ROOTS]]}
            gates = {"cost": all(v["mean_actual_cost_delta"] < 0 for s in validation.values() for v in s["by_root"].values()),
                "wait": all(v["mean_wait_delta_min"] <= 0 for s in validation.values() for v in s["by_root"].values())}
            full = cells[mode, "full", 0]
            variants[f"{mode}:{policy}"] = {"validation": validation, "old_state_subset": old_subset,
                "gates": gates, "apparent_training": validation_summary(full["predictions"][policy])}
            coverage_effect[f"{mode}:{policy}"] = effects
    sequence, scalar = (variants[f"{m}:wait_feasible"] for m in ("timetable_sequence", "tail_summary"))
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "fits": len(cells), "critic_updates": len(cells)*spec.STEPS, "actor_updates": 0, "new_native_steps": 0,
        "training_states": 600, "independent_scene_groups": 100, "paired_labels": 3600,
        "variants": variants, "same_old_state_coverage_effect": coverage_effect,
        "matched_representation_effect": {s: {k: sequence["validation"][s]["overall"][k]-scalar["validation"][s]["overall"][k]
            for k in FIELDS} for s in spec.SPLITS},
        "runtime": {m: {"fit_wall_s_sum": sum(c["fit_wall_s"] for (mode, _, _), c in cells.items() if mode == m),
            "evaluation_wall_s_sum": sum(c["evaluation_wall_s"] for (mode, _, _), c in cells.items() if mode == m)} for m in spec.MODES},
        "third_scene_screen_eligible": all(sequence["gates"].values()), "native_deployment_eligible": False,
        "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    cells = {(m, f["split"], f["index"]): json.loads(result_path(directory, m, f).read_text())
        for m in spec.MODES for f in registered["folds"]}
    old = ROOT/"results"/spec.OLD_RUN
    old_cells = {"tail_summary": json.loads((old/"inputs/control.json").read_text()),
        "timetable_sequence": json.loads((old/"fit/result.json").read_text())}
    summary = summarize(cells, registered, old_cells)
    write_json(directory/"summary.json", summary)
    print(json.dumps({"fits": summary["fits"], "critic_updates": summary["critic_updates"],
        "gates": {k: v["gates"] for k, v in summary["variants"].items()},
        "third_scene_screen_eligible": summary["third_scene_screen_eligible"]}, indent=2))


if __name__ == "__main__":
    main()
