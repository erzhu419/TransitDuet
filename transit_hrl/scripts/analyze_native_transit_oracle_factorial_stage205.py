#!/usr/bin/env python3
"""Separate wait filtering, cost ranking and candidate headroom across future halves."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_decision_wait_stage204 as source
from scripts.analyze_native_transit_mean_signal_stage196 import load_panels
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_oracle_factorial_stage205_v1"
SOURCE_RUN = "native_transit_decision_wait_stage204_development_20261013_r1"
PARTITIONS = {"first_last": ((0, 1, 2, 3), (4, 5, 6, 7)),
    "even_odd": ((0, 2, 4, 6), (1, 3, 5, 7))}
SELECTORS = ("LL", "LO", "OL", "OO", "unconstrained")
CONTRASTS = {"wait_repair": ("LO", "LL"), "cost_repair": ("OL", "LL"),
    "joint_repair": ("OO", "LL"), "cost_gap_given_oracle_wait": ("LO", "OO"),
    "wait_gap_given_oracle_cost": ("OL", "OO")}


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "draw_source": "native_transit_mean_coverage_stage195_development_20261012_r1",
        "data": "same240_states_40_prefix_groups_8_conditional_future_draws_47_frozen_outer_folds",
        "partitions": {k: [list(a), list(b)] for k, (a, b) in PARTITIONS.items()},
        "primary_partition": "first_last", "sensitivity_partition": "even_odd",
        "primary_variant": source.contract()["primary"], "variants": list(source.VARIANTS),
        "selectors": {"LL": "learned_cost_learned_wait", "LO": "learned_cost_selection_half_wait",
            "OL": "selection_half_cost_learned_wait", "OO": "selection_half_cost_selection_half_wait",
            "unconstrained": "selection_half_cost_no_wait_filter"},
        "evaluation": "both_directions_other_half_only_mean8_hindsight_separate_identity_ties_retained",
        "errors": "wait_signs_and_feasible_cost_pair_order_repeated_in_both_halves_not_CI",
        "gate": "negative_cross_cost_nonpositive_cross_wait_both_roots_all_three_splits",
        "budget": {"new_fits": 0, "critic_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "existing_draws_and_frozen_predictions_server_only_small_summary_sync",
        "boundary": "conditional_future_diagnostic_not_independent_prefix_seeds_or_learned_HRL_proof"}


def choices(cost, wait, truth, original):
    choose = source.paired.choice_from_cost
    return {"LL": choose(np.where(wait <= 0, cost, np.inf), original),
        "LO": choose(np.where(truth[:, 1] <= 0, cost, np.inf), original),
        "OL": choose(np.where(wait <= 0, truth[:, 0], np.inf), original),
        "OO": choose(np.where(truth[:, 1] <= 0, truth[:, 0], np.inf), original),
        "unconstrained": choose(truth[:, 0], original)}


def panel(values, predicted_cost, predicted_wait, original, partition):
    truth = values[..., [0, 3]]
    halves = [truth[list(indices)].mean(axis=0) for indices in PARTITIONS[partition]]
    actions = [choices(predicted_cost, predicted_wait, h, original) for h in halves]
    hindsight = choices(predicted_cost, predicted_wait, truth.mean(axis=0), original)
    selectors = {}
    for selector in SELECTORS:
        selected = [a[selector] for a in actions]
        same = [halves[i][selected[i]] for i in range(2)]
        cross = [halves[1-i][selected[i]] for i in range(2)]
        full = truth.mean(axis=0)[hindsight[selector]]
        selectors[selector] = {"selected": selected,
            "selection_cost": float(np.mean([x[0] for x in same])),
            "selection_wait_min": float(np.mean([x[1] for x in same])),
            "cross_cost": float(np.mean([x[0] for x in cross])),
            "cross_wait_min": float(np.mean([x[1] for x in cross])),
            "direction_cost": [float(x[0]) for x in cross],
            "direction_wait_min": [float(x[1]) for x in cross],
            "half_agreement": int(selected[0] == selected[1]),
            "directional_changes": sum(s != original for s in selected),
            "hindsight_cost": float(full[0]), "hindsight_wait_min": float(full[1])}
    a, b = halves
    nonidentity = np.arange(6) != original
    safe = (a[:, 1] <= 0) & (b[:, 1] <= 0)
    unsafe = (a[:, 1] > 0) & (b[:, 1] > 0)
    left, right = np.triu_indices(6, 1)
    ga, gb, gp = a[left, 0]-a[right, 0], b[left, 0]-b[right, 0], predicted_cost[left]-predicted_cost[right]
    stable_pairs = safe[left] & safe[right] & (abs(ga) > 1e-12) & (abs(gb) > 1e-12) & (ga*gb > 0)
    selected = actions[0]["LL"]
    errors = {"stable_false_safe_candidates": int(np.sum(nonidentity & (predicted_wait <= 0) & unsafe)),
        "stable_false_unsafe_candidates": int(np.sum(nonidentity & (predicted_wait > 0) & safe)),
        "wait_half_disagreement_candidates": int(np.sum(nonidentity & ((a[:, 1] > 0) != (b[:, 1] > 0)))),
        "learned_selected_both_halves_wait_harm": int(unsafe[selected]),
        "learned_selected_wait_half_disagreement": int((a[selected, 1] > 0) != (b[selected, 1] > 0)),
        "stable_feasible_cost_pairs": int(np.sum(stable_pairs)),
        "stable_cost_inversions": int(np.sum(stable_pairs & (ga*gp < 0))),
        "stable_cost_prediction_ties": int(np.sum(stable_pairs & (gp == 0)))}
    mean = truth.mean(axis=0)
    joint = lambda h: (h[:, 0] < 0) & (h[:, 1] <= 0)
    opportunities = {"mean8_cost_opportunity_states": int(np.any(mean[:, 0] < 0)),
        "mean8_joint_opportunity_states": int(np.any(joint(mean))),
        "both_halves_same_joint_action_states": int(np.any(joint(a) & joint(b))),
        "mean8_cost_only_conflict_states": int(np.any(mean[:, 0] < 0) and not np.any(joint(mean)))}
    return {"selectors": selectors, "errors": errors, "opportunities": opportunities}


def summarize(rows):
    result = {"states": len(rows), "prefix_groups": len({(r["root"], r["scene_seed"]) for r in rows}),
        "directional_evaluations": 2*len(rows), "selectors": {}}
    metrics = ("selection_cost", "selection_wait_min", "cross_cost", "cross_wait_min", "hindsight_cost", "hindsight_wait_min")
    for selector in SELECTORS:
        values = [r["selectors"][selector] for r in rows]
        result["selectors"][selector] = {**{"mean_"+m: float(np.mean([v[m] for v in values])) for m in metrics},
            **{m: sum(v[m] for v in values) for m in ("half_agreement", "directional_changes")},
            "directions": {name: {"mean_cost": float(np.mean([v["direction_cost"][i] for v in values])),
                "mean_wait_min": float(np.mean([v["direction_wait_min"][i] for v in values]))}
                for i, name in enumerate(("A_to_B", "B_to_A"))}}
    for group in ("errors", "opportunities"):
        result[group] = {key: sum(r[group][key] for r in rows) for key in rows[0][group]}
    result["contrasts"] = {name: {m: result["selectors"][a]["mean_"+m]-result["selectors"][b]["mean_"+m]
        for m in ("cross_cost", "cross_wait_min")} for name, (a, b) in CONTRASTS.items()}
    result["contrasts"]["interaction"] = {m: sum(sign*result["selectors"][s]["mean_"+m]
        for s, sign in (("OO", 1), ("LO", -1), ("OL", -1), ("LL", 1))) for m in ("cross_cost", "cross_wait_min")}
    return result


def scopes(rows, detailed):
    result = {"overall": summarize(rows), "by_root": {str(root): summarize([r for r in rows if r["root"] == root])
        for root in source.mean.ROOTS}}
    if detailed:
        result["by_regime"] = {s: summarize([r for r in rows if r["scenario"] == s]) for s in source.mean.SCENARIOS}
    return result


def gates(validation):
    roots = [v for split in validation.values() for v in split["by_root"].values()]
    return {s: {"cost": all(r["selectors"][s]["mean_cross_cost"] < 0 for r in roots),
        "wait": all(r["selectors"][s]["mean_cross_wait_min"] <= 0 for r in roots),
        "each_direction_cost": all(d["mean_cost"] < 0 for r in roots for d in r["selectors"][s]["directions"].values()),
        "each_direction_wait": all(d["mean_wait_min"] <= 0 for r in roots for d in r["selectors"][s]["directions"].values())}
        for s in SELECTORS}


def validate_panel_prediction(p, prediction):
    original, values = p["original_choice"], p["values"]
    if prediction["original_choice"] != original or np.any(values[:, original] != 0):
        raise ValueError("Changed original action or its exact paired-zero draw labels")
    cost = np.asarray(prediction["predicted_heads"]).sum(axis=1)
    wait = np.asarray(prediction["predicted_wait_delta_min"])
    selected = choices(cost, wait, values[..., [0, 3]].mean(axis=0), original)["LL"]
    if selected != prediction["selected"]:
        raise ValueError("Frozen learned selector changed")
    observed = values[:, selected, [0, 3]].mean(axis=0)
    if not np.allclose(observed, [prediction["actual_cost_delta"], prediction["wait_delta_min"]], atol=1e-12, rtol=0):
        raise ValueError("Future panels do not match the frozen prediction's mean8 labels")
    return cost, wait


def analyze(panels, cells, registered):
    keys = {source.state_key(r) for r in registered["mean_states"]}
    if len(panels) != 240 or set(panels) != keys or set(cells) != {(f["split"], f["index"]) for f in registered["folds"]} or len(cells) != 47:
        raise ValueError("Require all240 existing states and47 frozen grouped fits")
    for fold in registered["folds"]:
        source.qualify(cells[fold["split"], fold["index"]], fold, registered)
    variants = {}
    for variant in source.VARIANTS:
        partitions = {}
        for partition in PARTITIONS:
            validation = {}
            for split in source.SPLITS:
                predictions = [r for f in registered["folds"] if f["split"] == split
                    for r in cells[split, f["index"]]["predictions"][variant]]
                if len(predictions) != 240 or {source.state_key(r) for r in predictions} != keys:
                    raise ValueError("Lost a state or duplicated a grouped evaluation")
                rows = []
                for prediction in predictions:
                    p = panels[source.state_key(prediction)]
                    cost, wait = validate_panel_prediction(p, prediction)
                    rows.append({**{k: p[k] for k in ("root", "scenario", "scene_seed", "probe_index")},
                        **panel(p["values"], cost, wait, p["original_choice"], partition)})
                validation[split] = scopes(rows, variant == contract()["primary_variant"])
            partitions[partition] = {"validation": validation, "gates": gates(validation)}
        variants[variant] = partitions
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "states": 240, "prefix_groups": 40, "reused_fits": 47, **contract()["budget"], "variants": variants,
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/SOURCE_RUN
    prior = json.loads((directory/"report/result.json").read_text())
    if not (prior["protocol"] == source.EXPERIMENT_PROTOCOL and prior["contract"] == source.contract()
            and prior["software_qualified"] and prior["new_closed_form_fits"] == 1197):
        raise ValueError("Complete the qualified Stage204 report before factorial diagnosis")
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    cells = {(f["split"], f["index"]): json.loads((raw_directory(source.result_path(SOURCE_RUN, f))/"result.json").read_text())
        for f in registered["folds"]}
    result = analyze(load_panels(), cells, registered)
    write_json(args.output, result)
    print(json.dumps({p: d["gates"] for p, d in result["variants"][contract()["primary_variant"]].items()}, indent=2), flush=True)
    print("NATIVE_ORACLE_FACTORIAL_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
