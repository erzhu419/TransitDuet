#!/usr/bin/env python3
"""Report objective and weighting effects on the unchanged development cohort."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_weighted_credit_stage197 as spec
from scripts.analyze_native_transit_coverage_value_stage188 import group_key, FIELDS
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json


def qualify(cell, fold, registered):
    train = [g for g in registered["groups"] if group_key(g, fold["split"]) != fold["held_out"]]
    train_groups = sorted([[g["root"], g["scene_seed"]] for g in train])
    keys = {spec.state_key(r) for r in registered["mean_states"]
        if group_key(r, fold["split"]) == fold["held_out"]}
    train_means = len(registered["mean_states"])-len(keys)
    if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["fold"] == fold and cell["critic_updates"] == 300
            and cell["actor_updates"] == cell["new_native_episodes"] == 0
            and cell["training_mean_states"] == train_means and cell["evaluation_mean_states"] == len(keys)
            and set(cell["fits"]) == set(cell["predictions"]) == set(spec.VARIANTS)):
        raise ValueError("Changed development fold, variants or frozen budget")
    reference = cell["fits"]["anchored_uniform"]
    for variant in spec.VARIANTS:
        fit = cell["fits"][variant]
        if not (fit["training_groups"] == train_groups and fit["training_states"] == 6*len(train)
                and fit["steps"] == 100 and fit["parameter_count"] == 1635
                and fit["initialization_seed"] == 178 and fit["representation"] == "tail_summary"
                and fit["objective"] == "proper_mse"
                and fit["absolute_fleet_forecast"] == variant.startswith("anchored")
                and fit["sample_weight_sum"] == 6*len(train)+(7*train_means if variant.endswith("count_weighted") else 0)
                and fit["center"] == reference["center"] and fit["scale"] == reference["scale"]):
            raise ValueError("Changed training groups, normalization, sample weights or critic capacity")
        for label in ("original", "mean8"):
            for policy in spec.value.POLICIES:
                rows = cell["predictions"][variant][label][policy]
                if len(rows) != len(keys) or {spec.state_key(r) for r in rows} != keys:
                    raise ValueError("Lost or repeated held-out callbacks")
                for row in rows:
                    heads, waits = np.asarray(row["predicted_heads"]), np.asarray(row["predicted_wait_delta_min"])
                    original = row["original_choice"]
                    cost = heads.sum(axis=1)
                    if policy == "wait_feasible": cost = np.where(waits <= 0, cost, np.inf)
                    if not (row["selected"] == spec.paired.choice_from_cost(cost, original)
                            and np.all(heads[original] == 0) and waits[original] == 0):
                        raise ValueError("Changed identity or fixed action selector")
                    if variant.startswith("advantage") and "absolute_fleet_cost" in row:
                        raise ValueError("Relative action credit is not an absolute fleet-risk forecast")
        frozen = cell["predictions"][variant]["original"]["cost_only"]
        for label in ("original", "mean8"):
            for policy in spec.value.POLICIES:
                for a, b in zip(frozen, cell["predictions"][variant][label][policy]):
                    if any(a[k] != b[k] for k in ("predicted_heads", "predicted_wait_delta_min")):
                        raise ValueError("Labels or selectors changed model predictions")


def choice_summary(rows):
    changed = [r for r in rows if r["selected"] != r["original_choice"]]
    return {**spec.paired.summarize_choices(rows),
        "false_feasible_choices": sum(r["predicted_wait_delta_min"][r["selected"]] <= 0 and r["wait_delta_min"] > 0
            for r in changed)}


def summarize(cells, registered, panels):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete all47 grouped fits before reporting")
    if set(panels) != {spec.state_key(r) for r in registered["mean_states"]}:
        raise ValueError("Changed conditional-draw evaluation cohort")
    for fold in registered["folds"]:
        qualify(cells[fold["split"], fold["index"]], fold, registered)
    variants = {}
    for variant in spec.VARIANTS:
        for policy in spec.value.POLICIES:
            evaluation, pair_error = {}, {}
            for label in ("original", "mean8"):
                evaluation[label] = {}
                for split in spec.SPLITS:
                    rows = [r for f in registered["folds"] if f["split"] == split
                        for r in cells[split, f["index"]]["predictions"][variant][label][policy]]
                    if len(rows) != 240 or len({spec.state_key(r) for r in rows}) != 240:
                        raise ValueError("Aggregate lost the matched240 evaluation states")
                    evaluation[label][split] = {"overall": choice_summary(rows),
                        "by_root": {str(root): choice_summary([r for r in rows if r["root"] == root]) for root in spec.source.ROOTS},
                        "by_regime": {s: choice_summary([r for r in rows if r["scenario"] == s]) for s in spec.source.SCENARIOS}}
                    if label == "mean8":
                        pair_rows = [{**{k: r[k] for k in ("root", "scenario", "scene_seed", "probe_index")},
                            **spec.diagnostic.prediction_panel(panels[spec.state_key(r)]["values"], r)} for r in rows]
                        pair_error[split] = spec.diagnostic.scopes(pair_rows)
            gates = {"cost": all(v["mean_actual_cost_delta"] < 0 for s in evaluation["mean8"].values() for v in s["by_root"].values()),
                "wait": all(v["mean_wait_delta_min"] <= 0 for s in evaluation["mean8"].values() for v in s["by_root"].values())}
            variants[f"{variant}:{policy}"] = {"evaluation": evaluation, "pair_error": pair_error, "gates": gates}
    effects = {}
    for name, previous, current in (("weighting_with_anchor", "anchored_uniform", "anchored_count_weighted"),
            ("weighting_with_advantage", "advantage_uniform", "advantage_count_weighted"),
            ("objective_uniform", "anchored_uniform", "advantage_uniform"),
            ("objective_weighted", "anchored_count_weighted", "advantage_count_weighted")):
        effects[name] = {p: {s: {scope: {k: variants[f"{current}:{p}"]["evaluation"]["mean8"][s][scope][k]
            -variants[f"{previous}:{p}"]["evaluation"]["mean8"][s][scope][k] for k in (*FIELDS, "false_feasible_choices")}
            for scope in ("overall",)} for s in spec.SPLITS} for p in spec.value.POLICIES}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        **spec.contract()["budget"], "evaluation_states": 240, "evaluation_prefix_groups": 40,
        "variants": variants, "factor_effects": effects,
        "fit_wall_s_sum": sum(sum(c["fit_wall_s"].values()) for c in cells.values()),
        "development_gate": all(variants[spec.contract()["primary"]]["gates"].values()),
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    cells = {(f["split"], f["index"]): json.loads((raw_directory(spec.result_path(args.run_name, f))/"result.json").read_text())
        for f in registered["folds"]}
    result = summarize(cells, registered, spec.diagnostic.load_panels())
    write_json(args.output, result)
    print(json.dumps({k: v["gates"] for k, v in result["variants"].items()}, indent=2), flush=True)
    print("NATIVE_WEIGHTED_CREDIT_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
