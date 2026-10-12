#!/usr/bin/env python3
"""Separate loss-scaling effects from reference-risk/action-credit separation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_fleet_credit_stage194 as spec
from scripts.analyze_native_transit_coverage_value_stage188 import group_key, validation_summary
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

COMPARISONS = {"scale_shared": ("shared_pair_scaled", "fleet_from_zero"),
    "scale_separated": ("separated_pair_scaled", "separated_shared_scale"),
    "separation_shared_units": ("separated_shared_scale", "fleet_from_zero"),
    "separation_pair_units": ("separated_pair_scaled", "shared_pair_scaled"),
    "primary_vs_cached": ("separated_pair_scaled", "cached_absolute")}


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if group_key(g, fold["split"]) != fold["held_out"]]
    expected = [[g["root"], g["scene_seed"]] for g in groups]
    test = {spec.expected.state_key(r) for r in registered["mean_states"] if group_key(r, fold["split"]) == fold["held_out"]}
    if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["fold"] == fold and cell["critic_updates"] == 3*spec.STEPS
            and cell["actor_updates"] == cell["new_native_steps"] == 0
            and cell["training_mean_states"] == 30-len(test) and cell["evaluation_mean_states"] == len(test)
            and set(cell["fits"]) == {"fleet_from_zero", *spec.value.METHODS}
            and set(cell["predictions"]) == set(spec.value.VARIANTS)):
        raise ValueError("Changed fleet-credit fold or update budget")
    control = cell["fits"]["fleet_from_zero"]
    for method, fit in cell["fits"].items():
        if not (fit["method"] == method and fit["training_groups"] == expected and fit["training_states"] == 6*len(groups)
                and fit["steps"] == spec.STEPS and fit["parameter_count"] == 1601
                and fit["initialization_seed"] == spec.value.floor.occupancy.SEED
                and all(fit[k] == control[k] for k in ("center", "scale", "target_scale"))):
            raise ValueError("Fleet-credit comparison changed groups, inputs, units or capacity")
        if method in spec.value.METHODS and not (
                fit["initialization"] == "same16_hidden_rows_zero_output_shared16_or_disjoint8plus8"
                and fit["baseline_pair_gradient"] == (method == "shared_pair_scaled")
                and fit["paired_loss_scale"] == fit["target_scale" if method == "separated_shared_scale" else "paired_target_scale"]):
            raise ValueError("Fleet-credit fit changed its gradient separation or loss units")
    frozen, predictions = {}, {}
    for variant, by_label in cell["predictions"].items():
        if set(by_label) != {"original", "mean8"}:
            raise ValueError("Missing matched original/mean target evaluations")
        for by_policy in by_label.values():
            if set(by_policy) != set(spec.coverage.value.POLICIES):
                raise ValueError("Missing shared-fit selector")
            for policy, rows in by_policy.items():
                if len(rows) != len(test) or {spec.expected.state_key(r) for r in rows} != test:
                    raise ValueError("Fleet-credit evaluation changed held-out states")
                for row in rows:
                    key, original = spec.expected.state_key(row), row["original_choice"]
                    heads, waits = np.asarray(row["predicted_heads"]), np.asarray(row["predicted_wait_delta_min"])
                    absolute = np.asarray(row["absolute_fleet_cost"])
                    costs = heads.sum(axis=1)
                    if policy == "wait_feasible": costs = np.where(waits <= 0, costs, np.inf)
                    if not (row["selected"] == spec.coverage.paired.choice_from_cost(costs, original)
                            and np.all(heads[original] == 0) and waits[original] == 0
                            and np.all(absolute >= row["causal_fleet_floor"])
                            and np.allclose(absolute-absolute[original], heads[:, 0], rtol=0, atol=1e-12)
                            and np.isclose(row["actual_cost_delta"], row["fleet_cost_delta"]+row["nonfleet_cost_delta"], rtol=0, atol=1e-12)):
                        raise ValueError("Changed physical floor, identity, cost decomposition or selector")
                    shared = (heads[:, 1].tolist(), waits.tolist())
                    if key in frozen and frozen[key] != shared:
                        raise ValueError("A fleet-credit variant changed nonfleet/wait predictions")
                    frozen[key] = shared
                    prediction = (row["predicted_heads"], waits.tolist(), absolute.tolist())
                    pkey = variant, key
                    if pkey in predictions and predictions[pkey] != prediction:
                        raise ValueError("Evaluation labels or selectors changed predictions")
                    predictions[pkey] = prediction


def summary(rows):
    return {**validation_summary(rows), **{k: float(np.mean([r[k] for r in rows])) for k in
        ("fleet_absolute_mse", "fleet_pair_mse", "original_fleet_mse", "selected_fleet_delta_error")}}


def summarize(cells, registered):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Missing registered fleet-credit folds")
    for fold in registered["folds"]:
        qualify(cells[fold["split"], fold["index"]], fold, registered)
    variants = {}
    for variant in spec.value.VARIANTS:
        for policy in spec.coverage.value.POLICIES:
            evaluations = {}
            for label in ("original", "mean8"):
                evaluations[label] = {}
                for split in spec.SPLITS:
                    rows = [r for f in registered["folds"] if f["split"] == split
                        for r in cells[split, f["index"]]["predictions"][variant][label][policy]]
                    if len(rows) != 30 or len({spec.expected.state_key(r) for r in rows}) != 30:
                        raise ValueError("Fleet-credit aggregate lost a held-out state")
                    evaluations[label][split] = summary(rows)
            gates = {"cost": all(v["mean_actual_cost_delta"] < 0 for s in evaluations["mean8"].values() for v in s["by_root"].values()),
                "wait": all(v["mean_wait_delta_min"] <= 0 for s in evaluations["mean8"].values() for v in s["by_root"].values())}
            variants[f"{variant}:{policy}"] = {"evaluation": evaluations, "expected_label_gates": gates}
    effects = {}
    for policy in spec.coverage.value.POLICIES:
        effects[policy] = {}
        for name, (a, b) in COMPARISONS.items():
            effects[policy][name] = {}
            for split in spec.SPLITS:
                new, control = (variants[f"{v}:{policy}"]["evaluation"]["mean8"][split] for v in (a, b))
                effects[policy][name][split] = {scope: {k: x[k]-y[k] for k in ("mean_actual_cost_delta", "mean_wait_delta_min")}
                    for scope, x, y in [("overall", new["overall"], control["overall"]),
                        *[(str(r), new["by_root"][str(r)], control["by_root"][str(r)]) for r in spec.expected.noise.ROOTS]]}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "new_fits": 3*len(cells), "reused_fits": 2*len(cells), "critic_updates": 3*len(cells)*spec.STEPS,
        "actor_updates": 0, "new_native_steps": 0, "evaluation_states": 30, "evaluation_prefix_scene_groups": 10,
        "variants": variants, "factorial_effects": effects,
        "fit_wall_s_sum": {m: sum(c["fit_wall_s"][m] for c in cells.values()) for m in spec.value.METHODS},
        "expected_target_development_gate": all(variants[spec.contract()["primary"]]["expected_label_gates"].values()),
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    cells = {(f["split"], f["index"]): json.loads((raw_directory(spec.result_path(args.run_name, f))/"result.json").read_text())
        for f in registered["folds"]}
    result = summarize(cells, registered)
    write_json(args.output, result)
    print(json.dumps({"fits": result["new_fits"], "gates": {k: v["expected_label_gates"] for k, v in result["variants"].items()}}, indent=2))
    print("NATIVE_FLEET_CREDIT_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
