#!/usr/bin/env python3
"""Report every fleet-only comparison without reopening reserved native labels."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_fleet_residual_stage193 as spec
from scripts.analyze_native_transit_coverage_value_stage188 import group_key, validation_summary
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if group_key(g, fold["split"]) != fold["held_out"]]
    expected = [[g["root"], g["scene_seed"]] for g in groups]
    test = {spec.source.state_key(r) for r in registered["mean_states"] if group_key(r, fold["split"]) == fold["held_out"]}
    if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["fold"] == fold and cell["critic_updates"] == 2*spec.STEPS
            and cell["actor_updates"] == cell["new_native_steps"] == 0
            and cell["training_mean_states"] == 30-len(test) and cell["evaluation_mean_states"] == len(test)
            and set(cell["fits"]) == set(spec.value.METHODS) and set(cell["predictions"]) == set(spec.value.VARIANTS)):
        raise ValueError("Changed fleet experiment coverage or update budget")
    fits = list(cell["fits"].values())
    for method, fit in cell["fits"].items():
        if not (fit["method"] == method and fit["training_groups"] == expected and fit["training_states"] == 6*len(groups)
                and fit["steps"] == spec.STEPS and fit["parameter_count"] == 1601
                and fit["initialization_seed"] == spec.source.source.floor.occupancy.SEED
                and fit["objective"] == "absolute_fleet_MSE_plus_paired_fleet_MSE"
                and fit["initialization"] == "shared_scalar_hidden_zero_fleet_projection"):
            raise ValueError("Fleet-only fit changed grouping, initialization or matched method")
    if any(fits[0][k] != fits[1][k] for k in ("center", "scale", "target_scale", "training_groups", "training_states")):
        raise ValueError("Residual and zero-prior fits used different normalizations or groups")
    frozen = {}
    for variant, by_label in cell["predictions"].items():
        if set(by_label) != {"original", "mean8"}:
            raise ValueError("Missing same-state original and mean-label evaluations")
        for label, by_policy in by_label.items():
            if set(by_policy) != set(spec.source.source.value.POLICIES):
                raise ValueError("Missing shared-fit selector")
            for policy, rows in by_policy.items():
                if len(rows) != len(test) or {spec.source.state_key(r) for r in rows} != test:
                    raise ValueError("Fleet comparison lost held-out target states")
                for row in rows:
                    key = spec.source.state_key(row)
                    heads, waits = np.asarray(row["predicted_heads"]), np.asarray(row["predicted_wait_delta_min"])
                    absolute, original = np.asarray(row["absolute_fleet_cost"]), row["original_choice"]
                    costs = heads.sum(axis=1)
                    if policy == "wait_feasible": costs = np.where(waits <= 0, costs, np.inf)
                    if not (row["selected"] == spec.source.source.paired.choice_from_cost(costs, original)
                            and np.all(heads[original] == 0) and waits[original] == 0
                            and np.all(absolute >= row["causal_fleet_floor"])
                            and np.allclose(absolute-absolute[original], heads[:, 0], rtol=0, atol=1e-12)
                            and np.isclose(row["actual_cost_delta"], row["fleet_cost_delta"]+row["nonfleet_cost_delta"], rtol=0, atol=1e-12)):
                        raise ValueError("Changed fleet floor, identity, units or feasible selector")
                    shared = (heads[:, 1].tolist(), waits.tolist())
                    if key in frozen and frozen[key] != shared:
                        raise ValueError("A fleet variant changed the frozen nonfleet/wait controls")
                    frozen[key] = shared
                    prediction_key = variant, key
                    prediction = (row["predicted_heads"], row["predicted_wait_delta_min"], row["absolute_fleet_cost"])
                    if prediction_key in frozen and frozen[prediction_key] != prediction:
                        raise ValueError("Evaluation labels or selectors changed the fitted fleet prediction")
                    frozen[prediction_key] = prediction


def summary(rows):
    return {**validation_summary(rows),
        "fleet_absolute_mse": float(np.mean([r["fleet_absolute_mse"] for r in rows])),
        "mean_selected_fleet_delta_error": float(np.mean([r["selected_fleet_delta_error"] for r in rows]))}


def summarize(cells, registered):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Missing registered fleet-only folds")
    for fold in registered["folds"]:
        qualify(cells[fold["split"], fold["index"]], fold, registered)
    variants, comparisons = {}, {}
    for variant in spec.value.VARIANTS:
        for policy in spec.source.source.value.POLICIES:
            evaluations = {}
            for label in ("original", "mean8"):
                evaluations[label] = {}
                for split in spec.SPLITS:
                    rows = [r for f in registered["folds"] if f["split"] == split
                        for r in cells[split, f["index"]]["predictions"][variant][label][policy]]
                    if len(rows) != 30 or len({spec.source.state_key(r) for r in rows}) != 30:
                        raise ValueError("Fleet aggregate changed its thirty test states")
                    evaluations[label][split] = summary(rows)
            gates = {"cost": all(v["mean_actual_cost_delta"] < 0 for s in evaluations["mean8"].values() for v in s["by_root"].values()),
                "wait": all(v["mean_wait_delta_min"] <= 0 for s in evaluations["mean8"].values() for v in s["by_root"].values())}
            variants[f"{variant}:{policy}"] = {"evaluation": evaluations, "expected_label_gates": gates}
    fields = ("mean_actual_cost_delta", "mean_wait_delta_min", "mean_fleet_cost_delta", "mean_nonfleet_cost_delta")
    for policy in spec.source.source.value.POLICIES:
        learned = variants[f"fleet_residual:{policy}"]["evaluation"]["mean8"]
        comparisons[policy] = {}
        for control in ("physics_prior", "fleet_from_zero", "cached_absolute"):
            base = variants[f"{control}:{policy}"]["evaluation"]["mean8"]
            comparisons[policy][control] = {split: {scope: {k: a[k]-b[k] for k in fields}
                for scope, a, b in [("overall", learned[split]["overall"], base[split]["overall"]),
                    *[(str(r), learned[split]["by_root"][str(r)], base[split]["by_root"][str(r)]) for r in spec.source.noise.ROOTS]]}
                for split in spec.SPLITS}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "new_fits": 2*len(cells), "reused_fits": len(cells), "critic_updates": 2*len(cells)*spec.STEPS,
        "actor_updates": 0, "new_native_steps": 0, "evaluation_states": 30, "evaluation_prefix_scene_groups": 10,
        "variants": variants, "residual_increment": comparisons,
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
    print("NATIVE_FLEET_RESIDUAL_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
