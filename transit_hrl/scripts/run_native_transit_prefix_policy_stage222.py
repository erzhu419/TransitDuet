#!/usr/bin/env python3
"""Train-internal prefix/future calibration with unchanged outer control models."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_gap_calibration_stage221 as previous
from freq_hrl.domains.transit import native_prefix_policy_calibration as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_prefix_policy_stage222_v1"
SOURCE_RUN = "native_transit_gap_calibration_stage221_development_20261014_r1"
VIEWS, PARTITIONS, SCOPES = previous.VIEWS, previous.PARTITIONS, previous.SCOPES
KEYS = tuple((v, f, p, d) for v in VIEWS for f in (0, 1) for p in PARTITIONS for d in (0, 1))


def contract():
    parent = previous.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cohort": parent["cohort"], "outer": parent["outer"],
        "partitions": parent["partitions"], "final_heads": "reuse_exact32_stage221_models_no_refitting",
        "inner": "leave_one_complete_training_prefix_out54_train6_validate_crossed_with2_complementary2_future_splits",
        "inner_models": "own54_state2_draw_native_ridge_and_GLM_plus_three600_update_wait_residuals",
        "normalization": "inner_training_prefixes_only_no_outer_held_features_or_labels",
        "risk_control": "prefix_and_future_OOF_physical_mass_logloss_alpha_in0_1",
        "policy_constraint": "same_risk_objective_subject_selected_cost_and_wait_le_raw_controller_OOF_means_for_each_source_root",
        "policy_solver": "zero_logit_menu_event_sweep_boundary_ties_interval_risk_optima_identity_first_exact_ties",
        "inference": "one_fixed_alpha_per_architecture_no_root_ids_or_future_outcomes",
        "parameters": {v: previous.PARAMETERS[v] for v in learning.WAIT_VARIANTS},
        "optimizer": parent["optimizer"], "variants": list(learning.VARIANTS),
        "evaluation": "all_states_four_scopes_rootwise_and_future_sensitivity_controls_reproduced_no_CI",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "training_cells": 32, "reused_final_cells": 32,
            "inner_folds": 640, "analytic_cost_fits": 640, "glm_fits": 640,
            "inner_wait_fits": 1920, "calibration_scalar_fits": 192,
            "wait_residual_updates": 1920*learning.base.STEPS,
            "supervised_decision_updates": 0, "new_native_episodes": 0, "rl_actor_updates": 0},
        "storage": "inner_OOF_predictions_and_reused_final_models_remote_only_compact_report_pull",
        "boundary": "development_same240_states_and8_futures_relative_OOF_floors_not_a_held_control_guarantee"}


def result_path(run_name, view, fold, partition, direction):
    return ROOT/"results"/run_name/"cells"/view/f"fold_{fold}"/partition/str(direction)/"receipt.json"


def source_models(view, fold, partition, direction):
    path = previous.result_path(SOURCE_RUN, view, fold)
    return json.loads((raw_directory(path)/f"{partition}_{direction}_models.json").read_text())


def fit_cell(train, held, panels, models, partition, direction):
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    train_batch, held_batch = learning.base.base.neural.batches(train), learning.base.base.neural.batches(held)
    calibration, inner, data = learning.calibrate(train, panels, a, batch=train_batch)
    models["prefix_calibration"] = calibration
    train_prediction = learning.predict(train, models, batch=train_batch)
    held_prediction = learning.predict(held, models, batch=held_batch)
    scopes = {s: learning.base.base.records(train if s.startswith("train") else held, panels,
        a if s.endswith("same") else b, b if s.endswith("same") else a,
        train_prediction if s.startswith("train") else held_prediction) for s in SCOPES}
    groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
    fit = {"training_states": len(train), "training_groups": groups(train), "inner_fits": inner,
        "calibration": calibration, "inner_folds": len(inner), "analytic_cost_fits": len(inner),
        "glm_fits": len(inner), "wait_residual_updates": len(inner)*3*learning.base.STEPS,
        "calibration_scalar_fits": 6, "supervised_decision_updates": 0, "new_native_episodes": 0, "rl_actor_updates": 0}
    return {"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
        "training_groups": groups(train), "held_groups": groups(held), "fit": fit, "scopes": scopes}, models, data


def qualify(result, key, views):
    view, fold, partition, direction = key
    train, held = previous.previous.split_rows(views[view], fold)
    groups = lambda rs: {(r["root"], r["scene_seed"]) for r in rs}
    cell, fit = result["cell"], result["cell"]["fit"]
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    if (result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract()
            or result["key"] != list(key) or not result["software_qualified"]
            or cell["partition"] != partition or cell["direction"] != direction
            or cell["training_draws"] != list(a) or cell["evaluation_draws"] != list(b)
            or set(map(tuple, cell["training_groups"])) != groups(train)
            or set(map(tuple, cell["held_groups"])) != groups(held)
            or set(map(tuple, fit["training_groups"])) != groups(train) or fit["training_states"] != 60
            or fit["inner_folds"] != 20 or len(fit["inner_fits"]) != 20
            or fit["analytic_cost_fits"] != 20 or fit["glm_fits"] != 20
            or fit["wait_residual_updates"] != 60*learning.base.STEPS
            or fit["calibration_scalar_fits"] != 6 or fit["supervised_decision_updates"] != 0
            or fit["new_native_episodes"] != 0 or fit["rl_actor_updates"] != 0):
        raise ValueError("Changed prefix/future registration, grouping or fit budget")
    expected = {(g, d) for g in groups(train) for d in (0, 1)}
    if {(tuple(i["held_group"]), i["direction"]) for i in fit["inner_fits"]} != expected:
        raise ValueError("Incomplete leave-one-prefix future crossfit")
    for inner in fit["inner_fits"]:
        g, d = tuple(inner["held_group"]), inner["direction"]
        halves = (list(a[:2]), list(a[2:]))
        if (inner["training_draws"] != halves[d] or inner["validation_draws"] != halves[1-d]
                or set(map(tuple, inner["training_groups"])) != groups(train)-{g}
                or set(map(tuple, inner["wait_training_groups"])) != groups(train)-{g}
                or inner["training_states"] != 54 or inner["held_states"] != 6):
            raise ValueError("Held prefix/future contaminated an inner training fit")
        for v in learning.WAIT_VARIANTS:
            f = inner["fits"][v]
            if f["parameter_count"] != previous.PARAMETERS[v] or f["steps"] != learning.base.STEPS:
                raise ValueError("Changed matched inner architecture or updates")
    for v in learning.WAIT_VARIANTS:
        risk, policy = fit["calibration"]["risk"][v], fit["calibration"]["policy"][v]
        if not 0 <= risk["alpha"] <= 1 or not 0 <= policy["alpha"] <= 1 or risk["candidate_cases"] != 600:
            raise ValueError("Changed scalar calibration or candidate coverage")
        if (set(policy["raw_bounds_by_root"]) != {str(r) for r, _ in groups(train)}
                or not policy["selected_action_constraints_met"]
                or any(policy["selected_by_root"][r][m] > bound[m]
                    for r, bound in policy["raw_bounds_by_root"].items() for m in ("cost", "wait_min"))):
            raise ValueError("Selected-action calibration violated a source-specific floor")
    if set(cell["scopes"]) != set(SCOPES): raise ValueError("Missing train/held scope")
    for scope, rows in cell["scopes"].items():
        expected_rows = train if scope.startswith("train") else held
        if (len(rows) != 60 or {learning.base.base.state_key(r) for r in rows} != {learning.base.base.state_key(r) for r in expected_rows}
                or any(set(r["decisions"]) != set(learning.VARIANTS) or set(r["gates"]) != set(learning.GATES) for r in rows)):
            raise ValueError("Dropped a state or missing a selector control")


def metrics(rows):
    return {"prediction_cases": len(rows), "unique_states": len({learning.base.base.state_key(r) for r in rows}),
        "unique_prefixes": len({(r["root"], r["scene_seed"]) for r in rows}),
        "variants": {v: {"mean_cost": float(np.mean([r["decisions"][v]["cost"] for r in rows])),
            "mean_wait_min": float(np.mean([r["decisions"][v]["wait"] for r in rows])),
            "changed_fraction": float(np.mean([r["decisions"][v]["changed"] for r in rows])),
            "unsafe_fraction": float(np.mean([r["decisions"][v]["unsafe"] for r in rows]))} for v in learning.VARIANTS},
        "wait_heads": {g: {"physical_mass_loss_min": float(np.mean([r["gates"][g]["mass_loss_min"] for r in rows]))}
            for g in learning.GATES}}


def summarize(results, views):
    if set(results) != set(KEYS): raise ValueError("Complete all32 cells before reporting")
    for key, result in results.items(): qualify(result, key, views)
    output, records = {}, {}
    for view in VIEWS:
        cells = [r["cell"] for k, r in results.items() if k[0] == view]
        scopes = {s: [r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        if any(len(rs) != 480 or len({learning.base.base.state_key(r) for r in rs}) != 120 for rs in scopes.values()):
            raise ValueError("Changed complete held-state or future sensitivity coverage")
        output[view] = {"scopes": {s: metrics(rs) for s, rs in scopes.items()},
            "held_other_by_root": {str(root): metrics([r for r in scopes["held_other"] if r["root"] == root])
                for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction": {f"{p}/{d}": metrics([r for c in cells if c["partition"] == p and c["direction"] == d
                for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0, 1)},
            "calibration": {mode: {v: {"alpha_mean": float(np.mean([c["fit"]["calibration"][mode][v]["alpha"] for c in cells])),
                "cells": [c["fit"]["calibration"][mode][v] for c in cells]} for v in learning.WAIT_VARIANTS}
                for mode in ("risk", "policy")}}
        records[view] = {(*learning.base.base.state_key(r), c["partition"], c["direction"]): r
            for c in cells for r in c["scopes"]["held_other"]}
    mixed = {k: r for v in ("mixed_a", "mixed_b") for k, r in records[v].items()}
    comparisons = {}
    for root in previous.previous.source.prefixes.ROOTS:
        single, matched = records[f"root_{root}"], {k: r for k, r in mixed.items() if r["root"] == root}
        if set(single) != set(matched) or len(single) != 480: raise ValueError("Changed single/mixed held-state matching")
        comparisons[str(root)] = {"prediction_cases": len(single), "unique_states": 120,
            "variants": {v: {"cost_delta": float(np.mean([matched[k]["decisions"][v]["cost"]-r["decisions"][v]["cost"] for k, r in single.items()])),
                "wait_delta_min": float(np.mean([matched[k]["decisions"][v]["wait"]-r["decisions"][v]["wait"] for k, r in single.items()]))}
                for v in learning.VARIANTS}}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "coverage_qualified": True, "views": output, "single_vs_mixed": comparisons, **contract()["budget"],
        "native_deployment_eligible": False, "third_scene_screen_eligible": False}


def reproduce_controls(result, prior):
    differences = []
    for view in VIEWS:
        for scope in SCOPES:
            a, b = result["views"][view]["scopes"][scope], prior["views"][view]["scopes"][scope]
            for v in learning.base.VARIANTS:
                for m in ("mean_cost", "mean_wait_min", "changed_fraction", "unsafe_fraction"):
                    differences.append(abs(a["variants"][v][m]-b["variants"][v][m]))
            for g in learning.base.GATES:
                differences.append(abs(a["wait_heads"][g]["physical_mass_loss_min"]-b["wait_heads"][g]["physical_mass_loss_min"]))
    if max(differences) > 1e-12: raise ValueError("Reused Stage221 control did not reproduce")
    return {"comparisons": len(differences), "max_abs_difference": max(differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("fit", "report"), required=True)
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--partition", choices=PARTITIONS)
    parser.add_argument("--direction", type=int, choices=(0, 1))
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != previous.EXPERIMENT_PROTOCOL or prior["contract"] != previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage221 before prefix-policy calibration")
    bank, _, registered, _ = previous.previous.source.source.load_data()
    views = previous.previous.source.matched_views(bank["training_data"], registered)
    if args.operation == "fit":
        key = args.view, args.fold, args.partition, args.direction
        output = result_path(args.run_name, *key)
        raw = raw_directory(output)
        train, held = previous.previous.split_rows(views[args.view], args.fold)
        panels = previous.previous.source.source.wait_source.source.diagnostic.load_panels()
        cell, models, data = fit_cell(train, held, panels, source_models(*key), args.partition, args.direction)
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "software_qualified": True, "cell": cell}
        qualify(result, key, views)
        write_json(raw/"models.json", models)
        write_json(raw/"inner_oof.json", {k: v.tolist() for k, v in data.items()})
        write_json(raw/"result.json", result)
        write_json(output, {k: v for k, v in result.items() if k != "cell"} | {"training_cells": 1,
            "wait_residual_updates": 60*learning.base.STEPS, "supervised_decision_updates": 0,
            "new_native_episodes": 0, "rl_actor_updates": 0})
        print("NATIVE_PREFIX_POLICY_FIT_COMPLETE", flush=True)
    else:
        results = {k: json.loads((raw_directory(result_path(args.run_name, *k))/"result.json").read_text()) for k in KEYS}
        result = summarize(results, views)
        result["control_reproduction"] = reproduce_controls(result, prior)
        write_json(ROOT/"results"/args.run_name/"report/result.json", result)
        print("NATIVE_PREFIX_POLICY_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
