#!/usr/bin/env python3
"""Bind calibration floors to OOF and actual final-controller predictions."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_precision_stage223 as previous
from freq_hrl.domains.transit import native_refit_binding as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_refit_binding_stage224_v1"
SOURCE_RUN = "native_transit_precision_stage223_development_20261014_r1"
VIEWS, PARTITIONS, SCOPES, KEYS = previous.VIEWS, previous.PARTITIONS, previous.SCOPES, previous.KEYS
ENFORCED = {"final": ("final",), "joint": ("oof", "final")}


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cohort": previous.contract()["cohort"], "outer": previous.contract()["outer"],
        "partitions": previous.contract()["partitions"], "models": "reuse_exact_stage223_final4_all_three_wait_architectures_no_weight_updates",
        "oof": "reuse_c2v4_inner2_predictions_on_four_draw_excluded_prefix_truths",
        "objective": "same_OOF_physical_mass_logloss_not_final_training_loss",
        "constraints": "cost_and_wait_le_own_raw_controller_means_each_source_root",
        "modes": {k: list(v) for k, v in ENFORCED.items()}, "controls": "GLM_raw_risk_only_and_stage223_c2v4_OOF_only",
        "solver": "union_of_OOF_and_final_zero_logit_events_boundary_ties_interval_risk_minima_alpha1_exact_raw_feasible",
        "labels": "final_constraints_only60_outer_training_states_training4_draws_no_outer_held_labels",
        "inference": "fixed_alpha_causal_final4_predictions_only_no_true_safe_menu",
        "variants": list(learning.VARIANTS), "coverage": "four_scopes60_cases_per_cell_all_states_no_subhalf_repetition",
        "budget": {"calibration_tasks": 32, "report_tasks": 1, "training_cells": 32,
            "reused_final4_cells": 32, "reused_oof_cells": 32, "calibration_scalar_fits": 192,
            "analytic_cost_fits": 0, "glm_fits": 0, "wait_residual_updates": 0,
            "new_native_episodes": 0, "supervised_decision_updates": 0, "rl_actor_updates": 0},
        "boundary": "final_training_noninferiority_is_constructed_only_outer_evaluation_tests_transfer_no_new_independent_CI",
        "storage": "models_predictions_training_constraints_remote_only_compact_report_pull"}


def result_path(run_name, *key):
    return previous.result_path(run_name, *key)


def fit_cell(train, held, panels, bundle, oof, reused, partition, direction):
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    train_batch, held_batch = learning.learned.neural.batches(train), learning.learned.neural.batches(held)
    final = learning.final_data(train, panels, a, bundle, batch=train_batch)
    calibration, diagnostic = learning.calibrate(oof, final, reused)
    train_prediction = learning.predict(train, bundle, calibration, batch=train_batch)
    held_prediction = learning.predict(held, bundle, calibration, batch=held_batch)
    scopes = {s: learning.learned.records(train if s.startswith("train") else held, panels,
        a if s.endswith("same") else b, b if s.endswith("same") else a,
        train_prediction if s.startswith("train") else held_prediction) for s in SCOPES}
    groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
    fit = {"calibration": calibration, "oof_only_final_diagnostic": diagnostic,
        "final_constraint_states": len(train), "oof_candidate_cases": int(len(oof["originals"])*5),
        "calibration_scalar_fits": 6, "analytic_cost_fits": 0, "glm_fits": 0,
        "wait_residual_updates": 0, "new_native_episodes": 0, "supervised_decision_updates": 0, "rl_actor_updates": 0}
    cell = {"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
        "training_groups": groups(train), "held_groups": groups(held), "fit": fit, "scopes": scopes}
    return cell, {"bundle": bundle, "calibration": calibration}, final


def qualify(result, key, views):
    view, fold, partition, direction = key
    train, held = previous.previous.previous.previous.split_rows(views[view], fold)
    groups = lambda rs: {(r["root"], r["scene_seed"]) for r in rs}
    cell, fit = result["cell"], result["cell"]["fit"]
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    if (result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract()
            or result["key"] != list(key) or not result["software_qualified"]
            or cell["partition"] != partition or cell["direction"] != direction
            or cell["training_draws"] != list(a) or cell["evaluation_draws"] != list(b)
            or set(map(tuple, cell["training_groups"])) != groups(train) or set(map(tuple, cell["held_groups"])) != groups(held)
            or fit["final_constraint_states"] != 60 or fit["oof_candidate_cases"] != 600 or fit["calibration_scalar_fits"] != 6
            or any(fit[k] != 0 for k in ("analytic_cost_fits", "glm_fits", "wait_residual_updates", "new_native_episodes", "supervised_decision_updates", "rl_actor_updates"))):
        raise ValueError("Changed refit binding inputs, grouping or budget")
    expected_roots = {str(r) for r, _ in groups(train)}
    for mode, enforced in ENFORCED.items():
        for v in learning.WAIT_VARIANTS:
            fit_v = fit["calibration"][mode][v]
            if not 0 <= fit_v["alpha"] <= 1 or fit_v["enforced_scopes"] != list(enforced):
                raise ValueError("Changed refit calibration mode")
            for scope in ("oof", "final"):
                if set(fit_v["raw_bounds"][scope]) != expected_roots:
                    raise ValueError("Missing source-specific constraint")
                met = learning.floors_met(fit_v["selected"][scope], fit_v["raw_bounds"][scope])
                if met != fit_v["floor_met"][scope] or (scope in enforced and not met):
                    raise ValueError("Enforced selected-action floor violated")
    if set(cell["scopes"]) != set(SCOPES): raise ValueError("Missing train/held evaluation scope")
    for scope, rows in cell["scopes"].items():
        expected = train if scope.startswith("train") else held
        if (len(rows) != 60 or {learning.learned.state_key(r) for r in rows} != {learning.learned.state_key(r) for r in expected}
                or any(set(r["decisions"]) != set(learning.VARIANTS) or set(r["gates"]) != set(learning.GATES) for r in rows)):
            raise ValueError("Dropped a state or a causal selector")
    for mode in ENFORCED:
        for v in learning.WAIT_VARIANTS:
            for root in expected_roots:
                rs = [r for r in cell["scopes"]["train_same"] if str(r["root"]) == root]
                for metric in ("cost", "wait"):
                    selected = np.mean([r["decisions"][f"{mode}_{v}"][metric] for r in rs])
                    raw = np.mean([r["decisions"][f"raw_{v}"][metric] for r in rs])
                    if selected > raw+1e-12:
                        raise ValueError("Actual final training decisions violated the bound")


def metrics(rows):
    return {"prediction_cases": len(rows), "unique_states": len({learning.learned.state_key(r) for r in rows}),
        "unique_prefixes": len({(r["root"], r["scene_seed"]) for r in rows}),
        "variants": {v: {"mean_cost": float(np.mean([r["decisions"][v]["cost"] for r in rows])),
            "mean_wait_min": float(np.mean([r["decisions"][v]["wait"] for r in rows])),
            "changed_fraction": float(np.mean([r["decisions"][v]["changed"] for r in rows])),
            "unsafe_fraction": float(np.mean([r["decisions"][v]["unsafe"] for r in rows]))} for v in learning.VARIANTS},
        "wait_heads": {g: {"physical_mass_loss_min": float(np.mean([r["gates"][g]["mass_loss_min"] for r in rows]))} for g in learning.GATES}}


def summarize(results, views):
    if set(results) != set(KEYS): raise ValueError("Complete all32 binding cells before reporting")
    for key, result in results.items(): qualify(result, key, views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for k, r in results.items() if k[0] == view]
        scopes = {s: [r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        if any(len(rs) != 480 or len({learning.learned.state_key(r) for r in rs}) != 120 for rs in scopes.values()):
            raise ValueError("Changed full-state/future coverage")
        output[view] = {"scopes": {s: metrics(rs) for s, rs in scopes.items()},
            "held_other_by_root": {str(root): metrics([r for r in scopes["held_other"] if r["root"] == root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction": {f"{p}/{d}": metrics([r for c in cells if c["partition"] == p and c["direction"] == d
                for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0, 1)},
            "calibration": {mode: {v: {"alpha_mean": float(np.mean([c["fit"]["calibration"][mode][v]["alpha"] for c in cells])),
                "cells": [c["fit"]["calibration"][mode][v] for c in cells]} for v in learning.WAIT_VARIANTS}
                for mode in ("final", "joint")},
            "oof_only_final_floor_violations": {v: sum(not learning.floors_met(c["fit"]["oof_only_final_diagnostic"][v]["oof_only_on_final"],
                c["fit"]["oof_only_final_diagnostic"][v]["raw_final_bounds"]) for c in cells) for v in learning.WAIT_VARIANTS}}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "coverage_qualified": True, "views": output, **contract()["budget"],
        "native_deployment_eligible": False, "third_scene_screen_eligible": False}


def reproduce_controls(result, prior):
    mapping = {"glm": "f4_glm", **{f"raw_{v}": f"f4_raw_{v}" for v in learning.WAIT_VARIANTS},
        **{f"{m}_{v}": f"f4_c2v4_{old}_{v}" for m, old in (("risk", "risk"), ("oof", "policy")) for v in learning.WAIT_VARIANTS}}
    differences = []
    for view in VIEWS:
        for scope in SCOPES:
            a, b = result["views"][view]["scopes"][scope], prior["views"][view]["scopes"][scope]
            for name, old in mapping.items():
                for m in ("mean_cost", "mean_wait_min", "changed_fraction", "unsafe_fraction"):
                    differences.append(abs(a["variants"][name][m]-b["variants"][old][m]))
                differences.append(abs(a["wait_heads"][name]["physical_mass_loss_min"]-b["wait_heads"][old]["physical_mass_loss_min"]))
    if max(differences) > 1e-12: raise ValueError("Stage223 primary control failed reproduction")
    return {"comparisons": len(differences), "max_abs_difference": max(differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("calibrate", "report"), required=True)
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--partition", choices=PARTITIONS)
    parser.add_argument("--direction", type=int, choices=(0, 1))
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != previous.EXPERIMENT_PROTOCOL or prior["contract"] != previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage223 before final-model binding")
    bank, _, registered, _ = previous.previous.previous.previous.source.source.load_data()
    views = previous.previous.previous.previous.source.matched_views(bank["training_data"], registered)
    if args.operation == "calibrate":
        key = args.view, args.fold, args.partition, args.direction
        source = raw_directory(previous.result_path(SOURCE_RUN, *key))
        previous.qualify(json.loads((source/"result.json").read_text()), key, views)
        models = json.loads((source/"models.json").read_text())
        oof = {k: np.asarray(v) for k, v in json.loads((source/"inner_oof.json").read_text())["c2v4"].items()}
        train, held = previous.previous.previous.previous.split_rows(views[args.view], args.fold)
        panels = previous.previous.previous.previous.source.source.wait_source.source.diagnostic.load_panels()
        cell, model, final = fit_cell(train, held, panels, models["final4"], oof, models["calibration"]["c2v4"], args.partition, args.direction)
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "software_qualified": True, "cell": cell}
        qualify(result, key, views)
        output = result_path(args.run_name, *key)
        raw = raw_directory(output)
        write_json(raw/"models.json", model)
        write_json(raw/"final_training_constraints.json", {k: v.tolist() for k, v in final.items()})
        write_json(raw/"result.json", result)
        write_json(output, {k: v for k, v in result.items() if k != "cell"} | {"calibration_scalar_fits": 6, "weight_updates": 0})
        print("NATIVE_REFIT_BINDING_CALIBRATION_COMPLETE", flush=True)
    else:
        results = {k: json.loads((raw_directory(result_path(args.run_name, *k))/"result.json").read_text()) for k in KEYS}
        result = summarize(results, views)
        result["control_reproduction"] = reproduce_controls(result, prior)
        write_json(ROOT/"results"/args.run_name/"report/result.json", result)
        print("NATIVE_REFIT_BINDING_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
