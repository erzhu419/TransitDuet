#!/usr/bin/env python3
"""Cross inner/final target precision without new rollouts or outer-label tuning."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_prefix_policy_stage222 as previous
from freq_hrl.domains.transit import native_precision_calibration as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_precision_stage223_v1"
SOURCE_RUN = "native_transit_prefix_policy_stage222_development_20261014_r1"
VIEWS, PARTITIONS, SCOPES, KEYS = previous.VIEWS, previous.PARTITIONS, previous.SCOPES, previous.KEYS


def contract():
    parent = previous.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cohort": parent["cohort"], "outer": parent["outer"],
        "partitions": parent["partitions"], "calibration_factors": {
            "legacy": "reused_stage222_inner2_validation_other2",
            "c2v4": "reused_inner2_predictions_validated_on_all4_draws_of_excluded_prefix",
            "c4v4": "new_inner4_predictions_validated_on_same4_draws_of_excluded_prefix"},
        "final_factors": "same60_training_states_final2_each_complementary_subhalf_or_reused_final4",
        "normalization": "own_training_prefix_features_and_targets_only54_inner60_final",
        "validation": "prefix_held_in_c2v4_c4v4_not_inner_future_held_outer_other4_futures_never_used_for_fitting",
        "guard": parent["policy_constraint"], "solver": parent["policy_solver"],
        "parameters": parent["parameters"], "optimizer": parent["optimizer"],
        "variants": list(learning.VARIANTS), "coverage": "each_scope120_cases60_unique_states_two_subhalf_models_final4_repeated_not_ensembled",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "training_cells": 32,
            "reused_inner2_cells": 32, "reused_final4_cells": 32, "inner4_fits": 320, "final2_fits": 64,
            "analytic_cost_fits": 384, "glm_fits": 384, "wait_fits": 1152,
            "wait_residual_updates": 1152*learning.base.base.STEPS, "calibration_scalar_fits": 384,
            "new_native_episodes": 0, "supervised_decision_updates": 0, "rl_actor_updates": 0},
        "boundary": "precision_factorial_development_original240_states8_futures_no_new_independent_CI",
        "storage": "server_only_models_OOF_records_compact_report_pull"}


def result_path(run_name, *key):
    return previous.result_path(run_name, *key)


def fit_cell(train, held, panels, reused_models, reused_data, partition, direction):
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    train_batch, held_batch = learning.base.base.base.neural.batches(train), learning.base.base.base.neural.batches(held)
    calibration, final2, fit, data = learning.fit_cell_models(train, panels, a, reused_data, batch=train_batch)
    calibration["legacy"] = reused_models["prefix_calibration"]
    final4 = {k: reused_models[k] for k in ("ridge", "wait", "best")}
    scopes = {s: [] for s in SCOPES}
    for subhalf, model in enumerate(final2):
        train_prediction = learning.predict(train, model, final4, calibration, batch=train_batch)
        held_prediction = learning.predict(held, model, final4, calibration, batch=held_batch)
        for s in SCOPES:
            records = learning.base.base.base.records(train if s.startswith("train") else held, panels,
                a if s.endswith("same") else b, b if s.endswith("same") else a,
                train_prediction if s.startswith("train") else held_prediction)
            scopes[s].extend({**r, "deployment_subhalf": subhalf} for r in records)
    groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
    fit.update(calibration=calibration, wait_residual_updates=36*learning.base.base.STEPS,
        analytic_cost_fits=12, glm_fits=12, calibration_scalar_fits=12,
        new_native_episodes=0, supervised_decision_updates=0, rl_actor_updates=0)
    cell = {"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
        "training_groups": groups(train), "held_groups": groups(held), "fit": fit, "scopes": scopes}
    return cell, {"final2": final2, "final4": final4, "calibration": calibration}, data


def qualify(result, key, views):
    view, fold, partition, direction = key
    train, held = previous.previous.previous.split_rows(views[view], fold)
    groups = lambda rs: {(r["root"], r["scene_seed"]) for r in rs}
    cell, fit = result["cell"], result["cell"]["fit"]
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    if (result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract()
            or result["key"] != list(key) or not result["software_qualified"]
            or cell["partition"] != partition or cell["direction"] != direction
            or cell["training_draws"] != list(a) or cell["evaluation_draws"] != list(b)
            or set(map(tuple, cell["training_groups"])) != groups(train) or set(map(tuple, cell["held_groups"])) != groups(held)
            or len(fit["inner_fits"]) != 10 or len(fit["final_fits"]) != 2
            or fit["wait_residual_updates"] != 36*learning.base.base.STEPS
            or fit["analytic_cost_fits"] != 12 or fit["glm_fits"] != 12 or fit["calibration_scalar_fits"] != 12
            or any(fit[k] != 0 for k in ("new_native_episodes", "supervised_decision_updates", "rl_actor_updates"))):
        raise ValueError("Changed precision factorial grouping or fit budget")
    if {tuple(i["held_group"]) for i in fit["inner_fits"]} != groups(train):
        raise ValueError("Missing complete held prefix")
    for f in fit["inner_fits"]+fit["final_fits"]:
        inner = "held_group" in f
        expected = groups(train)-{tuple(f["held_group"])} if inner else groups(train)
        if (set(map(tuple, f["training_groups"])) != expected or set(map(tuple, f["wait_training_groups"])) != expected
                or f["training_states"] != (54 if inner else 60)):
            raise ValueError("Held prefix entered a fit or training size changed")
        for v in learning.WAIT_VARIANTS:
            if f["fits"][v]["parameter_count"] != contract()["parameters"][v] or f["fits"][v]["steps"] != learning.base.base.STEPS:
                raise ValueError("Changed architecture or per-model updates")
    if (any(f["draws"] != list(a) for f in fit["inner_fits"])
            or [f["draws"] for f in fit["final_fits"]] != [list(a[:2]), list(a[2:])]
            or set(fit["calibration"]) != set(learning.CALIBRATIONS)):
        raise ValueError("Changed target precision or used outer evaluation futures")
    for cal in ("c2v4", "c4v4"):
        for v in learning.WAIT_VARIANTS:
            risk, policy = fit["calibration"][cal]["risk"][v], fit["calibration"][cal]["policy"][v]
            if not 0 <= risk["alpha"] <= 1 or risk["candidate_cases"] != 600 or not 0 <= policy["alpha"] <= 1:
                raise ValueError("Changed scalar calibration coverage")
            if (not policy["selected_action_constraints_met"] or set(policy["raw_bounds_by_root"]) != {str(r) for r, _ in groups(train)}
                    or any(policy["selected_by_root"][r][m] > bound[m] for r, bound in policy["raw_bounds_by_root"].items() for m in ("cost", "wait_min"))):
                raise ValueError("OOF selected-action floor violated")
    if set(cell["scopes"]) != set(SCOPES): raise ValueError("Missing evaluation scope")
    for scope, rows in cell["scopes"].items():
        expected = train if scope.startswith("train") else held
        expected_keys = {(*learning.base.base.base.state_key(r), s) for r in expected for s in (0, 1)}
        if (len(rows) != 120 or {(*learning.base.base.base.state_key(r), r["deployment_subhalf"]) for r in rows} != expected_keys
                or any(set(r["decisions"]) != set(learning.VARIANTS) or set(r["gates"]) != set(learning.GATES) for r in rows)):
            raise ValueError("Changed state/subhalf coverage or missing control")


def metrics(rows):
    return {"prediction_cases": len(rows), "unique_states": len({learning.base.base.base.state_key(r) for r in rows}),
        "unique_prefixes": len({(r["root"], r["scene_seed"]) for r in rows}),
        "variants": {v: {"mean_cost": float(np.mean([r["decisions"][v]["cost"] for r in rows])),
            "mean_wait_min": float(np.mean([r["decisions"][v]["wait"] for r in rows])),
            "changed_fraction": float(np.mean([r["decisions"][v]["changed"] for r in rows])),
            "unsafe_fraction": float(np.mean([r["decisions"][v]["unsafe"] for r in rows]))} for v in learning.VARIANTS},
        "wait_heads": {g: {"physical_mass_loss_min": float(np.mean([r["gates"][g]["mass_loss_min"] for r in rows]))} for g in learning.GATES}}


def summarize(results, views):
    if set(results) != set(KEYS): raise ValueError("Complete all32 precision cells before reporting")
    for key, result in results.items(): qualify(result, key, views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for k, r in results.items() if k[0] == view]
        scopes = {s: [r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        if any(len(rs) != 960 or len({learning.base.base.base.state_key(r) for r in rs}) != 120 for rs in scopes.values()):
            raise ValueError("Changed correlated subhalf/future coverage")
        output[view] = {"scopes": {s: metrics(rs) for s, rs in scopes.items()},
            "by_deployment_subhalf": {str(h): metrics([r for r in scopes["held_other"] if r["deployment_subhalf"] == h]) for h in (0, 1)},
            "held_other_by_root": {str(root): metrics([r for r in scopes["held_other"] if r["root"] == root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction": {f"{p}/{d}": metrics([r for c in cells if c["partition"] == p and c["direction"] == d
                for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0, 1)},
            "calibration": {cal: {mode: {v: {"alpha_mean": float(np.mean([c["fit"]["calibration"][cal][mode][v]["alpha"] for c in cells])),
                "cells": [c["fit"]["calibration"][cal][mode][v] for c in cells]} for v in learning.WAIT_VARIANTS}
                for mode in ("risk", "policy")} for cal in learning.CALIBRATIONS}}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "coverage_qualified": True, "views": output, **contract()["budget"], "native_deployment_eligible": False,
        "third_scene_screen_eligible": False}


def reproduce_controls(result, prior):
    mapping = {"f4_glm": "ridge_glm", **{f"f4_raw_{v}": f"ridge_{v}" for v in learning.WAIT_VARIANTS},
        **{f"f4_legacy_{m}_{v}": f"ridge_prefix_{m}_{v}" for m in ("risk", "policy") for v in learning.WAIT_VARIANTS}}
    differences = []
    for view in VIEWS:
        for scope in SCOPES:
            a, b = result["views"][view]["scopes"][scope], prior["views"][view]["scopes"][scope]
            for name, old in mapping.items():
                for metric in ("mean_cost", "mean_wait_min", "changed_fraction", "unsafe_fraction"):
                    differences.append(abs(a["variants"][name][metric]-b["variants"][old][metric]))
                gate = old.removeprefix("ridge_")
                differences.append(abs(a["wait_heads"][name]["physical_mass_loss_min"]-b["wait_heads"][gate]["physical_mass_loss_min"]))
    if max(differences) > 1e-12: raise ValueError("Stage222 primary control failed reproduction")
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
        raise ValueError("Complete Stage222 before precision matching")
    bank, _, registered, _ = previous.previous.previous.source.source.load_data()
    views = previous.previous.previous.source.matched_views(bank["training_data"], registered)
    if args.operation == "fit":
        key = args.view, args.fold, args.partition, args.direction
        source = raw_directory(previous.result_path(SOURCE_RUN, *key))
        old = json.loads((source/"result.json").read_text())
        previous.qualify(old, key, views)
        reused_models = json.loads((source/"models.json").read_text())
        reused_data = {k: np.asarray(v) for k, v in json.loads((source/"inner_oof.json").read_text()).items()}
        train, held = previous.previous.previous.split_rows(views[args.view], args.fold)
        panels = previous.previous.previous.source.source.wait_source.source.diagnostic.load_panels()
        cell, models, data = fit_cell(train, held, panels, reused_models, reused_data, args.partition, args.direction)
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "software_qualified": True, "cell": cell}
        qualify(result, key, views)
        output = result_path(args.run_name, *key)
        raw = raw_directory(output)
        write_json(raw/"models.json", models)
        write_json(raw/"inner_oof.json", {name: {k: v.tolist() for k, v in dat.items()} for name, dat in data.items()})
        write_json(raw/"result.json", result)
        write_json(output, {k: v for k, v in result.items() if k != "cell"} | {"training_cells": 1, "wait_residual_updates": 36*learning.base.base.STEPS})
        print("NATIVE_PRECISION_FIT_COMPLETE", flush=True)
    else:
        results = {k: json.loads((raw_directory(result_path(args.run_name, *k))/"result.json").read_text()) for k in KEYS}
        result = summarize(results, views)
        result["control_reproduction"] = reproduce_controls(result, prior)
        write_json(ROOT/"results"/args.run_name/"report/result.json", result)
        print("NATIVE_PRECISION_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
