#!/usr/bin/env python3
"""Compare feasible and ranking objectives on fixed independent-future critics."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_disjoint_critic_stage244 as previous
from freq_hrl.domains.transit import native_feasible_critic as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_feasible_critic_stage245_v1"
SOURCE_RUN = "native_transit_disjoint_critic_stage244_development_20261015_r1"
KEYS, VIEWS, PARTITIONS = previous.KEYS, previous.VIEWS, previous.PARTITIONS
PREFLIGHT, SCOPES, LABELS = previous.PREFLIGHT, previous.SCOPES, previous.LABELS
training, result_path = previous.training, previous.result_path
PRIMARY = "rank_regret_pooled_safe"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "reference_run": previous.contract()["reference_run"], "refit_run": previous.contract()["refit_run"],
        "cohort": previous.contract()["cohort"], "objectives": list(learning.OBJECTIVES), "variants": list(learning.VARIANTS), "primary": PRIMARY,
        "data": "fixed_Stage244_B_target_and_B_variance_signal_calibration_2_inner_4_full_draws_no_extra_labels",
        "inner": previous.contract()["inner"], "outer": previous.contract()["outer"],
        "capacity": "same198_features_same_TSVD_subspace_same_k_cost_sum_dimensions_cached_head_difference_frozen",
        "budget_constraint": "explicit_A_MSE_RMS_radius_not_the_smaller_cached_BB_initial_norm",
        "all_mse": "frozen_Stage244_disjoint_variance_group_component_GLS_no_refit",
        "feasible_mse": "equal_state_feasible_nonanchor_component_MSE_frozen_B_precision_and_shrunk_service_risk_targets",
        "rank_regret": "same_shrunk_physical_service_plus_risk_target_gap_weighted_feasible_pair_logistic_loss_fixed_A_scale",
        "component_allocation": "each_new_sum_update_split_equally_between_heads_cached_head_difference_unchanged_not_independent_head_retraining",
        "optimizer": "existing_convex_quadratic_or_logistic_dual_ball_solver_explicit_radius_KKT_1e-8",
        "grid": learning.strength.CANDIDATES, "filters": list(learning.FILTERS),
        "selection": "unchanged_prefix_future_OOF_rules_each_objective_separately_no_joint_objective_search",
        "gate": previous.contract()["gate"], "anchor": "original",
        "labels": list(LABELS), "scopes": list(SCOPES),
        "label_exposure": "OOF_old_same_inner_A2_old_other_opposite_A2_full_old_same_A4_old_other_B4_B_is_fit_exposed_only_on_training_prefixes",
        "diagnostic": "root401_cost_wait_and_feasible_MSE_diagnostic_only_no_root_ID_parameters_or_posthoc_gate",
        "controls": "all_Stage244_BB_GLS_OOF_selected_keys_and_raw_all_mse_metrics_reproduce",
        "sampling": previous.contract()["sampling"],
        "budget": {"fit_tasks": 32, "report_tasks": 1, "inner_blocks": 640, "critic_subspace_fits": 672,
            "new_objective_solves": 5376, "new_raw_reference_fits": 0, "new_native_episodes": 0, "rl_actor_updates": 0},
        "storage": "reuse_remote_panels_and_models_pull_small_aggregate_only",
        "boundary": "reused_development_futures_not_new_confirmation_or_training_seed_CI_Stage169_reserves_closed"}


def qualify_fit(fit):
    expected = {(o, l) for o in learning.OBJECTIVES[1:] for l in learning.strength.lowrank.LEVELS[1:]}
    if len(fit["fits"]) != len(expected) or {(f["objective"], f["level"]) for f in fit["fits"]} != expected:
        raise ValueError("Dropped decision objective or rank")
    if fit["subspace_max_abs_difference"] > 1e-12: raise ValueError("Frozen subspace changed")
    for f in fit["fits"]:
        if (not f["success"] or f["optimality"] > 1e-8 or f["all_mse_control_optimality"] > 1e-8
                or f["norm"] > f["radius"]+1e-10 or f["new_parameters"] != f["retained_rank"]
                or f["loss_after"] > f["loss_before"]+1e-10*max(1., f["loss_before"])):
            raise ValueError("Decision objective optimization or capacity check failed")


def fit_cell(run_name, key, views):
    load = lambda run, name: json.loads((raw_directory(result_path(run, *key))/name).read_text())
    old = load(SOURCE_RUN, "aggregate.json")
    previous.qualify(old)
    cached = load(SOURCE_RUN, "models.json")
    reference = load(contract()["reference_run"], "models.json")["reference"]
    refit = load(contract()["refit_run"], "result.json")["cell"]
    references = load(contract()["refit_run"], "models.json")["references"]
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    panels, fresh = previous.previous.source.learning.read_panel_cache(previous.previous.source.panel_path(previous.previous.SOURCE_RUN))
    a, b = PARTITIONS[key[2]][key[3]], PARTITIONS[key[2]][1-key[3]]
    lookup = {(tuple(t["removed_group"]), t["half"]): str(i) for i, t in enumerate(refit["trials"])
        if t["removed_group"] and t["half"] is not None}
    blocks, models_cache, probes = [], {}, []
    for group, indices, validation in learning.joint.binding.policy.prefix_folds(train):
        rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
        for half in (0, 1):
            index = lookup[tuple(group), half]
            da, db, dv = previous.draw_assignment(a, b, half)
            data, val = learning.base.causal_data(rows, references[index]), learning.base.causal_data(val_rows, references[index])
            models, fit = learning.fit_paths(rows, data, learning.base.training_draws(rows, panels, db), cached[index])
            qualify_fit(fit)
            paths = learning.candidates(val, models)
            truth = learning.base.training_draws(val_rows, panels, dv)
            curves = {o: {c: learning.base.curve(val, truth, p) for c, p in path.items()} for o, path in paths.items()}
            blocks.append({"held_group": list(group), "half": half, "A": da, "B": db, "validation": dv,
                "scales": cached[index]["same"]["combined"]["scalar_path"]["scales"], "fits": fit, "curves": curves})
            models_cache[index] = models
            probes.append((val_rows, val, paths, da, dv))
            print(f"FEASIBLE_CRITIC_INNER_COMPLETE {group}/{half}", flush=True)
    selection, summaries = learning.select(blocks)
    if selection["all_mse"] != old["selection"]["disjoint"]["variance_group"]: raise ValueError("Frozen BB GLS selection changed")
    data = learning.base.causal_data(train, reference)
    models, full_fit = learning.fit_paths(train, data, learning.base.training_draws(train, panels, b), cached["full"])
    qualify_fit(full_fit)
    models_cache["full"] = models
    path = result_path(run_name, *key)
    write_json(raw_directory(path)/"models.json", models_cache)
    frozen = [("oof6", rows, d, learning.predict(paths, selection), same, other) for rows, d, paths, same, other in probes]
    for scope, rows in (("train60", train), ("held60", held)):
        d = data if scope == "train60" else learning.base.causal_data(rows, reference)
        frozen.append((scope, rows, d, learning.predict(learning.candidates(d, models), selection), a, b))
    write_json(raw_directory(path)/"actions.json", [{"scope": scope,
        "states": [list(learning.joint.binding.learned.state_key(r)) for r in rows],
        "choices": {v: p["chosen"].tolist() for v, p in pred.items()}} for scope, rows, _, pred, _, _ in frozen])
    pieces = {s: {label: [] for label in LABELS} for s in SCOPES}
    for scope, rows, d, pred, same, other in frozen:
        for label, draws in previous.previous.label_draws(rows, panels, fresh, same, other).items():
            pieces[scope][label].append(learning.previous.metrics(d, draws, pred))
    matrix = {s: {label: learning.aggregate(xs) for label, xs in labels.items()} for s, labels in pieces.items()}
    reproduction = learning.compare_cached(matrix, old["matrix"])
    write_json(raw_directory(path)/"fits.json", {"blocks": blocks, "full": full_fit, "summaries": summaries})
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "coverage_qualified": True,
        "selection": selection, "matrix": matrix, "full_fits": full_fit, "reproduction": reproduction}
    qualify(result)
    write_json(raw_directory(path)/"aggregate.json", result)
    write_json(path, {k: v for k, v in result.items() if k not in ("matrix", "full_fits")})
    print(f"FEASIBLE_CRITIC_CELL_COMPLETE {key}", flush=True)


def qualify(r):
    if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) not in KEYS or not r["coverage_qualified"]:
        raise ValueError("Changed feasible critic protocol")
    if set(r["matrix"]) != set(SCOPES): raise ValueError("Dropped prefix scope")
    for scope, labels in r["matrix"].items():
        if set(labels) != set(LABELS): raise ValueError("Dropped future half")
        for stats in labels.values():
            if stats["all"]["comparisons"] != (20 if scope == "oof6" else 1) or stats["all"]["mean_cases"] != (6 if scope == "oof6" else 60):
                raise ValueError("Dropped matched prefix states")
            for piece in (stats["all"], *stats["by_root"].values()):
                if set(piece["variants"]) != set(learning.VARIANTS): raise ValueError("Dropped decision objective")
    qualify_fit(r["full_fits"])
    if r["reproduction"]["max_abs_difference"] > 1e-12: raise ValueError("Frozen control changed")


def merge(results):
    def piece(xs):
        return {"cells": len(xs), "variants": {v: {f: float(np.mean([x["variants"][v][f] for x in xs]))
            for f in xs[0]["variants"][v]} for v in learning.VARIANTS}}
    return {s: {label: {"all": piece([r["matrix"][s][label]["all"] for r in results]),
        "by_root": {root: piece([r["matrix"][s][label]["by_root"][root] for r in results if root in r["matrix"][s][label]["by_root"]])
            for root in sorted({root for r in results for root in r["matrix"][s][label]["by_root"]})}}
        for label in LABELS} for s in SCOPES}


def report(run_name):
    results = {k: json.loads((raw_directory(result_path(run_name, *k))/"aggregate.json").read_text()) for k in KEYS}
    for r in results.values(): qualify(r)
    matrix = merge([r for k, r in results.items() if k[0].startswith("mixed")])
    _, fresh = previous.previous.source.learning.read_panel_cache(previous.previous.source.panel_path(previous.previous.SOURCE_RUN))
    mc = {}
    for scope in ("oof6", "held60"):
        cells = []
        for key in KEYS:
            if not key[0].startswith("mixed"): continue
            actions = json.loads((raw_directory(result_path(run_name, *key))/"actions.json").read_text())
            cells.append({"blocks": [{"states": [dict(zip(("root", "scene_seed", "probe_index"), state)) for state in b["states"]],
                "choices": b["choices"]} for b in actions if b["scope"] == scope]})
        mc[scope] = previous.sampling.conditional_mc(cells, fresh, variants=learning.VARIANTS)
        for name, stats in mc[scope]["variants"].items():
            if abs(stats["fields"]["cost"]["mean"]-matrix[scope]["fresh8"]["all"]["variants"][name]["delta_cost"]) > 1e-12:
                raise ValueError("Conditional MC mean differs from registered aggregate")
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "mixed_pooled": matrix, "views": {v: merge([r for k, r in results.items() if k[0] == v]) for v in VIEWS},
        "selections": [{"key": list(k), "selection": r["selection"]} for k, r in results.items()],
        "conditional_mc": mc, "full_fit_summary": [{"key": list(k), "fits": r["full_fits"]} for k, r in results.items()],
        "reproduction": {"records": sum(r["reproduction"]["records"] for r in results.values()),
            "max_abs_difference": max(r["reproduction"]["max_abs_difference"] for r in results.values())},
        **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}
    write_json(ROOT/"results"/run_name/"report/result.json", payload)
    print("FEASIBLE_CRITIC_REPORT_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("fit", "report"), required=True)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--fold", type=int, choices=(0, 1))
    p.add_argument("--partition", choices=PARTITIONS)
    p.add_argument("--direction", type=int, choices=(0, 1))
    args = p.parse_args()
    if args.operation == "fit":
        bank, _, registered, _ = training.cohort.source.source.load_data()
        views = training.cohort.source.matched_views(bank["training_data"], registered)
        fit_cell(args.run_name, (args.view, args.fold, args.partition, args.direction), views)
    else: report(args.run_name)


if __name__ == "__main__": main()
