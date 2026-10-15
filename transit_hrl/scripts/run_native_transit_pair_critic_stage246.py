#!/usr/bin/env python3
"""Compare covariance-aware paired-margin regressions under frozen controls."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_feasible_critic_stage245 as previous
from freq_hrl.domains.transit import native_pair_critic as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_pair_critic_stage246_v1"
SOURCE_RUN = "native_transit_feasible_critic_stage245_development_20261015_r1"
fixed = previous.previous
KEYS, VIEWS, PARTITIONS = previous.KEYS, previous.VIEWS, previous.PARTITIONS
PREFLIGHT, SCOPES, LABELS = previous.PREFLIGHT, previous.SCOPES, previous.LABELS
training, result_path = previous.training, previous.result_path
PRIMARY = "pair_gls_pooled_safe"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "reference_run": previous.contract()["reference_run"], "refit_run": previous.contract()["refit_run"],
        "cohort": previous.contract()["cohort"], "objectives": list(learning.OBJECTIVES), "variants": list(learning.VARIANTS), "primary": PRIMARY,
        "data": previous.contract()["data"], "inner": previous.contract()["inner"], "outer": previous.contract()["outer"],
        "capacity": previous.contract()["capacity"], "budget_constraint": previous.contract()["budget_constraint"],
        "controls": "all_Stage245_objectives_models_selected_keys_and_old_metrics_frozen_no_refits",
        "pair_mse": "equal_state_all_feasible_pairs_component_margin_MSE_same_shrunk_B_service_risk_targets",
        "pair_gls": "pair_mse_with_separate_service_risk_inverse_MC_variance_weights",
        "precision": "variance_of_same_future_action_difference_not_sum_of_action_variances_all15_pairs_prefix_pool_normalized_per_head",
        "variance_pool": "same_group_plus_global_prior_half_then_half_with_pair_variance_no_new_strength_parameter",
        "zero_variance": "unit_precision_for_head_with_no_observed_variance_not_zero_tail_risk_claim",
        "component_allocation": previous.contract()["component_allocation"],
        "optimizer": "existing_quadratic_ball_solver_explicit_frozen_A_radius_KKT_1e-8",
        "grid": learning.strength.CANDIDATES, "filters": list(learning.FILTERS),
        "selection": previous.contract()["selection"], "gate": previous.contract()["gate"], "anchor": "original",
        "labels": list(LABELS), "scopes": list(SCOPES), "label_exposure": previous.contract()["label_exposure"],
        "diagnostic": "feasible_pair_sign_error_margin_MSE_changed_harm_rate_and_fleet_MC_events_no_fresh_selection",
        "sampling": previous.contract()["sampling"],
        "budget": {"fit_tasks": 32, "report_tasks": 1, "inner_blocks": 640, "critic_subspace_fits": 672,
            "new_objective_solves": 5376, "new_raw_reference_fits": 0, "new_native_episodes": 0, "rl_actor_updates": 0},
        "storage": previous.contract()["storage"], "boundary": previous.contract()["boundary"]}


def panel_cache():
    source = fixed.previous.source
    return source.learning.read_panel_cache(source.panel_path(fixed.previous.SOURCE_RUN))


def qualify_fit(fit):
    expected = {(o, l) for o in learning.NEW_OBJECTIVES for l in learning.strength.lowrank.LEVELS[1:]}
    if len(fit["fits"]) != len(expected) or {(f["objective"], f["level"]) for f in fit["fits"]} != expected:
        raise ValueError("Dropped paired objective or rank")
    if fit["subspace_max_abs_difference"] > 1e-12: raise ValueError("Frozen subspace changed")
    for f in fit["fits"]:
        if (not f["success"] or f["optimality"] > 1e-8 or f["norm"] > f["radius"]+1e-10
                or f["new_parameters"] != f["retained_rank"]
                or f["loss_after"] > f["loss_before"]+1e-10*max(1., f["loss_before"])):
            raise ValueError("Paired objective optimization or capacity check failed")


def fit_cell(run_name, key, views):
    load = lambda run, name: json.loads((raw_directory(result_path(run, *key))/name).read_text())
    old = load(SOURCE_RUN, "aggregate.json")
    previous.qualify(old)
    cached, ledger = load(SOURCE_RUN, "models.json"), load(SOURCE_RUN, "fits.json")
    old_blocks = {(tuple(b["held_group"]), b["half"]): b for b in ledger["blocks"]}
    reference = load(contract()["reference_run"], "models.json")["reference"]
    refit = load(contract()["refit_run"], "result.json")["cell"]
    references = load(contract()["refit_run"], "models.json")["references"]
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    panels, fresh = panel_cache()
    a, b = PARTITIONS[key[2]][key[3]], PARTITIONS[key[2]][1-key[3]]
    lookup = {(tuple(t["removed_group"]), t["half"]): str(i) for i, t in enumerate(refit["trials"])
        if t["removed_group"] and t["half"] is not None}
    blocks, models_cache, probes = [], {}, []
    for group, indices, validation in learning.joint.binding.policy.prefix_folds(train):
        rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
        for half in (0, 1):
            index, old_block = lookup[tuple(group), half], old_blocks[tuple(group), half]
            da, db, dv = fixed.draw_assignment(a, b, half)
            data, val = learning.base.causal_data(rows, references[index]), learning.base.causal_data(val_rows, references[index])
            models, fit = learning.fit_paths(rows, data, learning.base.training_draws(rows, panels, db), cached[index], old_block["fits"])
            qualify_fit(fit)
            paths, truth = learning.candidates(val, models), learning.base.training_draws(val_rows, panels, dv)
            curves = {o: {c: learning.base.curve(val, truth, p) for c, p in path.items()} for o, path in paths.items()}
            blocks.append({"held_group": list(group), "half": half, "A": da, "B": db, "validation": dv,
                "scales": old_block["scales"], "fits": fit, "curves": curves})
            models_cache[index] = models
            probes.append((val_rows, val, paths, da, dv))
            print(f"PAIR_CRITIC_INNER_COMPLETE {group}/{half}", flush=True)
    selection, summaries = learning.select(blocks)
    if any(selection[o] != old["selection"][o] for o in learning.previous.OBJECTIVES): raise ValueError("Frozen Stage245 selection changed")
    data = learning.base.causal_data(train, reference)
    models, full_fit = learning.fit_paths(train, data, learning.base.training_draws(train, panels, b), cached["full"], ledger["full"])
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
        for label, draws in fixed.previous.label_draws(rows, panels, fresh, same, other).items():
            pieces[scope][label].append(learning.metrics(d, draws, pred))
    matrix = {s: {label: learning.aggregate(xs) for label, xs in labels.items()} for s, labels in pieces.items()}
    reproduction = learning.compare_cached(matrix, old["matrix"])
    write_json(raw_directory(path)/"fits.json", {"blocks": blocks, "full": full_fit, "summaries": summaries})
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "coverage_qualified": True,
        "selection": selection, "matrix": matrix, "full_fits": full_fit, "reproduction": reproduction}
    qualify(result)
    write_json(raw_directory(path)/"aggregate.json", result)
    write_json(path, {k: v for k, v in result.items() if k not in ("matrix", "full_fits")})
    print(f"PAIR_CRITIC_CELL_COMPLETE {key}", flush=True)


def qualify(r):
    if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) not in KEYS or not r["coverage_qualified"]:
        raise ValueError("Changed paired critic protocol")
    if set(r["matrix"]) != set(SCOPES): raise ValueError("Dropped prefix scope")
    for scope, labels in r["matrix"].items():
        if set(labels) != set(LABELS): raise ValueError("Dropped future half")
        for stats in labels.values():
            if stats["all"]["comparisons"] != (20 if scope == "oof6" else 1) or stats["all"]["mean_cases"] != (6 if scope == "oof6" else 60):
                raise ValueError("Dropped matched prefix states")
            for piece in (stats["all"], *stats["by_root"].values()):
                if set(piece["variants"]) != set(learning.VARIANTS): raise ValueError("Dropped paired objective")
    qualify_fit(r["full_fits"])
    if r["reproduction"]["max_abs_difference"] > 1e-12: raise ValueError("Frozen controls changed")


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
    _, fresh = panel_cache()
    mc = {}
    for scope in ("oof6", "held60"):
        cells = []
        for key in KEYS:
            if not key[0].startswith("mixed"): continue
            actions = json.loads((raw_directory(result_path(run_name, *key))/"actions.json").read_text())
            cells.append({"blocks": [{"states": [dict(zip(("root", "scene_seed", "probe_index"), state)) for state in b["states"]],
                "choices": b["choices"]} for b in actions if b["scope"] == scope]})
        mc[scope] = fixed.sampling.conditional_mc(cells, fresh, variants=learning.VARIANTS)
        for name, stats in mc[scope]["variants"].items():
            if abs(stats["fields"]["cost"]["mean"]-matrix[scope]["fresh8"]["all"]["variants"][name]["delta_cost"]) > 1e-12:
                raise ValueError("Conditional MC mean differs from aggregate")
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "mixed_pooled": matrix, "views": {v: merge([r for k, r in results.items() if k[0] == v]) for v in VIEWS},
        "selections": [{"key": list(k), "selection": r["selection"]} for k, r in results.items()],
        "conditional_mc": mc, "full_fit_summary": [{"key": list(k), "fits": r["full_fits"]} for k, r in results.items()],
        "reproduction": {"records": sum(r["reproduction"]["records"] for r in results.values()),
            "max_abs_difference": max(r["reproduction"]["max_abs_difference"] for r in results.values())},
        **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}
    write_json(ROOT/"results"/run_name/"report/result.json", payload)
    print("PAIR_CRITIC_REPORT_COMPLETE", flush=True)


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
