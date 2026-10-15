#!/usr/bin/env python3
"""Separate critic target reuse from variance/signal calibration reuse."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_variance_critic_stage243 as previous
from freq_hrl.domains.transit import native_disjoint_critic as learning
from freq_hrl.domains.transit import native_frozen_future as sampling
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_disjoint_critic_stage244_v1"
SOURCE_RUN = "native_transit_variance_critic_stage243_development_20261015_r1"
KEYS, VIEWS, PARTITIONS = previous.KEYS, previous.VIEWS, previous.PARTITIONS
PREFLIGHT, SCOPES, LABELS = previous.PREFLIGHT, previous.SCOPES, previous.LABELS
training, result_path = previous.training, previous.result_path
PRIMARY = "disjoint_variance_group_pooled_safe"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "reference_run": previous.contract()["reference_run"], "refit_run": previous.contract()["refit_run"],
        "cohort": previous.contract()["cohort"], "modes": {m: list(v) for m, v in learning.MODES.items()},
        "arms": list(learning.base.ARMS), "variants": list(learning.VARIANTS), "primary": PRIMARY,
        "A": "old_futures_used_to_train_cached_raw_reference_2_inner_or4_full",
        "B": "matched_half_of_old_outer_unused4_for_inner_or_all4_full_same_training_prefixes_not_held_or_Stage241_futures",
        "inner": "same20_prefix_future_blocks_54states_fit2_draws_validate6held_states_opposite_A2",
        "outer": "same60_training_prefix_states_4draws_then60_outer_held_states",
        "factorial": "target_A_orB_cross_calibration_A_orB_matched_counts_per_estimate_not_matched_unique_raw_plus_critic_draw_union",
        "capacity": "fixed198_features_TSVD_subspace_and_A_scale_A_MSE_sum_RMS_cap_every_mode_and_arm",
        "combined": "B_MSE_recomputed_analytically_and_projected_to_A_ball_same_and_calibration_only_cost_match_cached_A",
        "variance_shrinkage": previous.contract()["pooling"]+";"+previous.contract()["shrinkage"],
        "optimizer": previous.contract()["optimizer"], "grid": learning.strength.CANDIDATES,
        "filters": list(learning.base.FILTERS), "selection": "unchanged_OOF_rules_each_mode_arm_separately_no_joint_mode_selection",
        "gate": previous.contract()["gate"], "anchor": "original", "scopes": list(SCOPES), "labels": list(LABELS),
        "exposure": "old_same_is_A_old_other_is_B_target_fit_exposure_depends_on_mode_not_uniform_across_modes",
        "controls": "all_Stage243_same_models_selections_and9_metric_fields_reproduce_full_matrix_no_refitting_same_mode",
        "metrics": "physical_cost_wait_service_fleet_unsafe_score_MSE_plus_equal_state_feasible_cost_MSE_and_gate_regret",
        "sampling": "shared_state_future_effects_accumulated_before_conditional_MC_SE_for_mixed_OOF_and_full_held_no_training_seed_CI",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "inner_blocks": 640, "critic_subspace_fits": 672,
            "new_component_quadratic_solves": 40320, "new_raw_reference_fits": 0, "new_native_episodes": 0, "rl_actor_updates": 0},
        "storage": "reuse_remote_panels_and_models_pull_pooled_filter_matrix_plus_all_registered_fresh8_aggregate_only",
        "boundary": "reused_development_old_and_Stage241_futures_not_new_confirmation_two_Transit_mixtures_Stage169_reserves_closed"}


def draw_assignment(a, b, half=None):
    if len(a) != 4 or len(b) != 4 or set(a) & set(b): raise ValueError("A/B futures must be disjoint matched banks")
    if half is None: return list(a), list(b), list(b)
    return list(a[2*half:2*half+2]), list(b[2*half:2*half+2]), list(a[2*(1-half):2*(1-half)+2])


def qualify_fit(fit):
    expected = {(m, a, l) for m in learning.MODES if m != "same" for a in learning.base.ARMS for l in learning.strength.lowrank.LEVELS[1:]}
    if len(fit["fits"]) != len(expected) or {(f["mode"], f["arm"], f["level"]) for f in fit["fits"]} != expected:
        raise ValueError("Dropped factorial fit")
    if fit["cached_MSE_max_abs_difference"] > 1e-12: raise ValueError("Cached A basis changed")
    for f in fit["fits"]:
        if (f["optimality"] > 1e-8 or f["norm"] > f["radius"]+1e-10
                or f["loss_after"] > f["loss_before"]+1e-10*max(1., f["loss_before"])):
            raise ValueError("Factorial optimization or matched RMS budget failed")


def fit_cell(run_name, key, views):
    load = lambda run, name: json.loads((raw_directory(result_path(run, *key))/name).read_text())
    old = load(SOURCE_RUN, "aggregate.json")
    previous.qualify(old)
    cached = load(SOURCE_RUN, "models.json")
    reference = load(contract()["reference_run"], "models.json")["reference"]
    refit = load(contract()["refit_run"], "result.json")["cell"]
    references = load(contract()["refit_run"], "models.json")["references"]
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    past, fresh = previous.source.learning.read_panel_cache(previous.source.panel_path(previous.SOURCE_RUN))
    a, b = PARTITIONS[key[2]][key[3]], PARTITIONS[key[2]][1-key[3]]
    lookup = {(tuple(t["removed_group"]), t["half"]): str(i) for i, t in enumerate(refit["trials"])
        if t["removed_group"] and t["half"] is not None}
    blocks, models_cache, probes = [], {}, []
    for group, indices, validation in learning.joint.binding.policy.prefix_folds(train):
        rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
        for half in (0, 1):
            index = lookup[tuple(group), half]
            da, db, dv = draw_assignment(a, b, half)
            data = learning.base.causal_data(rows, references[index])
            val = learning.base.causal_data(val_rows, references[index])
            models, fit = learning.fit_paths(rows, data, learning.base.training_draws(rows, past, da),
                learning.base.training_draws(rows, past, db), cached[index])
            qualify_fit(fit)
            paths = learning.candidates(val, models)
            truth = learning.base.training_draws(val_rows, past, dv)
            curves = {m: {arm: {c: learning.base.curve(val, truth, p) for c, p in path.items()}
                for arm, path in arms.items()} for m, arms in paths.items()}
            blocks.append({"held_group": list(group), "half": half, "scales": cached[index]["combined"]["scalar_path"]["scales"],
                "A": da, "B": db, "validation": dv, "fits": fit, "curves": curves})
            models_cache[index] = models
            probes.append((val_rows, val, paths, da, dv))
            print(f"DISJOINT_CRITIC_INNER_COMPLETE {group}/{half}", flush=True)
    selection, summaries = {}, {}
    for mode in learning.MODES:
        selection[mode], summaries[mode] = learning.base.select([{**block, "curves": block["curves"][mode]} for block in blocks])
    if selection["same"] != old["selection"]: raise ValueError("Stage243 OOF selection changed")
    for arm in ("combined", "split"):
        if (selection["same"][arm] != selection["calibration_only"][arm]
                or selection["target_only"][arm] != selection["disjoint"][arm]):
            raise ValueError("Calibration changed an unweighted OLS control")
    data = learning.base.causal_data(train, reference)
    da, db, _ = draw_assignment(a, b)
    models, full_fit = learning.fit_paths(train, data, learning.base.training_draws(train, past, da),
        learning.base.training_draws(train, past, db), cached["full"])
    qualify_fit(full_fit)
    models_cache["full"] = models
    path = result_path(run_name, *key)
    write_json(raw_directory(path)/"models.json", models_cache)
    frozen = [("oof6", rows, d, learning.predict(paths, selection), same, other) for rows, d, paths, same, other in probes]
    for scope, rows in (("train60", train), ("held60", held)):
        d = data if scope == "train60" else learning.base.causal_data(rows, reference)
        frozen.append((scope, rows, d, learning.predict(learning.candidates(d, models), selection), da, db))
    write_json(raw_directory(path)/"actions.json", [{"scope": scope,
        "states": [list(learning.joint.binding.learned.state_key(r)) for r in rows],
        "choices": {v: p["chosen"].tolist() for v, p in pred.items()}} for scope, rows, _, pred, _, _ in frozen])
    pieces = {s: {label: [] for label in LABELS} for s in SCOPES}
    for scope, rows, d, pred, same, other in frozen:
        for label, draws in previous.label_draws(rows, past, fresh, same, other).items():
            pieces[scope][label].append(learning.metrics(d, draws, pred))
    matrix = {s: {label: learning.aggregate(xs) for label, xs in labels.items()} for s, labels in pieces.items()}
    reproduction = learning.compare_cached(matrix, old["matrix"])
    reproduction["basis_max_abs_difference"] = max(x["fits"]["cached_MSE_max_abs_difference"] for x in blocks+[{"fits": full_fit}])
    write_json(raw_directory(path)/"fits.json", {"blocks": blocks, "full": full_fit, "summaries": summaries})
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "coverage_qualified": True,
        "matrix": matrix, "selection": selection, "full_noise": full_fit["noise"], "full_fit_summary": full_fit["fits"], "reproduction": reproduction}
    qualify(result)
    write_json(raw_directory(path)/"aggregate.json", result)
    write_json(path, {k: v for k, v in result.items() if k not in ("matrix", "full_fit_summary", "full_noise")})
    print(f"DISJOINT_CRITIC_CELL_COMPLETE {key}", flush=True)


def qualify(r):
    if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) not in KEYS or not r["coverage_qualified"]:
        raise ValueError("Changed disjoint critic protocol")
    if set(r["matrix"]) != set(SCOPES): raise ValueError("Dropped prefix scope")
    for scope, labels in r["matrix"].items():
        if set(labels) != set(LABELS): raise ValueError("Dropped future half")
        for stats in labels.values():
            if stats["all"]["comparisons"] != (20 if scope == "oof6" else 1) or stats["all"]["mean_cases"] != (6 if scope == "oof6" else 60):
                raise ValueError("Dropped matched prefix states")
            for piece in (stats["all"], *stats["by_root"].values()):
                if set(piece["variants"]) != set(learning.VARIANTS): raise ValueError("Dropped factorial arm")
    if r["reproduction"]["max_abs_difference"] > 1e-12 or r["reproduction"]["basis_max_abs_difference"] > 1e-12:
        raise ValueError("Cached control changed")


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
    mixed = [r for k, r in results.items() if k[0].startswith("mixed")]
    matrix = merge(mixed)
    _, fresh = previous.source.learning.read_panel_cache(previous.source.panel_path(previous.SOURCE_RUN))
    mc = {}
    for scope in ("oof6", "held60"):
        cells = []
        for key in KEYS:
            if not key[0].startswith("mixed"): continue
            actions = json.loads((raw_directory(result_path(run_name, *key))/"actions.json").read_text())
            cells.append({"blocks": [{"states": [dict(zip(("root", "scene_seed", "probe_index"), state)) for state in b["states"]],
                "choices": b["choices"]} for b in actions if b["scope"] == scope]})
        mc[scope] = sampling.conditional_mc(cells, fresh, variants=learning.VARIANTS)
        for name, stats in mc[scope]["variants"].items():
            if abs(stats["fields"]["cost"]["mean"]-matrix[scope]["fresh8"]["all"]["variants"][name]["delta_cost"]) > 1e-12:
                raise ValueError("Shared-future MC mean differs from registered aggregate")
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "mixed_pooled": matrix, "views": {v: merge([r for k, r in results.items() if k[0] == v]) for v in VIEWS},
        "selections": [{"key": list(k), "selection": r["selection"]} for k, r in results.items()],
        "full_calibration": [{"key": list(k), "noise": r["full_noise"], "fits": r["full_fit_summary"]} for k, r in results.items()],
        "conditional_mc": mc, "reproduction": {"records": sum(r["reproduction"]["records"] for r in results.values()),
            "max_abs_difference": max(r["reproduction"]["max_abs_difference"] for r in results.values())},
        **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}
    write_json(raw_directory(ROOT/"results"/run_name/"report/result.json")/"result.json", payload)
    # Keep complete cell fit diagnostics remote; pull scalar noise summaries, not every coefficient or ledger.
    payload["full_calibration"] = [{"key": x["key"], "noise": x["noise"], "cap_activity": {m: sum(
        f["OLS_budget_clipped"] for f in x["fits"] if f["mode"] == m and f["arm"] == "combined") for m in learning.MODES if m != "same"}}
        for x in payload["full_calibration"]]
    payload["all_registered_fresh8"] = {s: matrix[s]["fresh8"] for s in SCOPES}
    keep = {"raw", PRIMARY.replace("pooled_safe", "half_safe"),
        *(f"{m}_{a}_pooled_safe" for m in learning.MODES for a in learning.base.ARMS if a != "split")}
    def compact(matrix):
        return {s: {label: {"all": {**p["all"], "variants": {v: x for v, x in p["all"]["variants"].items() if v in keep}},
            "by_root": {r: {**piece, "variants": {v: x for v, x in piece["variants"].items() if v in keep}}
                for r, piece in p["by_root"].items()}} for label, p in labels.items()} for s, labels in matrix.items()}
    payload["mixed_pooled"] = compact(matrix)
    payload["views"] = {v: compact(m) for v, m in payload["views"].items()}
    write_json(ROOT/"results"/run_name/"report/result.json", payload)
    print("DISJOINT_CRITIC_REPORT_COMPLETE", flush=True)


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
