#!/usr/bin/env python3
"""Fit service/risk residual critics using cached paired future panels only."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_fresh_transfer_stage242 as source
from freq_hrl.domains.transit import native_variance_critic as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_variance_critic_stage243_v1"
SOURCE_RUN = "native_transit_fresh_transfer_stage242_development_20261015_r1"
KEYS, VIEWS, PARTITIONS = source.KEYS, source.VIEWS, source.source.prior.PARTITIONS
training, result_path = source.training, source.result_path
SCOPES = ("oof6", "train60", "held60")
LABELS = ("old_same", "old_other", "fresh_half0", "fresh_half1", "fresh8")
PRIMARY = "variance_group_pooled_safe"
PREFLIGHT = ("mixed_a", 0, "even_odd", 0)


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "reference_run": source.source.prior.contract()["reference_run"], "refit_run": source.source.REFIT_RUN,
        "selection_run": source.source.SOURCE_RUN, "cohort": source.contract()["cohort"],
        "arms": list(learning.ARMS), "variants": list(learning.VARIANTS), "primary": PRIMARY,
        "heads": {"service": "cached_nonfleet_cost_delta", "risk": "cached_fleet_cost_delta_not_wait_violation_probability"},
        "variance": "paired_action_minus_original_future_variance_ddof1_divided_by_training_draw_count",
        "pooling": "equal_prefix_pooled_variance_group_half_pooled_then_state_half_group_prior_no_tuned_hyperparameters",
        "shrinkage": "prefix_residual_vector_raw_centred_pseudo_target_signal_energy_from_other_training_prefixes_minus_MC_variance_clamped_at_zero",
        "reliability": "leave_group_out_signal_over_signal_plus_prefix_half_pooled_MC_variance_unchanged_at_inference_no_group_ID_predictor",
        "zero_variance": "all_zero_observed_channel_uses_unit_precision_and_is_reported_as_unidentified_not_certain",
        "capacity": "same198_causal_features_TSVD_basis_k_total_function_df_2k_component_df_all_arms_same_cached_sum_residual_RMS_ball",
        "optimizer": "two_head_convex_GLS_dual_sum_ball_solve_KKT_1e-8_no_gradient_training",
        "anchor": "original_fixed_from_prior_development_not_an_anchor_search",
        "gate": "frozen_cached_raw_wait_gate_no_new_safety_threshold",
        "grid": learning.strength.CANDIDATES, "filters": list(learning.FILTERS),
        "selection": "same_Stage239_pooled_safe_and_half_safe_nonharm_rules_separate_per_arm_no_joint_arm_search",
        "inner": "20_blocks_54states_2old_draws_fit_6held_prefix_states_opposite2old_draws_validate_cached_native_inner_reference",
        "outer": "full60states_4old_draws_fit_cached_native_full_reference_then_train60_and_held60_evaluate",
        "scopes": list(SCOPES), "labels": list(LABELS),
        "old_same": "inner_same2_is_unseen_prefix_but_same_future_indices_full_train_same4_is_fit_exposed",
        "controls": "split_OLS_sum_reproduces_cached_all_MSE_all_ranks_and_strengths_combined_OOF_curves_selections_exact",
        "fresh": "Stage241_eight_draw_cache_already_inspected_development_only_never_used_for_fit_or_selection",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "inner_blocks": 640, "new_raw_reference_fits": 0,
            "new_native_episodes": 0, "raw_wait_updates": 0, "rl_actor_updates": 0,
            "critic_subspace_fits": 672, "new_component_GLS_solves": 8064},
        "storage": "reuse_remote_Stage242_npz_no_panel_or_model_download_small_aggregate_only",
        "boundary": "240_reused_states_40_prefixes_two_Transit_mixtures_not_new_seed_CI_or_end_to_end_HRL_reserved_Stage169_closed"}


def label_draws(rows, old, fresh, same, other):
    past = learning.training_draws(rows, old, range(8))
    future = learning.training_draws(rows, fresh, range(8))
    return {"old_same": past[:, same], "old_other": past[:, other],
        "fresh_half0": future[:, :4], "fresh_half1": future[:, 4:], "fresh8": future}


def qualify_fit(fit):
    expected = {(a, l) for a in learning.ARMS for l in learning.strength.lowrank.LEVELS[1:]}
    if {(f["arm"], f["level"]) for f in fit["fits"]} != expected or len(fit["fits"]) != len(expected):
        raise ValueError("Dropped critic arm or rank")
    if fit["combined_max_abs_difference"] > 1e-12: raise ValueError("Combined control changed")
    for f in fit["fits"]:
        if (f["optimality"] > 1e-8 or f["norm"] > f["radius"]+1e-10
                or f["total_cost_function_df"] != f["retained_rank"]
                or f["component_function_df"] != 2*f["retained_rank"]
                or f["loss_after"] > f["loss_before"]+1e-10*max(1., f["loss_before"])):
            raise ValueError("Critic fit exceeds objective or capacity contract")


def fit_cell(run_name, key, views):
    load = lambda run, name: json.loads((raw_directory(result_path(run, *key))/name).read_text())
    selected = load(source.source.SOURCE_RUN, "result.json")
    source.source.prior.previous.qualify(selected, key, views)
    refit = load(source.source.REFIT_RUN, "result.json")["cell"]
    references = load(source.source.REFIT_RUN, "models.json")["references"]
    cached = load(source.source.SOURCE_RUN, "models.json")
    reference = load(contract()["reference_run"], "models.json")["reference"]
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    old, fresh = source.learning.read_panel_cache(source.panel_path(SOURCE_RUN))
    a, b = selected["cell"]["training_draws"], selected["cell"]["evaluation_draws"]
    lookup = {(tuple(t["removed_group"]), t["half"]): str(i) for i, t in enumerate(refit["trials"])
        if t["removed_group"] is not None and t["half"] is not None}
    blocks, cache, predictions = [], {}, []
    curve_difference, comparisons = 0., 0
    for group, indices, validation in learning.joint.binding.policy.prefix_folds(train):
        rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
        for half in (0, 1):
            index = lookup[tuple(group), half]
            data = learning.causal_data(rows, references[index])
            val = learning.causal_data(val_rows, references[index])
            same, other = a[2*half:2*half+2], a[2*(1-half):2*(1-half)+2]
            models, fit = learning.fit_paths(rows, data, learning.training_draws(rows, old, same), cached[index]["original"]["all_mse"])
            qualify_fit(fit)
            paths = {arm: learning.candidates(val, models[arm]) for arm in learning.ARMS}
            truth = learning.training_draws(val_rows, old, other)
            curves = {arm: {c: learning.curve(val, truth, p) for c, p in path.items()} for arm, path in paths.items()}
            expected = next(x for x in selected["cell"]["blocks"] if x["held_group"] == list(group) and x["half"] == half)
            for c in learning.strength.CANDIDATES:
                actual, previous = curves["combined"][c], expected["curves"]["original"]["all_mse"][c]
                if actual["changed_count"] != previous["changed_count"]: raise ValueError("Cached combined choices changed")
                for root, fields in actual["by_root"].items():
                    curve_difference = max(curve_difference, *(abs(x-previous["by_root"][root][f]) for f, x in fields.items()))
                comparisons += len(val_rows)
            if curve_difference > 1e-12: raise ValueError("Cached combined OOF outcomes changed")
            blocks.append({"held_group": list(group), "half": half, "scales": cached[index]["original"]["all_mse"]["scales"],
                "training_draws": same, "validation_draws": other, "fits": fit, "curves": curves})
            predictions.append((val_rows, val, paths, same, other))
            cache[index] = models
            print(f"VARIANCE_CRITIC_INNER_COMPLETE {group}/{half}", flush=True)
    selection, summaries = learning.select(blocks)
    for f in learning.FILTERS:
        if selection["combined"][f] != selected["cell"]["selection"]["original"]["all_mse"][f]["candidate"]:
            raise ValueError("Combined frozen OOF selection changed")
        if selection["split"][f] != selection["combined"][f]: raise ValueError("Unweighted split OLS selection changed")
    data = learning.causal_data(train, reference)
    models, fit = learning.fit_paths(train, data, learning.training_draws(train, old, a), cached["full"]["original"]["all_mse"])
    qualify_fit(fit)
    cache["full"] = models
    path = result_path(run_name, *key)
    write_json(raw_directory(path)/"models.json", cache)
    frozen = []
    for rows, d, paths, same, other in predictions:
        pred = {"raw": paths["combined"]["zero"],
            **{f"{arm}_{f}": paths[arm][selection[arm][f] or "zero"] for arm in learning.ARMS for f in learning.FILTERS}}
        frozen.append(("oof6", rows, d, pred, same, other))
    for scope, rows in (("train60", train), ("held60", held)):
        d = data if scope == "train60" else learning.causal_data(rows, reference)
        frozen.append((scope, rows, d, learning.predict(d, models, selection), a, b))
    # Seal every fit, selection and action before evaluating independent futures.
    write_json(raw_directory(path)/"actions.json", [{"scope": scope,
        "states": [list(learning.joint.binding.learned.state_key(r)) for r in rows],
        "choices": {v: p["chosen"].tolist() for v, p in pred.items()}} for scope, rows, _, pred, _, _ in frozen])
    pieces = {s: {label: [] for label in LABELS} for s in SCOPES}
    for scope, rows, d, pred, same, other in frozen:
        for label, draws in label_draws(rows, old, fresh, same, other).items():
            pieces[scope][label].append(learning.metrics(d, draws, pred))
    matrix = {s: {label: learning.aggregate(xs) for label, xs in labels.items()} for s, labels in pieces.items()}
    write_json(raw_directory(path)/"fits.json", {"blocks": blocks, "full": fit, "summaries": summaries})
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "coverage_qualified": True,
        "selection": selection, "matrix": matrix, "full_noise": fit["noise"], "reproduction": {"combined_curve_records": comparisons,
            "combined_curve_max_abs_difference": curve_difference,
            "split_OLS_max_abs_difference": max(f["fits"]["combined_max_abs_difference"] for f in blocks+[{"fits": fit}])}}
    qualify(result)
    write_json(raw_directory(path)/"aggregate.json", result)
    write_json(path, {k: v for k, v in result.items() if k != "matrix"})
    print(f"VARIANCE_CRITIC_CELL_COMPLETE {key}", flush=True)


def qualify(r):
    if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) not in KEYS or not r["coverage_qualified"]:
        raise ValueError("Changed variance critic protocol")
    if set(r["matrix"]) != set(SCOPES): raise ValueError("Dropped prefix scope")
    for scope, labels in r["matrix"].items():
        if set(labels) != set(LABELS): raise ValueError("Dropped future half")
        for stats in labels.values():
            p = stats["all"]
            if p["comparisons"] != (20 if scope == "oof6" else 1) or p["mean_cases"] != (6 if scope == "oof6" else 60):
                raise ValueError("Dropped states or prefix blocks")
            for piece in (p, *stats["by_root"].values()):
                if set(piece["variants"]) != set(learning.VARIANTS): raise ValueError("Dropped critic ablation")
    if r["reproduction"]["combined_curve_records"] != 2040 or max(r["reproduction"][f] for f in (
            "combined_curve_max_abs_difference", "split_OLS_max_abs_difference")) > 1e-12:
        raise ValueError("Cached all-MSE control changed")


def merge(results):
    def piece(xs):
        return {"cells": len(xs), "variants": {v: {f: float(np.mean([x["variants"][v][f] for x in xs]))
            for f in xs[0]["variants"][v]} for v in learning.VARIANTS}}
    return {s: {label: {"all": piece([r["matrix"][s][label]["all"] for r in results]),
        "by_root": {root: piece([r["matrix"][s][label]["by_root"][root] for r in results if root in r["matrix"][s][label]["by_root"]])
            for root in sorted({root for r in results for root in r["matrix"][s][label]["by_root"]})}}
        for label in LABELS} for s in SCOPES}


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
    else:
        results = {key: json.loads((raw_directory(result_path(args.run_name, *key))/"aggregate.json").read_text()) for key in KEYS}
        for r in results.values(): qualify(r)
        report = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
            "mixed_pooled": merge([r for k, r in results.items() if k[0].startswith("mixed")]),
            "views": {v: merge([r for k, r in results.items() if k[0] == v]) for v in VIEWS},
            "selections": [{"key": list(k), "selection": r["selection"]} for k, r in results.items()],
            "full_noise_by_cell": [{"key": list(k), "noise": r["full_noise"]} for k, r in results.items()],
            "reproduction": {"combined_curve_records": sum(r["reproduction"]["combined_curve_records"] for r in results.values()),
                **{f: max(r["reproduction"][f] for r in results.values()) for f in ("combined_curve_max_abs_difference", "split_OLS_max_abs_difference")}},
            **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}
        write_json(ROOT/"results"/args.run_name/"report/result.json", report)
        print("VARIANCE_CRITIC_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
