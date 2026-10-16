#!/usr/bin/env python3
"""Infer frozen prefix54 heads on the same held60 states as full60 heads."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_local_risk_stage272 as source
from scripts.run_native_transit_prefix_influence_stage259 import read_cell
from freq_hrl.domains.transit import native_held_removal as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_held_removal_stage273_v1"
SOURCE_RUN = "native_transit_local_risk_stage272_20261016_r1"
cache, KEYS, VIEWS, PENALTIES = source.cache, source.KEYS, source.VIEWS, source.PENALTIES
ARMS = source.learning.ARMS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arms": list(ARMS), "penalties": list(PENALTIES), "primary_penalty": source.PRIMARY_PENALTY,
        "comparison": "each_cached_prefix54_and_full60_head_on_identical_outer_held60_features_eligibility_and_assigned_or_opposite16_labels; no_refitting_normalization_radius_or_clipping_changes",
        "matched_size": "for_each_held_query_average_single_model_losses_of_matching_root_regime_deletions_vs_other_deletions; both54_states; then_state_balanced_average; not_error_of_ensembled_predictions",
        "support": "count_distinct_training_prefixes_per_root_regime_before_and_after_deletion; mixed1_to0_vs_single_root2_to1; scenario_metadata_descriptive_only_not_feature_gate_or_selection",
        "attribution": "other_deletion_minus_full60_is_count_plus_composition_reference_not_pure_count; matched_minus_other_is_equal_size_condition_specific_contrast_not_randomized_causal_effect; check_mixture_identity",
        "priors": source.source.contract()["prior"], "bound": source.contract()["bound"],
        "controls": "all28_existing_full60_held_predictions_and_prefix54_retained_predictions_reproduce_Stage272; full_held_error_sums_and_group_MSE_reproduce_same_parent; exact_states_folds_bank_assignments_and_pointwise_caps",
        "frozen": "all_coefficients_normalizers_priors_reference_feature_basis_actions_selectors_and_service_path; no_optimizer_or_policy_choice",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_full_contexts": 192, "cached_prefix_contexts": 1920,
            "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "held_predictions_support_masks_and_cell_ledgers_server_only; pull_scalar_report",
        "boundary": "reused_development_labels_historical_reference_conditioning_and_repeated_models_not_seed_CI_new_prefix_confirmation_or_policy_gain; no_pure_sample_count_identification_or_new_gate; Stage253_248_169_closed; no_default_change"}


def infer(local, residual, cold, priors, z, originals, eligible):
    cached = {"cold_mse": (z@np.asarray(cold["direct_mse"]))[..., 0]}
    cached["cold_mse"][np.arange(len(z)), originals] = 0.
    for j, p in enumerate(source.learning.PRIORS):
        for suffix, field in (("residual", "unbounded"), ("bounded", "bounded")):
            cached[p+"_"+suffix] = priors[..., j]+(z@np.asarray(residual["priors"][p][field]))[..., 0]
    return source.learning.predict(local, z, priors, cached, originals, eligible, local["radius"])


def analyze_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    load = lambda run: json.loads((here(run)/"models.json").read_text())
    local, residual, cold = load(SOURCE_RUN), load(source.SOURCE_RUN), load(source.source.SOURCE_RUN)
    queries = json.loads((here(SOURCE_RUN)/"queries.json").read_text())
    errors = json.loads((here(SOURCE_RUN)/"errors.json").read_text())
    with np.load(here(source.source.SOURCE_RUN)/"predictions.npz") as p: basis = p["basis"].copy()
    with np.load(here(SOURCE_RUN)/"predictions.npz") as p:
        saved = {f'{q["index"]}_{q["scope"]}_{a}': p[f'{q["index"]}_{q["scope"]}_{a}'].copy()
            for q in queries if q["scope"] in ("retained", "held") for a in ARMS}
    reference = read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    full_priors = read_cell(cache.cached.SOURCE_RUN, key, "models.json")
    prefix_priors = read_cell(source.source.PREFIX_RUN, key, "models.json")
    prior_lookup = {(p["bank"], tuple(p["held_group"])): p for p in prefix_priors}
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held};data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    if {(r["root"], r["scene_seed"]) for r in train}&{(r["root"], r["scene_seed"]) for r in held}: raise ValueError("Overlapping outer prefixes")
    x = {s: cache.learning.joint.relative(d["x"][..., :198], d["originals"]) for s, d in data.items()}
    z = {s: np.concatenate((np.ones((*a.shape[:2], 1)), a@basis), axis=-1) for s, a in x.items()}
    raw = {s: cache.learning.joint.relative(d["components"], d["originals"])[..., 1] for s, d in data.items()}
    eligible = {s: (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None]) for s, d in data.items()}
    folds = {tuple(g): np.asarray(ix) for g, ix, _ in cache.learning.joint.binding.policy.prefix_folds(train)}
    if len(local) != 66 or len(residual) != 66 or len(cold) != 66: raise ValueError("Incomplete cached contexts")
    predictions, maximum = [], 0.
    for i, (m, r, c) in enumerate(zip(local, residual, cold)):
        for f in ("bank", "removed", "penalty", "training_draws", "fit_indices"):
            if m[f] != r[f] or m[f] != c[f]: raise ValueError("Changed cached model alignment")
        ds, _ = cache.sampling.learning.draw_assignment(m["bank"], "expanded")
        group = m["removed"]
        indices = np.arange(60) if group is None else folds[tuple(group)]
        if m["training_draws"] != ds or not np.array_equal(m["fit_indices"], indices): raise ValueError("Changed cached fit population")
        prior_models = full_priors[str(m["bank"])] if group is None else prior_lookup[m["bank"], tuple(group)]["models"]
        if group is not None and prior_lookup[m["bank"], tuple(group)]["training_draws"] != ds: raise ValueError("Changed prefix prior bank")
        if any(r["priors"][p]["radius"] != m["radius"] for p in source.learning.PRIORS): raise ValueError("Changed correction radius")
        def priors(scope, ids):
            return np.stack([raw[scope][ids] if p == "raw" else raw[scope][ids]+(x[scope][ids]@np.asarray(prior_models[p]["weights"]["8"]).T)[..., 1]*prior_models[p]["scale"] for p in source.learning.PRIORS], axis=-1)
        cost, _ = infer(m, r, c, priors("held", np.arange(60)), z["held"], data["held"]["originals"], eligible["held"])
        predictions.append(cost)
        scope, label, ids = ("held", "held", np.arange(60)) if group is None else ("retained", "train", indices)
        replay = cost if group is None else infer(m, r, c, priors(label, ids), z[label][ids], data[label]["originals"][ids], eligible[label][ids])[0]
        for a in ARMS: maximum = max(maximum, float(np.max(np.abs(replay[a]-saved[f"{i}_{scope}_{a}"]))))
    if maximum > 1e-12: raise ValueError("Stage272 held/retained predictions changed")
    groups = sorted(folds)
    masks, support = learning.conditions(train, held, groups)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"queries.json", {"held_states": [list(cache.sampling.old_future.learning.state_key(r)) for r in held],
        "removed_groups": groups, "support_prefixes_before": support, "masks": masks})
    np.savez_compressed(directory/"held_predictions.npz", **{a: np.asarray([p[a] for p in predictions]) for a in ARMS})
    print(f"HELD_REMOVAL_PREDICTIONS_FROZEN {key}", flush=True)
    lookup = {(m["bank"], m["penalty"], tuple(m["removed"]) if m["removed"] else None): i for i, m in enumerate(local)}
    old_errors = {(e["bank"], e["penalty"], e["scope"]): e for e in errors if e["removed"] is None and e["scope"].startswith("held_")}
    records, metric_error, mixture, bound_error = [], 0., 0., 0.
    w = source.learning.base.state_weights(eligible["held"]);valid = w.sum(axis=1) > 0
    for bank in (0, 1):
        for penalty in PENALTIES:
            full_i = lookup[bank, penalty, None];prefix_i = [lookup[bank, penalty, g] for g in groups]
            if len(prefix_i) != 10: raise ValueError("Incomplete equal-size prefix comparisons")
            for a in ARMS:
                if a.endswith("_clip"):
                    p = next(p for p in source.learning.PRIORS if a.startswith(p+"_"))
                    for i in (full_i, *prefix_i):
                        delta = np.abs(predictions[i][a]-predictions[i][p+"_prior"])
                        bound_error = max(bound_error, float(np.max(delta[eligible["held"]], initial=0.))-local[i]["radius"])
            for which, ev_bank in (("same", bank), ("opposite", 1-bank)):
                ds, _ = cache.sampling.learning.draw_assignment(ev_bank, "expanded")
                values = cache.learning.base.training_draws(held, panels, ds)[..., 1]
                y = values-values[np.arange(60), :, data["held"]["originals"]][:, :, None]
                target = y.mean(axis=1);metrics = {}
                for a in ARMS:
                    metrics[a], error = learning.paired_stats(predictions[full_i][a], np.asarray([predictions[i][a] for i in prefix_i]), target, eligible["held"], masks)
                    mixture = max(mixture, error)
                    previous = old_errors[bank, penalty, "held_"+which]["metrics"][a]
                    current = metrics[a]["all"]
                    metric_error = max(metric_error, abs(current["full_squared_error_sum"]-previous["squared_error_sum"]), abs(current["state_exposures"]-previous["state_exposures"]))
                records.append({"bank": bank, "penalty": penalty, "future": which, "metrics": metrics})
    if metric_error > 1e-10 or bound_error > 1e-12: raise ValueError("Held baseline replay or local cap failed")
    write_json(directory/"errors.json", records)
    return {"key": list(key), "records": records,
        "support_before": {str(n): int(np.count_nonzero(valid&(support == n))) for n in sorted(set(support.tolist()))},
        "prediction_max_abs_error": maximum, "metric_max_abs_error": metric_error, "mixture_max_abs_error": mixture, "bound_excess": max(bound_error, 0.)}


def report(cells, before):
    output, error = {}, 0.
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    for group, predicate in groups.items():
        selected = [c for c in cells if predicate(c["key"])];records = [r for c in selected for r in c["records"]]
        output[group] = {"support_prefixes_before": {str(n): sum(c["support_before"].get(str(n), 0) for c in selected) for n in (1, 2)}, "banks": {}}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group]["banks"][label] = {}
            for penalty in PENALTIES:
                output[group]["banks"][label][str(penalty)] = {}
                for future in ("same", "opposite"):
                    rs = [r for r in records if r["penalty"] == penalty and r["future"] == future and (bank is None or r["bank"] == bank)]
                    summaries = {a: learning.summarize([r["metrics"][a] for r in rs]) for a in ARMS}
                    previous = before["groups"][group][label][str(penalty)]["held_"+future]
                    for a, s in summaries.items():
                        error = max(error, abs(s["all"]["full60_MSE"]-previous["arms"][a]["MSE"]), abs(s["all"]["state_exposures"]-previous["state_exposures"]))
                    output[group]["banks"][label][str(penalty)][future] = summaries
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or error > 1e-10: raise ValueError("Incomplete Stage273 held replay")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"prediction_max_abs_error": max(c["prediction_max_abs_error"] for c in cells), "cell_metric_max_abs_error": max(c["metric_max_abs_error"] for c in cells),
            "group_metric_max_abs_error": error, "mixture_max_abs_error": max(c["mixture_max_abs_error"] for c in cells), "max_checked_bound_excess": max(c["bound_excess"] for c in cells)},
        "default_changed": False, "native_deployment_eligible": False}


def compact_report(payload):
    def summary(s):
        return {"state_exposures": s["all"]["state_exposures"], "full60_MSE": s["all"]["full60_MSE"],
            "prefix54_MSE_by_removal": {r: v["prefix54_MSE"] for r, v in s.items() if isinstance(v, dict)},
            "matched_minus_other_MSE": s["matched_minus_other_MSE"]}
    return {**payload, "groups": {g: {**group, "banks": {b: {p: {f: {a: summary(s) for a, s in arms.items()} for f, arms in futures.items()}
        for p, futures in penalties.items()} for b, penalties in group["banks"].items()}} for g, group in payload["groups"].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(analyze_cell(directory/("_".join(map(str, key))), key, views, panels));print(f"HELD_REMOVAL_CELL_COMPLETE {key}", flush=True)
    write_json(directory/"cells.json", cells)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    payload = report(cells, before)
    write_json(directory/"aggregate.json", payload);write_json(path, compact_report(payload));print("HELD_REMOVAL_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
