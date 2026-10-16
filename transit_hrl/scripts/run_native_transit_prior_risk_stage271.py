#!/usr/bin/env python3
"""Keep frozen risk priors, fit residuals, and isolate prior span from shrinkage."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_shared_risk_stage270 as source
from scripts.run_native_transit_prefix_influence_stage259 import read_cell
from freq_hrl.domains.transit import native_prior_risk as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_prior_risk_stage271_v1"
SOURCE_RUN = "native_transit_shared_risk_stage270_20261016_r1"
PREFIX_RUN = "native_transit_training_count_stage258_20261016_r1"
cache, KEYS, VIEWS, SCOPES = source.cache, source.KEYS, source.VIEWS, source.SCOPES


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL, "prefix_run": PREFIX_RUN,
        "cohort": source.contract()["cohort"], "priors": list(learning.PRIORS), "arms": list(learning.ARMS),
        "representation": "verbatim_Stage270_rank8_basis_intercept_cold_coefficients_and_queries; no_cold_refit",
        "penalties": list(source.learning.PENALTIES), "primary_penalty": source.learning.PRIMARY_PENALTY,
        "prior": "raw_reference_or_cached_free_old_shrunk_rank8_alpha1_risk_slice; full60_Stage251_for_full_fit_prefix54_Stage258_for_excluded_fit; no_selector_change_or_new_prior_fit",
        "fit": "same_state_balanced_physical_ridge_on_assigned16_mean_minus_prior; same_trainable_capacity_as_cold; unbounded_and_convex_RMS_constrained_corrections; zero_correction_reproduces_prior",
        "radius": "existing_training_independent_historical_rank8_old_radius_times_cached_physical_scale; same_extra_correction_budget_for_all3_priors; no_query_or_evaluation_rescaling; not_original_joint_service+risk_cap_or_global_safety_bound",
        "diagnostic": "training_only_weighted_OLS_projection_of_frozen_prior; prior_residual_minus_cached_cold_equals_outside_span_prior_plus_ridge_shrink_return; outside_span_added_to_cold_is_frozen_information_control; preserve_signed_cross_term",
        "controls": "reproduce_Stage270_raw_and_cold_errors_all_cells_banks_penalties_scopes_groups; identical_states_queries_and_assignments; prefix54_current_bank_excluded_labels_never_enter_prior_or_residual_fit; historical_reference_remains_conditioned",
        "evaluation": source.contract()["evaluation"], "frozen": "service_predictions_raw_reference_feature_basis_actions_and_selectors; only_risk_label_error_no_native_control",
        "budget": {"fit_tasks": 1, "cached_cells": 32, "cached_cold_models": 2112, "new_residual_ridge_fits": 6336,
            "constrained_paths": 6336, "prior_projection_solves": 2112, "new_prior_or_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "models_predictions_projection_coefficients_and_all_error_ledgers_server_only; pull_compact_scalar_report",
        "boundary": "development_label_prediction_not_new_prefix_confirmation_seed_CI_policy_gain_or_proven_no_harm; learned_priors_share_assigned_training_futures; no_selection_or_default_change; Stage253_248_169_closed"}


def cached_cell(directory, key, views, panels):
    old_dir = ROOT/"results"/SOURCE_RUN/"report_raw"/("_".join(map(str, key)))
    old = json.loads((old_dir/"models.json").read_text())
    queries = json.loads((old_dir/"queries.json").read_text())
    with np.load(old_dir/"predictions.npz") as saved:
        basis = saved["basis"].copy()
        old_costs = {k: saved[k].copy() for k in saved.files if k.endswith("_direct_mse_cost") or k.endswith("_raw_reference_cost")}
    previous = json.loads((old_dir/"errors.json").read_text())
    reference = read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    full = read_cell(cache.cached.SOURCE_RUN, key, "models.json")
    prefix = read_cell(PREFIX_RUN, key, "models.json")
    lookup = {(p["bank"], tuple(p["held_group"])): p for p in prefix}
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held};data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    x = {s: cache.learning.joint.relative(d["x"][..., :198], d["originals"]) for s, d in data.items()}
    z = {s: np.concatenate((np.ones((*a.shape[:2], 1)), a@basis), axis=-1) for s, a in x.items()}
    raw = {s: cache.learning.joint.relative(d["components"], d["originals"])[..., 1] for s, d in data.items()}
    eligible = {s: (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None]) for s, d in data.items()}
    assignment = {tuple(g): ix for g, ix, _ in cache.learning.joint.binding.policy.prefix_folds(train)}
    radius_values = [next(f["old_radius"] for f in p["fit"]["fits"] if f["level"] == "8") for p in prefix]
    if len(old) != 66 or len(prefix) != 20 or len(set(radius_values)) != 1: raise ValueError("Changed cached prior coverage/radius")
    new, predictions, projection_cache, arrays = [], [], {}, {}
    max_error = 0.
    for index, m in enumerate(old):
        bank, group, indices = m["bank"], m["removed"], np.asarray(m["fit_indices"])
        ds, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
        expected_indices = np.arange(60) if group is None else np.asarray(assignment[tuple(group)])
        if m["training_draws"] != ds or not np.array_equal(indices, expected_indices): raise ValueError("Changed cold fit population/bank")
        saved_prior = full[str(bank)] if group is None else lookup[bank, tuple(group)]["models"]
        if group is not None and lookup[bank, tuple(group)]["training_draws"] != ds: raise ValueError("Changed prefix prior bank")
        scale = saved_prior["shrunk"]["scale"]
        if any(saved_prior[p]["scale"] != scale for p in ("free_old", "shrunk")): raise ValueError("Changed physical prior scales")
        radius = radius_values[0]*scale
        values = cache.learning.base.training_draws(train, panels, ds)[..., 1]
        y = values-values[np.arange(60), :, data["train"]["originals"]][:, :, None]
        priors = {s: {"raw": raw[s], **{p: raw[s]+(x[s]@np.asarray(saved_prior[p]["weights"]["8"]).T)[..., 1]*scale for p in ("free_old", "shrunk")}} for s in rows}
        fitted = {}
        for p in learning.PRIORS:
            identity = bank, tuple(group) if group else None, p
            if identity not in projection_cache:
                projection_cache[identity] = learning.project(z["train"][indices], priors["train"][p][indices], eligible["train"][indices])
            fitted[p] = learning.fit(z["train"][indices], y[indices], priors["train"][p][indices], eligible["train"][indices], m["penalty"], radius)
        new.append({"bank": bank, "removed": group, "penalty": m["penalty"], "training_draws": ds, "fit_indices": indices,
            "priors": fitted, "projections": {p: projection_cache[bank, tuple(group) if group else None, p] for p in learning.PRIORS}})
        for q in (q for q in queries if q["index"] == index):
            label, ids, scope = q["label"], np.asarray(q["ids"]), q["scope"]
            mask = eligible[label][ids]
            if not np.array_equal(q["eligible"], mask): raise ValueError("Changed eligible query actions")
            cold = (z[label][ids]@np.asarray(m["direct_mse"]))[..., 0];cold[np.arange(len(ids)), data[label]["originals"][ids]] = 0.
            max_error = max(max_error, float(np.max(np.abs(cold-old_costs[f"{index}_{scope}_direct_mse_cost"]))),
                float(np.max(np.abs(raw[label][ids]-old_costs[f"{index}_{scope}_raw_reference_cost"]))))
            costs, responses = {"cold_mse": cold}, {}
            for p in learning.PRIORS:
                pred, parts = learning.predict(z[label][ids], priors[label][p][ids], np.asarray(m["direct_mse"]), fitted[p],
                    projection_cache[bank, tuple(group) if group else None, p], data[label]["originals"][ids])
                max_error = max(max_error, parts["identity_error"])
                costs.update({p+"_"+a: value for a, value in pred.items()})
                responses[p] = learning.response_stats(parts, pred["prior"], pred["bounded"], mask, radius)
            predictions.append({"index": index, "scope": scope, "label": label, "ids": ids, "eligible": mask, "costs": costs, "responses": responses})
            arrays.update({f"{index}_{scope}_{arm}": c for arm, c in costs.items()})
    if max_error > 1e-10 or len(projection_cache) != 66: raise ValueError("Cached prediction/decomposition replay failed")
    directory.mkdir(parents=True, exist_ok=True);write_json(directory/"models.json", new)
    np.savez_compressed(directory/"predictions.npz", **arrays)
    print(f"PRIOR_RISK_PREDICTIONS_FROZEN {key}", flush=True)
    records, replay_error = [], 0.
    before = {(p["bank"], tuple(p["removed"]) if p["removed"] else None, p["penalty"], p["scope"]): p for p in previous}
    for q in predictions:
        m = old[q["index"]];rs = [rows[q["label"]][i] for i in q["ids"]];originals = data[q["label"]]["originals"][q["ids"]]
        for which, bank in (("same", m["bank"]), ("opposite", 1-m["bank"])):
            ds, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws(rs, panels, ds)[..., 1]
            y = values-values[np.arange(len(rs)), :, originals][:, :, None]
            metrics = {a: source.learning.metrics(cost, None, y, q["eligible"], np.asarray(m["amplitudes"])) for a, cost in q["costs"].items()}
            scope = q["scope"]+"_"+which
            prior_record = before[m["bank"], tuple(m["removed"]) if m["removed"] else None, m["penalty"], scope]
            for actual, expected in (("raw_prior", "raw_reference"), ("cold_mse", "direct_mse")):
                for f, v in metrics[actual].items(): replay_error = max(replay_error, abs(v-prior_record["metrics"][expected][f]))
            records.append({"bank": m["bank"], "penalty": m["penalty"], "scope": scope, "metrics": metrics, "responses": q["responses"]})
    if replay_error > 1e-10: raise ValueError("Stage270 errors changed")
    write_json(directory/"errors.json", records)
    checks = [p["checks"] for m in new for p in m["priors"].values()]
    return {"key": list(key), "records": records, "fits": len(checks), "projection_solves": len(projection_cache),
        "constraint_active": sum(c["active"] for c in checks), "constraint_linear_solves": sum(c["constraint_linear_solves"] for c in checks),
        "zero_radius_paths": sum(c["zero_radius"] for c in checks),
        "max_kkt": max((c["kkt"] for c in checks if c["kkt"] is not None), default=None), "max_radius_excess": max(p["checks"]["bounded_RMS"]-p["radius"] for m in new for p in m["priors"].values()),
        "max_prediction_error": max_error, "max_metric_error": replay_error}


def report(cells, before):
    output, error = {}, 0.
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    for group, predicate in groups.items():
        records = [p for c in cells if predicate(c["key"]) for p in c["records"]];output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for penalty in source.learning.PENALTIES:
                output[group][label][str(penalty)] = {}
                for scope in SCOPES:
                    selected = [p for p in records if p["scope"] == scope and p["penalty"] == penalty and (bank is None or p["bank"] == bank)]
                    arms = {a: source.learning.summarize([p["metrics"][a] for p in selected]) for a in learning.ARMS}
                    old = before["groups"][group][label][str(penalty)][scope]
                    for actual, expected in (("raw_prior", "raw_reference"), ("cold_mse", "direct_mse")):
                        error = max(error, *(abs(arms[actual][f]-old[expected][f]) for f in ("MSE", "bias", "state_exposures", "action_exposures")))
                    output[group][label][str(penalty)][scope] = {"state_exposures": arms["cold_mse"]["state_exposures"],
                        "action_exposures": arms["cold_mse"]["action_exposures"], "arms": {a: {"MSE": s["MSE"], "bias": s["bias"]} for a, s in arms.items()},
                        "responses": {p: learning.summarize_response([r["responses"][p] for r in selected]) for p in learning.PRIORS}}
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["fits"] for c in cells) != 6336 or sum(c["projection_solves"] for c in cells) != 2112 or error > 1e-10:
        raise ValueError("Incomplete Stage271 coverage/replay")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "fits": {"residual_ridge_fits": sum(c["fits"] for c in cells), "constrained_paths": sum(c["fits"] for c in cells),
            "projection_solves": sum(c["projection_solves"] for c in cells), "active_constraints": sum(c["constraint_active"] for c in cells),
            "constraint_linear_solves": sum(c["constraint_linear_solves"] for c in cells), "zero_radius_paths": sum(c["zero_radius_paths"] for c in cells),
            "max_kkt": max((c["max_kkt"] for c in cells if c["max_kkt"] is not None), default=None), "max_radius_excess": max(c["max_radius_excess"] for c in cells)},
        "reproduction": {"prediction_identity_max_abs_error": max(c["max_prediction_error"] for c in cells), "cell_metric_max_abs_error": max(c["max_metric_error"] for c in cells), "group_metric_max_abs_error": error},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(cached_cell(directory/("_".join(map(str, key))), key, views, panels));print(f"PRIOR_RISK_CELL_COMPLETE {key}", flush=True)
    write_json(directory/"cells.json", cells)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(path, report(cells, before));print("PRIOR_RISK_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
