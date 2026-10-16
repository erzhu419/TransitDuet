#!/usr/bin/env python3
"""Separate observable-context residual learning from fixed per-action clipping."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_prior_risk_stage271 as source
from freq_hrl.domains.transit import native_local_risk as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_local_risk_stage272_v1"
SOURCE_RUN = "native_transit_prior_risk_stage271_20261016_r1"
cache, KEYS, VIEWS, SCOPES = source.cache, source.KEYS, source.VIEWS, source.SCOPES
PENALTIES = source.source.learning.PENALTIES
PRIMARY_PENALTY = source.source.learning.PRIMARY_PENALTY


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "priors": list(learning.PRIORS), "arms": list(learning.ARMS),
        "penalties": list(PENALTIES), "primary_penalty": PRIMARY_PENALTY,
        "features": "same_frozen_rank8_action_relative_causal_features; linear_x_vs_interaction_x_times_tanh_current_state_eligible_action_centroid; no_labels_root_scenario_seed_or_prefix_ID_in_map",
        "normalization": "each_fit_subset_state_balanced_weighted_center_and_RMS_per_coordinate_both_maps; no_held_or_opposite_labels_or_features_in_stats; 8slopes_plus_intercept_each; no_superset_capacity_claim",
        "fit": "convex_physical_residual_MSE_same_state_weights_penalties_and_assigned16_for_both_maps; share_matrix_solve_but_not_coefficients_across_priors; zero_residual_preserves_prior",
        "bound": "each_nonoriginal_action_correction_clipped_to_plus_minus_existing_Stage271_physical_radius; both_new_maps_and_frozen_old_residual_RMS_controls; no_labels_query_population_rescaling_or_new_threshold_selection; cap_not_no_harm",
        "controls": "reproduce_Stage271_cold_prior_residual_and_RMS_predictions_and_error_sums_all_cells_banks_scopes_penalties; 2x2_new_map_vs_clip_plus_old_fit_clip_only; exact_pointwise_cap_and_original_anchor",
        "split": source.source.contract()["split"], "evaluation": source.contract()["evaluation"],
        "frozen": "all_cold_prior_raw_service_models_actions_and_selectors; prefix54_prior_indices_and_assigned_bank_verbatim; no_native_control_or_selection",
        "budget": {"fit_tasks": 1, "cached_cells": 32, "new_residual_head_fits": 12672, "ridge_solver_calls": 4224,
            "cached_context_models": 2112, "new_raw_cold_or_prior_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "features_models_queries_predictions_and_error_ledgers_server_only; pull_compact_scalar_report",
        "boundary": "development_label_accuracy_not_new_prefix_seed_CI_policy_gain_or_necessity_of_nonlinearity; fixed_interaction_is_one_alternative_not_superset; repeated_contexts_not_independent; no_penalty_or_arm_selection_default_change; Stage253_248_169_closed"}


def fit_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    old = json.loads((here(SOURCE_RUN)/"models.json").read_text())
    queries = json.loads((here(source.SOURCE_RUN)/"queries.json").read_text())
    previous = json.loads((here(SOURCE_RUN)/"errors.json").read_text())
    with np.load(here(source.SOURCE_RUN)/"predictions.npz") as saved: basis = saved["basis"].copy()
    with np.load(here(SOURCE_RUN)/"predictions.npz") as saved: cached = {k: saved[k].copy() for k in saved.files}
    reference = source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held};data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    z, eligible = {}, {}
    for s, d in data.items():
        x = cache.learning.joint.relative(d["x"][..., :198], d["originals"])@basis
        z[s] = np.concatenate((np.ones((*x.shape[:2], 1)), x), axis=-1)
        eligible[s] = (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None])
    folds = {tuple(g): np.asarray(ix) for g, ix, _ in cache.learning.joint.binding.policy.prefix_folds(train)}
    models, predictions, arrays, maximum = [], [], {}, 0.
    for index, m in enumerate(old):
        group = m["removed"];indices = np.asarray(m["fit_indices"])
        ds, _ = cache.sampling.learning.draw_assignment(m["bank"], "expanded")
        if m["training_draws"] != ds or not np.array_equal(indices, np.arange(60) if group is None else folds[tuple(group)]): raise ValueError("Changed fit bank/population")
        qs = [q for q in queries if q["index"] == index]
        training = next(q for q in qs if q["scope"] == ("train" if group is None else "retained"))
        if training["label"] != "train" or not np.array_equal(training["ids"], indices): raise ValueError("Changed fit query order")
        prior = np.stack([cached[f'{index}_{training["scope"]}_{p}_prior'] for p in learning.PRIORS], axis=-1)
        rs = [train[i] for i in indices];values = cache.learning.base.training_draws(rs, panels, ds)[..., 1]
        originals = data["train"]["originals"][indices]
        y = values-values[np.arange(len(rs)), :, originals][:, :, None]
        model = learning.fit(z["train"][indices], y, prior, eligible["train"][indices], m["penalty"])
        radii = [m["priors"][p]["radius"] for p in learning.PRIORS]
        if len(set(radii)) != 1: raise ValueError("Changed common prior correction budget")
        radius = radii[0]
        models.append({"bank": m["bank"], "removed": group, "penalty": m["penalty"], "training_draws": ds,
            "fit_indices": indices, "radius": radius, **model})
        for q in qs:
            label, ids, scope = q["label"], np.asarray(q["ids"]), q["scope"]
            mask = eligible[label][ids]
            if not np.array_equal(mask, q["eligible"]): raise ValueError("Changed query feasibility")
            old_costs = {a: cached[f"{index}_{scope}_{a}"] for a in source.learning.ARMS}
            priors = np.stack([old_costs[p+"_prior"] for p in learning.PRIORS], axis=-1)
            costs, clips = learning.predict(model, z[label][ids], priors, old_costs, data[label]["originals"][ids], mask, radius)
            for a in ("cold_mse", *(p+"_"+s for p in learning.PRIORS for s in ("prior", "residual", "bounded"))):
                maximum = max(maximum, float(np.max(np.abs(costs[a]-old_costs[a]))))
            predictions.append({**q, "costs": costs, "clips": clips, "radius": radius})
            arrays.update({f"{index}_{scope}_{a}": cost for a, cost in costs.items()})
    if len(old) != 66 or len(queries) != 132 or maximum > 1e-12: raise ValueError("Incomplete Stage271 replay")
    directory.mkdir(parents=True, exist_ok=True);write_json(directory/"models.json", models)
    write_json(directory/"queries.json", queries);np.savez_compressed(directory/"predictions.npz", **arrays)
    print(f"LOCAL_RISK_PREDICTIONS_FROZEN {key}", flush=True)
    records, metric_error = [], 0.
    for qi, q in enumerate(predictions):
        m = models[q["index"]];label, ids = q["label"], np.asarray(q["ids"])
        rs = [rows[label][i] for i in ids];originals = data[label]["originals"][ids]
        for b, (which, bank) in enumerate((("same", m["bank"]), ("opposite", 1-m["bank"]))):
            ds, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws(rs, panels, ds)[..., 1]
            y = values-values[np.arange(len(rs)), :, originals][:, :, None]
            metrics = {}
            for a, cost in q["costs"].items():
                p = "raw" if a == "cold_mse" else next(p for p in learning.PRIORS if a.startswith(p+"_"))
                metrics[a] = learning.metrics(cost, q["costs"][p+"_prior"], y, np.asarray(q["eligible"]), q["radius"], q["clips"].get(a))
            old_record = previous[2*qi+b]
            if (old_record["bank"], old_record["penalty"], old_record["scope"]) != (m["bank"], m["penalty"], q["scope"]+"_"+which): raise ValueError("Changed cached error order")
            for a in ("cold_mse", *(p+"_"+s for p in learning.PRIORS for s in ("prior", "residual", "bounded"))):
                prior_metrics = old_record["metrics"][a]
                for current, old_field in (("squared_error_sum", "state_balanced_squared_error_sum"), ("signed_error_sum", "state_balanced_signed_error_sum"), ("state_exposures", "state_exposures"), ("action_exposures", "action_exposures")):
                    metric_error = max(metric_error, abs(metrics[a][current]-prior_metrics[old_field]))
            records.append({"bank": m["bank"], "removed": m["removed"], "penalty": m["penalty"], "scope": q["scope"]+"_"+which, "metrics": metrics})
    if metric_error > 1e-10: raise ValueError("Stage271 error sums changed")
    write_json(directory/"errors.json", records)
    return {"key": list(key), "records": records, "head_fits": sum(m["head_fits"] for m in models),
        "ridge_solver_calls": sum(m["ridge_solver_calls"] for m in models), "max_normal_error": max(m["normal_error"] for m in models),
        "prediction_max_abs_error": maximum, "metric_max_abs_error": metric_error,
        "max_checked_bound_excess": max(s["checked_bound_excess"] for r in records for s in r["metrics"].values())}


def report(cells, before):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, error = {}, 0.
    for group, predicate in groups.items():
        records = [p for c in cells if predicate(c["key"]) for p in c["records"]];output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for penalty in PENALTIES:
                output[group][label][str(penalty)] = {}
                for scope in SCOPES:
                    selected = [r for r in records if r["penalty"] == penalty and r["scope"] == scope and (bank is None or r["bank"] == bank)]
                    arms = {a: learning.summarize([r["metrics"][a] for r in selected], penalty == PRIMARY_PENALTY) for a in learning.ARMS}
                    old = before["groups"][group][label][str(penalty)][scope]
                    n = sum(r["metrics"]["cold_mse"]["state_exposures"] for r in selected)
                    count = sum(r["metrics"]["cold_mse"]["action_exposures"] for r in selected)
                    error = max(error, abs(n-old["state_exposures"]), abs(count-old["action_exposures"]))
                    for a in ("cold_mse", *(p+"_"+s for p in learning.PRIORS for s in ("prior", "residual", "bounded"))):
                        for f in ("MSE", "bias"): error = max(error, abs(arms[a][f]-old["arms"][a][f]))
                    output[group][label][str(penalty)][scope] = {"state_exposures": n, "action_exposures": count, "arms": arms}
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["head_fits"] for c in cells) != 12672 or sum(c["ridge_solver_calls"] for c in cells) != 4224 or error > 1e-10:
        raise ValueError("Incomplete Stage272 coverage/replay")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "fits": {"residual_head_fits": sum(c["head_fits"] for c in cells), "ridge_solver_calls": sum(c["ridge_solver_calls"] for c in cells), "max_normal_error": max(c["max_normal_error"] for c in cells)},
        "reproduction": {"prediction_max_abs_error": max(c["prediction_max_abs_error"] for c in cells), "cell_metric_max_abs_error": max(c["metric_max_abs_error"] for c in cells),
            "group_metric_max_abs_error": error, "max_checked_bound_excess": max(c["max_checked_bound_excess"] for c in cells)},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(fit_cell(directory/("_".join(map(str, key))), key, views, panels));print(f"LOCAL_RISK_CELL_COMPLETE {key}", flush=True)
    write_json(directory/"cells.json", cells)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(path, report(cells, before));print("LOCAL_RISK_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
