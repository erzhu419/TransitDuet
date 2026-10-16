#!/usr/bin/env python3
"""Compare feature-only effective-df matching with frozen Stage277 controls."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_state_risk_stage277 as source
from freq_hrl.domains.transit import native_effective_risk as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_effective_risk_stage278_v1"
SOURCE_RUN = "native_transit_state_risk_stage277_20261017_r2"
cache, KEYS, VIEWS, SCOPES = source.cache, source.KEYS, source.VIEWS, source.SCOPES


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arms": list(learning.ARMS),
        "representation": "reuse_each_Stage277_73_coefficient_map_and_all_training_normalizers_verbatim",
        "rule": "train_feature_only_weighted_centered_Gram_eigenvalues; df=1+sum_eigen_over_eigen_plus_lambda; match_each73_head_to_same_subset_Linear9_df_at_lambda0.01; lambda_at_least0.01; no_label_selection_or_search_grid",
        "fit": "two_new_residual_heads_per_context; same_raw_prior_assigned16_physicalMSE_weights_and_anchors; no_raw_prior_cold_reference_or_basis_refit",
        "clip": source.contract()["clip"], "evaluation": source.contract()["evaluation"],
        "controls": "reproduce_all_Stage277_cached_predictions_for_both73_maps; all7_old_arm_error_sums; frozen_query_ids_masks_fit_indices_banks_radii; predictions_written_before_evaluation",
        "budget": {"fit_tasks": 1, "cached_cells": 32, "cached_contexts": 704, "new_residual_head_fits": 1408,
            "new_raw_cold_prior_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "models_predictions_and_error_ledgers_remote_only; pull_scalar_report",
        "boundary": "EDF_controls_training_design_sensitivity_not_OOD_capacity_or_correlated_simulator_noise; changed_penalty_not_changed_observations; development_diagnostic_not_policy_gain_or_seed_CI; Stage253_248_169_closed; no_default_change"}


def fit_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    load = lambda run, name: json.loads((here(run)/name).read_text())
    old, queries, previous = (load(SOURCE_RUN, f) for f in ("models.json", "queries.json", "errors.json"))
    linear_models = load(source.SOURCE_RUN, "models.json")
    with np.load(here(source.BASIS_RUN)/"predictions.npz") as p: basis = p["basis"].copy()
    reference = source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    if {(r["root"], r["scene_seed"]) for r in train}&{(r["root"], r["scene_seed"]) for r in held}: raise ValueError("Outer prefix overlap")
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    observed = {s: learning.previous.observed_context(rs) for s, rs in rows.items()}
    z, eligible = {}, {}
    for s, d in data.items():
        x = cache.learning.joint.relative(d["x"][..., :198], d["originals"])@basis
        z[s] = np.concatenate((np.ones((*x.shape[:2], 1)), x), axis=-1)
        eligible[s] = (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None])
    folds = {tuple(g): np.asarray(ix) for g, ix, _ in cache.learning.joint.binding.policy.prefix_folds(train)}
    models, predictions, arrays, maximum = [], [], {}, 0.
    with np.load(here(SOURCE_RUN)/"predictions.npz") as saved:
        for m in old:
            i, ix = m["index"], np.asarray(m["fit_indices"])
            draws, _ = cache.sampling.learning.draw_assignment(m["bank"], "expanded")
            if m["training_draws"] != draws or not np.array_equal(ix, np.arange(60) if m["removed"] is None else folds[tuple(m["removed"])]): raise ValueError("Changed cached fit bank/population")
            old_linear = linear_models[i]
            for f in ("bank", "removed", "fit_indices", "training_draws"):
                if m[f] != old_linear[f]: raise ValueError("Changed Linear9 context")
            if old_linear["penalty"] != .01 or m["radius"] != old_linear["radius"]: raise ValueError("Changed baseline penalty/radius")
            qs = [q for q in queries if q["index"] == i]
            training = next(q for q in qs if q["scope"] == ("train" if m["removed"] is None else "retained"))
            if training["label"] != "train" or not np.array_equal(training["ids"], ix): raise ValueError("Changed training query")
            prior = saved[f'{i}_{training["scope"]}_raw_prior'].copy()
            values = cache.learning.base.training_draws([train[j] for j in ix], panels, draws)[..., 1]
            y = values-values[np.arange(len(ix)), :, data["train"]["originals"][ix]][:, :, None]
            model = learning.fit(m, old_linear["maps"]["linear"], z["train"][ix], observed["train"][ix], y.mean(axis=1), prior, eligible["train"][ix])
            models.append({**m, **model})
            for q in qs:
                label, ids, scope = q["label"], np.asarray(q["ids"]), q["scope"]
                mask = eligible[label][ids]
                if not np.array_equal(mask, q["eligible"]) or q["radius"] != m["radius"]: raise ValueError("Changed cached query")
                costs = {a: saved[f"{i}_{scope}_{a}"].copy() for a in learning.previous.ARMS}
                args = (z[label][ids], observed[label][ids], costs["raw_prior"], mask, data[label]["originals"][ids], m["radius"])
                replay, old_clips = learning.previous.predict(m, *args)
                maximum = max(maximum, *(float(np.max(np.abs(c-costs[a]))) for a, c in replay.items()))
                new, clips = learning.predict(model, *args);costs.update(new);clips.update(old_clips)
                predictions.append({**q, "costs": costs, "clips": clips})
                arrays.update({f"{i}_{scope}_{a}": c for a, c in costs.items()})
    if len(models) != 22 or len(predictions) != 64 or maximum > 1e-12: raise ValueError("Incomplete Stage277 replay")
    directory.mkdir(parents=True, exist_ok=True);write_json(directory/"models.json", models)
    np.savez_compressed(directory/"predictions.npz", **arrays);write_json(directory/"queries.json", queries)
    print(f"EFFECTIVE_RISK_PREDICTIONS_FROZEN {key}", flush=True)
    lookup = {(r["bank"], r["scope"]): [] for r in previous}
    # Retained/excluded/matched-held repeat per removed prefix; retain cache order.
    for r in previous: lookup[r["bank"], r["scope"]].append(r)
    positions = dict.fromkeys(lookup, 0)
    records, metric_error = [], 0.
    for q in predictions:
        rs = [rows[q["label"]][j] for j in q["ids"]];originals = data[q["label"]]["originals"][q["ids"]]
        for future, bank in (("same", q["bank"]), ("opposite", 1-q["bank"])):
            draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws(rs, panels, draws)[..., 1]
            y = values-values[np.arange(len(rs)), :, originals][:, :, None]
            scope = q["scope"]+"_"+future;pair = q["bank"], scope
            before = lookup[pair][positions[pair]];positions[pair] += 1
            metrics = {}
            for a, cost in q["costs"].items():
                if a in ("raw_prior", "raw_bounded_clip", "raw_linear_clip"):
                    metrics[a] = before["metrics"][a]
                    replay = learning.previous.base.metrics(cost, q["costs"]["raw_prior"], y, np.asarray(q["eligible"]), q["radius"])
                    for f in ("squared_error_sum", "signed_error_sum", "state_exposures", "action_exposures"):
                        metric_error = max(metric_error, abs(replay[f]-metrics[a][f]))
                else:
                    metrics[a] = learning.previous.base.metrics(cost, q["costs"]["raw_prior"], y, np.asarray(q["eligible"]), q["radius"], q["clips"].get(a))
                    if a in learning.previous.ARMS:
                        for f in metrics[a]:
                            if metrics[a][f] is not None: metric_error = max(metric_error, abs(metrics[a][f]-before["metrics"][a][f]))
            records.append({"bank": q["bank"], "scope": scope, "metrics": metrics})
    if metric_error > 1e-10: raise ValueError("Stage277 scalar replay failed")
    write_json(directory/"errors.json", records)
    return {"key": list(key), "records": records, "complexity": [{"bank": m["bank"], "removed": m["removed"], **m["complexity"]} for m in models],
        "head_fits": sum(m["head_fits"] for m in models), "normal_error": max(m["normal_error"] for m in models), "prediction_error": maximum, "metric_error": metric_error}


def report(cells, before):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["head_fits"] for c in cells) != 1408: raise ValueError("Incomplete Stage278 fits")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, group_error = {}, 0.
    for g, predicate in groups.items():
        selected = [c for c in cells if predicate(c["key"])];output[g] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            rs = [r for c in selected for r in c["records"] if bank is None or r["bank"] == bank]
            output[g][label] = {}
            for scope in SCOPES:
                scoped = [r for r in rs if r["scope"] == scope]
                summaries = {a: learning.previous.base.summarize([r["metrics"][a] for r in scoped], True) for a in learning.ARMS}
                for a in learning.previous.ARMS:
                    for f, value in summaries[a].items():
                        if value is not None: group_error = max(group_error, abs(value-before["groups"][g][label][scope]["arms"][a][f]))
                output[g][label][scope] = {"state_exposures": sum(r["metrics"]["raw_prior"]["state_exposures"] for r in scoped), "arms": summaries}
    if group_error > 1e-10: raise ValueError("Stage277 group summaries changed")
    complexity = {}
    for kind in learning.MAPS:
        entries = [d[kind] for c in cells for d in c["complexity"]]
        complexity[kind] = {f: {"min": min(d[f] for d in entries), "mean": float(np.mean([d[f] for d in entries])), "max": max(d[f] for d in entries)} for f in entries[0]}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "fits": {"residual_head_fits": 1408, "max_normal_error": max(c["normal_error"] for c in cells), "complexity": complexity},
        "reproduction": {"prediction_error": max(c["prediction_error"] for c in cells), "metric_error": max(c["metric_error"] for c in cells), "group_metric_error": group_error},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(fit_cell(directory/("_".join(map(str, key))), key, views, panels));print(f"EFFECTIVE_RISK_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(directory/"cells.json", cells);write_json(path, report(cells, before));print("EFFECTIVE_RISK_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
