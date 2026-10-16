#!/usr/bin/env python3
"""Fit matched-capacity state-conditioned residuals on existing cached futures."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_training_count_stage258 as cache
from scripts.run_native_transit_prefix_influence_stage259 import read_cell
from freq_hrl.domains.transit import native_state_risk as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_state_risk_stage277_v1"
SOURCE_RUN = "native_transit_local_risk_stage272_20261016_r1"
HELD_RUN = "native_transit_held_removal_stage273_20261016_r1"
BASIS_RUN = "native_transit_shared_risk_stage270_20261016_r1"
GATE_RUN = "native_transit_alignment_scale_stage276_20261016_r1"
KEYS, VIEWS, PENALTY = cache.KEYS, cache.VIEWS, .01
SCOPES = tuple(s+"_"+f for s in ("train", "held", "retained", "excluded", "matched_held") for f in ("same", "opposite"))


def contract():
    return {"source_run": SOURCE_RUN, "held_run": HELD_RUN, "basis_run": BASIS_RUN,
        "cohort": "unchanged32_cells_four_views_two_outer_folds_two_partitions_two_directions; full60_prefix54_both_banks_assigned16; all_development_states_no_selected_subgroups",
        "maps": "both_8_action_slopes_plus_full8x8_context_interactions_plus_intercept=73_coefficients; baseline_context_is_eligible_action_centroid; treatment_context_is_actual_observed_state",
        "observed_context": list(learning.CONTEXT_NAMES),
        "normalization": "per_fit_subset_only_state_balanced_context_center_RMS_then_tanh; state_balanced_lift_center_RMS; constant_scale1; same_penalty_0.01_physical_MSE_and_raw_prior",
        "clip": "same_Stage272_cached_radius_per_fit; original_action_correction0; no_query_rescaling_or_selection",
        "evaluation": "cached_assigned_and_opposite16; full_train_held_prefix_retained_excluded_plus_same_prefix54_heads_on_outer_held; predictions_written_before_evaluation; primary_mixed_pooled_excluded_opposite_and_held_opposite",
        "controls": "cached_raw_bounded_clip_and_linear9; regenerate_linear9_predictions_from_frozen_weights_and_reproduce_Stage272_error_sums; matched_held_controls_reused_Stage273",
        "arms": list(learning.ARMS), "penalty": PENALTY,
        "budget": {"fit_tasks": 1, "cached_cells": 32, "cached_contexts": 704, "new_residual_head_fits": 1408,
            "new_raw_cold_prior_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "models_features_queries_predictions_and_error_ledgers_server_only; pull_scalar_report_only",
        "boundary": "matched_parameter_count_not_equal_effective_degrees_of_freedom; reused_development_prefixes_and_futures_historical_reference_conditioning_not_seed_CI_confirmation_or_policy_gain; no_default_change; Stage253_248_169_closed"}


def fit_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    old = json.loads((here(SOURCE_RUN)/"models.json").read_text())
    queries = json.loads((here(SOURCE_RUN)/"queries.json").read_text())
    previous = json.loads((here(SOURCE_RUN)/"errors.json").read_text())
    with np.load(here(BASIS_RUN)/"predictions.npz") as p: basis = p["basis"].copy()
    reference = read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    if {(r["root"], r["scene_seed"]) for r in train}&{(r["root"], r["scene_seed"]) for r in held}: raise ValueError("Outer prefix overlap")
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    observed = {s: learning.observed_context(rs) for s, rs in rows.items()}
    z, eligible = {}, {}
    for s, d in data.items():
        x = cache.learning.joint.relative(d["x"][..., :198], d["originals"])@basis
        z[s] = np.concatenate((np.ones((*x.shape[:2], 1)), x), axis=-1)
        eligible[s] = (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None])
    folds = {tuple(g): np.asarray(ix) for g, ix, _ in cache.learning.joint.binding.policy.prefix_folds(train)}
    models, predictions, arrays, maximum = [], [], {}, 0.
    with np.load(here(SOURCE_RUN)/"predictions.npz") as saved, np.load(here(HELD_RUN)/"held_predictions.npz") as matched:
        for i, m in enumerate(old):
            if m["penalty"] != PENALTY: continue
            ix = np.asarray(m["fit_indices"]);group = m["removed"]
            draws, _ = cache.sampling.learning.draw_assignment(m["bank"], "expanded")
            if m["training_draws"] != draws or not np.array_equal(ix, np.arange(60) if group is None else folds[tuple(group)]): raise ValueError("Changed fit bank/population")
            qs = [q for q in queries if q["index"] == i]
            training = next(q for q in qs if q["scope"] == ("train" if group is None else "retained"))
            if training["label"] != "train" or not np.array_equal(training["ids"], ix): raise ValueError("Changed training query")
            prior = saved[f'{i}_{training["scope"]}_raw_prior'].copy()
            values = cache.learning.base.training_draws([train[j] for j in ix], panels, draws)[..., 1]
            y = values-values[np.arange(len(ix)), :, data["train"]["originals"][ix]][:, :, None]
            model = learning.fit(z["train"][ix], observed["train"][ix], y.mean(axis=1), prior, eligible["train"][ix], PENALTY)
            models.append({"index": i, "bank": m["bank"], "removed": group, "fit_indices": ix, "training_draws": draws, "radius": m["radius"], **model})
            if group is not None: qs += [{"index": i, "scope": "matched_held", "label": "held", "ids": list(range(60)), "eligible": eligible["held"]}]
            for q in qs:
                label, ids, scope = q["label"], np.asarray(q["ids"]), q["scope"]
                mask = eligible[label][ids]
                if not np.array_equal(mask, q["eligible"]): raise ValueError("Changed feasibility")
                costs = {a: (matched[a][i].copy() if scope == "matched_held" else saved[f"{i}_{scope}_{a}"].copy()) for a in learning.ARMS[:3]}
                linear = m["maps"]["linear"]
                delta = (learning.base.transform(z[label][ids][..., 1:], linear)@np.asarray(linear["weights"]))[..., 0]
                delta[np.arange(len(ids)), data[label]["originals"][ids]] = 0.
                replay = costs["raw_prior"]+np.clip(delta, -m["radius"], m["radius"])
                maximum = max(maximum, float(np.max(np.abs(replay-costs["raw_linear_clip"]))))
                new, clips = learning.predict(model, z[label][ids], observed[label][ids], costs["raw_prior"], mask, data[label]["originals"][ids], m["radius"])
                for a in ("raw_bounded_clip", "raw_linear_clip"):
                    uncut = matched[a[:-5]][i] if scope == "matched_held" else saved[f"{i}_{scope}_{a[:-5]}"]
                    clips[a] = np.abs(uncut-costs["raw_prior"]) > m["radius"]
                costs.update(new)
                predictions.append({**q, "bank": m["bank"], "removed": group, "costs": costs, "clips": clips, "radius": m["radius"]})
                arrays.update({f"{i}_{scope}_{a}": c for a, c in costs.items()})
    if len(models) != 22 or len(predictions) != 64 or maximum > 1e-12: raise ValueError("Incomplete cached context replay")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"models.json", models)
    write_json(directory/"queries.json", [{k: v for k, v in q.items() if k not in ("costs", "clips")} for q in predictions])
    np.savez_compressed(directory/"predictions.npz", **arrays)
    print(f"STATE_RISK_PREDICTIONS_FROZEN {key}", flush=True)
    old_errors = {(r["bank"], tuple(r["removed"] or []), r["scope"]): r for r in previous if r["penalty"] == PENALTY}
    records, metric_error = [], 0.
    for q in predictions:
        rs = [rows[q["label"]][j] for j in q["ids"]]
        originals = data[q["label"]]["originals"][q["ids"]]
        for future, bank in (("same", q["bank"]), ("opposite", 1-q["bank"])):
            draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws(rs, panels, draws)[..., 1]
            y = values-values[np.arange(len(rs)), :, originals][:, :, None]
            metrics = {a: learning.base.metrics(c, q["costs"]["raw_prior"], y, np.asarray(q["eligible"]), q["radius"], q["clips"].get(a)) for a, c in q["costs"].items()}
            scope = q["scope"]+"_"+future
            if q["scope"] != "matched_held":
                before = old_errors[q["bank"], tuple(q["removed"] or []), scope]["metrics"]
                for a in learning.ARMS[:3]:
                    for f in ("squared_error_sum", "signed_error_sum", "state_exposures", "action_exposures"):
                        metric_error = max(metric_error, abs(metrics[a][f]-before[a][f]))
            records.append({"bank": q["bank"], "scope": scope, "metrics": metrics})
    if metric_error > 1e-10: raise ValueError("Cached scalar errors changed")
    write_json(directory/"errors.json", records)
    return {"key": list(key), "records": records, "head_fits": sum(m["head_fits"] for m in models),
        "normal_error": max(m["normal_error"] for m in models), "prediction_error": maximum, "metric_error": metric_error}


def report(cells):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["head_fits"] for c in cells) != 1408: raise ValueError("Incomplete Stage277 fits")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output = {}
    for g, predicate in groups.items():
        rs = [r for c in cells if predicate(c["key"]) for r in c["records"]];output[g] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[g][label] = {}
            for scope in SCOPES:
                selected = [r for r in rs if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                output[g][label][scope] = {"state_exposures": sum(r["metrics"]["raw_prior"]["state_exposures"] for r in selected),
                    "arms": {a: learning.base.summarize([r["metrics"][a] for r in selected], True) for a in learning.ARMS}}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "fits": {"residual_head_fits": 1408, "parameters_per_new_head": 73, "max_normal_error": max(c["normal_error"] for c in cells)},
        "reproduction": {f: max(c[f] for c in cells) for f in ("prediction_error", "metric_error")},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(fit_cell(directory/("_".join(map(str, key))), key, views, panels))
        print(f"STATE_RISK_CELL_COMPLETE {key}", flush=True)
    write_json(directory/"cells.json", cells);write_json(path, report(cells));print("STATE_RISK_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
