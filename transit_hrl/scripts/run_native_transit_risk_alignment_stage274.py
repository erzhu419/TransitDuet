#!/usr/bin/env python3
"""Decompose cached prefix54 errors on excluded and condition-matched held queries."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_training_count_stage258 as cache
from freq_hrl.domains.transit import native_risk_alignment as learning
from freq_hrl.domains.transit.native_held_removal import conditions
from freq_hrl.domains.transit.native_shared_risk import PENALTIES
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_risk_alignment_stage274_v1"
SOURCE_RUN = "native_transit_held_removal_stage273_20261016_r1"
SOURCE_PROTOCOL = "native_transit_held_removal_stage273_v1"
PREDICTION_RUN = "native_transit_local_risk_stage272_20261016_r1"
KEYS, VIEWS = cache.KEYS, cache.VIEWS
ARMS = ("raw_prior", "cold_mse", "raw_bounded", "raw_bounded_clip", "raw_linear_clip", "raw_interaction_clip")
POPULATIONS = ("excluded", "matched_held", "paired_excluded", "paired_held")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": SOURCE_PROTOCOL,
        "prediction_run": PREDICTION_RUN, "cohort": cache.contract()["cohort"], "arms": list(ARMS),
        "penalties": list(PENALTIES), "primary": {"arm": "raw_bounded_clip", "penalty": .01, "group": "mixed_pooled", "future": "opposite"},
        "identity": "delta=prediction-raw; residual=raw-mean_of_assigned_or_opposite16_physical_fleet_labels; delta_MSE=E_delta_squared+2E_delta_times_residual; negative_alignment_is_beneficial",
        "natural": "each_frozen_prefix54_context_excluded6_vs_held_queries_matching_removed_root_regime; original_state_balanced_eligible_action_weights; reproduce_Stage272_excluded_and_Stage273_matched_held_MSE",
        "paired": "same_model_root_regime_probe_and_intersection_of_both_frozen_eligible_action_sets; one_unit_per_supported_model_probe; average_valid_held_replicas_within_probe_before_pooling; report_pair_coverage",
        "frozen": "read_saved_predictions_only; no_model_inference_fits_solves_normalization_radius_actions_selectors_or_reference_changes; metadata_for_diagnosis_only",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_prefix_contexts": 1920, "new_fits": 0,
            "new_model_inferences": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "cached_predictions_and_panels_stay_remote; new_per_context_error_and_pair_ledgers_remote_only; pull_scalar_report",
        "boundary": "reused_development_labels_and_historical_raw_reference_conditioning; repeated_contexts_not_seed_CI_or_policy_gain; population_contrast_not_causal_transfer_or_pure_count_effect; no_selection_or_default_change; Stage253_248_169_closed"}


def analyze_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    load = lambda run, name: json.loads((here(run)/(name+".json")).read_text())
    models, queries = load(PREDICTION_RUN, "models"), load(PREDICTION_RUN, "queries")
    old_errors, held_queries, held_errors = load(PREDICTION_RUN, "errors"), load(SOURCE_RUN, "queries"), load(SOURCE_RUN, "errors")
    qs = {(q["index"], q["scope"]): q for q in queries}
    previous = {(r["bank"], tuple(r["removed"]), r["penalty"], r["scope"]): r for r in old_errors if r["removed"] is not None and r["scope"].startswith("excluded_")}
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    if {(r["root"], r["scene_seed"]) for r in train}&{(r["root"], r["scene_seed"]) for r in held}: raise ValueError("Overlapping outer prefixes")
    state_key = cache.sampling.old_future.learning.state_key
    if held_queries["held_states"] != [list(state_key(r)) for r in held]: raise ValueError("Changed held query order")
    groups = [tuple(g) for g in held_queries["removed_groups"]]
    masks, support = conditions(train, held, groups)
    if not np.array_equal(masks["matched"], held_queries["masks"]["matched"]) or not np.array_equal(support, held_queries["support_prefixes_before"]): raise ValueError("Changed held condition matching")
    if len(models) != 66 or len(queries) != 132: raise ValueError("Incomplete cached contexts")
    with np.load(here(PREDICTION_RUN)/"predictions.npz") as p:
        excluded = {(i, a): p[f"{i}_excluded_{a}"].copy() for i, m in enumerate(models) if m["removed"] is not None for a in ARMS}
        full_raw = {i: p[f"{i}_held_raw_prior"].copy() for i, m in enumerate(models) if m["removed"] is None}
    with np.load(here(SOURCE_RUN)/"held_predictions.npz") as p: held_predictions = {a: p[a].copy() for a in ARMS}
    targets = {}
    for bank in (0, 1):
        ds, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
        for label, rows in (("train", train), ("held", held)):
            values = cache.learning.base.training_draws(rows, panels, ds)[..., 1]
            originals = np.asarray([r["original_choice"] for r in rows])
            targets[label, bank] = (values-values[np.arange(len(rows)), :, originals][:, :, None]).mean(axis=1)
    records, pairing, identity_error, replay_error = [], [], 0., 0.
    for i, m in enumerate(models):
        if m["removed"] is None: continue
        group, bank, penalty = tuple(m["removed"]), m["bank"], m["penalty"]
        ds, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
        q = qs[i, "excluded"];ex = np.asarray(q["ids"])
        expected = np.asarray([j for j, r in enumerate(train) if (r["root"], r["scene_seed"]) == group])
        if q["label"] != "train" or not np.array_equal(ex, expected) or not np.array_equal(m["fit_indices"], np.setdiff1d(np.arange(60), ex)) or m["training_draws"] != ds: raise ValueError("Changed prefix54 model/query binding")
        full_i = next(j for j, v in enumerate(models) if v["removed"] is None and v["bank"] == bank and v["penalty"] == penalty)
        held_q = qs[full_i, "held"]
        if held_q["label"] != "held" or held_q["ids"] != list(range(60)): raise ValueError("Changed held feature order")
        he = np.flatnonzero(masks["matched"][groups.index(group)])
        ex_rows, he_rows = [train[j] for j in ex], [held[j] for j in he]
        ex_eligible, he_eligible = np.asarray(q["eligible"], bool), np.asarray(held_q["eligible"], bool)[he]
        for rows, eligible in ((ex_rows, ex_eligible), (he_rows, he_eligible)):
            if eligible[np.arange(len(rows)), [r["original_choice"] for r in rows]].any(): raise ValueError("Original action included in diagnostic weights")
        ex_weights, he_weights = map(learning.state_weights, (ex_eligible, he_eligible))
        pe, ph, weights, counts = learning.common_pairs(ex_rows, he_rows, ex_eligible, he_eligible)
        pairing.append({"index": i, "bank": bank, "penalty": penalty, "removed": group, **counts})
        raw_ex, raw_he = excluded[i, "raw_prior"], held_predictions["raw_prior"][i, he]
        if not np.array_equal(raw_he, full_raw[full_i][he]): raise ValueError("Raw reference changed under prefix removal")
        for future, b in (("same", bank), ("opposite", 1-bank)):
            ex_y, he_y = targets["train", b][ex], targets["held", b][he]
            metrics = {}
            for a in ARMS:
                pred_ex, pred_he = excluded[i, a], held_predictions[a][i, he]
                populations = ((pred_ex, raw_ex, ex_y, ex_weights, int(ex_eligible.any(axis=1).sum())),
                    (pred_he, raw_he, he_y, he_weights, int(he_eligible.any(axis=1).sum())),
                    (pred_ex[pe], raw_ex[pe], ex_y[pe], weights, counts["units"]),
                    (pred_he[ph], raw_he[ph], he_y[ph], weights, counts["units"]))
                metrics[a] = {}
                for label, args in zip(POPULATIONS, populations):
                    metrics[a][label], error = learning.components(*args);identity_error = max(identity_error, error)
                old = previous[bank, group, penalty, "excluded_"+future]["metrics"][a]
                current = metrics[a]["excluded"]
                replay_error = max(replay_error, abs(current["model_MSE"]-old["squared_error_sum"]), abs(current["units"]-old["state_exposures"]))
            records.append({"index": i, "bank": bank, "penalty": penalty, "future": future, "removed": group, "metrics": metrics})
    for r in held_errors:
        rs = [x for x in records if (x["bank"], x["penalty"], x["future"]) == (r["bank"], r["penalty"], r["future"])]
        for a in ARMS:
            current = learning.summarize([x["metrics"][a]["matched_held"] for x in rs])
            old = r["metrics"][a]["matched"]
            replay_error = max(replay_error, abs(current["model_MSE"]-old["prefix54_squared_error_sum"]/old["state_exposures"]), abs(current["units"]-old["prefix_state_exposures"]))
    if len(records) != 120 or replay_error > 1e-10: raise ValueError("Cached excluded/held error replay failed")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"errors.json", records);write_json(directory/"pairs.json", pairing)
    return {"key": list(key), "records": records, "pairing": pairing,
        "identity_max_abs_error": identity_error, "metric_max_abs_error": replay_error}


def report(cells, before_excluded, before_held):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, replay_error, identity_error = {}, 0., 0.
    for group, predicate in groups.items():
        selected = [c for c in cells if predicate(c["key"])];records = [r for c in selected for r in c["records"]]
        pairs = [p for c in selected for p in c["pairing"]]
        output[group] = {"pairing": {k: sum(p[k] for p in pairs) for k in ("units", "attempted_state_pairs", "retained_state_pairs", "shared_action_exposures")}, "banks": {}}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group]["banks"][label] = {}
            for penalty in PENALTIES:
                output[group]["banks"][label][str(penalty)] = {}
                for future in ("same", "opposite"):
                    rs = [r for r in records if r["penalty"] == penalty and r["future"] == future and (bank is None or r["bank"] == bank)]
                    arms = {}
                    for a in ARMS:
                        s = {p: learning.summarize([r["metrics"][a][p] for r in rs]) for p in POPULATIONS}
                        s["natural_gap"] = learning.contrast(s["excluded"], s["matched_held"])
                        s["paired_gap"] = learning.contrast(s["paired_excluded"], s["paired_held"])
                        for pop in POPULATIONS:
                            v = s[pop]
                            if v["units"]: identity_error = max(identity_error, abs(v["delta_MSE"]-v["correction_squared"]-v["alignment"]), abs(v["model_MSE"]-v["raw_MSE"]-v["delta_MSE"]))
                        old = before_excluded["groups"][group][label][str(penalty)]["excluded_"+future]
                        old_held = before_held["groups"][group]["banks"][label][str(penalty)][future][a]
                        replay_error = max(replay_error, abs(s["excluded"]["model_MSE"]-old["arms"][a]["MSE"]), abs(s["excluded"]["units"]-old["state_exposures"]),
                            abs(s["matched_held"]["model_MSE"]-old_held["prefix54_MSE_by_removal"]["matched"]))
                        arms[a] = s
                    output[group]["banks"][label][str(penalty)][future] = arms
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(len(c["pairing"]) for c in cells) != 1920 or replay_error > 1e-10 or identity_error > 1e-12: raise ValueError("Incomplete Stage274 attribution/replay")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"cell_metric_max_abs_error": max(c["metric_max_abs_error"] for c in cells), "group_metric_max_abs_error": replay_error,
            "pointwise_identity_max_abs_error": max(c["identity_max_abs_error"] for c in cells), "group_identity_max_abs_error": identity_error},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(analyze_cell(directory/("_".join(map(str, key))), key, views, panels));print(f"RISK_ALIGNMENT_CELL_COMPLETE {key}", flush=True)
    before_excluded = json.loads((ROOT/"results"/PREDICTION_RUN/"report/result.json").read_text())
    before_held = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(directory/"cells.json", cells);write_json(path, report(cells, before_excluded, before_held));print("RISK_ALIGNMENT_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
