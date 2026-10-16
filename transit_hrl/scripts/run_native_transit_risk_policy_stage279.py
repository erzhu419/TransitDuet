#!/usr/bin/env python3
"""Evaluate frozen risk-head substitutions on cached candidate outcomes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_effective_risk_stage278 as source
from freq_hrl.domains.transit import native_risk_policy as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_risk_policy_stage279_v1"
SOURCE_RUN = "native_transit_effective_risk_stage278_20261017_r1"
cache, KEYS, VIEWS, SCOPES = source.cache, source.KEYS, source.VIEWS, source.SCOPES


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arms": list(learning.ARMS),
        "selection": "freeze_all_Stage278_heads_and_clip_predictions; raw_total_plus_cached_risk_minus_raw_risk; raw_service_and_wait_logits_unchanged; original_identity_first_then_ascending_index_ties; no_strength_gate_or_arm_selection",
        "populations": "same64_queries_per_cell_full_train_held_prefix_retained_excluded_and_matched_outer_held; both_assigned_and_opposite16; outcomes_include_all_states_including_no_eligible_actions; risk_MSE_uses_existing_state_balanced_eligible_mask",
        "primary": "mixed_pooled_excluded_opposite_and_held_opposite; observed_df_vs_action_df_and_Linear9; cost_wait_fleet_service_deltas_to_frozen_raw_choice; no_winner_selection",
        "future_halves": "fixed_first8_last8_of_each16_reported_descriptively_no_tuning_or_gating",
        "controls": "exact_raw_choice_reproduction; cached_raw_risk_anchor_masks_radii_and_all7_risk_MSE_sums_replay_Stage278; total_cost_equals_fleet_plus_service; all_actions_written_before_future_outcome_access",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_contexts": 704, "cached_queries": 2048,
            "model_state_exposures": 84480, "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "actions_and_cell_outcome_ledgers_remote_only; pull_scalar_group_report",
        "boundary": "cached_single_intervention_plugin_test_not_full_episode_learning_new_prefix_confirmation_seed_CI_or_deployment; reused_development_futures; historical_reference_geometry_unchanged; Stage253_248_169_closed; no_default_change"}


def analyze_cell(directory, key, views, panels):
    here = ROOT/"results"/SOURCE_RUN/"report_raw"/("_".join(map(str, key)))
    models, queries, previous = (json.loads((here/f).read_text()) for f in ("models.json", "queries.json", "errors.json"))
    reference = source.source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    if {(r["root"], r["scene_seed"]) for r in train}&{(r["root"], r["scene_seed"]) for r in held}: raise ValueError("Outer prefix overlap")
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    lookup = {m["index"]: m for m in models}
    folds = {tuple(g): np.asarray(ix) for g, ix, _ in cache.learning.joint.binding.policy.prefix_folds(train)}
    for m in models:
        draws, _ = cache.sampling.learning.draw_assignment(m["bank"], "expanded")
        ix = np.arange(60) if m["removed"] is None else folds[tuple(m["removed"])]
        if m["training_draws"] != draws or not np.array_equal(m["fit_indices"], ix): raise ValueError("Changed cached model population")
    predictions, raw_error, cap_error = [], 0., 0.
    with np.load(here/"predictions.npz") as saved:
        for q in queries:
            m = lookup[q["index"]];label, ids = q["label"], np.asarray(q["ids"])
            if q["bank"] != m["bank"] or q["removed"] != m["removed"] or q["radius"] != m["radius"]: raise ValueError("Changed cached query context")
            d = {k: v[ids] for k, v in data[label].items()}
            eligible = (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None])
            if not np.array_equal(q["eligible"], eligible): raise ValueError("Changed feasibility mask")
            risks = {a: saved[f'{q["index"]}_{q["scope"]}_{a}'].copy() for a in learning.ARMS}
            for a, risk in risks.items():
                if np.any(risk[np.arange(len(ids)), d["originals"]] != 0): raise ValueError("Original risk anchor changed")
                cap_error = max(cap_error, float(np.max(np.abs((risk-risks["raw_prior"])[eligible]), initial=0.))-q["radius"])
            chosen, error = learning.decisions(d, risks);raw_error = max(raw_error, error)
            predictions.append({**q, "choices": chosen, "risks": risks})
    if len(models) != 22 or len(queries) != 64 or cap_error > 1e-12: raise ValueError("Incomplete context coverage or changed clip")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"actions.json", [{k: v for k, v in p.items() if k != "risks"} for p in predictions])
    print(f"RISK_POLICY_ACTIONS_FROZEN {key}", flush=True)
    records, metric_error = [], 0.
    for qi, q in enumerate(predictions):
        ids, label = np.asarray(q["ids"]), q["label"]
        rs = [rows[label][j] for j in ids];originals = data[label]["originals"][ids]
        for fi, (future, bank) in enumerate((("same", q["bank"]), ("opposite", 1-q["bank"]))):
            draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws(rs, panels, draws)
            target = values[..., 1]-values[np.arange(len(rs)), :, originals, 1][:, :, None]
            old = previous[2*qi+fi]
            if old["bank"] != q["bank"] or old["scope"] != q["scope"]+"_"+future: raise ValueError("Changed cached error order")
            metrics = {}
            for a, risk in q["risks"].items():
                stat = source.learning.previous.base.metrics(risk, q["risks"]["raw_prior"], target, np.asarray(q["eligible"]), q["radius"])
                for f in ("squared_error_sum", "state_exposures", "action_exposures"):
                    metric_error = max(metric_error, abs(stat[f]-old["metrics"][a][f]))
                metrics[a] = stat
            records.append({"bank": q["bank"], "scope": q["scope"]+"_"+future,
                "risk_metrics": metrics, "outcomes": learning.outcomes(values, q["choices"])})
    if len(records) != 128 or metric_error > 1e-10: raise ValueError("Stage278 risk error replay failed")
    write_json(directory/"outcomes.json", records)
    return {"key": list(key), "records": records, "model_state_exposures": sum(len(q["ids"]) for q in queries),
        "raw_risk_error": raw_error, "risk_metric_error": metric_error, "cap_excess": max(cap_error, 0.)}


def report(cells, before):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["model_state_exposures"] for c in cells) != 84480: raise ValueError("Incomplete Stage279 coverage")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, error = {}, 0.
    for g, predicate in groups.items():
        rs = [r for c in cells if predicate(c["key"]) for r in c["records"]];output[g] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[g][label] = {}
            for scope in SCOPES:
                selected = [r for r in rs if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                risk_mse = {}
                for a in learning.ARMS:
                    n = sum(r["risk_metrics"][a]["state_exposures"] for r in selected)
                    risk_mse[a] = sum(r["risk_metrics"][a]["squared_error_sum"] for r in selected)/n
                    error = max(error, abs(risk_mse[a]-before["groups"][g][label][scope]["arms"][a]["MSE"]))
                output[g][label][scope] = {"risk_MSE": risk_mse, "outcomes": {d: {a: learning.summarize([r["outcomes"][d][a] for r in selected]) for a in learning.ARMS} for d in learning.DRAW_GROUPS}}
    if error > 1e-10: raise ValueError("Stage278 group risk MSE changed")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"raw_risk_error": max(c["raw_risk_error"] for c in cells), "risk_metric_error": max(c["risk_metric_error"] for c in cells),
            "group_risk_MSE_error": error, "cap_excess": max(c["cap_excess"] for c in cells)},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(analyze_cell(directory/("_".join(map(str, key))), key, views, panels));print(f"RISK_POLICY_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(directory/"cells.json", cells);write_json(path, report(cells, before));print("RISK_POLICY_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
