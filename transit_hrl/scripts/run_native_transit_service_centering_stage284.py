#!/usr/bin/env python3
"""Diagnose frozen service calibration versus within-state action ranking."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_offset_stage283 as source
from freq_hrl.domains.transit import native_service_centering as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_service_centering_stage284_v1"
SOURCE_RUN = "native_transit_service_offset_stage283_20261017_r1"
cache, KEYS, VIEWS, SCOPES, policy = source.cache, source.KEYS, source.VIEWS, source.SCOPES, source.policy
PRIMARY = source.PRIMARY


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arms": list(learning.ARMS), "report_arms": list(learning.REPORT_ARMS),
        "decomposition": "uncut_delta=eligible_nonoriginal_action_mean_per_state+zero_mean_action_contrast; original_anchor0; raw_and_all_service_errors_MSE=state_level_MSE+within_state_contrast_MSE_with_state_balanced_eligible_weights",
        "ablation": "all3_cached_maps; level_only_centered_only_and_cached_offset_plus_centered; same_Stage282_training_radius_after_decomposition; no_fit_or_penalty_or_query_label_calibration; centering_uses_only_saved_predictions_and_raw_feasibility",
        "ranking": "state_balanced_pair_inversion_against_evaluation_mean_service_on_eligible_nonoriginal_pairs; empirical_truth_ties_omitted_predicted_ties_half_error; assigned_opposite16_and_fixed8+8; descriptive_MC_ranking_not_noise_free_ground_truth",
        "selection": "all3_frozen_risks_raw_wait_gate_and_original_first_ties_unchanged; freeze_all_choices_before_evaluation; uncut_rank_preserved_by_level_removal_but_clipping_can_change_contrasts",
        "evaluation": "same_train_retained_excluded_full60_held_prefix54_held_all_views_banks; primary_mixed_excluded_opposite_held_opposite; no_winner_selection; outcomes_relative_raw_and_switches_relative_own_frozen_risk_control",
        "controls": "reproduce_all42_Stage283_choices_cell_outcomes_and_service_MSE_bias_clip_fraction; reproduce_all_compact_group_metrics_and_outcomes; exact_MSE_and_switch_partition_identities",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_queries": 2048, "model_state_exposures": 84480,
            "new_ablation_decisions": 2280960, "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "full_predictions_and_query_ledgers_server_only; scalar_report_pull",
        "boundary": "reused_development_frozen_single_intervention_not_new_prefix_confirmation_training_seed_CI_or_full_HRL; MC_not_deployable; Stage253_248_169_closed; defaults_unchanged"}


def analyze_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    queries = json.loads((here(SOURCE_RUN)/"predictions.json").read_text())
    reference = source.source.geometry.source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    predictions = []
    with np.load(here(source.source.RISK_RUN)/"predictions.npz") as saved:
        for q in queries:
            ids, label = np.asarray(q["ids"]), q["label"]
            d = {k: v[ids] for k, v in data[label].items()}
            eligible = (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None])
            if not np.array_equal(eligible, q["eligible"]):
                raise ValueError("Changed raw feasibility")
            risks = {a: saved[f'{q["index"]}_{q["scope"]}_{a}'].copy() for a in learning.CONTROLS}
            services = {a: np.asarray(v) for a, v in q["services"].items()}
            services, clips = learning.ablate(services, q["parts"], eligible, d["originals"], q["radius"])
            choices = learning.previous.previous.decisions(d, risks, services)
            if any(not np.array_equal(choices[a], q["choices"][a]) for a in learning.previous.ARMS):
                raise ValueError("Stage283 frozen choice replay failed")
            predictions.append({**q, "services": services, "choices": choices,
                "clips": {**{a: np.asarray(v) for a, v in q["clips"].items()}, **clips}})
    if len(predictions) != 64:
        raise ValueError("Incomplete Stage284 query coverage")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"predictions.json", predictions)
    print(f"SERVICE_CENTERING_CHOICES_FROZEN {key}", flush=True)
    before = json.loads((here(SOURCE_RUN)/"outcomes.json").read_text())
    records, error, identity, switch_error = [], 0., 0., 0.
    for qi, q in enumerate(predictions):
        label, ids = q["label"], np.asarray(q["ids"])
        originals, eligible = data[label]["originals"][ids], np.asarray(q["eligible"])
        for fi, (future, bank) in enumerate((("same", q["bank"]), ("opposite", 1-q["bank"]))):
            draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws([rows[label][i] for i in ids], panels, draws)
            outcomes = policy.outcomes(values, q["choices"])
            metrics = learning.previous.previous.metrics(q["services"], q["clips"], q["services"]["raw"], values, originals, eligible, q["radius"])
            geometry, residual = learning.error_geometry(values, q["services"], originals, eligible)
            identity = max(identity, residual)
            scope = q["scope"]+"_"+future
            switches = learning.previous.switch_stats(values, {a: q["choices"][a] for a in learning.REPORT_ARMS}, originals, learning.ARM_RISK) if scope in PRIMARY else {}
            old = before[2*qi+fi]
            if (old["bank"], old["scope"]) != (q["bank"], scope):
                raise ValueError("Changed cached outcome order")
            for h, arms in outcomes.items():
                for a in learning.previous.ARMS:
                    for f in ("cases", "changed_count", "linear9_disagreement_count"):
                        error = max(error, abs(arms[a][f]-old["outcomes"][h][a][f]))
                    for f in policy.FIELDS:
                        error = max(error, abs(arms[a]["sums"][f]-old["outcomes"][h][a]["sums"][f]))
                for a, entry in switches.get(h, {}).items():
                    control = arms[learning.ARM_RISK[a]]
                    switch_error = max(switch_error, abs(sum(t["cases"] for t in entry["types"].values())-entry["cases"]))
                    for f in policy.FIELDS:
                        switch_error = max(switch_error, abs(sum(t["sums"][f] for t in entry["types"].values())-(arms[a]["sums"][f]-control["sums"][f])))
            for a in learning.previous.SERVICES:
                for f, v in metrics[a].items():
                    if v is not None:
                        error = max(error, abs(v-old["metrics"][a][f]))
            for a in learning.SERVICES:
                error = max(error, abs(geometry["all16"][a]["sums"]["MSE"]-metrics[a]["squared_error_sum"]))
            records.append({"bank": q["bank"], "scope": scope, "metrics": metrics,
                "outcomes": outcomes, "geometry": geometry, "switches": switches})
    if max(error, switch_error) > 1e-10:
        raise ValueError("Stage283 replay or switch partition failed")
    write_json(directory/"outcomes.json", records)
    return {"key": list(key), "records": records, "model_state_exposures": sum(len(q["ids"]) for q in queries),
        "replay_error": error, "identity_error": identity, "switch_error": switch_error}


def report(cells, before):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["model_state_exposures"] for c in cells) != 84480:
        raise ValueError("Incomplete Stage284 coverage")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, error = {}, 0.
    for g, predicate in groups.items():
        rows = [r for c in cells if predicate(c["key"]) for r in c["records"]]
        output[g] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[g][label] = {}
            for scope in SCOPES:
                rs = [r for r in rows if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                old = before["groups"][g][label][scope]
                halves = policy.DRAW_GROUPS if g == "mixed_pooled" and scope in (*PRIMARY, "matched_held_opposite") else ("all16",)
                metrics = {a: learning.previous.previous.base.summarize([r["metrics"][a] for r in rs], True) for a in learning.SERVICES}
                outcomes = {h: {a: policy.summarize([r["outcomes"][h][a] for r in rs]) for a in learning.REPORT_ARMS} for h in halves}
                geometry = {h: {a: learning.summarize([r["geometry"][h][a] for r in rs]) for a in learning.SERVICES} for h in halves}
                for a, m in old["service_metrics"].items():
                    error = max(error, *(abs(metrics[a][f]-v) for f, v in m.items() if v is not None))
                for h, arms in old["outcomes"].items():
                    for a, m in arms.items():
                        error = max(error, *(abs(outcomes[h][a][f]-v) for f, v in m.items()))
                result = {"service_metrics": {a: {f: m[f] for f in ("MSE", "bias", "clipped_action_fraction")} for a, m in metrics.items()},
                    "outcomes": outcomes, "error_geometry": geometry}
                if g == "mixed_pooled" and scope in PRIMARY:
                    result["switches"] = {a: learning.previous.summarize_switches([r["switches"]["all16"][a] for r in rs]) for a in learning.REPORT_ARMS}
                output[g][label][scope] = result
    if error > 1e-10:
        raise ValueError("Stage283 aggregate replay failed")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"cell_replay_error": max(c["replay_error"] for c in cells), "group_replay_error": error,
            "MSE_identity_error": max(c["identity_error"] for c in cells), "switch_partition_error": max(c["switch_error"] for c in cells)},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json"
    directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(analyze_cell(directory/("_".join(map(str, key))), key, views, panels))
        print(f"SERVICE_CENTERING_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(directory/"cells.json", [{k: v for k, v in c.items() if k != "records"} for c in cells])
    write_json(path, report(cells, before))
    print("SERVICE_CENTERING_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
