#!/usr/bin/env python3
"""Attribute frozen Stage285 calibration and control differences on cached futures."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_level_stage285 as source
from freq_hrl.domains.transit import native_service_decision_error as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_service_decision_error_stage286_v1"
SOURCE_RUN = "native_transit_service_level_stage285_20261017_r1"
cache, KEYS, VIEWS, policy = source.cache, source.KEYS, source.VIEWS, source.policy
SCOPES = ("train_opposite", "excluded_opposite", "held_opposite", "matched_held_opposite")
GROUPS = ("mixed_pooled", *VIEWS)


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "scopes": list(SCOPES), "arms": list(learning.ARM_RISK), "comparisons": learning.PAIRS,
        "candidate": "best_raw_total_plus_each_frozen_risk_nonoriginal_feasible_action; raw_gate_and_lowest_candidate_index_ties; fixed_without_new_service_or_labels; no_candidate_for_empty_menu; store_predictions_before_evaluation",
        "errors": "service_risk_signed_cross_and_total_margin_MSE_at_same_fixed_candidate_plus_empirical_margin_sign_error; original_relative_targets; eligible_state_denominator; compare_to_Stage285_all_action_errors_without_replacing_objective",
        "paired": "action_level_vs_uniform_observed_level_vs_uniform_observed_vs_action_and_each_frozen_contrast_added_to_observed_level; all_state_cost_fleet_service_wait; all16_fixed8+8; all_views_banks_scopes_no_winner_selection",
        "MC": "opposite_bank_only_per_fit_bank; sum_model_view_fold_contributions_for_each_unique_state_future_before_variance; independent_continuation_seed_per_state_draw; conditional_normal95_MC_error_bars_per_bank_only_no_pooled_bank_CI; development_diagnostic_not_confirmation_or_seed_CI",
        "concentration": "prefix_contributions_with_full_state_denominator; fleet_up_down_exposure_counts_and_net_unique_seed_counts_separately; repeated_model_exposures_not_independent_samples",
        "controls": "used21_frozen_controller_outcomes_reproduce_Stage285_cells_and_reported_group_halves; paired_unique_unit_means_reproduce_controller_contrasts; service_plus_risk_margin_MSE_identity",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_queries": 1408, "model_state_exposures": 49920,
            "new_fits": 0, "new_policies": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "fixed_candidate_predictions_and_unique_unit_arrays_server_only_scalar_report_pull",
        "boundary": "reused_development_supervised_frozen_single_intervention_not_unseen_prefix_training_seed_CI_or_full_HRL; Stage253_248_169_closed; defaults_unchanged"}


def group_names(key):
    return ("mixed_pooled", key[0]) if key[0].startswith("mixed") else (key[0],)


def analyze_cell(directory, key, views, panels, units):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    queries = json.loads((here(SOURCE_RUN)/"predictions.json").read_text())
    reference = source.transfer.geometry.source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    frozen = []
    with np.load(here(source.transfer.RISK_RUN)/"predictions.npz") as saved:
        for qi, q in enumerate(queries):
            scope = q["scope"]+"_opposite"
            if scope not in SCOPES:
                continue
            ids = np.asarray(q["ids"])
            d = {k: v[ids] for k, v in data[q["label"]].items()}
            risks = {r: saved[f'{q["index"]}_{q["scope"]}_{r}'].copy() for r in learning.service.CONTROLS}
            services = {s: np.asarray(q["services"][s]) for s in learning.SERVICES}
            eligible, candidates, predictions = learning.candidates(d, risks, services)
            if not np.array_equal(eligible, q["eligible"]):
                raise ValueError("Changed frozen menu")
            frozen.append({"source_query": qi, "bank": q["bank"], "scope": scope, "label": q["label"], "ids": q["ids"],
                "eligible": eligible, "candidates": candidates, "predictions": predictions,
                "choices": {a: np.asarray(q["choices"][a]) for a in learning.ARM_RISK}})
    if len(frozen) != 44 or sum(len(q["ids"]) for q in frozen) != 1560:
        raise ValueError("Incomplete Stage286 frozen query coverage")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"candidate_predictions.json", frozen)
    print(f"SERVICE_DECISION_ERROR_CANDIDATES_FROZEN {key}", flush=True)
    old_records = json.loads((here(SOURCE_RUN)/"outcomes.json").read_text())
    records, maximum, replay = [], 0., 0.
    for q in frozen:
        ids, label = np.asarray(q["ids"]), q["label"]
        rs = [rows[label][i] for i in ids]
        draws, _ = cache.sampling.learning.draw_assignment(1-q["bank"], "expanded")
        values = cache.learning.base.training_draws(rs, panels, draws)
        originals = data[label]["originals"][ids]
        errors, residual = learning.errors(values, originals, q["eligible"], q["candidates"], q["predictions"])
        maximum = max(maximum, residual)
        outcomes = policy.outcomes(values, q["choices"])
        old = old_records[2*q["source_query"]+1]
        if (old["bank"], old["scope"]) != (q["bank"], q["scope"]):
            raise ValueError("Changed cached record order")
        for h, arms in outcomes.items():
            for a, entry in arms.items():
                before = old["outcomes"][h][a]
                for f in ("cases", "changed_count", "linear9_disagreement_count"):
                    replay = max(replay, abs(entry[f]-before[f]))
                replay = max(replay, *(abs(v-before["sums"][f]) for f, v in entry["sums"].items()))
        ix = np.arange(len(ids))
        deltas = np.stack([values[ix, :, q["choices"][a]]-values[ix, :, q["choices"][b]] for a, b in learning.PAIRS.values()], axis=1)
        states = [(r["root"], r["scene_seed"], r["probe_index"]) for r in rs]
        for g in group_names(key):
            bucket = units.setdefault((g, q["bank"], q["scope"]), learning.new_bucket())
            learning.add_units(bucket, states, deltas)
        records.append({"bank": q["bank"], "scope": q["scope"], "candidate_errors": errors, "outcomes": outcomes})
    if replay > 1e-10:
        raise ValueError("Stage285 cell outcome replay failed")
    write_json(directory/"outcomes.json", records)
    return {"key": list(key), "records": records, "model_state_exposures": 1560, "identity_error": maximum, "replay_error": replay}


def report(cells, before, units):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["model_state_exposures"] for c in cells) != 49920:
        raise ValueError("Incomplete Stage286 coverage")
    output, replay, unit_error = {}, 0., 0.
    for g in GROUPS:
        records = [r for c in cells if g in group_names(c["key"]) for r in c["records"]]
        output[g] = {}
        for bank in (0, 1):
            output[g][f"bank{bank}"] = {}
            for scope in SCOPES:
                rs = [r for r in records if r["bank"] == bank and r["scope"] == scope]
                outcomes = {h: {a: policy.summarize([r["outcomes"][h][a] for r in rs]) for a in learning.ARM_RISK} for h in policy.DRAW_GROUPS}
                errors = {h: {a: learning.summarize_errors([r["candidate_errors"][h][a] for r in rs]) for a in learning.ARM_RISK} for h in policy.DRAW_GROUPS}
                old = before["groups"][g][f"bank{bank}"][scope]
                for h, arms in outcomes.items():
                    if h in old["outcomes"]:
                        for a, entry in arms.items():
                            replay = max(replay, *(abs(v-old["outcomes"][h][a][f]) for f, v in entry.items()))
                paired = learning.summarize_units(units[g, bank, scope])
                for name, (a, b) in learning.PAIRS.items():
                    for f in policy.FIELDS:
                        expected = outcomes["all16"][a][f]-outcomes["all16"][b][f]
                        unit_error = max(unit_error, abs(paired[name]["mean"][f]-expected),
                            abs(sum(p[f] for p in paired[name]["prefix_contributions"])-expected))
                output[g][f"bank{bank}"][scope] = {"candidate_errors": errors, "outcomes": outcomes, "paired": paired}
    if max(replay, unit_error) > 1e-10:
        raise ValueError("Stage285 group replay or shared-future accounting failed")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"cell_replay_error": max(c["replay_error"] for c in cells), "group_replay_error": replay,
            "candidate_MSE_identity_error": max(c["identity_error"] for c in cells), "unique_unit_or_prefix_sum_error": unit_error},
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
    cells, units = [], {}
    for key in KEYS:
        cells.append(analyze_cell(directory/("_".join(map(str, key))), key, views, panels, units))
        print(f"SERVICE_DECISION_ERROR_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(directory/"cells.json", [{k: v for k, v in c.items() if k != "records"} for c in cells])
    for (g, bank, scope), bucket in units.items():
        states = sorted(bucket["units"])
        np.savez_compressed(directory/f"units_{g}_{bank}_{scope}.npz", states=np.asarray(states),
            values=np.stack([bucket["units"][s] for s in states]), exposures=np.asarray([bucket["state_cases"][s] for s in states]))
    write_json(path, report(cells, before, units))
    print("SERVICE_DECISION_ERROR_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
