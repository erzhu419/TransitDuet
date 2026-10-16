#!/usr/bin/env python3
"""Attribute frozen Stage279 selected-pair errors to risk and service margins."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_risk_policy_stage279 as source
from scripts.run_native_transit_prefix_influence_stage259 import read_cell
from freq_hrl.domains.transit import native_decision_margin as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_decision_margin_stage280_v1"
SOURCE_RUN = "native_transit_risk_policy_stage279_20261017_r1"
cache, KEYS, VIEWS, SCOPES = source.cache, source.KEYS, source.VIEWS, source.SCOPES
ACTUAL = dict(zip(("actual_cost", "actual_fleet", "actual_service", "actual_wait_min"), source.learning.FIELDS))


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "pairs": {k: list(v) for k, v in learning.PAIRS.items()},
        "frozen": "all_Stage279_choices_and_Stage278_risk_predictions; identical_raw_service_wait_gate_samples_and_draw_assignments; replay_only_no_new_action_rule_or_choice_changes",
        "identity": "actual_cost=predicted_total+fleet_error+service_error_on_same_proposed_minus_control_actions; each_controller_uses_same_service_but_own_risk_prediction; check_opposite_predicted_preferences",
        "pair_error": "both_risk_heads_evaluated_on_identical_saved_action_pairs; conditional_pair_MSE_and_MC_variance_of_mean_divide_by_disagreement_cases; component_means_divide_by_all_query_states; no_subtracting_noise_or_CI_claim",
        "events": "saved_pair_fleet_add_save_draw_counts_at_existing1e-12_zero_threshold; repeats_and_event_counts_not_independent_seeds",
        "evaluation": "same_assigned_opposite16_and_fixed8+8_halves; identical_train_retained_excluded_full_held_prefix_held_queries; primary_mixed_excluded_opposite_and_held_opposite",
        "controls": "all7_cached_choices_replay_exactly; selected_pair_outcomes_reproduce_Stage279_contrasts_per_query_and_group_all_views_banks_scopes_halves; predicted_margins_written_before_label_access",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_queries": 2048, "model_state_exposures": 84480,
            "new_fits": 0, "new_policy_choices": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "full_margin_ledgers_and_bank_half_report_remote_only; compact_scalar_pull",
        "boundary": "descriptive_development_selected_pair_diagnosis_not_causal_feature_attribution_new_prefix_confirmation_or_policy_training; Stage253_248_169_closed; no_default_change"}


def analyze_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    queries = json.loads((here(SOURCE_RUN)/"actions.json").read_text())
    reference = read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    predictions, risk_error = [], 0.
    with np.load(here(source.SOURCE_RUN)/"predictions.npz") as saved:
        for q in queries:
            label, ids = q["label"], np.asarray(q["ids"])
            d = {k: v[ids] for k, v in data[label].items()}
            choices = {a: np.asarray(v) for a, v in q["choices"].items()}
            risks = {a: saved[f'{q["index"]}_{q["scope"]}_{a}'].copy() for a in source.learning.ARMS}
            replay, error = source.learning.decisions(d, risks);risk_error = max(risk_error, error)
            if any(not np.array_equal(replay[a], choices[a]) for a in source.learning.ARMS): raise ValueError("Cached Stage279 choices changed")
            predictions.append({**q, "predicted": learning.predictions(d, risks, choices)})
    if len(predictions) != 64: raise ValueError("Incomplete cached query coverage")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"predicted_margins.json", predictions)
    print(f"DECISION_MARGINS_FROZEN {key}", flush=True)
    previous = json.loads((here(SOURCE_RUN)/"outcomes.json").read_text())
    records, replay_error, identity_error = [], 0., 0.
    for qi, q in enumerate(predictions):
        rs = [rows[q["label"]][j] for j in q["ids"]]
        for fi, (future, bank) in enumerate((("same", q["bank"]), ("opposite", 1-q["bank"]))):
            draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws(rs, panels, draws)
            stats, error = learning.audit(values, q["predicted"]);identity_error = max(identity_error, error)
            old = previous[2*qi+fi]
            if old["bank"] != q["bank"] or old["scope"] != q["scope"]+"_"+future: raise ValueError("Changed cached outcome order")
            for h, pairs in stats.items():
                for name, s in pairs.items():
                    a, b = learning.PAIRS[name];control = old["outcomes"][h]
                    replay_error = max(replay_error, abs(s["cases"]-control[a]["cases"]))
                    for f, g in ACTUAL.items(): replay_error = max(replay_error, abs(s["sums"][f]-(control[a]["sums"][g]-control[b]["sums"][g])))
            records.append({"bank": q["bank"], "scope": q["scope"]+"_"+future, "margins": stats})
    if replay_error > 1e-10 or len(records) != 128: raise ValueError("Stage279 pair-outcome replay failed")
    write_json(directory/"margins.json", records)
    return {"key": list(key), "records": records, "model_state_exposures": sum(len(q["ids"]) for q in queries),
        "raw_risk_error": risk_error, "outcome_error": replay_error, "identity_error": identity_error}


def report(cells, before):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["model_state_exposures"] for c in cells) != 84480: raise ValueError("Incomplete Stage280 coverage")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, error = {}, 0.
    for g, predicate in groups.items():
        rs = [r for c in cells if predicate(c["key"]) for r in c["records"]];output[g] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[g][label] = {}
            for scope in SCOPES:
                selected = [r for r in rs if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                output[g][label][scope] = {}
                for h in learning.policy.DRAW_GROUPS:
                    pairs = {p: learning.summarize([r["margins"][h][p] for r in selected]) for p in learning.PAIRS}
                    for p, s in pairs.items():
                        a, b = learning.PAIRS[p];old = before["groups"][g][label][scope]["outcomes"][h]
                        error = max(error, abs(s["cases"]-old[a]["cases"]))
                        for f, old_field in ACTUAL.items(): error = max(error, abs(s[f]-(old[a][old_field]-old[b][old_field])))
                    output[g][label][scope][h] = pairs
    if error > 1e-10: raise ValueError("Stage279 group contrast replay failed")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"raw_risk_error": max(c["raw_risk_error"] for c in cells), "cell_outcome_error": max(c["outcome_error"] for c in cells),
            "group_outcome_error": error, "margin_identity_error": max(c["identity_error"] for c in cells)},
        "default_changed": False, "native_deployment_eligible": False}


def compact_report(payload):
    def compact(g, scope, result):
        if g == "mixed_pooled" and scope in ("excluded_opposite", "held_opposite", "matched_held_opposite"): return result
        return {"all16": result["all16"]}
    return {**payload, "groups": {g: {b: {s: compact(g, s, v) for s, v in scopes.items()} for b, scopes in banks.items()} for g, banks in payload["groups"].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(analyze_cell(directory/("_".join(map(str, key))), key, views, panels));print(f"DECISION_MARGIN_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text());payload = report(cells, before)
    write_json(directory/"cells.json", cells);write_json(directory/"aggregate.json", payload)
    write_json(path, compact_report(payload));print("DECISION_MARGIN_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
