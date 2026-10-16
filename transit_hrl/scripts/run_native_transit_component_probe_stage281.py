#!/usr/bin/env python3
"""Separate fleet and service information using disjoint cached future banks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_decision_margin_stage280 as source
from freq_hrl.domains.transit import native_component_probe as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_component_probe_stage281_v1"
SOURCE_RUN = "native_transit_decision_margin_stage280_20261017_r1"
cache, KEYS, VIEWS, SCOPES = source.cache, source.KEYS, source.VIEWS, source.SCOPES


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arms": list(learning.ARMS),
        "information": "assigned_bank_only_finite_MC_action_means; fleet_only_raw_service; service_only_cross_raw_Linear9_observed_df_risk; both_components; no_outcome_based_winner_selection",
        "sample_counts": "MC_components_use_assigned16_or_fixed_first8_or_last8; frozen_model_training_unchanged_not_total_information_matching; exact_same_MC_selection_draws_for_fleet_service_and_joint; opposite16_and_its8+8_halves_never_enter_propose; same_bank_results_are_in_sample_descriptions",
        "frozen": "reuse_Stage279_queries_and_all7_choices_Stage278_risks_raw_service_wait_logits_original_first_ties_and_feasibility; no_new_gate_clip_or_model_fit",
        "evaluation": "unchanged_full_train_prefix_retained_excluded_full_outer_held_prefix_head_outer_held; primary_mixed_excluded_opposite_and_held_opposite; all_views_banks_reported; policy_denominator_all_states_component_MSE_state_balanced_eligible_actions",
        "controls": "all7_cached_choices_replay_exactly; three_reported_controls_reproduce_Stage279_all_group_bank_scope_half_outcomes; physical_cost=fleet+service; proposal_ledger_written_before_each_query_opposite_outcome_access",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_queries": 2048,
            "model_state_exposures": 84480, "new_probe_decisions": 1267200,
            "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "full_proposals_components_and_outcomes_remote_only; compact_scalar_report_pull",
        "boundary": "label_informed_information_diagnostic_not_deployable_oracle_upper_bound_or_policy_validation; finite16_or8_MC_risk_is_noisy; reused_development_futures_not_seed_CI; Stage253_248_169_closed; no_default_change"}


def analyze_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    queries = json.loads((here(source.SOURCE_RUN)/"actions.json").read_text())
    previous = json.loads((here(source.SOURCE_RUN)/"outcomes.json").read_text())
    reference = source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    records, raw_error, replay_error = [], 0., 0.
    directory.mkdir(parents=True, exist_ok=True)
    with np.load(here(source.source.SOURCE_RUN)/"predictions.npz") as saved:
        for qi, q in enumerate(queries):
            label, ids = q["label"], np.asarray(q["ids"])
            d = {k: v[ids] for k, v in data[label].items()}
            rs = [rows[label][j] for j in ids]
            risks = {a: saved[f'{q["index"]}_{q["scope"]}_{a}'].copy() for a in learning.policy.ARMS}
            replay, error = learning.policy.decisions(d, risks)
            raw_error = max(raw_error, error)
            if any(not np.array_equal(replay[a], q["choices"][a]) for a in learning.policy.ARMS):
                raise ValueError("Stage279 frozen choices changed")
            draws, _ = cache.sampling.learning.draw_assignment(q["bank"], "expanded")
            selection = cache.learning.base.training_draws(rs, panels, draws)
            choices, components, error = learning.propose(d, risks, selection)
            raw_error = max(raw_error, error)
            write_json(directory/f"proposal_{qi}.json", {**q, "selection_draws": draws,
                "choices": choices, "components": components})
            for fi, (future, bank) in enumerate((("same", q["bank"]), ("opposite", 1-q["bank"]))):
                evaluation_draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
                if future == "opposite" and set(draws)&set(evaluation_draws):
                    raise ValueError("Selection and opposite futures overlap")
                values = selection if future == "same" else cache.learning.base.training_draws(rs, panels, evaluation_draws)
                outcomes = learning.policy.outcomes(values, choices)
                old = previous[2*qi+fi]
                if old["bank"] != q["bank"] or old["scope"] != q["scope"]+"_"+future:
                    raise ValueError("Changed cached outcome order")
                for h in learning.policy.DRAW_GROUPS:
                    for a in learning.CONTROLS:
                        new, before = outcomes[h][a], old["outcomes"][h][a]
                        for f in ("cases", "changed_count", "linear9_disagreement_count"):
                            replay_error = max(replay_error, abs(new[f]-before[f]))
                        for f in learning.policy.FIELDS:
                            replay_error = max(replay_error, abs(new["sums"][f]-before["sums"][f]))
                records.append({"bank": q["bank"], "scope": q["scope"]+"_"+future,
                    "component_errors": learning.component_errors(d, components, values),
                    "selection_stability": learning.selection_stability(choices), "outcomes": outcomes})
    if len(queries) != 64 or len(records) != 128 or replay_error > 1e-10:
        raise ValueError("Incomplete Stage281 coverage or control replay")
    write_json(directory/"outcomes.json", records)
    return {"key": list(key), "records": records,
        "model_state_exposures": sum(len(q["ids"]) for q in queries),
        "raw_risk_error": raw_error, "outcome_error": replay_error}


def report(cells, before):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["model_state_exposures"] for c in cells) != 84480:
        raise ValueError("Incomplete Stage281 coverage")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, error = {}, 0.
    for g, predicate in groups.items():
        rs = [r for c in cells if predicate(c["key"]) for r in c["records"]]
        output[g] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[g][label] = {}
            for scope in SCOPES:
                selected = [r for r in rs if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                outcomes = {h: {a: learning.policy.summarize([r["outcomes"][h][a] for r in selected])
                    for a in learning.ARMS} for h in learning.policy.DRAW_GROUPS}
                for h, arms in outcomes.items():
                    old = before["groups"][g][label][scope]["outcomes"][h]
                    for a in learning.CONTROLS:
                        error = max(error, *(abs(value-old[a][f]) for f, value in arms[a].items()))
                if g != "mixed_pooled" or scope not in ("excluded_opposite", "held_opposite", "matched_held_opposite"):
                    outcomes = {"all16": outcomes["all16"]}
                output[g][label][scope] = {"outcomes": outcomes,
                    "selection_disagreement_rates": learning.summarize_stability([r["selection_stability"] for r in selected]),
                    "component_errors": learning.summarize_errors([r["component_errors"] for r in selected])}
    if error > 1e-10:
        raise ValueError("Stage279 aggregate control replay failed")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "groups": output, "reproduction": {"raw_risk_error": max(c["raw_risk_error"] for c in cells),
            "cell_outcome_error": max(c["outcome_error"] for c in cells), "group_outcome_error": error},
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
        print(f"COMPONENT_PROBE_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/source.SOURCE_RUN/"report/result.json").read_text())
    write_json(directory/"cells.json", cells)
    write_json(path, report(cells, before))
    print("COMPONENT_PROBE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
