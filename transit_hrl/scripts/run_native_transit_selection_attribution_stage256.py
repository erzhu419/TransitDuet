#!/usr/bin/env python3
"""Separate admission loss from rekeying in the frozen Stage255 comparison."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_deployment_bundle_stage255 as source
from freq_hrl.domains.transit import native_selection_attribution as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_selection_attribution_stage256_v1"
SOURCE_RUN = "native_transit_deployment_bundle_stage255_20261016_r1"
KEYS, VIEWS = source.KEYS, source.VIEWS
ARMS, FILTERS = source.learning.ARMS, source.learning.FILTERS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arms": list(ARMS), "filters": list(FILTERS),
        "attribution": "unchanged_removed_added_rekey; disjoint_exact_paired_decomposition_on_entire_group_bank_cell_denominator",
        "units": "equal60_state_bank_cell_means_reused_across_views_not_independent_seeds; within_category_means_separate_from_contributions",
        "rekey": "rank_change_vs_same_rank_strength_change; frozen_full_models_and_references_identical_in_both_arms",
        "gate": "explain_old_key_under_new_inner_validation; existing_strict_max_root_cost_wait_score_nonharm_unsafe_and_optional_half_guards; no_reselection",
        "report": "four_views_mixed_pooled_both_banks_seen_held; reproduce_all_Stage255_group_means_and_contrasts",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "new_critic_fits": 0,
            "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "read_cached_cell_aggregate_keys_summaries_metrics_only; no_panels_models_actions_or_trajectories; scalar_summary_pull",
        "boundary": "posthoc_development_accounting_not_causal_identification_fresh_confirmation_or_policy_selection; Stage253_248_outcomes_and_169_reserves_closed; no_default_change"}


def analyze_cell(key):
    load = lambda run: json.loads((raw_directory(source.result_path(run, *key))/"aggregate.json").read_text())
    now, prior = load(SOURCE_RUN), load(source.SOURCE_RUN)
    if (now["protocol"] != source.EXPERIMENT_PROTOCOL or now["contract"] != source.contract()
            or tuple(now["key"]) != key or not now["coverage_qualified"]):
        raise ValueError("Require qualified Stage255 cell")
    rows = []
    for bank in now["banks"]:
        old_bank = next(b for b in prior["banks"] if b["bank"] == bank["bank"])
        for arm in ARMS:
            for mode in FILTERS:
                old, new = (bank["selection"][bundle][arm][mode] for bundle in source.learning.BUNDLES)
                gate_counts = {bundle: learning.gate_counts(summaries, selected, mode)
                    for bundle, summaries, selected in (("prefix_refit", old_bank["summaries"][arm], old), ("fixed_bundle", bank["summaries"][arm], new))}
                status = learning.gate(bank["summaries"][arm][old], mode) if old is not None else None
                category = learning.transition(old, new)
                rekey = ("rank" if source.learning.strength.CANDIDATES[old]["level"] != source.learning.strength.CANDIDATES[new]["level"] else "strength") if category == "rekey" else None
                for scope in ("train60", "held60"):
                    variants = bank["matrix"][scope]["all"]["variants"]
                    before, after = (variants[f"{bundle}_{arm}_{mode}"] for bundle in source.learning.BUNDLES)
                    rows.append({"bank": bank["bank"], "arm": arm, "filter": mode, "scope": scope,
                        "old_key": old, "new_key": new, "category": category, "rekey_kind": rekey,
                        "old_key_under_new_gate": status, "gate_candidates": gate_counts,
                        "before": {f: before[f] for f in learning.FIELDS}, "after": {f: after[f] for f in learning.FIELDS}})
    return {"key": list(key), "pieces": rows}


def summarize(records, expected):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, maximum = {}, 0.
    for group, predicate in groups.items():
        pieces = [p for r in records if predicate(r["key"]) for p in r["pieces"]]
        output[group] = {}
        for scope in ("train60", "held60"):
            output[group][scope] = {}
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
                arms = {}
                for arm in ARMS:
                    for mode in FILTERS:
                        variant = f"{arm}_{mode}"
                        xs = [p for p in pieces if p["scope"] == scope and p["arm"] == arm and p["filter"] == mode and (bank is None or p["bank"] == bank)]
                        actual = learning.account(xs)
                        prior = expected["groups"][group][scope][label]
                        for side, bundle in (("before", "prefix_refit"), ("after", "fixed_bundle")):
                            for f in learning.FIELDS:
                                maximum = max(maximum, abs(actual[side][f]-prior["outcomes"][f"{bundle}_{variant}"][f]))
                        for f, value in prior["alignment_contrasts"][variant].items():
                            maximum = max(maximum, abs(actual["change"][f]-value))
                        arms[variant] = actual
                output[group][scope][label] = arms
    if maximum > 1e-12: raise ValueError("Attribution failed Stage255 group-mean reproduction")
    return output, maximum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    records = [analyze_cell(key) for key in KEYS]
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    groups, maximum = summarize(records, expected)
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "groups": groups, "reproduction": {"max_abs_difference": maximum, "cached_cells": len(records)},
        "default_changed": False, "native_deployment_eligible": False}
    path = ROOT/"results"/args.run_name/"report/result.json"
    write_json(raw_directory(path)/"cells.json", records)
    write_json(path, payload)
    print("SELECTION_ATTRIBUTION_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
