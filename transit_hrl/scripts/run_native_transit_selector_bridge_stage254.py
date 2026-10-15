#!/usr/bin/env python3
"""Audit fixed Stage252 OOF/full actions under common cached future outcomes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_selector_information_stage252 as source
from scripts import run_native_transit_selector_confirmation_stage253 as confirmation
from freq_hrl.domains.transit import native_selector_bridge as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_selector_bridge_stage254_v1"
SOURCE_RUN, CONFIRMATION_RUN = confirmation.SOURCE_RUN, "native_transit_selector_confirmation_stage253_20261015_r1"
ARMS, FILTERS = source.learning.ARMS, source.learning.FILTERS
KEYS, VIEWS = source.KEYS, source.VIEWS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL, "confirmation_run": CONFIRMATION_RUN,
        "cohort": "same32_cells4_views_two_banks_60_seen_states20_prefix_half_blocks_per_cell_bank",
        "actions": "Stage252_selected_keys_full_actions_and_cached_inner_candidate_actions_verbatim_both_filters_all_six_protocols",
        "data": "old_A2_or_new2_new8_validation_vs_common_opposite16_Stage249_futures_for_inner_and_full; never_impute_missing_Stage253_sparse_actions",
        "decomposition": "OOF_future_minus_validation_optimism; full_future_correction_minus_OOF_future_correction; full_raw_minus_inner_raw_reference; selected_action_gap=sum_last_two",
        "references": "cached_prefix_excluded_inner_vs_full_reference_inference_only; intentional_CV_difference_not_presumed_bug",
        "strata": list(learning.STRATA), "controls": "reproduce_Stage252_full_train60_cost_wait_service_fleet_and_selected_OOF_validation_cost_wait_by_root",
        "report": "both_banks_all_four_views_prefix_and_scenario_means; raw_choice_gate_and_selected_action_disagreement",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "new_native_episodes": 0, "new_critic_fits": 0,
            "new_raw_reference_fits": 0, "inner_reconstruction_solves": 0, "actor_updates": 0, "new_prefixes": 0},
        "boundary": "post_confirmation_development_diagnostic_reused_training_crossfit_futures_no_CI_no_tuning_no_default_change; Stage248_outcomes_and_Stage169_reserves_closed",
        "storage": "cached_actions_labels_and_references_server_only_small_scalar_summary_pull"}


def analyze_cell(key, views, panels, past):
    load = lambda run, name: json.loads((raw_directory(source.result_path(run, *key))/name).read_text())
    info, actions, cached = (load(SOURCE_RUN, name) for name in ("aggregate.json", "actions.json", "oof_choices.json"))
    if info["protocol"] != source.EXPERIMENT_PROTOCOL or info["contract"] != source.contract() or not info["coverage_qualified"]:
        raise ValueError("Require complete Stage252 cache")
    fixed, base = source.fixed, source.learning.base
    reference = load(fixed.contract()["reference_run"], "models.json")["reference"]
    refs = load(fixed.contract()["refit_run"], "models.json")["references"]
    trials = load(fixed.contract()["refit_run"], "result.json")["cell"]["trials"]
    indices = {(tuple(t["removed_group"]), t["half"]): str(i) for i, t in enumerate(trials)
        if t["removed_group"] and t["half"] is not None}
    train, _ = source.training.cohort.split_rows(views[key[0]], key[1])
    rows = {source.information.old_future.learning.state_key(r): r for r in train}
    ordered = list(rows)
    full_data = base.causal_data(train, reference)
    full_gate = {st: g for st, g in zip(ordered, full_data["raw_logits"] <= 0)}
    full_actions = {b: next(a for a in actions if a["bank"] == b and a["scope"] == "train60") for b in (0, 1)}
    full_maps = {}
    for bank, a in full_actions.items():
        if a["states"] != [list(st) for st in ordered] or a["choices"]["raw"] != full_data["raw_choices"].tolist():
            raise ValueError("Changed full raw reference or state order")
        full_maps[bank] = {v: dict(zip(ordered, cs)) for v, cs in a["choices"].items()}
    a, b = fixed.PARTITIONS[key[2]][key[3]], fixed.PARTITIONS[key[2]][1-key[3]]
    blocks, maximum, comparisons = [], 0., 0
    expected = {(bank, tuple(group), half) for bank in (0, 1)
        for group, _, _ in source.learning.joint.binding.policy.prefix_folds(train) for half in (0, 1)}
    if len(cached) != len(expected) or {(c["bank"], tuple(c["held_group"]), c["half"]) for c in cached} != expected:
        raise ValueError("Dropped cached inner fold")
    selections = {p["bank"]: p["selection"] for p in info["banks"]}
    for c in cached:
        bank, half, group = c["bank"], c["half"], tuple(c["held_group"])
        val_rows = [rows[tuple(st)] for st in c["states"]]
        data = base.causal_data(val_rows, refs[indices[group, half]])
        if data["raw_choices"].tolist() != c["raw_choices"]: raise ValueError("Changed cached OOF raw choices")
        _, evaluation = source.information.learning.draw_assignment(bank, "expanded")
        ev = base.training_draws(val_rows, panels, evaluation)
        _, _, old_draws = fixed.fixed.draw_assignment(a, b, half)
        validation = {"old_A2": base.training_draws(val_rows, past, old_draws),
            **{s: base.training_draws(val_rows, panels, source.learning.validation_draws(bank, half, s)) for s in source.learning.SELECTORS[1:]}}
        states = [tuple(st) for st in c["states"]]
        fg = np.stack([full_gate[st] for st in states])
        fr = [full_maps[bank]["raw"][st] for st in states]
        pieces = {}
        for m in source.learning.MODELS:
            for s in source.learning.SELECTORS:
                arm = f"{m}_{s}"
                for mode in FILTERS:
                    variant = f"{arm}_{mode}"
                    chosen = c["choices"][m][selections[bank][arm][mode] or "zero"]
                    full = [full_maps[bank][variant][st] for st in states]
                    pieces[variant] = learning.compare(validation[s], ev, chosen, full, c["raw_choices"], fr, data["raw_logits"] <= 0, fg)
        blocks.append({"bank": bank, "prefix": list(group), "half": half, "scenario": val_rows[0]["scenario"], "pieces": pieces})
        comparisons += len(states)*len(pieces)
    for bank in (0, 1):
        prior = next(x for x in info["banks"] if x["bank"] == bank)
        for arm in ARMS:
            for mode in FILTERS:
                variant = f"{arm}_{mode}"
                bs = [x for x in blocks if x["bank"] == bank]
                measured = learning.combine([x["pieces"][variant] for x in bs])["all"]["means"]
                expected_full = prior["matrix"]["train60"]["all"]["variants"][variant]
                for f, old in (("cost", "delta_cost"), ("fleet", "delta_risk"), ("nonfleet", "delta_service"), ("wait_min", "delta_wait_min")):
                    maximum = max(maximum, abs(measured["full_future"][f]-expected_full[old]))
                selected = selections[bank][arm][mode] or "zero"
                expected_validation = prior["summaries"][arm][selected]["pooled"]["by_root"]
                for root, old in expected_validation.items():
                    value = learning.combine([x["pieces"][variant] for x in bs if str(x["prefix"][0]) == root])["all"]["means"]["validation"]
                    maximum = max(maximum, abs(value["cost"]-old["delta_cost"]), abs(value["wait_min"]-old["delta_wait_min"]))
    if maximum > 1e-12: raise ValueError("Changed cached full or validation outcomes")
    print(f"SELECTOR_BRIDGE_CELL_COMPLETE {key} reproduction={maximum}", flush=True)
    return {"key": list(key), "blocks": blocks, "reproduction_max_abs_difference": maximum, "action_exposures": comparisons}


def summarize(records):
    variants = [f"{a}_{f}" for a in ARMS for f in FILTERS]
    def summary(blocks):
        return {v: learning.combine([b["pieces"][v] for b in blocks]) for v in variants}
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output = {v: {label: summary([b for r in records if pred(r["key"]) for b in r["blocks"] if bank is None or b["bank"] == bank])
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1))} for v, pred in groups.items()}
    mixed = [b for r in records if r["key"][0].startswith("mixed") for b in r["blocks"]]
    return {"groups": output, "mixed_by_scenario": {s: summary([b for b in mixed if b["scenario"] == s]) for s in sorted({b["scenario"] for b in mixed})},
        "mixed_by_prefix": {f"{p[0]}/{p[1]}": {v: s["all"] for v, s in summary([b for b in mixed if tuple(b["prefix"]) == p]).items()}
            for p in sorted({tuple(b["prefix"]) for b in mixed})}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    data, _, registered, _ = source.training.cohort.source.source.load_data()
    views = source.training.cohort.source.matched_views(data["training_data"], registered)
    panels = source.information.read_panels(source.previous.DATA_RUN)
    past = source.fixed.panel_cache()[0]
    records = [analyze_cell(key, views, panels, past) for key in KEYS]
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, **summarize(records),
        "case_unit": "state_prefix_half_model_exposures_not_independent_samples",
        "reproduction": {"action_exposures": sum(r["action_exposures"] for r in records),
            "max_abs_difference": max(r["reproduction_max_abs_difference"] for r in records)},
        "default_changed": False, "native_deployment_eligible": False}
    write_json(raw_directory(ROOT/"results"/args.run_name/"report/result.json")/"blocks.json", records)
    write_json(ROOT/"results"/args.run_name/"report/result.json", payload)
    print("SELECTOR_BRIDGE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
