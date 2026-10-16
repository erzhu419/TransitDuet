#!/usr/bin/env python3
"""Trace frozen old keys through aligned inner and full response paths."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_selection_attribution_stage256 as previous
from freq_hrl.domains.transit import native_old_key_response as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

source = previous.source
EXPERIMENT_PROTOCOL = "native_transit_old_key_response_stage257_v1"
SOURCE_RUN = "native_transit_selection_attribution_stage256_20261016_r1"
KEYS, VIEWS, ARMS, FILTERS = previous.KEYS, previous.VIEWS, previous.ARMS, previous.FILTERS
CATEGORIES = ("all", "unchanged", "removed", "removed_point_harm", "removed_no_strict_gain", "added", "rekey")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cache_run": previous.SOURCE_RUN, "cohort": source.contract()["cohort"], "arms": list(ARMS), "filters": list(FILTERS),
        "keys": "Stage252_prefix_refit_selected_old_keys_fixed_in_new_Stage255_inner_and_frozen_full_actions; None_is_raw",
        "path": list(learning.PATHS), "steps": list(learning.STEPS), "categories": list(CATEGORIES),
        "futures": "assigned_new2_or_new8_validation_vs_same_inner_actions_on_opposite16; inner_full_seen_share_state_and_future; opposite16_split_first8_last8",
        "prefix": "full_held_minus_full_seen_uses_same_model_bank_cell_but_different_prefixes_not_paired_state_or_causal_effect",
        "diagnostics": "pooled_rootwise_positive_cost_wait_unsafe_flags_half_replication_and_inner_zero_action_full_nonzero_response; half_safe_categories_verbatim_Stage256",
        "controls": "reproduce_old_key_new_inner_validation_by_root_Stage255; full_seen_and_held_Stage255; all_and_category_before_means_and_denominators_Stage256",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "new_critic_fits": 0, "new_raw_reference_fits": 0,
            "new_native_episodes": 0, "reference_inference": 0, "actor_updates": 0},
        "storage": "server_actions_keys_and_complete_Stage249_panels_only; no_models_fits_or_new_rollouts; pull_scalar_report_only",
        "boundary": "posthoc_reused_development_response_diagnostic; validation_is_selection_exposed; opposite_futures_independent_per_bank_not_independent_seeds; half_signs_not_CI; no_unique_causal_attribution_or_arm_selection; Stage253_248_outcomes_and_169_reserves_closed; no_default_change"}


def analyze_cell(key, panels):
    load = lambda name: json.loads((raw_directory(source.result_path(previous.SOURCE_RUN, *key))/name).read_text())
    info, actions, cached = (load(name) for name in ("aggregate.json", "actions.json", "oof_choices.json"))
    if info["protocol"] != source.EXPERIMENT_PROTOCOL or info["contract"] != source.contract() or not info["coverage_qualified"]:
        raise ValueError("Require qualified Stage255 actions")
    values = lambda states, draws: np.stack([panels[tuple(st)]["values"][draws] for st in states])
    pieces, maximum = [], 0.
    for bank in info["banks"]:
        number = bank["bank"]
        full = {scope: next(a for a in actions if a["bank"] == number and a["scope"] == scope) for scope in ("train60", "held60")}
        positions = {tuple(st): i for i, st in enumerate(full["train60"]["states"])}
        _, ev = source.source.information.learning.draw_assignment(number, "expanded")
        held_values = values(full["held60"]["states"], ev)
        for model in source.learning.MODELS:
            for selector in source.learning.SELECTORS:
                arm = f"{model}_{selector}"
                for mode in FILTERS:
                    old, new = (bank["selection"][b][arm][mode] for b in source.learning.BUNDLES)
                    variant = f"prefix_refit_{arm}_{mode}"
                    path_parts = {p: [] for p in learning.PATHS[:3]}
                    half_parts = [{p: [] for p in learning.PATHS[1:3]} for _ in (0, 1)]
                    disagree, cases = 0, 0
                    for c in (c for c in cached if c["bank"] == number):
                        roots = [st[0] for st in c["states"]]
                        ix = [positions[tuple(st)] for st in c["states"]]
                        raw = c["raw_choices"]
                        if raw != [full["train60"]["choices"]["raw"][i] for i in ix]:
                            raise ValueError("Aligned old-key reference actions differ")
                        inner = c["choices"][model][old or "zero"]
                        chosen = [full["train60"]["choices"][variant][i] for i in ix]
                        val_draws = source.source.learning.validation_draws(number, c["half"], selector)
                        path_parts["validation"].append(learning.action_stats(values(c["states"], val_draws), inner, raw, roots))
                        opposite = values(c["states"], ev)
                        for p, ch in (("inner_opposite", inner), ("full_seen", chosen)):
                            path_parts[p].append(learning.action_stats(opposite, ch, raw, roots))
                            for h in (0, 1): half_parts[h][p].append(learning.action_stats(opposite[:, h*8:(h+1)*8], ch, raw, roots))
                        disagree += int(np.count_nonzero(np.asarray(inner) != chosen)); cases += len(inner)
                    paths = {p: learning.combine(ps) for p, ps in path_parts.items()}
                    halves = [{p: learning.combine(ps) for p, ps in hp.items()} for hp in half_parts]
                    held = full["held60"]
                    for target, v in [(paths, held_values), *((halves[h], held_values[:, h*8:(h+1)*8]) for h in (0, 1))]:
                        target["full_held"] = learning.combine([learning.action_stats(v, held["choices"][variant], held["choices"]["raw"], [st[0] for st in held["states"]])])
                    summary = bank["summaries"][arm][old or "zero"]["pooled"]["by_root"]
                    for root, before in summary.items():
                        actual = paths["validation"]["by_root"][root]
                        if actual["cases"] != before["cases"]: raise ValueError("Dropped validation states or inner half")
                        for f in ("delta_cost", "delta_wait_min", "delta_unsafe"):
                            maximum = max(maximum, abs(actual[f]-before[f]))
                    for p, scope in (("full_seen", "train60"), ("full_held", "held60")):
                        for f in learning.FIELDS:
                            maximum = max(maximum, abs(paths[p]["all"][f]-bank["matrix"][scope]["all"]["variants"][variant][f]))
                    status = previous.learning.gate(bank["summaries"][arm][old], mode) if old is not None else None
                    pieces.append({"bank": number, "arm": arm, "filter": mode,
                        "old_key": old, "new_key": new, "category": previous.learning.transition(old, new),
                        "old_key_new_gate": status, "paths": paths, "halves": halves,
                        "action_disagreement": disagree/cases, "diagnosis": learning.diagnose(paths, halves)})
    if maximum > 1e-12: raise ValueError("Frozen-key validation or full outcomes changed")
    return {"key": list(key), "pieces": pieces, "reproduction_max_abs_difference": maximum}


def summarize(records, expected):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, maximum = {}, 0.
    for group, predicate in groups.items():
        pieces = [p for r in records if predicate(r["key"]) for p in r["pieces"]]
        output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for arm in ARMS:
                for mode in FILTERS:
                    xs = [p for p in pieces if p["arm"] == arm and p["filter"] == mode and (bank is None or p["bank"] == bank)]
                    variant = f"{arm}_{mode}"
                    partitions = {cat: [p for p in xs if cat == "all" or p["category"] == cat or
                        (cat == "removed_point_harm" and p["category"] == "removed" and not p["old_key_new_gate"]["strict_gain_only"]) or
                        (cat == "removed_no_strict_gain" and p["category"] == "removed" and p["old_key_new_gate"]["strict_gain_only"])] for cat in CATEGORIES}
                    summaries = {cat: learning.summarize(ps, len(xs)) for cat, ps in partitions.items()}
                    for scope, path in (("train60", "full_seen"), ("held60", "full_held")):
                        prior = expected["groups"][group][scope][label][variant]
                        for cat in ("all", *previous.learning.CATEGORIES):
                            before = prior["before"] if cat == "all" else prior["categories"][cat]["before_conditional_mean"]
                            n = prior["bank_cell_exposures"] if cat == "all" else prior["categories"][cat]["bank_cell_exposures"]
                            if n != summaries[cat]["bank_cell_exposures"]: raise ValueError("Changed Stage256 category denominator")
                            if n:
                                for f in learning.FIELDS:
                                    maximum = max(maximum, abs(summaries[cat]["conditional"]["path_means"][path][f]-before[f]))
                    output[group][label][variant] = summaries
    if maximum > 1e-12: raise ValueError("Frozen-key path failed Stage256 category reproduction")
    return output, maximum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    panels = source.source.information.read_panels(source.source.previous.DATA_RUN)
    records = [analyze_cell(key, panels) for key in KEYS]
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    groups, maximum = summarize(records, expected)
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": groups,
        "reproduction": {"cached_cells": len(records), "cell_max_abs_difference": max(r["reproduction_max_abs_difference"] for r in records),
            "category_max_abs_difference": maximum}, "default_changed": False, "native_deployment_eligible": False}
    path = ROOT/"results"/args.run_name/"report/result.json"
    write_json(raw_directory(path)/"cells.json", records)
    write_json(path, payload)
    print("OLD_KEY_RESPONSE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
