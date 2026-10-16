#!/usr/bin/env python3
"""Audit signed action-level fleet expectations in two cached future banks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_risk_influence_stage268 as source
from freq_hrl.domains.transit import native_signed_risk as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_signed_risk_stage269_v1"
SOURCE_RUN = "native_transit_risk_influence_stage268_20261016_r1"
cache, KEYS, VIEWS, PRIMARY = source.cache, source.KEYS, source.VIEWS, source.PRIMARY


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "banks": [list(range(16)), list(range(16, 32))],
        "target": "physical_fleet_action_minus_original_margin; added_and_saved_events_separate_at_existing_1e-12_tolerance; near_zero_remainder_preserved",
        "factorization": "empirical_p_times_conditional_amplitude_equals_direct_bank_mean; empty_amplitude_unknown_not_zero; factorization_alone_is_not_a_new_estimator",
        "gap": "bank1_minus_bank0; symmetric_probability_and_amplitude_terms_only_when_both_banks_support_that_sign; one_sided_support_reported_separately; all_signed_cross_terms_retained",
        "unit": "unique_root_scene_probe_action; eligible_in_at_least_one_training_context_per_group; reused_cells_folds_banks_and_model_exposures_are_not_independent_replications",
        "groups": "all_eligible_and_active_removed_prefix_states_by_frozen_family; mixed_pooled_and_four_views; all_existing_prefixes_descriptive; no_outcome_selection",
        "controls": "reproduce_Stage268_cell_states_and_any_eligible_pair_events; original_action_zero; unique_raw_targets_identical_across_reused_contexts; exact_mean_gap_and_scatter_identities",
        "focus_prefix": list(source.FOCUS_PREFIX), "focus_basis": source.contract()["focus_basis"],
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "parent_prefix_model_records": 1280,
            "new_critic_fits": 0, "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "panels_references_context_ledgers_action_arrays_and_full_aggregate_server_only; pull_compact_scalar_summary",
        "boundary": "training_label_stability_audit_not_new_futures_estimator_gain_seed_CI_KKT_variance_or_policy_performance; no_fit_selection_or_default_change; Stage253_248_169_closed"}


def analyze_cell(key, views, panels, before):
    reference = source.source.source.source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, _ = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    data = cache.learning.base.causal_data(train, reference)
    states = [list(cache.sampling.old_future.learning.state_key(r)) for r in train]
    values = cache.learning.base.training_draws(train, panels, list(range(32)))
    physical = values[..., (2, 1)]
    paired = physical-physical[np.arange(len(train)), :, data["originals"]][:, :, None]
    eligible = data["raw_logits"] <= 0
    events = source.learning.fleet_events(paired, eligible)
    masks = {"all_eligible": np.ones(len(train), bool), **{v: np.zeros(len(train), bool) for v in PRIMARY}}
    for p in before["pieces"]:
        if p["_states"] != states or not np.array_equal(p["_events"], events): raise ValueError("Stage268 events/states changed")
        if source.source.source.source.in_category(p, "removed") and p["active"]:
            masks[p["variant"]] |= ~np.asarray(p["_retained"], bool)
    if np.any(paired[np.arange(len(train)), :, data["originals"]] != 0): raise ValueError("Original anchor changed")
    feasible = eligible&(np.arange(6)[None] != data["originals"][:, None])
    return {"key": list(key), "states": states, "originals": data["originals"],
        "margins": paired[..., 1], "feasible": feasible, "masks": masks}


def collect(cells):
    targets, memberships, unions = {}, {}, {}
    for cell in cells:
        group = cell["key"][0]
        for i, state in enumerate(cell["states"]):
            for action in range(6):
                if action == int(cell["originals"][i]): continue
                key = (*state, action)
                y = cell["margins"][i, :, action]
                if key in targets and not np.array_equal(targets[key], y): raise ValueError("Reused physical action target changed")
                targets[key] = y
                if not cell["feasible"][i, action]: continue
                for label, mask in cell["masks"].items():
                    if mask[i]:
                        memberships.setdefault((group, label), set()).add(key)
                        unions.setdefault(label, set()).add(key)
    keys = sorted(targets)
    data = learning.audit(np.stack([targets[k] for k in keys]))
    return keys, data, memberships, unions


def report(cells):
    keys, data, memberships, unions = collect(cells)
    groups = {"mixed_pooled": ["mixed_a", "mixed_b"], **{v: [v] for v in VIEWS}}
    result, selected = {}, {}
    for name, views in groups.items():
        result[name] = {}
        for label in ("all_eligible", *PRIMARY):
            ids = set().union(*(memberships.get((v, label), set()) for v in views))
            selected[name, label] = ids
            result[name][label] = learning.summarize(data, [k in ids for k in keys])
    all_eligible = unions.get("all_eligible", set())
    prefixes = sorted({k[:2] for k in all_eligible})
    by_prefix = {"_".join(map(str, p)): learning.summarize(data, [k in all_eligible and k[:2] == p for k in keys]) for p in prefixes}
    focal = {v: learning.summarize(data, [k in selected["mixed_pooled", v] and k[:2] == source.FOCUS_PREFIX for k in keys]) for v in PRIMARY}
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS): raise ValueError("Incomplete Stage269 cells")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "groups": result, "all_unique_eligible": learning.summarize(data, [k in all_eligible for k in keys]),
        "prefixes": by_prefix, "focus_mixed_active_removed": focal,
        "reproduction": {"cached_cells": len(cells), "unique_state_actions_before_gate": len(keys),
            "unique_states": len({k[:3] for k in keys}), "unique_prefixes": len(prefixes), **data["errors"]},
        "default_changed": False, "native_deployment_eligible": False}, keys, data


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report_raw/cells.json").read_text())
    lookup = {tuple(c["key"]): c for c in before}
    if len(before) != 32 or set(lookup) != set(KEYS): raise ValueError("Incomplete Stage268 ledger")
    cells = []
    for key in KEYS:
        cells.append(analyze_cell(key, views, panels, lookup[key]));print(f"SIGNED_RISK_CELL_COMPLETE {key}", flush=True)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    payload, keys, arrays = report(cells)
    write_json(directory/"contexts.json", cells)
    np.savez_compressed(directory/"action_targets.npz", state_actions=np.asarray(keys), **{k: v for k, v in arrays.items() if k != "errors"})
    write_json(directory/"aggregate.json", payload);write_json(path, payload)
    print("SIGNED_RISK_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
