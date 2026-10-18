#!/usr/bin/env python3
"""Describe the independent transfer of frozen screening signals, without tuning."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_abstention_transfer_stage294 as source
from scripts import run_native_transit_abstention_attribution_stage293 as attribution
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_screen_transfer_stage295_v1"
SOURCE_RUN = "native_transit_abstention_transfer_stage294_20261018_r1"
read = source.read


def directory(run):
    return ROOT/"results"/run


def contract():
    return {"source_run": SOURCE_RUN,
        "frozen": "all_Stage294_choices_matching_weights_profiles_prefixes_and_primary_conclusions",
        "calibration": "compare_selected_S8_means_with_independent_E32_and_both_E16_halves; S8_is_selection_not_performance_evidence",
        "categories": "retained_negative_S8_fleet_mean; retained_signed_cancellation; expected_thinning_on_zero_event_rejects_and_nonzero_retained",
        "support": "only_frozen_E32_action_union; missing_zero_weight_proposals_never_imputed_as_observed; all1440states_in_mean_denominator",
        "attribution": "reuse_Stage293_prefix_and_shared_future_variance_formula; descriptive_groups_extremes_leave_one_prefix; no_pruning",
        "events": "net_physical_state_future_signs_after_16profile_average; profile_state_counts_are_not_independent_units",
        "budget": {"tasks": 1, "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "boundary": "closed_confirmation_diagnostic_not_policy_selection_or_new_confirmation; no_threshold_grid_or_controller_promotion"}


def qualify_parent(receipt):
    path = source.directory(SOURCE_RUN)
    parent = read(path/"report/result.json")
    if (parent["contract"] != source.contract() or not parent["coverage_qualified"]
            or receipt["scheduler"]["collection_status_counts"] != {"done": 4196}
            or receipt["scheduler"]["collection_exit_code_counts"] != {"0": 4196}
            or receipt["scheduler"]["report"]["status"] != "done"
            or receipt["scheduler"]["report"]["exit_code"] != 0):
        raise ValueError("Complete the qualified independent Stage294 evaluation first")
    return parent


def load_cached():
    path = source.directory(SOURCE_RUN)
    sealed, plan = read(path/"sealed_raw/primary.json"), read(path/"seal/result.json")
    rows = [r for key in source.keys() for r in read(source.prefix_path(SOURCE_RUN, key)/"rows.json")]
    screen = np.zeros((len(rows), 8, 6, 4))
    evaluation = np.zeros((len(rows), 32, 6, 4))
    # Stage294 already audited receipts, seed identity, finiteness and components.
    for i, state in enumerate(sealed["states"]):
        key = tuple(state["key"])
        for d, label in enumerate(read(source.menu_path(SOURCE_RUN, key, state["probe"], "screen")/"labels.json")):
            for action, value in label["values"].items():
                screen[i, d, int(action)] = value
    for i in plan["active_indices"]:
        for chunk in range(4):
            for label in read(source.evaluation_path(SOURCE_RUN, i, chunk)/"labels.json"):
                for action, value in label["values"].items():
                    evaluation[i, label["draw"]-16, int(action)] = value
    return sealed, rows, screen, evaluation


def diagnostics(sealed, screen, evaluation, cube, rows, parent):
    rule = source.rule
    actual, selected = rule.effects(sealed, evaluation), rule.effects(sealed, screen)
    bridge_error = 0.
    for a, b in rule.PAIRS:
        delta = actual[:, :, rule.ARMS.index(a)]-actual[:, :, rule.ARMS.index(b)]
        for f, value in zip(rule.FIELDS, delta.mean(axis=(0, 1))):
            bridge_error = max(bridge_error, abs(value-parent["comparisons"][a+"_vs_"+b]["mean"][f]))
    for name, ds in (("first_half", slice(0, 16)), ("last_half", slice(16, 32))):
        for v, arm in enumerate(rule.ARMS):
            for f, value in zip(rule.FIELDS, actual[:, ds, v].mean(axis=(0, 1))):
                bridge_error = max(bridge_error, abs(value-parent["halves"][name][arm][f]))
    if bridge_error > 1e-13:
        raise ValueError("Frozen Stage294 means or E16 halves changed")
    choices = np.asarray(sealed["choices"])
    raw, proposal = choices[:, 0], np.asarray(sealed["proposal"])
    keep = choices[:, 2] != raw
    s_pair = rule.paired_values(screen, proposal, raw)
    e_pair = rule.paired_values(evaluation, proposal, raw)
    fleet_mean = s_pair[..., 1].mean(axis=2)
    probability = np.asarray(sealed["matched_thin_probabilities"])
    weights = {"retained_negative_mean": keep*(fleet_mean < 0),
        "retained_signed_cancellation": keep*(fleet_mean == 0),
        "thinning_zero_event_rejects": probability*~keep,
        "thinning_nonzero_retained": probability*keep}
    def mean(values):
        return dict(zip(rule.FIELDS, map(float, values.mean(axis=(0, 1)))))
    categories = {}
    for name, weight in weights.items():
        s = (s_pair*weight[:, :, None, None]).mean(axis=0)
        e = (e_pair*weight[:, :, None, None]).mean(axis=0)
        categories[name] = {"profile_state_support": int(np.count_nonzero(weight)),
            "expected_changes": float(weight.sum()), "screen8_contribution": mean(s),
            "evaluation32_contribution": mean(e), "first16_contribution": mean(e[:, :16]),
            "last16_contribution": mean(e[:, 16:])}
    transferred = {arm: {"screen8_selected_mean": mean(selected[:, :, v]),
        "evaluation32_mean": mean(actual[:, :, v])} for v, arm in enumerate(rule.ARMS)}
    attributed, prefixes = attribution.attribution(actual, cube, rows)
    events = {}
    for a, b in attribution.PAIRS:
        delta = actual[:, :, rule.ARMS.index(a)]-actual[:, :, rule.ARMS.index(b)]
        events[a+"_vs_"+b] = {name: {"net_fleet_positive": int(np.count_nonzero(delta[:, ds, 1] > 0)),
            "net_fleet_negative": int(np.count_nonzero(delta[:, ds, 1] < 0)),
            "net_fleet_zero": int(np.count_nonzero(delta[:, ds, 1] == 0))}
            for name, ds in (("all32", slice(None)), ("first16", slice(0, 16)), ("last16", slice(16, 32)))}
    return {"frozen_statistics_replayed": True, "bridge_max_abs_error": bridge_error,
        "screen_to_evaluation": transferred, "categories": categories,
        "physical_net_fleet_events": events, "prefix_attribution": attributed,
        "retained_profile_state_changes": int(keep.sum()),
        "retained_physical_states": int(np.any(keep, axis=0).sum()),
        "retained_prefixes": int(np.any(keep[:, cube], axis=(0, 3)).sum())}, prefixes


def analyze(run):
    registration = read(directory(run)/"inputs/preregistration.json")
    if registration["contract"] != contract():
        raise ValueError("Changed screen transfer diagnostic contract")
    parent = qualify_parent(registration["parent_completion"])
    sealed, rows, screen, evaluation = load_cached()
    result, prefixes = diagnostics(sealed, screen, evaluation, source.cube(rows), rows, parent)
    write_json(directory(run)/"report_raw/prefixes.json", prefixes)
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        **result, "primary_conclusion_unchanged": {k: parent[k] for k in
            ("joint_transfer_supported", "cost_information_supported", "fleet_information_supported")},
        "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0, "controller_promoted": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    analyze(parser.parse_args().run_name)
    print("SCREEN_TRANSFER_DIAGNOSTIC_COMPLETE", flush=True)
