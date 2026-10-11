#!/usr/bin/env python3
"""Fit paired native action costs with scene/regime/lower-root held out."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_direction_stage175 as source
from freq_hrl.domains.transit import native_action_value as value
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_action_value_stage176_v1"
SOURCE_RUN = "native_transit_value_direction_stage175_frozen_20261011_r1"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "target": "paired_actual_terminal_fleet_and_nonfleet_cost_not_forecast_or_episode_reward",
        "features": "9_causal_LF_physical_context_by_8_committed_geometry_cost_features_no_scene_or_root_ID",
        "fit": "pooled_72_parameter_per_head_linear_pairwise_ridge_RMS_training_only",
        "ridge": value.RIDGE, "hyperparameter_search": False,
        "validation": ["10_leave_scene_out", "5_leave_regime_out", "2_leave_lower_root_out"],
        "grouping": "all_6_callbacks_and_6_candidates_of_a_scene_remain_together",
        "native_screen_gate": "mean_cost_delta_negative_for_both_roots_in_all_three_splits",
        "native_episodes": 0, "policy_updates": 0,
        "boundary": "cross_fitted_single_intervention_value_screen_not_closed_loop_or_end_to_end_HRL_proof"}


def training_rows(summary):
    if not (summary["protocol"] == source.EXPERIMENT_PROTOCOL and summary["software_qualified"]
            and summary["contract"] == source.contract() and summary["training_states"] == 60
            and summary["training_updates"] == 0 and summary["native_steps"] == 420*61380
            and summary["source_control_reproductions"] == 120
            and summary["window_ledger_qualified_episodes"] == 420):
        raise ValueError("Action values require the qualified bidirectional training summary")
    rows = summary["training_data"]
    expected = {(g["root"], g["scene_seed"], i) for g in summary["fitting_groups"] for i in g["probes"]}
    actual = [(r["root"], r["scene_seed"], r["probe_index"]) for r in rows]
    if not (len(rows) == len(set(actual)) == 60 and set(actual) == expected
            and len({value.scene_group(r) for r in rows}) == 10):
        raise ValueError("Missing or duplicate training states/scene groups")
    for r in rows:
        labels = {l["candidate"]: l for l in r["labels"]}
        if not (len(r["labels"]) == len(r["candidates"]) == 6 and set(labels) == set(range(6))
                and len(r["native_state_v2"]) == 16 and r["direction"] == bool(r["native_state_v2"][9])):
            raise ValueError("Changed native training action or state geometry")
        for l in labels.values():
            if not np.isclose(l["actual_cost_delta"], sum(l[h] for h in value.HEADS), rtol=0, atol=1e-12):
                raise ValueError("Fleet/nonfleet labels do not add to physical terminal cost")
        if any(labels[r["original_choice"]][k] != 0 for k in ("actual_cost_delta", *value.HEADS)):
            raise ValueError("Identity action must have exactly zero paired physical cost")
        if not np.all(np.isfinite(value.action_features(r))):
            raise ValueError("Nonfinite causal action features")
    return rows


def fit_and_validate(summary):
    rows = training_rows(summary)
    validation = {split: value.grouped_validation(rows, split) for split in ("scene", "regime", "lower_root")}
    passed = all(v["mean_actual_cost_delta"] < 0 for split in validation.values() for v in split["by_root"].values())
    model = value.fit_action_value(rows)
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "training_states": len(rows), "training_scene_groups": 10, "fitted_models": 18,
        "new_native_steps": 0, "new_native_episodes": 0, "policy_updates": 0,
        "model": model, "validation": validation,
        "apparent_training": value.summarize_choices(value.evaluate_action_value(rows, model)),
        "native_screen_eligible": passed,
        "next_step": "separately_registered_native_closed_loop_screen" if passed
            else "diagnose_heldout_cost_heads_and_collect_missing_causal_value_context_before_native_deployment"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit_and_validate(json.loads(args.data.read_text()))
    write_json(args.output, result)
    print(json.dumps({"native_screen_eligible": result["native_screen_eligible"],
        "validation": {k: v["overall"] for k, v in result["validation"].items()}}, indent=2), flush=True)
    print("NATIVE_ACTION_VALUE_GROUPED_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
