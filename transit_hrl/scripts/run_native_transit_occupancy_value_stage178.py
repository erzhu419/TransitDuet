#!/usr/bin/env python3
"""Matched representation/objective screen using qualified native cost labels."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_context_stage177 as source
from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_occupancy_value as value
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_occupancy_value_stage178_v1"
SOURCE_RUN = "native_transit_value_context_stage177_frozen_20261011_r1"
PRIMARY = "occupancy_regret"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "methods": list(value.METHODS), "primary": PRIMARY,
        "representation": "old72_plus24_causal_floor_active_trip_and_committed_departure_overlap_features",
        "control": "extra24_zero_mask_same_96_16_2_tanh_architecture_initialization_and_training_budget",
        "loss": "paired_fleet_nonfleet_MSE_with_or_without_unit_weight_soft_expected_native_cost_regret",
        "steps": value.STEPS, "initialization_seed": value.SEED,
        "learning_rate": value.LEARNING_RATE, "weight_decay": value.WEIGHT_DECAY,
        "normalization": "feature_mean_std_and_shared_native_cost_RMS_training_scenes_only",
        "validation": ["10_leave_scene_out", "5_leave_regime_out", "2_leave_lower_root_out"],
        "grouping": "all_six_callbacks_and_candidates_of_each_scene_remain_together",
        "native_screen_gate": "registered_primary_negative_mean_cost_for_both_roots_in_all_three_splits",
        "budget": {"fits_per_method": 18, "critic_updates_per_method": 18*value.STEPS,
            "new_native_episodes": 0, "actor_updates": 0},
        "boundary": "development_one_action_value_screen_under_original_greedy_continuation_not_HRL_or_policy_confirmation"}


def training_rows(summary, labels):
    original = source.prior.training_rows(labels)
    if not (summary["protocol"] == source.EXPERIMENT_PROTOCOL and summary["contract"] == source.contract()
            and summary["software_qualified"] and summary["training_updates"] == 0
            and summary["new_native_steps"] == 613800 and summary["source_control_reproductions"] == 10
            and summary["training_states"] == 60 and summary["paired_labels_reused"] == 360
            and summary["fitting_groups"] == labels["fitting_groups"]):
        raise ValueError("Occupancy value fitting requires qualified Stage177 contexts")
    key = lambda r: (r["root"], r["scene_seed"], r["probe_index"])
    bank = {key(r): r for r in original}
    rows = summary["training_data"]
    if len(rows) != 60 or len({key(r) for r in rows}) != 60 or {key(r) for r in rows} != set(bank):
        raise ValueError("Missing or duplicate occupancy training states")
    for row in rows:
        if {k: v for k, v in row.items() if k != "occupancy_context"} != bank[key(row)]:
            raise ValueError("Occupancy fitting changed the qualified original native value labels")
        if not np.all(np.isfinite(value.features(row, PRIMARY))):
            raise ValueError("Nonfinite causal occupancy features")
    return rows


def fit_and_validate(summary, labels, method):
    rows, ledger = training_rows(summary, labels), []
    def fit(train):
        model = value.fit_value(train, method)
        ledger.append({k: model[k] for k in ("training_groups", "training_states", "steps", "parameter_count",
            "cost_scale", "parameter_update_l2", "loss_ledger")})
        print(f"OCCUPANCY_VALUE_FIT method={method} fit={len(ledger)} states={len(train)} "
            f"updates={model['steps']} parameter_update={model['parameter_update_l2']:.6f}", flush=True)
        return model
    validation = {split: paired.grouped_validation(rows, split, fit_model=fit, predictor=value.predict_value)
        for split in ("scene", "regime", "lower_root")}
    model = fit(rows)
    passed = all(v["mean_actual_cost_delta"] < 0 for split in validation.values() for v in split["by_root"].values())
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "method": method, "software_qualified": True,
        "model": model, "validation": validation, "fit_ledger": ledger,
        "critic_updates": sum(r["steps"] for r in ledger), "actor_updates": 0, "new_native_steps": 0,
        "apparent_training": paired.summarize_choices(paired.evaluate_action_value(rows, model, predictor=value.predict_value)),
        "value_screen_passed": passed, "native_screen_eligible": method == PRIMARY and passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=value.METHODS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = fit_and_validate(json.loads(args.data.read_text()), json.loads(args.labels.read_text()), args.method)
    write_json(args.output, result)
    print(json.dumps({"method": args.method, "native_screen_eligible": result["native_screen_eligible"],
        "validation": {k: v["overall"] for k, v in result["validation"].items()}}, indent=2), flush=True)
    print("NATIVE_OCCUPANCY_VALUE_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
