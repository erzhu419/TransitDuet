#!/usr/bin/env python3
"""Expand native action-value coverage and test matched head normalization."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_transfer_stage179 as source
from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_occupancy_value as value
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_value_balance_stage180_v1"
SOURCE_RUN = "native_transit_value_transfer_stage179_frozen_20261011_r1"
NORMALIZATIONS = ("joint_native", "paired_heads")
PRIMARY = "paired_heads"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "normalizations": list(NORMALIZATIONS), "primary": PRIMARY,
        "data": "all_120_states_720_labels_first_two_registered_scenes_20_groups_no_outcome_selection",
        "representation": "unchanged_causal_96_features_96_16_2_tanh_1586_parameters",
        "loss": "paired_MSE_plus_unit_weight_expected_regret_native_total_cost",
        "head_scaling": "joint_native_RMS_control_or_per_head_paired_RMS_training_scenes_only_floor_1e-6",
        "decision_units": "restore_both_heads_before_summing_for_regret_and_inference_no_new_cost_weights",
        "steps": value.STEPS, "initialization_seed": value.SEED,
        "learning_rate": value.LEARNING_RATE, "weight_decay": value.WEIGHT_DECAY,
        "validation": ["20_leave_scene_out", "5_leave_regime_out", "2_leave_lower_root_out"],
        "grouping": "all_six_callbacks_and_six_candidates_of_each_scene_remain_together",
        "screen": "primary_negative_mean_cost_and_nonpositive_wait_for_both_roots_in_all_three_splits",
        "budget": {"fits_per_normalization": 28, "critic_updates_per_normalization": 28*value.STEPS,
            "new_native_episodes": 0, "actor_updates": 0},
        "next": "separately_register_third_scene_frozen_transfer_before_repeated_learned_actions",
        "boundary": "expanded_development_fitting_and_grouped_screen_not_independent_confirmation_or_learned_HRL"}


def training_rows(first, original_labels, second):
    old = source.fitted.training_rows(first, original_labels)
    if not (second["protocol"] == source.EXPERIMENT_PROTOCOL and second["contract"] == source.contract()
            and second["software_qualified"] and second["evaluation_states"] == 60
            and second["native_steps"] == 420*61380 and second["source_control_reproductions"] == 120
            and second["paired_interventions"] == 360 and second["window_ledger_qualified_episodes"] == 420
            and second["training_updates"] == second["actor_updates"] == second["critic_updates"] == 0):
        raise ValueError("Expanded fitting requires complete qualified frozen Stage179 labels")
    new = second["evaluation_data"]
    old_groups = {paired.scene_group(r) for r in old}
    expected = {(g["root"], g["scenario"], g["scene_seed"], i)
        for g in second["evaluation_groups"] for i in source.PROBES}
    keys = [(r["root"], r["scenario"], r["scene_seed"], r["probe_index"]) for r in new]
    if len(new) != 60 or len(set(keys)) != 60 or set(keys) != expected:
        raise ValueError("Expanded fitting dropped or duplicated a registered transfer state")
    rows = sorted([*old, *new], key=lambda r: (r["root"], r["scene_seed"], r["probe_index"]))
    if len(old_groups) != 10 or len({paired.scene_group(r) for r in rows}) != 20:
        raise ValueError("Expanded fitting overlaps original and second-scene groups")
    for row in new:
        if paired.scene_group(row) in old_groups:
            raise ValueError("Expanded fitting reused a first-bank scene")
        labels = {l["candidate"]: l for l in row["labels"]}
        if len(row["labels"]) != 6 or set(labels) != set(range(6)):
            raise ValueError("Expanded fitting requires all candidates, not realized winners")
        for label in labels.values():
            if not (np.isclose(label["actual_cost_delta"], sum(label[h] for h in paired.HEADS), rtol=0, atol=1e-12)
                    and np.isclose(label["physical_return_delta"], -100*label["actual_cost_delta"], rtol=0, atol=1e-10)):
                raise ValueError("Expanded fitting changed native cost units")
        if any(labels[row["original_choice"]][k] != 0 for k in
                (*paired.HEADS, "actual_cost_delta", "wait_delta_min", "episode_reward_delta")):
            raise ValueError("Expanded fitting changed an exact identity label")
        if not np.all(np.isfinite(value.features(row, source.fitted.PRIMARY))):
            raise ValueError("Nonfinite expanded causal features")
    return rows


def fit_and_validate(first, original_labels, second, normalization):
    rows, ledger = training_rows(first, original_labels, second), []
    def fit(train):
        model = value.fit_value(train, source.fitted.PRIMARY, normalization=normalization)
        ledger.append({k: model[k] for k in ("training_groups", "training_states", "steps", "parameter_count",
            "cost_scale", "regret_cost_scale", "normalization", "parameter_update_l2", "loss_ledger")})
        print(f"BALANCED_VALUE_FIT normalization={normalization} fit={len(ledger)} "
            f"states={len(train)} updates={model['steps']}", flush=True)
        return model
    validation = {split: paired.grouped_validation(rows, split, fit_model=fit, predictor=value.predict_value)
        for split in ("scene", "regime", "lower_root")}
    model = fit(rows)
    cost_passed = all(v["mean_actual_cost_delta"] < 0 for s in validation.values() for v in s["by_root"].values())
    wait_passed = all(v["mean_wait_delta_min"] <= 0 for s in validation.values() for v in s["by_root"].values())
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "normalization": normalization, "model": model, "fit_ledger": ledger, "validation": validation,
        "critic_updates": sum(r["steps"] for r in ledger), "actor_updates": 0, "new_native_steps": 0,
        "apparent_training": paired.summarize_choices(paired.evaluate_action_value(rows, model, predictor=value.predict_value)),
        "cost_screen_passed": cost_passed, "wait_screen_passed": wait_passed,
        "third_scene_screen_eligible": normalization == PRIMARY and cost_passed and wait_passed,
        "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalization", choices=NORMALIZATIONS, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = fit_and_validate(json.loads(args.first.read_text()), json.loads(args.labels.read_text()),
        json.loads(args.second.read_text()), args.normalization)
    write_json(args.output, result)
    print(json.dumps({"normalization": args.normalization, "third_scene_screen_eligible": result["third_scene_screen_eligible"],
        "validation": {k: v["overall"] for k, v in result["validation"].items()}}, indent=2), flush=True)
    print("NATIVE_VALUE_BALANCE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
