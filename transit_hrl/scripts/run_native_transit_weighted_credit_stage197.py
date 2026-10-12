#!/usr/bin/env python3
"""Separate replicate-count weighting from direct action-credit supervision."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_mean_coverage_stage195 as source
from scripts import analyze_native_transit_mean_signal_stage196 as diagnostic
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_weighted_credit_stage197_v1"
SOURCE_RUN = diagnostic.SOURCE_RUN
DIAGNOSTIC_RUN = "native_transit_mean_signal_stage196_diagnostic_20261013_r1"
VARIANTS = ("anchored_uniform", "anchored_count_weighted", "advantage_uniform", "advantage_count_weighted")
SPLITS, STEPS = source.SPLITS, source.STEPS
value, paired = source.source.value, source.source.paired
state_key = source.state_key
FIT_FIELDS = ("method", "training_groups", "training_states", "steps", "parameter_count", "center", "scale",
    "target_scale", "initialization_seed", "representation", "objective", "target", "loss_ledger",
    "parameter_update_l2")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "diagnostic_run": DIAGNOSTIC_RUN, "data": "same600_inputs_240_mean8_targets_360_single_targets",
        "variants": list(VARIANTS), "primary": "advantage_count_weighted:wait_feasible",
        "weighting": "training_states_only_mean8_weight8_single_weight1_normalized_weighted_state_mean",
        "weight_limit": "replicate_count_proxy_not_estimated_heteroskedastic_inverse_variance",
        "anchored": "unchanged_softplus_absolute_fleet_anchor_plus_three_head_pair_MSE",
        "advantage": "linear_three_head_pair_MSE_only_fleet_pair_RMS_no_absolute_risk_forecast",
        "normalization": "same_unweighted_train_only_input_and_target_scales_within_each_objective",
        "training": "all_three_heads_refit_1635_parameters_100steps_seed178_AdamW_lr0.003_wd0.01",
        "initialization": "same_network_weights_not_same_decoded_fleet_predictions_across_objectives",
        "folds": "same40_scene_5_regime_2_lower_root_folds_all_callbacks_grouped",
        "evaluation": "same240_mean8_states_40_prefix_groups_original_and_mean_labels_both_fixed_selectors",
        "policies": list(value.POLICIES),
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean_label_splits",
        "budget": {"new_fits": 141, "reused_fits": 47, "critic_updates": 14100,
            "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_mean_map_models_predictions_remote_only_small_receipts_and_report",
        "boundary": "development_action_credit_screen_original_greedy_suffix_not_full_learned_upper_or_HRL"}


def manifest(previous):
    if previous["protocol"] != source.EXPERIMENT_PROTOCOL or previous["contract"] != source.contract():
        raise ValueError("Stage197 requires the registered Stage195 cohort")
    return {**previous, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def sample_counts(rows, targets):
    keys = {state_key(r) for r in targets}
    return [8 if state_key(r) in keys else 1 for r in rows]


def fit_fold(bank, targets, fold, reference, prior):
    train_old, test_old, _ = source.source.split_rows(bank, fold["split"], fold["index"])
    mean_keys = {state_key(r) for r in targets}
    test_old = [r for r in test_old if state_key(r) in mean_keys]
    train = source.expected.replace_targets(train_old, targets)
    test_mean = source.expected.replace_targets(test_old, targets)
    models = {VARIANTS[0]: reference}
    timings = {}
    for variant in VARIANTS[1:]:
        start = time.monotonic()
        weights = sample_counts(train, targets) if variant.endswith("count_weighted") else None
        if variant.startswith("anchored"):
            model = value.fit_value(train, "floor_absolute", steps=STEPS, objective="proper_mse",
                representation="tail_summary", sample_weights=weights)
        else:
            model = value.fit_advantage(train, steps=STEPS, representation="tail_summary", sample_weights=weights)
        timings[variant] = time.monotonic()-start
        common = ("training_groups", "training_states", "steps", "parameter_count", "center", "scale",
            "initialization_seed", "representation", "objective")
        if any(model[k] != reference[k] for k in common):
            raise ValueError("Objective comparison changed capacity, inputs, training scenes or budget")
        scale_reference = reference if variant.startswith("anchored") else models["advantage_uniform"] if weights else model
        if model["target_scale"] != scale_reference["target_scale"]:
            raise ValueError("Weighting changed normalization rather than just the state loss")
        models[variant] = model
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "fold": fold, "fits": {v: {**{k: m[k] for k in FIT_FIELDS},
            "sample_weight_sum": m.get("sample_weight_sum", len(train)),
            "absolute_fleet_forecast": not v.startswith("advantage")} for v, m in models.items()},
        "training_mean_states": sum(state_key(r) in mean_keys for r in train),
        "evaluation_mean_states": len(test_mean), "fit_wall_s": timings,
        "critic_updates": 3*STEPS, "actor_updates": 0, "new_native_episodes": 0,
        "predictions": {v: {label: {p: source.source.floor.evaluate(rows, m, p) for p in value.POLICIES}
            for label, rows in (("original", test_old), ("mean8", test_mean))} for v, m in models.items()}}
    if result["predictions"]["anchored_uniform"] != prior["predictions"]["mean8_overlay"]:
        raise ValueError("Cached control no longer reproduces Stage195 predictions and choices")
    return result, models


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    previous = json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text())
    if registered != manifest(previous):
        raise ValueError("Changed registered development cohort")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    bank = json.loads(source.source.bank_path(source.SOURCE_RUN).read_text())
    target_file = json.loads(source.expected.targets_path(SOURCE_RUN).read_text())
    if target_file["protocol"] != source.EXPERIMENT_PROTOCOL or target_file["contract"] != source.contract():
        raise ValueError("Target map does not belong to completed Stage195")
    prior_path = raw_directory(source.result_path(SOURCE_RUN, "tail_summary", fold))
    prior, reference = (json.loads((prior_path/name).read_text()) for name in ("result.json", "model.json"))
    result, models = fit_fold(bank, target_file["targets"], fold, reference, prior)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", result)
    for variant, model in models.items():
        write_json(raw_directory(output)/f"{variant}.json", model)
    write_json(output, {k: v for k, v in result.items() if k not in ("fits", "predictions")})
    print(f"WEIGHTED_CREDIT_FIT split={args.split} fold={args.fold_index} "
        f"mean_train={result['training_mean_states']} fit_s={sum(result['fit_wall_s'].values()):.3f}", flush=True)
    print("NATIVE_WEIGHTED_CREDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
