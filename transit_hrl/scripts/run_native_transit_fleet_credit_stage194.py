#!/usr/bin/env python3
"""Cross-fit fleet reference-risk/action-credit separation on unchanged targets."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_fleet_residual_stage193 as previous
from freq_hrl.domains.transit import native_fleet_credit as value
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

expected, coverage = previous.source, previous.source.source
EXPERIMENT_PROTOCOL = "native_transit_fleet_credit_stage194_v1"
PREVIOUS_RUN = "native_transit_fleet_residual_stage193_fit_20261012_r1"
STEPS, SPLITS = previous.STEPS, previous.SPLITS


def contract():
    return {"previous_run": PREVIOUS_RUN, "previous_protocol": previous.EXPERIMENT_PROTOCOL,
        "data": "unchanged600_states_100_scenes_30_mean8_targets_570_single_targets",
        "controls": "reuse17_Stage193_from_zero_and17_Stage192_scalar_models_frozen_nonfleet_wait",
        "variants": list(value.VARIANTS), "primary": "separated_pair_scaled:wait_feasible",
        "factorial": "shared16_or_disjoint8plus8_crossed_with_absolute_or_pair_RMS_loss_units",
        "separated_target": "reference_future_risk_original_only_anchor_candidate_credit_all15_pairs",
        "prediction": "floor_plus_clamp_zero_reference_risk_plus_contrast_minus_reference_contrast",
        "gradient": "pair_loss_detaches_reference_risk_disjoint_networks_no_contrast_gradient_into_risk",
        "units": "output_absolute_RMS_all_variants_loss_pair_RMS_only_pair_scaled_train_only_floor1e-6",
        "parameters": {"fleet_from_zero": 1601, **value.PARAMETERS},
        "fitting": "100steps_seed178_same16_initial_hidden_rows_zero_projections_AdamW0.003_wd0.01",
        "matching": "same_features_groups_normalization_initial_predictions_updates_and1601_parameters",
        "folds": "same10_scene_5_regime_2_lower_root_all_callbacks_grouped",
        "evaluation": "same30_states_original_and_mean8_labels_all_variants_and_cells_retained",
        "policies": list(coverage.value.POLICIES),
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean8_splits",
        "budget": {"new_fits": 51, "new_critic_updates": 5100, "reused_fits": 34,
            "cached_reference_updates": 3400, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_targets_full_models_predictions_server_only_compact_receipts_summary_sync",
        "boundary": "ten_prefix_development_screen_not_conditional_coverage_confirmation_or_full_learned_HRL"}


def manifest(registered):
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        **{k: registered[k] for k in ("groups", "folds", "mean_states")}}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def fit_fold(bank, targets, fold, reference, cached):
    original_train, test, _ = coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {expected.state_key(r) for r in targets}
    train = expected.replace_targets(original_train, targets)
    test = [r for r in test if expected.state_key(r) in keys]
    groups = sorted(map(list, {coverage.paired.scene_group(r) for r in train}))
    for model in (reference, cached):
        if not (model["training_groups"] == groups and model["training_states"] == len(train)
                and model["steps"] == STEPS and model["initialization_seed"] == value.floor.occupancy.SEED):
            raise ValueError("Fleet-credit control changed its training membership or budget")
    if not (reference["representation"] == "tail_summary" and reference["objective"] == "proper_mse"
            and cached["method"] == "fleet_from_zero" and cached["parameter_count"] == 1601
            and reference["parameter_count"] == 1635):
        raise ValueError("Fleet-credit experiment requires its registered cached controls")
    models, timings = {"fleet_from_zero": cached}, {}
    for method in value.METHODS:
        start = time.monotonic()
        model = value.fit_fleet(train, method, steps=STEPS)
        timings[method] = time.monotonic()-start
        if any(model[k] != cached[k] for k in ("center", "scale", "target_scale", "training_groups", "training_states", "steps", "initialization_seed")):
            raise ValueError("Fleet-credit comparison changed its inputs, scales or grouped budget")
        models[method] = model
    fields = ("method", "center", "scale", "target_scale", "training_groups", "training_states", "steps",
        "parameter_count", "initialization_seed", "initialization", "parameter_update_l2", "loss_ledger")
    fits = {m: {k: model[k] for k in fields} for m, model in models.items()}
    for method in value.METHODS:
        fits[method].update({k: models[method][k] for k in ("paired_target_scale", "paired_loss_scale", "baseline_pair_gradient")})
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "fits": fits, "training_mean_states": sum(expected.state_key(r) in keys for r in train), "evaluation_mean_states": len(test),
        "predictions": {v: {label: {p: value.evaluate(rows, reference, v, models, p) for p in coverage.value.POLICIES}
            for label, rows in (("original", test), ("mean8", expected.replace_targets(test, targets)))} for v in value.VARIANTS},
        "critic_updates": 3*STEPS, "actor_updates": 0, "new_native_steps": 0, "fit_wall_s": timings}, models


def match_previous(result, old):
    for variant in ("cached_absolute", "fleet_from_zero"):
        for label in ("original", "mean8"):
            for policy in coverage.value.POLICIES:
                new = {expected.state_key(r): r for r in result["predictions"][variant][label][policy]}
                prior = {expected.state_key(r): r for r in old["predictions"][variant][label][policy]}
                fields = ("selected", "predicted_heads", "predicted_wait_delta_min", "absolute_fleet_cost", "actual_cost_delta", "wait_delta_min")
                if set(new) != set(prior) or any(new[k][f] != prior[k][f] for k in new for f in fields):
                    raise ValueError("Reused controls changed their Stage193 predictions, labels or choices")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    old_manifest = json.loads((ROOT/"results"/PREVIOUS_RUN/"inputs/manifest.json").read_text())
    if registered != manifest(old_manifest):
        raise ValueError("Fleet-credit worker changed its registered folds")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    bank = json.loads(coverage.bank_path(expected.SOURCE_RUN).read_text())
    targets = json.loads(expected.targets_path(previous.REFERENCE_RUN).read_text())
    if targets["contract"] != expected.contract():
        raise ValueError("Fleet-credit worker changed the prepared mean targets")
    reference = json.loads((raw_directory(expected.result_path(previous.REFERENCE_RUN, "tail_summary", fold))/"model.json").read_text())
    old_path = raw_directory(previous.result_path(PREVIOUS_RUN, fold))
    cached = json.loads((old_path/"models.json").read_text())["fleet_from_zero"]
    result, models = fit_fold(bank, targets["targets"], fold, reference, cached)
    match_previous(result, json.loads((old_path/"result.json").read_text()))
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", result)
    write_json(raw_directory(output)/"models.json", {m: models[m] for m in value.METHODS})
    write_json(output, {k: v for k, v in result.items() if k not in ("fits", "predictions")})
    print(f"FLEET_CREDIT_FIT split={args.split} fold={args.fold_index} fit_s={result['fit_wall_s']}", flush=True)
    print("NATIVE_FLEET_CREDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
