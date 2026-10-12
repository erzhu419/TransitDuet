#!/usr/bin/env python3
"""Fit two fleet-only heads per grouped fold on the existing mean-label overlay."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_expected_value_stage192 as source
from freq_hrl.domains.transit import native_fleet_residual as value
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_fleet_residual_stage193_v1"
REFERENCE_RUN = "native_transit_expected_value_stage192_fit_20261012_r1"
STEPS, SPLITS = source.STEPS, source.SPLITS


def contract():
    return {"reference_run": REFERENCE_RUN, "reference_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "same600_states_100_scenes_30_mean8_targets_570_single_targets",
        "controls": "reuse17_Stage192_scalar_mean8_models_freeze_nonfleet_and_wait_for_all_variants",
        "variants": list(value.VARIANTS), "primary": "fleet_residual:wait_feasible",
        "prior": "existing_causal_ensemble_full_candidate_fleet_cost_floored_at_historical_peak",
        "prediction": "floor_plus_max_zero_prior_future_plus_learned_residual_in_physical_units",
        "fleet_fit": "98_16_1_tanh_1601_parameters_100steps_seed178_shared_hidden_zero_output_AdamW0.003_wd0.01",
        "loss": "absolute_future_fleet_MSE_plus15_pair_fleet_MSE_train_only_absolute_RMS",
        "comparison": "from_zero_and_residual_have_same_features_groups_scales_loss_updates_and_initial_weights",
        "folds": "same10_target_scene_5_regime_2_lower_root_folds_all6_callbacks_grouped",
        "evaluation": "same30_states_original_and_mean8_labels_no_label_dependent_predictions",
        "policies": list(source.source.value.POLICIES),
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean8_splits",
        "increment": "residual_vs_prior_and_from_zero_same_state_cost_wait_differences_all_cells_reported",
        "budget": {"new_fits": 34, "reused_fits": 17, "new_critic_updates": 3400,
            "cached_reference_updates": 1700, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "existing_bank_targets_and_all_models_predictions_server_only_receipts_and_summary_sync",
        "boundary": "fleet_head_development_screen_not_full600_confirmation_or_learned_upper_authority"}


def manifest(registered):
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        **{k: registered[k] for k in ("groups", "folds", "mean_states")}}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def fit_fold(bank, targets, fold, reference):
    train_old, test, _ = source.source.split_rows(bank, fold["split"], fold["index"])
    keys = {source.state_key(r) for r in targets}
    train = source.replace_targets(train_old, targets)
    test = [r for r in test if source.state_key(r) in keys]
    expected = sorted(map(list, {source.source.paired.scene_group(r) for r in train}))
    if not (reference["training_groups"] == expected and reference["training_states"] == len(train)
            and reference["steps"] == STEPS and reference["representation"] == "tail_summary"
            and reference["objective"] == "proper_mse" and reference["parameter_count"] == 1635):
        raise ValueError("Fleet experiment requires the matched frozen scalar mean-label critic")
    models, timings = {}, {}
    for method in value.METHODS:
        start = time.monotonic()
        model = value.fit_fleet(train, method, steps=STEPS)
        timings[method] = time.monotonic()-start
        if (any(model[k] != reference[k] for k in ("center", "scale", "training_groups", "training_states", "initialization_seed", "steps"))
                or not np.isclose(model["target_scale"], reference["target_scale"][0], rtol=0, atol=1e-12)):
            raise ValueError("Fleet-only fit changed training membership or normalization")
        models[method] = model
    fields = ("method", "center", "scale", "target_scale", "training_groups", "training_states", "steps",
        "parameter_count", "initialization_seed", "initialization", "objective", "parameter_update_l2", "loss_ledger")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "fits": {k: {f: m[f] for f in fields} for k, m in models.items()},
        "training_mean_states": sum(source.state_key(r) in keys for r in train), "evaluation_mean_states": len(test),
        "predictions": {v: {label: {p: value.evaluate(rows, reference, v, models, p) for p in source.source.value.POLICIES}
            for label, rows in (("original", test), ("mean8", source.replace_targets(test, targets)))} for v in value.VARIANTS},
        "critic_updates": 2*STEPS, "actor_updates": 0, "new_native_steps": 0, "fit_wall_s": timings}, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    old_manifest = json.loads((ROOT/"results"/REFERENCE_RUN/"inputs/manifest.json").read_text())
    if registered != manifest(old_manifest):
        raise ValueError("Fleet residual worker changed registered scene folds")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    bank = json.loads(source.source.bank_path(source.SOURCE_RUN).read_text())
    targets = json.loads(source.targets_path(REFERENCE_RUN).read_text())
    if not (targets["protocol"] == source.EXPERIMENT_PROTOCOL and targets["contract"] == source.contract()):
        raise ValueError("Fleet residual requires unchanged prepared Stage192 targets")
    reference = json.loads((raw_directory(source.result_path(REFERENCE_RUN, "tail_summary", fold))/"model.json").read_text())
    result, models = fit_fold(bank, targets["targets"], fold, reference)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", result)
    write_json(raw_directory(output)/"models.json", models)
    write_json(output, {k: v for k, v in result.items() if k not in ("fits", "predictions")})
    print(f"FLEET_RESIDUAL_FIT split={args.split} fold={args.fold_index} fit_s={result['fit_wall_s']}", flush=True)
    print("NATIVE_FLEET_RESIDUAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
