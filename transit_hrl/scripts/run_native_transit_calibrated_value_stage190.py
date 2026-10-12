#!/usr/bin/env python3
"""Fit four proper-subset critics and calibrate on separate training scenes."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_coverage_value_stage188 as source
from freq_hrl.domains.transit import native_calibrated_value as calibrated
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_calibrated_value_stage190_v1"
SOURCE_RUN = "native_transit_coverage_value_stage188_fit_20261012_r1"
MODES, SPLITS = source.MODES, source.SPLITS
STEPS = source.STEPS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "unchanged_server_only_600_states_100_scenes_3600_labels",
        "modes": list(MODES), "primary": "timetable_sequence:calibrated",
        "outer_splits": list(SPLITS), "outer_folds": 107,
        "partition": "root_regime_stratified_seed190_ceil20percent_scene_calibration_then_four_leave_quarter_members",
        "members": calibrated.MEMBERS, "steps_per_member": STEPS,
        "method": "unchanged_floor_absolute_proper_mse_AdamW_lr0.003_wd0.01_seed178",
        "uncertainty": "std_of_paired_total_cost_and_wait_across_members_floor_10percent_proper_training_RMS_min1e-6",
        "calibration": "separate_cost_wait_90percent_upper_rank_max_standardized_residual_over_five_actions_and_six_callbacks_per_scene",
        "selection": "min_pessimistic_cost_subject_to_pessimistic_wait_nonpositive_identity_exact_zero_ties_identity",
        "policies": list(calibrated.POLICIES), "reference": "cached_Stage188_wait_feasible_same_test_states",
        "budget": {"fits": 856, "critic_updates": 856*STEPS, "actor_updates": 0, "new_native_episodes": 0},
        "gate": "strict_negative_mean_cost_and_nonpositive_mean_wait_both_roots_all_three_splits",
        "storage": "bank_models_detailed_predictions_server_only_compact_receipts_and_report_synced",
        "boundary": "development_single_action_screen_not_repeated_upper_or_HRL_no_OOD_coverage_guarantee"}


def manifest(registered):
    groups = registered["groups"]
    folds = []
    for fold in registered["folds"]:
        split = fold["split"]
        if split not in SPLITS:
            continue
        key = {"scene": lambda g: [g["root"], g["scene_seed"]], "regime": lambda g: g["scenario"],
            "lower_root": lambda g: g["root"]}[split]
        train = [g for g in groups if key(g) != fold["held_out"]]
        folds.append({**fold, "partition": calibrated.partition_groups(train)})
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "groups": groups, "folds": folds}


def result_path(run_name, mode, fold):
    return ROOT/"results"/run_name/"cells"/mode/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def fit_fold(bank, mode, fold):
    train, test, _ = source.split_rows(bank, fold["split"], fold["index"])
    groups = {source.paired.scene_group(r) for r in train}
    held = {tuple(g) for g in fold["partition"]["calibration_groups"]}
    cal_rows = [r for r in train if source.paired.scene_group(r) in held]
    proper = [r for r in train if source.paired.scene_group(r) not in held]
    floors = calibrated.training_scales(proper)
    models, ledgers = [], []
    start = time.monotonic()
    for member_groups in fold["partition"]["member_groups"]:
        member_keys = {tuple(g) for g in member_groups}
        if not member_keys or member_keys & held or not member_keys <= groups:
            raise ValueError("Member fit includes calibration or outer held-out scenes")
        rows = [r for r in proper if source.paired.scene_group(r) in member_keys]
        model = source.value.fit_value(rows, "floor_absolute", steps=STEPS, objective="proper_mse", representation=mode)
        if model["training_groups"] != member_groups:
            raise ValueError("Member fitted different training scenes")
        models.append(model)
        fields = ("training_groups", "training_states", "steps", "parameter_count", "initialization_seed",
            "objective", "representation", "parameter_update_l2")
        ledgers.append({k: model[k] for k in fields})
        if mode == "timetable_sequence":
            ledgers[-1].update({k: model[k] for k in
                ("encoder_parameter_update_l2", "sequence_projection_update_l2", "encoder_gradient_l2_max")})
    fit_seconds = time.monotonic()-start
    calibration = calibrated.calibrate(cal_rows, [calibrated.ensemble_prediction(r, models, floors) for r in cal_rows])
    predictions = {p: [] for p in calibrated.POLICIES}
    for row in test:
        prediction = calibrated.ensemble_prediction(row, models, floors)
        for policy in calibrated.POLICIES:
            multiplier = calibration["multipliers"] if policy == "calibrated" else [0., 0.]
            predictions[policy].append(calibrated.evaluate(row, prediction, multiplier))
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "mode": mode, "fold": fold, "fits": ledgers, "calibration": calibration,
        "uncertainty_floors": floors.tolist(), "predictions": predictions,
        "critic_updates": calibrated.MEMBERS*STEPS, "actor_updates": 0, "new_native_steps": 0,
        "fit_wall_s": fit_seconds, "evaluation_wall_s": time.monotonic()-start-fit_seconds}, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    bank = json.loads(source.bank_path(SOURCE_RUN).read_text())
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    if not (bank["protocol"] == source.EXPERIMENT_PROTOCOL and bank["contract"] == source.contract()
            and bank["software_qualified"] and registered == manifest(bank["manifest"])):
        raise ValueError("Calibration requires the frozen qualified bank and registered nested splits")
    fold = next(f for f in registered["folds"] if f["split"] == args.split and f["index"] == args.fold_index)
    output = result_path(args.run_name, args.mode, fold)
    result, models = fit_fold(bank, args.mode, fold)
    write_json(raw_directory(output)/"result.json", result)
    write_json(raw_directory(output)/"models.json", models)
    receipt = {k: v for k, v in result.items() if k not in ("fits", "predictions", "calibration", "fold")}
    receipt.update(fold={k: v for k, v in fold.items() if k != "partition"},
        fits=len(models), calibration_groups=len(result["calibration"]["groups"]),
        multipliers=result["calibration"]["multipliers"],
        outcomes={p: calibrated.summary(rows) for p, rows in result["predictions"].items()})
    write_json(output, receipt)
    print(f"CALIBRATED_FIT mode={args.mode} split={args.split} fold={args.fold_index} "
        f"multipliers={result['calibration']['multipliers']} fit_s={result['fit_wall_s']:.3f}", flush=True)
    print("NATIVE_CALIBRATED_VALUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
