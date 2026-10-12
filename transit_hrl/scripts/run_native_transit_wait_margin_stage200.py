#!/usr/bin/env python3
"""Calibrate only the wait filter over frozen Stage199 action-credit predictions."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_pair_ridge_stage199 as source
from freq_hrl.domains.transit import native_wait_calibration as calibrated
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_wait_margin_stage200_v1"
SOURCE_RUN = "native_transit_pair_ridge_stage199_development_20261013_r1"
SPLITS = source.SPLITS
POLICIES = calibrated.POLICIES


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "same600_inputs_240_mean8_targets_360_single_targets_47_grouped_outer_folds",
        "deployed_model": "frozen_Stage199_count_weighted_ridge_all_three_heads_294_coefficients",
        "partition": "four_whole_scene_inner_folds_seed200_stratified_root_regime_and_mean8_coverage",
        "inner_training": "own_training_only_input_and_pair_target_scales_ridge0.1_mean8_weight8_single_weight1",
        "calibration": "outer_training_mean8_states_once_selected_override_wait_residual_true_minus_predicted",
        "margin": "nonnegative_selected_residual_mean_or_mean_plus_z90_group_ratio_standard_error",
        "cluster_se": "sqrt(G/(G-1)*sum((R_g-bias*N_g)^2))/sum(N_g)_identity_only_groups_retained",
        "selection": "unchanged_cost_ranking_raw_wait_plus_global_training_margin_nonpositive_identity_exact_zero",
        "policies": list(POLICIES), "primary": "cluster_margin90",
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean_label_splits",
        "budget": {"new_closed_form_fits": 188, "reused_deployed_models": 47,
            "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_targets_models_detailed_calibration_predictions_server_only_receipts_and_small_report_sync",
        "boundary": "empirical_mean_wait_calibration_not_formal_OOD_coverage_CI_or_full_HRL_validation"}


def manifest(previous):
    if previous["protocol"] != source.EXPERIMENT_PROTOCOL or previous["contract"] != source.contract():
        raise ValueError("Wait calibration requires the exact Stage199 registered cohort")
    mean_groups = {source.source.paired.scene_group(r) for r in previous["mean_states"]}
    folds = []
    for fold in previous["folds"]:
        train = [g for g in previous["groups"] if source.group_key(g, fold["split"]) != fold["held_out"]]
        folds.append({**fold, "inner_partition": calibrated.partition_groups(train, mean_groups)})
    return {**previous, "protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "folds": folds}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def fit_fold(bank, targets, fold, deployed, prior):
    train, test, _ = source.diagnostic.coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {source.source.state_key(r) for r in targets}
    train = source.diagnostic.mean.expected.replace_targets(train, targets)
    test = source.diagnostic.mean.expected.replace_targets([r for r in test if source.source.state_key(r) in keys], targets)
    expected_groups = sorted(map(list, {source.source.paired.scene_group(r) for r in train}))
    if deployed["training_groups"] != expected_groups or deployed["training_states"] != len(train):
        raise ValueError("Frozen wait-calibration model changed outer training membership")
    start = time.monotonic()
    calibration = calibrated.crossfit(train, keys, fold["inner_partition"])
    reference = prior["predictions"]["ridge_count_weighted"]["wait_feasible"]
    frozen = {source.source.state_key(r): r for r in reference}
    predictions = {p: calibrated.evaluate(test, frozen, calibration["margins"][p]) for p in POLICIES}
    for old, current in zip(reference, predictions["uncalibrated"], strict=True):
        if any(old[k] != current[k] for k in ("selected", "predicted_heads", "predicted_wait_delta_min", "actual_cost_delta", "wait_delta_min")):
            raise ValueError("Zero-margin control changed Stage199 predictions, units or selections")
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "deployed_fit": {k: deployed[k] for k in source.FIT_FIELDS},
        "calibration": calibration, "predictions": predictions,
        "new_closed_form_fits": 4, "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0,
        "fit_and_evaluation_wall_s": time.monotonic()-start}
    return result


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if source.group_key(g, fold["split"]) != fold["held_out"]]
    membership = {source.source.paired.scene_group(g) for g in groups}
    mean_states = {source.source.state_key(r) for r in registered["mean_states"]}
    train_mean_groups = {k[:2] for k in mean_states if k[:2] in membership}
    test_keys = {source.source.state_key(r) for r in registered["mean_states"] if source.group_key(r, fold["split"]) == fold["held_out"]}
    calibration, fit = cell["calibration"], cell["deployed_fit"]
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["new_closed_form_fits"] == 4
            and cell["critic_optimizer_updates"] == cell["actor_updates"] == cell["new_native_episodes"] == 0
            and fit["training_groups"] == sorted(map(list, membership)) and fit["training_states"] == 6*len(membership)
            and fit["parameter_count"] == 294 and fit["steps"] == 0
            and calibration["partition"] == fold["inner_partition"]
            and {tuple(g) for g in calibration["groups"]} == train_mean_groups
            and calibration["calibration_states"] == 6*len(train_mean_groups)):
        raise ValueError("Changed frozen predictor, calibration population or budget")
    for held, inner in zip(fold["inner_partition"], calibration["fits"], strict=True):
        training = membership-{tuple(g) for g in held}
        if not (inner["held_groups"] == held and inner["training_groups"] == sorted(map(list, training))
                and inner["training_states"] == 6*len(training) and inner["parameter_count"] == 294):
            raise ValueError("Inner fit leaked a calibration or outer held-out scene")
    for policy in POLICIES:
        rows, margin = cell["predictions"][policy], calibration["margins"][policy]
        if margin < 0 or len(rows) != len(test_keys) or {source.source.state_key(r) for r in rows} != test_keys:
            raise ValueError("Changed global margin or held-out evaluation states")
        for row in rows:
            if row["selected"] != calibrated.choose(row, row["original_choice"], margin):
                raise ValueError("Wait calibration changed cost ranking or its registered filter")
            if row["calibrated_wait_delta_min"] != calibrated.corrected_wait(row, row["original_choice"], margin).tolist():
                raise ValueError("Wait correction changed physical units or identity")
    base = cell["predictions"]["uncalibrated"]
    for policy in POLICIES:
        for a, b in zip(base, cell["predictions"][policy], strict=True):
            if any(a[k] != b[k] for k in ("predicted_heads", "predicted_wait_delta_min")):
                raise ValueError("Wait calibration changed a deployed prediction head")


def summarize(cells, registered):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete all47 calibration folds before aggregation")
    for fold in registered["folds"]: qualify(cells[fold["split"], fold["index"]], fold, registered)
    variants = {}
    for policy in POLICIES:
        validation, margins = {}, {}
        for split in SPLITS:
            selected = [cells[split, f["index"]] for f in registered["folds"] if f["split"] == split]
            rows = [r for cell in selected for r in cell["predictions"][policy]]
            if len(rows) != 240 or len({source.source.state_key(r) for r in rows}) != 240:
                raise ValueError("Calibrated report lost a registered mean8 state")
            validation[split] = {"overall": source.choice_summary(rows),
                "by_root": {str(root): source.choice_summary([r for r in rows if r["root"] == root]) for root in source.diagnostic.mean.ROOTS}}
            values = [c["calibration"]["margins"][policy] for c in selected]
            margins[split] = {"min": min(values), "mean": float(np.mean(values)), "max": max(values)}
        gates = {"cost": all(x["mean_actual_cost_delta"] < 0 for s in validation.values() for x in s["by_root"].values()),
            "wait": all(x["mean_wait_delta_min"] <= 0 for s in validation.values() for x in s["by_root"].values())}
        variants[policy] = {"validation": validation, "margins_min": margins, "gates": gates}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "evaluation_states": 240, "evaluation_prefix_groups": 40, "variants": variants,
        "development_gate": all(variants[contract()["primary"]]["gates"].values()),
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("fit", "report"), required=True)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--fold-index", type=int)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    previous = json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text())
    if registered != manifest(previous): raise ValueError("Changed preregistered wait-calibration folds")
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered)
        write_json(directory/"report/result.json", result)
        print(json.dumps({k: v["gates"] for k, v in result["variants"].items()}, indent=2), flush=True)
        print("NATIVE_WAIT_MARGIN_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None: parser.error("Fit requires split and fold index")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    outer = {k: v for k, v in fold.items() if k != "inner_partition"}
    path = raw_directory(source.result_path(SOURCE_RUN, outer))
    deployed = json.loads((path/"ridge_count_weighted.json").read_text())
    prior = json.loads((path/"result.json").read_text())
    bank = json.loads(source.diagnostic.coverage.bank_path(source.diagnostic.mean.SOURCE_RUN).read_text())
    targets = json.loads(source.diagnostic.mean.expected.targets_path(source.source.SOURCE_RUN).read_text())["targets"]
    result = fit_fold(bank, targets, fold, deployed, prior)
    qualify(result, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", result)
    write_json(output, {k: v for k, v in result.items() if k not in ("calibration", "predictions", "deployed_fit")})
    print(f"WAIT_MARGIN_FIT split={args.split} fold={args.fold_index} margins={result['calibration']['margins']}", flush=True)
    print("NATIVE_WAIT_MARGIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
