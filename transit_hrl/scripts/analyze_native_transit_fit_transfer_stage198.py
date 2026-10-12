#!/usr/bin/env python3
"""Distinguish in-bag action-credit fitting from held-out transfer errors."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_weighted_credit_stage197 as source
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_fit_transfer_stage198_v1"
SOURCE_RUN = "native_transit_weighted_credit_stage197_development_20261013_r1"
FIELDS = source.diagnostic.FIELDS
COHORTS = ("train_mean8", "train_single", "heldout_mean8")
mean, coverage = source.source, source.source.source


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "unchanged600_inputs_240_mean8_targets_360_single_targets_all47_grouped_folds",
        "models": "188_frozen_Stage197_models_four_variants_no_refitting",
        "cohorts": list(COHORTS),
        "errors": "physical_pair_MSE_and_zero_gap_reference_in_each_cohort_unweighted_descriptive_errors",
        "training_objective": "final_model_training_weighted_pair_and_absolute_anchor_losses_in_fitted_scales",
        "sampling": "train_evaluations_repeat_states_across_folds_not_independent_samples",
        "noise": "no_training_noise_subtraction_labels_were_used_in_fitting",
        "heldout": "reuse_exact_archived_mean8_predictions_no_reselection_or_retuning",
        "shift": "mean8_train_vs_heldout_occupancy_and_plan_features_in_actual_training_input_scales",
        "budget": {"new_fits": 0, "critic_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_targets_models_and_fold_diagnostics_server_only_compact_summary_sync",
        "boundary": "descriptive_fit_transfer_diagnosis_not_CI_or_causal_lower_policy_attribution_or_HRL_proof"}


def arrays(rows, predictions):
    labels = [{l["candidate"]: l for l in r["labels"]} for r in rows]
    observed = np.asarray([[[l[i][f] for f in FIELDS] for i in range(6)] for l in labels])
    heads = np.asarray([p["predicted_heads"] for p in predictions])
    predicted = np.concatenate((heads.sum(axis=-1, keepdims=True), heads,
        np.asarray([p["predicted_wait_delta_min"] for p in predictions])[..., None]), axis=-1)
    return observed, predicted


def pair_records(rows, predictions):
    observed, predicted = arrays(rows, predictions)
    a, b = np.triu_indices(6, 1)
    gaps = observed[:, a]-observed[:, b]
    errors = (predicted[:, a]-predicted[:, b]-gaps)**2
    return [{**{k: row[k] for k in ("root", "scene_seed", "probe_index")},
        "pair_mse": errors[i].mean(axis=0).tolist(), "zero_pair_mse": (gaps[i]**2).mean(axis=0).tolist()}
        for i, row in enumerate(rows)]


def summary(records):
    mse = np.asarray([r["pair_mse"] for r in records]).mean(axis=0)
    zero = np.asarray([r["zero_pair_mse"] for r in records]).mean(axis=0)
    return {"model_state_evaluations": len(records),
        "unique_states": len({source.state_key(r) for r in records}),
        "unique_prefix_groups": len({source.paired.scene_group(r) for r in records}),
        "pair_metrics": {f: {"observed_pair_mse": float(mse[i]), "zero_gap_pair_mse": float(zero[i]),
            "mse_over_zero": float(mse[i]/zero[i]) if zero[i] > 0 else None,
            "error_change_vs_zero": float(mse[i]-zero[i])} for i, f in enumerate(FIELDS)}}


def fitted_objective(rows, predictions, model):
    observed, predicted = arrays(rows, predictions)
    a, b = np.triu_indices(6, 1)
    errors = ((predicted[:, a, 1:]-predicted[:, b, 1:]
        -(observed[:, a, 1:]-observed[:, b, 1:]))/model["target_scale"])**2
    weights = model.get("sample_weights", np.ones(len(rows)))
    paired_by_head = np.average(errors.mean(axis=1), axis=0, weights=weights)
    anchor = None
    if model["method"] != "paired_advantage":
        future = np.asarray([[l["future_fleet_cost"] for l in r["labels"]] for r in rows])
        absolute = np.asarray([p["absolute_fleet_cost"] for p in predictions])
        floors = np.asarray([p["causal_fleet_floor"] for p in predictions])[:, None]
        anchor = float(np.average((((absolute-floors-future)/model["target_scale"][0])**2).mean(axis=1), weights=weights))
    return {"final_training_loss": float(paired_by_head.mean())+(anchor if anchor is not None else 0),
        "final_paired_loss": float(paired_by_head.mean()), "absolute_fleet_anchor_loss": anchor,
        "paired_loss_by_head": dict(zip(FIELDS[1:], paired_by_head.tolist())),
        "first_step_preupdate_loss": model["loss_ledger"][0]["absolute_fleet_mse"]+model["loss_ledger"][0]["paired_mse"],
        "last_step_preupdate_loss": model["loss_ledger"][-1]["absolute_fleet_mse"]+model["loss_ledger"][-1]["paired_mse"]}


def physical_shift(train_features, test_features, scale):
    names = source.value.occupancy.OCCUPANCY_NAMES+list(source.value.TAIL_NAMES)
    train_mean, test_mean = train_features.mean(axis=(0, 1)), test_features.mean(axis=(0, 1))
    low, high = train_features.min(axis=(0, 1)), train_features.max(axis=(0, 1))
    outside = ((test_features < low) | (test_features > high)).mean(axis=(0, 1))
    return {name: {"train_mean": float(train_mean[i]), "heldout_mean": float(test_mean[i]),
        "standardized_mean_shift": float((test_mean[i]-train_mean[i])/scale[i]),
        "heldout_outside_train_range_fraction": float(outside[i])} for i, name in enumerate(names, start=72)}


def diagnose_fold(train, test, raw_train, raw_test, mean_keys, models, prior):
    membership = sorted(map(list, {source.paired.scene_group(r) for r in train}))
    mean_mask = np.asarray([source.state_key(r) in mean_keys for r in train])
    records, objectives = {}, {}
    for variant in source.VARIANTS:
        model = models[variant]
        if model["training_groups"] != membership or model["training_states"] != len(train):
            raise ValueError("Fit-transfer diagnostic loaded a different model training population")
        predictions = source.value.predict_many(train, model, features_batch=raw_train)
        training = pair_records(train, predictions)
        heldout = prior["predictions"][variant]["mean8"]["wait_feasible"]
        if [source.state_key(r) for r in heldout] != [source.state_key(r) for r in test]:
            raise ValueError("Archived held-out predictions changed their target population")
        records[variant] = {"train_mean8": [r for r, keep in zip(training, mean_mask) if keep],
            "train_single": [r for r, keep in zip(training, mean_mask) if not keep],
            "heldout_mean8": pair_records(test, heldout)}
        objectives[variant] = fitted_objective(train, predictions, model)
    shift = physical_shift(raw_train[mean_mask], raw_test, np.asarray(models["anchored_uniform"]["scale"]))
    return records, objectives, shift


def diagnose(bank, targets, registered, load_fold):
    if registered["protocol"] != source.EXPERIMENT_PROTOCOL or registered["contract"] != source.contract():
        raise ValueError("Fit-transfer diagnosis requires the registered Stage197 folds")
    rows = mean.expected.replace_targets(bank["training_data"], targets)
    by_key = {source.state_key(r): (i, r) for i, r in enumerate(rows)}
    mean_keys = {source.state_key(r) for r in targets}
    raw = np.stack([source.value.features(r, "tail_summary") for r in rows])
    pool = {v: {s: {c: [] for c in COHORTS} for s in source.SPLITS} for v in source.VARIANTS}
    ledgers, lower_folds, details = [], {}, []
    start = time.monotonic()
    for fold in registered["folds"]:
        old_train, old_test, _ = coverage.split_rows(bank, fold["split"], fold["index"])
        train = [by_key[source.state_key(r)][1] for r in old_train]
        test = [by_key[source.state_key(r)][1] for r in old_test if source.state_key(r) in mean_keys]
        train_indices = [by_key[source.state_key(r)][0] for r in train]
        test_indices = [by_key[source.state_key(r)][0] for r in test]
        models, prior = load_fold(fold)
        records, objectives, shift = diagnose_fold(train, test, raw[train_indices], raw[test_indices], mean_keys, models, prior)
        for variant in source.VARIANTS:
            for cohort in COHORTS: pool[variant][fold["split"]][cohort].extend(records[variant][cohort])
        ledgers.append({"fold": fold, "objectives": objectives})
        detail = {"fold": fold, "objectives": objectives, "physical_feature_shift": shift,
            "variants": {v: {c: summary(rs) for c, rs in scopes.items()} for v, scopes in records.items()}}
        details.append(detail)
        if fold["split"] == "lower_root": lower_folds[str(fold["held_out"])] = detail
        print(f"FIT_TRANSFER_DIAGNOSTIC split={fold['split']} fold={fold['index']}", flush=True)
    variants = {v: {s: {c: summary(records) for c, records in cohorts.items()}
        for s, cohorts in splits.items()} for v, splits in pool.items()}
    for splits in variants.values():
        for cohorts in splits.values():
            if cohorts["heldout_mean8"]["model_state_evaluations"] != 240:
                raise ValueError("Fit-transfer summary lost a registered held-out mean state")
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        **contract()["budget"], "reused_models": 188, "folds": 47, "variants": variants,
        "lower_root_folds": lower_folds, "training_objectives": ledgers, "inference_and_summary_wall_s": time.monotonic()-start,
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}
    return result, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/SOURCE_RUN
    completed = json.loads((directory/"report/result.json").read_text())
    if not (completed["protocol"] == source.EXPERIMENT_PROTOCOL and completed["contract"] == source.contract()
            and completed["software_qualified"] and completed["new_fits"] == 141):
        raise ValueError("Complete Stage197 before separating fit and transfer errors")
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    bank = json.loads(coverage.bank_path(mean.SOURCE_RUN).read_text())
    target_file = json.loads(mean.expected.targets_path(source.SOURCE_RUN).read_text())
    if target_file["protocol"] != mean.EXPERIMENT_PROTOCOL or target_file["contract"] != mean.contract():
        raise ValueError("Fit-transfer diagnosis changed the mean-target map")
    def load_fold(fold):
        path = raw_directory(source.result_path(SOURCE_RUN, fold))
        return ({v: json.loads((path/f"{v}.json").read_text()) for v in source.VARIANTS},
            json.loads((path/"result.json").read_text()))
    result, details = diagnose(bank, target_file["targets"], registered, load_fold)
    write_json(raw_directory(args.output)/"folds.json", details)
    write_json(args.output, result)
    print(json.dumps({v: {s: {c: d["pair_metrics"]["actual_cost_delta"]["mse_over_zero"]
        for c, d in cohorts.items()} for s, cohorts in splits.items()} for v, splits in result["variants"].items()}, indent=2), flush=True)
    print("NATIVE_FIT_TRANSFER_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
