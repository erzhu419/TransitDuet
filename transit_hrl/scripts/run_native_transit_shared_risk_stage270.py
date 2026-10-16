#!/usr/bin/env python3
"""Controlled shared risk fitting on cached futures, without policy selection."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_signed_risk_stage269 as source
from freq_hrl.domains.transit import native_shared_risk as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_shared_risk_stage270_v1"
SOURCE_RUN = "native_transit_signed_risk_stage269_20261016_r1"
cache, KEYS, VIEWS = source.cache, source.KEYS, source.VIEWS
SCOPES = ("train_same", "train_opposite", "held_same", "held_opposite",
    "retained_same", "retained_opposite", "excluded_same", "excluded_opposite")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "rank": "8_fixed_full60_causal_198feature_TSVD_plus_intercept; no_label_based_feature_selection",
        "arms": list(learning.ARMS), "penalties": list(learning.PENALTIES), "primary_penalty": learning.PRIMARY_PENALTY,
        "objective": "state_balanced_physical_MSE; same_capacity_signed_two_head_magnitude_MSE_and_simplex_projection_control_vs_three_class_event_CE; CE_scaled3_for_uniform_contrast_curvature; ridge_slopes_only",
        "amplitude": "same_two_training_only_pooled_signed_event_amplitudes_for_signed_MSE_and_event_heads; absent_sign_unidentified_not_imputed; evaluation_oracle_reports_pooling_bias_only",
        "capacity": "d_direct_coefficients_vs_2d+2_for_each_signed_arm; same_frozen_features; projected_MSE_reuses_identical_fit; no_claim_equal_function_class_to_linear_regression",
        "controls": "unpenalized_intercept_and_shared_ridge_make_signed_MSE_physical_risk_equal_direct_MSE; verify_on_all_queries; preserve_service_predictions; original_action_zero",
        "split": "both_disjoint16_future_directions; full60_train_and_held60; ten_prefix54_label_OOF_fits_and_excluded6; reference_and_feature_geometry_conditioned_on_full60_historical_data",
        "evaluation": "all_models_and_predictions_per_cell_written_before_error_evaluation; assigned_bank_and_opposite_bank_both_reported; independent_for_each_fit_not_globally_new_confirmation_data",
        "frozen": "raw_reference_service_head_actions_original_anchor_selector_old_keys_and_native_controller; no_policy_choice_or_cap_gate_changes; no_claim_new_heads_satisfy_old_deployment_RMS_caps",
        "budget": {"fit_tasks": 1, "cached_cells": 32, "future_banks": 2, "fits_per_bank": 11,
            "new_risk_solves": 6336, "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "models_basis_predictions_action_errors_and_cell_ledgers_server_only; pull_scalar_group_report",
        "boundary": "development_label_prediction_test_not_policy_gain_strict_unseen_prefix_seed_CI_or_complete_HRL; no_penalty_selection_or_default_change; Stage253_248_169_closed"}


def freeze_cell(directory, key, views, panels):
    read = source.source.source.source.source.read_cell
    reference = read(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    basis = cache.learning.deployment_geometry(data["train"])["8"][0]
    z, raw, eligible = {}, {}, {}
    for scope, d in data.items():
        x = cache.learning.joint.relative(d["x"][..., :198], d["originals"])@basis
        z[scope] = np.concatenate((np.ones((*x.shape[:2], 1)), x), axis=-1)
        raw[scope] = cache.learning.joint.relative(d["components"], d["originals"])[..., 1]
        eligible[scope] = (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None])
    groups = list(cache.learning.joint.binding.policy.prefix_folds(train))
    contexts = [(None, np.arange(60), {"train": np.arange(60), "held": np.arange(60)})]
    contexts += [(list(g), np.asarray(ix), {"retained": np.asarray(ix), "excluded": np.asarray(val)}) for g, ix, val in groups]
    models, predictions, arrays = [], [], {"basis": basis}
    for bank in (0, 1):
        draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
        values = cache.learning.base.training_draws(train, panels, draws)[..., 1]
        y = values-values[np.arange(60), :, data["train"]["originals"]][:, :, None]
        for group, indices, queries in contexts:
            for penalty in learning.PENALTIES:
                model = learning.fit(z["train"][indices], y[indices], eligible["train"][indices], penalty)
                model["checks"]["cost_identity_error"] = 0.
                index = len(models)
                models.append({"bank": bank, "removed": group, "penalty": penalty, "training_draws": draws, "fit_indices": indices, **model})
                for scope, ids in queries.items():
                    label = "held" if scope == "held" else "train"
                    costs, probabilities = learning.predict(model, z[label][ids], data[label]["originals"][ids], raw[label][ids])
                    error = float(np.max(np.abs(costs["direct_mse"]-costs["signed_mse"])))
                    if error > 1e-10: raise ValueError("Signed MSE changed the direct MSE physical function")
                    model["checks"]["cost_identity_error"] = max(error, model["checks"]["cost_identity_error"])
                    predictions.append({"index": index, "scope": scope, "label": label, "ids": ids,
                        "costs": costs, "probabilities": probabilities, "eligible": eligible[label][ids]})
                    for arm, cost in costs.items(): arrays[f"{index}_{scope}_{arm}_cost"] = cost
                    for arm, p in probabilities.items(): arrays[f"{index}_{scope}_{arm}_probability"] = p
        print(f"SHARED_RISK_BANK_FROZEN {key}/{bank}", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"models.json", models)
    np.savez_compressed(directory/"predictions.npz", **arrays)
    write_json(directory/"queries.json", [{k: v for k, v in p.items() if k not in ("costs", "probabilities")} for p in predictions])
    return models, predictions, rows, data


def evaluate(models, predictions, rows, data, panels):
    records = []
    for p in predictions:
        model = models[p["index"]];bank = model["bank"]
        rs = [rows[p["label"]][i] for i in p["ids"]]
        originals = data[p["label"]]["originals"][p["ids"]]
        for which, b in (("same", bank), ("opposite", 1-bank)):
            draws, _ = cache.sampling.learning.draw_assignment(b, "expanded")
            values = cache.learning.base.training_draws(rs, panels, draws)[..., 1]
            y = values-values[np.arange(len(rs)), :, originals][:, :, None]
            records.append({"bank": bank, "removed": model["removed"], "penalty": model["penalty"], "scope": p["scope"]+"_"+which,
                "metrics": {arm: learning.metrics(cost, p["probabilities"].get(arm), y, p["eligible"], model["amplitudes"]) for arm, cost in p["costs"].items()}})
    return records


def report(cells):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output = {}
    for group, test in groups.items():
        rows = [p for c in cells if test(c["key"]) for p in c["records"]];output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for penalty in learning.PENALTIES:
                output[group][label][str(penalty)] = {scope: {arm: learning.summarize([p["metrics"][arm] for p in rows
                    if p["penalty"] == penalty and p["scope"] == scope and (bank is None or p["bank"] == bank)]) for arm in learning.ARMS} for scope in SCOPES}
    solves = sum(c["solves"] for c in cells)
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or solves != contract()["budget"]["new_risk_solves"]:
        raise ValueError("Incomplete shared risk fitting coverage")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "fits": {"risk_solves": solves, "max_normal_error": max(c["normal_error"] for c in cells), "max_event_gradient": max(c["event_gradient"] for c in cells),
            "max_MSE_cost_identity_error": max(c["cost_identity_error"] for c in cells)},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        here = directory/("_".join(map(str, key)))
        models, predictions, rows, d = freeze_cell(here, key, views, panels)
        records = evaluate(models, predictions, rows, d, panels)
        write_json(here/"errors.json", records)
        cells.append({"key": list(key), "records": records, "solves": len(models)*3,
            "normal_error": max(m["checks"]["normal_error"] for m in models), "event_gradient": max(m["checks"]["event_gradient"] for m in models),
            "cost_identity_error": max(m["checks"]["cost_identity_error"] for m in models)})
        print(f"SHARED_RISK_CELL_COMPLETE {key}", flush=True)
    write_json(directory/"cells.json", cells);write_json(path, report(cells))
    print("SHARED_RISK_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
