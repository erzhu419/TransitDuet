#!/usr/bin/env python3
"""Explain frozen wait-margin failures without fitting or changing selections."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_wait_margin_stage200 as source
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_wait_selection_stage201_v1"
SOURCE_RUN = "native_transit_wait_margin_stage200_development_20261013_r1"
ROLES = ("identity", "retained", "cancelled", "rerouted")
state_key = source.source.source.state_key


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "same240_mean8_states_40_prefix_groups_47_grouped_folds",
        "selection": "reuse_all_three_Stage200_policies_and_all_frozen_predictions",
        "decomposition": "identity_retained_cancelled_rerouted_relative_to_zero_margin",
        "contributions": "sum_current_minus_baseline_cost_or_wait_divided_by_entire_scope_states",
        "wait_errors": "all_nonidentity_candidates_true_mean8_wait_minus_raw_prediction",
        "context": "candidate_id_queue0_0to50_over50_and_prefix_mean_gap_relative_to_identity",
        "queue_threshold": "50_passengers_existing_native_action_value_feature_unit",
        "budget": {"new_fits": 0, "critic_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_targets_predictions_and_state_details_server_only_small_report_sync",
        "boundary": "descriptive_selection_diagnosis_not_independent_confirmation_CI_or_HRL_proof"}


def transition(before, after):
    original, old, new = before["original_choice"], before["selected"], after["selected"]
    if old == new: return "identity" if new == original else "retained"
    if new == original: return "cancelled"
    if old == original: raise ValueError("Positive wait margin introduced an override from identity")
    return "rerouted"


def action_context(row, candidate):
    queue = row["context"]["queues"][str(row["direction"])]
    plans = [np.asarray(c["commitment"]["prefix_planned_s"]) for c in row["candidates"]]
    change = np.diff(plans[candidate]).mean()-np.diff(plans[row["original_choice"]]).mean()
    return {"candidate": candidate, "queue": "zero" if queue == 0 else "0to50" if queue <= 50 else "over50",
        "headway": "compressed" if change < 0 else "widened" if change > 0 else "same"}


def selection_record(row, before, after):
    if state_key(before) != state_key(after) or state_key(row) != state_key(before):
        raise ValueError("Wait diagnostic joined different held-out states")
    labels = {l["candidate"]: l for l in row["labels"]}
    for prediction in (before, after):
        label = labels[prediction["selected"]]
        if any(prediction[k] != label[k] for k in ("actual_cost_delta", "wait_delta_min")):
            raise ValueError("Wait diagnostic changed its mean8 target map")
    old, new = before["selected"], after["selected"]
    return {**{k: row[k] for k in ("root", "scenario", "scene_seed", "probe_index")},
        "role": transition(before, after), "old": old, "new": new,
        "override": new != row["original_choice"],
        "old_cost": before["actual_cost_delta"], "new_cost": after["actual_cost_delta"],
        "old_wait": before["wait_delta_min"], "new_wait": after["wait_delta_min"],
        "predicted_wait": after["predicted_wait_delta_min"][new],
        "residual": after["wait_delta_min"]-after["predicted_wait_delta_min"][new]}


def selection_summary(records, denominator):
    if not records: return {"states": 0}
    changed = [r for r in records if r["override"]]
    return {"states": len(records), "overrides": len(changed),
        "prefix_groups": len({(r["root"], r["scene_seed"]) for r in records}),
        "mean_old_cost": float(np.mean([r["old_cost"] for r in records])),
        "mean_new_cost": float(np.mean([r["new_cost"] for r in records])),
        "mean_old_wait_min": float(np.mean([r["old_wait"] for r in records])),
        "mean_new_wait_min": float(np.mean([r["new_wait"] for r in records])),
        "cost_change_contribution": sum(r["new_cost"]-r["old_cost"] for r in records)/denominator,
        "wait_change_contribution_min": sum(r["new_wait"]-r["old_wait"] for r in records)/denominator,
        "selected_residual_mean_min": float(np.mean([r["residual"] for r in changed])) if changed else None,
        "predicted_selected_wait_mean_min": float(np.mean([r["predicted_wait"] for r in changed])) if changed else None,
        "actual_selected_wait_mean_min": float(np.mean([r["new_wait"] for r in changed])) if changed else None,
        "harmful_wait_overrides": sum(r["new_wait"] > 0 for r in changed),
        "harmful_cost_overrides": sum(r["new_cost"] > 0 for r in changed)}


def error_records(row, prediction):
    labels = {l["candidate"]: l for l in row["labels"]}
    return [{**{k: row[k] for k in ("root", "scenario", "scene_seed", "probe_index")},
        **action_context(row, i), "actual": label["wait_delta_min"],
        "predicted": prediction["predicted_wait_delta_min"][i],
        "selected": i == prediction["selected"]}
        for i, label in sorted(labels.items()) if i != row["original_choice"]]


def error_summary(records):
    if not records: return {"candidate_evaluations": 0}
    actual = np.asarray([r["actual"] for r in records])
    predicted = np.asarray([r["predicted"] for r in records])
    selected = [r for r in records if r["selected"]]
    return {"candidate_evaluations": len(records), "unique_states": len({state_key(r) for r in records}),
        "mean_residual_min": float(np.mean(actual-predicted)),
        "rmse_min": float(np.sqrt(np.mean((actual-predicted)**2))),
        "predicted_safe": int(np.sum(predicted <= 0)),
        "false_safe": int(np.sum((predicted <= 0) & (actual > 0))),
        "predicted_unsafe_actual_safe": int(np.sum((predicted > 0) & (actual <= 0))),
        "selected_overrides": len(selected),
        "selected_residual_mean_min": float(np.mean([r["actual"]-r["predicted"] for r in selected])) if selected else None}


def scope_summary(selections, errors):
    return {"policies": {p: {"overall": selection_summary(rows, len(rows)),
        "transitions": {role: selection_summary([r for r in rows if r["role"] == role], len(rows)) for role in ROLES}}
        for p, rows in selections.items()},
        "all_nonidentity_wait_errors": {"overall": error_summary(errors), **{f"by_{field}": {
            str(value): error_summary([r for r in errors if r[field] == value])
            for value in sorted({r[field] for r in errors})} for field in ("candidate", "queue", "headway")}}}


def diagnose(bank, targets, registered, load_fold):
    if registered["protocol"] != source.EXPERIMENT_PROTOCOL or registered["contract"] != source.contract():
        raise ValueError("Wait-selection diagnosis requires the registered Stage200 folds")
    target_keys = {state_key(r) for r in targets}
    rows = source.source.diagnostic.mean.expected.replace_targets(bank["training_data"], targets)
    by_key = {state_key(r): r for r in rows if state_key(r) in target_keys}
    selections = {s: {p: [] for p in source.POLICIES} for s in source.SPLITS}
    errors, fold_calibration = defaultdict(list), []
    for fold in registered["folds"]:
        cell = load_fold(fold)
        source.qualify(cell, fold, registered)
        baseline = cell["predictions"]["uncalibrated"]
        for prediction in baseline:
            errors[fold["split"]].extend(error_records(by_key[state_key(prediction)], prediction))
        for policy in source.POLICIES:
            for before, after in zip(baseline, cell["predictions"][policy], strict=True):
                selections[fold["split"]][policy].append(selection_record(by_key[state_key(before)], before, after))
        calibration = cell["calibration"]
        fold_calibration.append({"fold": {k: fold[k] for k in ("split", "index", "held_out")},
            **{k: calibration[k] for k in ("mean_selected_residual_min", "group_ratio_standard_error_min", "changed_choices", "margins")}})
    validation = {}
    for split in source.SPLITS:
        for records in selections[split].values():
            if len(records) != 240 or {state_key(r) for r in records} != target_keys:
                raise ValueError("Wait-selection diagnostic lost a registered mean8 state")
        validation[split] = {"overall": scope_summary(selections[split], errors[split]),
            "by_root": {str(root): scope_summary({p: [r for r in rs if r["root"] == root]
                for p, rs in selections[split].items()}, [r for r in errors[split] if r["root"] == root])
                for root in source.source.diagnostic.mean.ROOTS}}
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        **contract()["budget"], "folds": len(registered["folds"]), "evaluation_states": 240,
        "evaluation_prefix_groups": 40, "validation": validation,
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}
    return result, {"selections": selections, "wait_errors": dict(errors), "fold_calibration": fold_calibration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/SOURCE_RUN
    completed = json.loads((directory/"report/result.json").read_text())
    if not (completed["protocol"] == source.EXPERIMENT_PROTOCOL and completed["contract"] == source.contract()
            and completed["software_qualified"] and completed["new_closed_form_fits"] == 188):
        raise ValueError("Complete Stage200 before explaining its wait-margin selections")
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    mean = source.source.diagnostic.mean
    bank = json.loads(source.source.diagnostic.coverage.bank_path(mean.SOURCE_RUN).read_text())
    target_file = json.loads(mean.expected.targets_path(source.source.source.SOURCE_RUN).read_text())
    if target_file["protocol"] != mean.EXPERIMENT_PROTOCOL or target_file["contract"] != mean.contract():
        raise ValueError("Wait-selection diagnosis changed the mean-target map")
    result, details = diagnose(bank, target_file["targets"], registered,
        lambda f: json.loads((raw_directory(source.result_path(SOURCE_RUN, f))/"result.json").read_text()))
    write_json(raw_directory(args.output)/"states.json", details)
    write_json(args.output, result)
    print(json.dumps({s: {p: v["overall"] for p, v in d["overall"]["policies"].items()}
        for s, d in result["validation"].items()}, indent=2), flush=True)
    print("NATIVE_WAIT_SELECTION_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
