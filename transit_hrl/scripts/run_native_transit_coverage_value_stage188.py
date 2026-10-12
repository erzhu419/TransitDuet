#!/usr/bin/env python3
"""Prepare a server-only expanded bank or fit one fixed-budget grouped fold."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_expansion_stage187 as collection
from scripts.analyze_native_transit_value_expansion_stage187 import load_summary
from scripts.analyze_native_transit_tail_value_stage184 import candidate_ceiling
from freq_hrl.domains.transit.native_timetable_context import event_features
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

prior = collection.prior
floor, value, paired = prior.floor_source, prior.value, prior.paired
EXPERIMENT_PROTOCOL = "native_transit_coverage_value_stage188_v1"
COLLECTION_RUN = "native_transit_value_expansion_stage187_development_20261012_r2"
OLD_RUN = collection.SOURCE_RUN
MODES = ("tail_summary", "timetable_sequence")
SPLITS = ("scene", "regime", "lower_root")
STEPS = 100


def contract():
    return {"collection_run": COLLECTION_RUN, "collection_protocol": collection.EXPERIMENT_PROTOCOL,
        "old_run": OLD_RUN, "old_protocol": prior.EXPERIMENT_PROTOCOL,
        "data": "120_unchanged_existing_states_plus480_fresh_development_states_100_scenes_3600_labels",
        "modes": list(MODES), "primary": "timetable_sequence:wait_feasible",
        "method": "floor_absolute", "objective": "proper_mse", "steps": STEPS,
        "initialization_seed": prior.floor_source.occupancy.SEED,
        "optimizer": "unchanged_AdamW_lr0.003_weight_decay0.01_full_batch",
        "parameters": {"tail_summary": 1635, "timetable_sequence": 2147},
        "matching": "same_training_groups_targets_base_target_scales_steps_and_initial_predictor",
        "validation": ["100_leave_scene_out", "5_leave_regime_out", "2_leave_lower_root_out"],
        "policies": list(value.POLICIES), "selectors_share_fit": True,
        "grouping": "six_callbacks_and_all_candidates_of_each_scene_kept_together",
        "coverage_comparison": "same_old120_heldout_states_against_Stage184_186_not_new600_vs_old120",
        "budget": {"fits": 216, "critic_updates": 216*STEPS, "new_native_episodes": 0, "actor_updates": 0},
        "gate": "negative_mean_cost_and_nonpositive_wait_both_roots_all_three_splits",
        "storage": "merged_timetable_bank_full_models_and_detailed_aggregate_server_only_compact_receipts_for_sync",
        "boundary": "development_grouped_single_action_value_screen_not_repeated_learned_upper_or_end_to_end_HRL"}


def manifest(old_rows):
    groups = {paired.scene_group(r): {"root": r["root"], "scene_seed": r["scene_seed"],
        "scenario": r["scenario"], "origin": "existing"} for r in old_rows}
    if len(old_rows) != 120 or len(groups) != 20:
        raise ValueError("Coverage manifest requires the twenty qualified existing scene groups")
    for r in collection.ROOTS:
        for s in collection.SCENARIOS:
            for i in collection.REPLICAS:
                key = r, collection.scene_seed(r, s, i)
                if key in groups:
                    raise ValueError("Fresh development group overlaps an existing group")
                groups[key] = {"root": r, "scene_seed": key[1], "scenario": s, "origin": "fresh"}
    folds = []
    for split, held_out in (("scene", [list(g) for g in sorted(groups)]),
            ("regime", sorted(collection.SCENARIOS)), ("lower_root", sorted(collection.ROOTS)), ("full", [None])):
        folds.extend({"split": split, "index": i, "held_out": g} for i, g in enumerate(held_out))
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "groups": [groups[g] for g in sorted(groups)], "folds": folds}


def bank_path(run_name):
    return ROOT/"results"/f"{run_name}_raw"/"bank.json"


def assemble_bank(old_rows, fresh, registered):
    if not (fresh["protocol"] == collection.EXPERIMENT_PROTOCOL and fresh["contract"] == collection.contract()
            and fresh["software_qualified"] and fresh["phase"] == "complete" and fresh["training_updates"] == 0
            and fresh["training_states"] == 480 and fresh["paired_labels"] == 2880
            and fresh["independent_scene_groups"] == 80 and fresh["exact_identity_reproductions"] == 480
            and fresh["new_native_steps"] == 3360*61380 and registered == manifest(old_rows)):
        raise ValueError("Coverage bank requires the complete qualified fixed development cohort")
    rows = [*old_rows, *fresh["training_data"]]
    expected = {(g["root"], g["scene_seed"]): g for g in registered["groups"]}
    keys = [(r["root"], r["scene_seed"], r["probe_index"]) for r in rows]
    if (len(keys) != 600 or len(set(keys)) != 600
            or Counter(paired.scene_group(r) for r in rows) != Counter({g: 6 for g in expected})):
        raise ValueError("Coverage bank changed scene grouping or duplicated preaction states")
    event_counts = []
    for row in rows:
        if (row["scenario"] != expected[paired.scene_group(row)]["scenario"]
                or row["probe_index"] not in collection.PROBES
                or [l["candidate"] for l in row["labels"]] != list(range(6))
                or not np.all(np.isfinite(value.features(row, "tail_summary")))):
            raise ValueError("Coverage bank changed candidate labels or causal base inputs")
        events = event_features(row)
        if events.shape[0] != 6 or events.shape[2] != 6 or not np.all(np.isfinite(events)):
            raise ValueError("Coverage bank lost complete finite candidate timetable events")
        event_counts.append(events.shape[1])
    bank = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "manifest": registered, "training_data": rows}
    receipt = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "training_states": 600, "paired_labels": 3600, "independent_scene_groups": 100,
        "existing_states": 120, "fresh_states": 480, "event_count_range": [min(event_counts), max(event_counts)],
        "collection": {k: v for k, v in fresh.items() if k not in ("training_data", "contract")},
        "candidate_ceiling": {"existing": candidate_ceiling(old_rows),
            "fresh": candidate_ceiling(fresh["training_data"]), "combined": candidate_ceiling(rows)},
        "critic_updates": 0, "actor_updates": 0, "new_native_steps": 0,
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}
    return bank, receipt


def prepare(run_name, output):
    old = ROOT/"results"/OLD_RUN
    old_rows = prior.training_rows(json.loads((old/"inputs/data.json").read_text()),
        json.loads((old/"inputs/original.json").read_text()))
    registered = json.loads((ROOT/"results"/run_name/"inputs/manifest.json").read_text())
    fresh = load_summary(ROOT/"results"/COLLECTION_RUN)
    bank, receipt = assemble_bank(old_rows, fresh, registered)
    write_json(bank_path(run_name), bank)
    receipt["server_bank_path"] = str(bank_path(run_name))
    receipt["server_bank_bytes"] = bank_path(run_name).stat().st_size
    write_json(output, receipt)
    print(json.dumps({k: receipt[k] for k in ("training_states", "paired_labels", "independent_scene_groups",
        "event_count_range", "server_bank_bytes", "candidate_ceiling")}, indent=2), flush=True)
    print("NATIVE_COVERAGE_BANK_QUALIFIED", flush=True)


def split_rows(bank, split, index):
    fold = next(f for f in bank["manifest"]["folds"] if f["split"] == split and f["index"] == index)
    rows = bank["training_data"]
    if split == "full":
        return rows, rows, fold
    key = {"scene": lambda r: list(paired.scene_group(r)), "regime": lambda r: r["scenario"],
        "lower_root": lambda r: r["root"]}[split]
    return [r for r in rows if key(r) != fold["held_out"]], [r for r in rows if key(r) == fold["held_out"]], fold


def fit_fold(bank, mode, split, index):
    train, test, fold = split_rows(bank, split, index)
    start = time.monotonic()
    model = value.fit_value(train, "floor_absolute", steps=STEPS, objective="proper_mse", representation=mode)
    fit_seconds = time.monotonic()-start
    expected = sorted(map(list, {paired.scene_group(r) for r in train}))
    if model["training_groups"] != expected:
        raise ValueError("Coverage fit included the wrong training scene groups")
    predictions = {p: floor.evaluate(test, model, p) for p in value.POLICIES}
    evaluation_seconds = time.monotonic()-start-fit_seconds
    fields = ("training_groups", "training_states", "steps", "parameter_count", "center", "scale", "target_scale",
        "regret_cost_scale", "parameter_update_l2", "loss_ledger", "initialization_seed", "objective", "representation")
    ledger = {k: model[k] for k in fields}
    if mode == "timetable_sequence":
        ledger.update({k: model[k] for k in ("encoder_parameter_update_l2", "sequence_projection_update_l2", "encoder_gradient_l2_max")})
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "mode": mode, "fold": fold, "fit": ledger, "predictions": predictions,
        "critic_updates": STEPS, "actor_updates": 0, "new_native_steps": 0,
        "fit_wall_s": fit_seconds, "evaluation_wall_s": evaluation_seconds}
    return result, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("prepare", "fit"), required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--split", choices=(*SPLITS, "full"))
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.operation == "prepare":
        prepare(args.run_name, args.output)
        return
    if args.mode is None or args.split is None or args.fold_index is None:
        parser.error("Fit requires mode, split and fold index")
    bank = json.loads(bank_path(args.run_name).read_text())
    if not (bank["protocol"] == EXPERIMENT_PROTOCOL and bank["contract"] == contract() and bank["software_qualified"]):
        raise ValueError("Coverage fit requires its qualified prepared bank")
    result, model = fit_fold(bank, args.mode, args.split, args.fold_index)
    write_json(raw_directory(args.output)/"model.json", model)
    write_json(args.output, result)
    print(f"COVERAGE_FIT mode={args.mode} split={args.split} fold={args.fold_index} "
        f"states={result['fit']['training_states']} steps={STEPS} fit_s={result['fit_wall_s']:.3f}", flush=True)
    print("NATIVE_COVERAGE_VALUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
