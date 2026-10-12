#!/usr/bin/env python3
"""Replace thirty noisy targets, retaining all six hundred causal input states."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_continuation_noise_stage191 as noise
from scripts import run_native_transit_coverage_value_stage188 as source
from scripts.submit_native_transit_coverage_value_stage188_scheduleurm import result_path as source_path
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_expected_value_stage192_v1"
NOISE_RUN = "native_transit_continuation_noise_stage191_development_20261012_r1"
SOURCE_RUN = noise.BANK_RUN
MODES, SPLITS, STEPS = source.MODES, source.SPLITS, source.STEPS
LABEL_FIELDS = ("actual_cost_delta", "physical_return_delta", "fleet_cost_delta", "nonfleet_cost_delta",
    "wait_delta_min", "episode_reward_delta", "future_fleet_cost")
VARIANTS = ("cached_original", "mean8_overlay")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "noise_run": NOISE_RUN, "noise_protocol": noise.EXPERIMENT_PROTOCOL,
        "data": "600_unchanged_input_states_30_existing_targets_replaced_by8_independent_future_means",
        "modes": list(MODES), "primary": "timetable_sequence:mean8_overlay:wait_feasible",
        "method": "floor_absolute_proper_mse_100steps_seed178_unchanged_AdamW_lr0.003_wd0.01",
        "normalization": "input_scales_unchanged_target_scales_recomputed_training_only_after_target_replacement",
        "label_mean": "average_physical_costs_not_cost_of_average_peak_identity_deltas_exact_zero",
        "folds": "10_leave_target_scene_out_5_leave_regime_out_2_leave_lower_root_out_all_scene_callbacks_grouped",
        "reference": "reuse34_qualified_Stage188_models_identical_groups_inputs_architecture_optimizer_and_steps",
        "evaluation": "same30_heldout_states_on_original_labels_and8_mean_labels_shared_models_selectors",
        "policies": list(source.value.POLICIES), "variants": list(VARIANTS),
        "budget": {"new_fits": 34, "new_critic_updates": 3400, "reused_fits": 34,
            "actor_updates": 0, "new_native_episodes": 0},
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_expected_label_splits",
        "storage": "feature_bank_mean_label_map_models_predictions_server_only_compact_receipts_report_sync",
        "boundary": "ten_scene_development_target_screen_not600_state_confirmation_or_end_to_end_HRL"}


def state_key(row):
    return row["root"], row["scene_seed"], row["probe_index"]


def manifest(registered):
    keys = {(r, noise.source.scene_seed(r, s, 0)) for r in noise.ROOTS for s in noise.SCENARIOS}
    folds = [f for f in registered["folds"] if f["split"] in SPLITS and
        (f["split"] != "scene" or tuple(f["held_out"]) in keys)]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "groups": registered["groups"],
        "folds": folds, "mean_states": [{"root": r, "scene_seed": noise.source.scene_seed(r, s, 0),
            "scenario": s, "probe_index": p} for r in noise.ROOTS for s in noise.SCENARIOS for p in noise.PROBES]}


def targets_path(run_name):
    return ROOT/"results"/f"{run_name}_raw"/"targets.json"


def build_targets(bank, cells):
    if set(cells) != set(noise.cells()):
        raise ValueError("Expected targets require all registered conditional continuations")
    by_state = {state_key(r): r for r in bank["training_data"]}
    targets, decomposition = [], []
    a, b = np.triu_indices(6, 1)
    for root in noise.ROOTS:
        for scenario in noise.SCENARIOS:
            for probe in noise.PROBES:
                draws = [cells[root, scenario, probe, d] for d in noise.DRAWS]
                old = by_state[state_key(draws[0])]
                if any(state_key(c) != state_key(old) or c["existing_labels"] != old["labels"] for c in draws):
                    raise ValueError("Mean targets do not describe the unchanged feature-bank state")
                labels = [{"candidate": i, **{field: float(np.mean([c["labels"][i][field] for c in draws]))
                    for field in LABEL_FIELDS}} for i in range(6)]
                if any(labels[old["original_choice"]][field] != 0 for field in LABEL_FIELDS if field != "future_fleet_cost"):
                    raise ValueError("Expected labels changed exact same-tape identity")
                targets.append({**{k: old[k] for k in ("root", "scenario", "scene_seed", "probe_index")}, "labels": labels})
                values = np.asarray([[[l[f] for f in ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta")]
                    for l in c["labels"]] for c in draws])
                gaps = values[:, a]-values[:, b]
                centered = gaps-gaps.mean(axis=0)
                variances = np.mean(np.sum(centered**2, axis=0)/(len(draws)-1), axis=0)
                twice_covariance = float(np.mean(2*np.sum(centered[..., 1]*centered[..., 2], axis=0)/(len(draws)-1)))
                decomposition.append({"root": root, "probe_index": probe,
                    "total_gap_variance": float(variances[0]), "fleet_gap_variance": float(variances[1]),
                    "nonfleet_gap_variance": float(variances[2]), "twice_gap_covariance": twice_covariance})
    return targets, {"overall": variance_summary(decomposition),
        "by_root": {str(r): variance_summary([v for v in decomposition if v["root"] == r]) for r in noise.ROOTS},
        "by_probe": {str(p): variance_summary([v for v in decomposition if v["probe_index"] == p]) for p in noise.PROBES}}


def variance_summary(rows):
    return {"states": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in
        ("total_gap_variance", "fleet_gap_variance", "nonfleet_gap_variance", "twice_gap_covariance")}}


def replace_targets(rows, targets):
    by_key = {state_key(r): r["labels"] for r in targets}
    return [{**r, "labels": copy.deepcopy(by_key[state_key(r)])} if state_key(r) in by_key else r for r in rows]


def prepare(run_name):
    completed = json.loads((ROOT/"results"/NOISE_RUN/"report/result.json").read_text())
    if not (completed["protocol"] == noise.EXPERIMENT_PROTOCOL and completed["contract"] == noise.contract()
            and completed["software_qualified"] and completed["native_episodes"] == 1920):
        raise ValueError("Expected targets require the completed qualified noise diagnostic")
    bank = json.loads(source.bank_path(SOURCE_RUN).read_text())
    registered = json.loads((ROOT/"results"/run_name/"inputs/manifest.json").read_text())
    if registered != manifest(bank["manifest"]):
        raise ValueError("Expected-value folds changed the registered scene coverage")
    cells = {k: json.loads((raw_directory(noise.result_path(ROOT/"results"/NOISE_RUN, *k))/"result.json").read_text())
        for k in noise.cells()}
    targets, decomposition = build_targets(bank, cells)
    write_json(targets_path(run_name), {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "targets": targets})
    write_json(ROOT/"results"/run_name/"prepare/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "input_states": 600, "changed_target_states": len(targets),
        "independent_prefix_groups": 10, "variance_decomposition": decomposition, "training_updates": 0,
        "new_native_steps": 0, "server_targets_bytes": targets_path(run_name).stat().st_size})
    print(json.dumps(decomposition, indent=2), flush=True)
    print("NATIVE_EXPECTED_VALUE_TARGETS_QUALIFIED", flush=True)


def fit_fold(bank, targets, mode, fold, reference):
    original_train, original_test, _ = source.split_rows(bank, fold["split"], fold["index"])
    keys = {state_key(r) for r in targets}
    test_old = [r for r in original_test if state_key(r) in keys]
    train, test_mean = replace_targets(original_train, targets), replace_targets(test_old, targets)
    expected = sorted(map(list, {source.paired.scene_group(r) for r in original_train}))
    if not (reference["training_groups"] == expected and reference["training_states"] == len(train)
            and reference["steps"] == STEPS and reference["representation"] == mode and reference["objective"] == "proper_mse"):
        raise ValueError("Cached reference changed training groups, steps or representation")
    start = time.monotonic()
    model = source.value.fit_value(train, "floor_absolute", steps=STEPS, objective="proper_mse", representation=mode)
    seconds = time.monotonic()-start
    if any(model[k] != reference[k] for k in ("training_groups", "training_states", "steps", "parameter_count",
            "center", "scale", "initialization_seed", "representation", "objective")):
        raise ValueError("Target comparison changed inputs, architecture or fixed optimizer budget")
    fields = ("training_groups", "training_states", "steps", "parameter_count", "center", "scale", "target_scale",
        "regret_cost_scale", "initialization_seed", "representation", "objective")
    if mode == "timetable_sequence":
        fields += ("encoder_parameter_update_l2", "sequence_projection_update_l2", "encoder_gradient_l2_max")
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "mode": mode, "fold": fold, "fits": {v: {k: m[k] for k in fields} for v, m in
            (("cached_original", reference), ("mean8_overlay", model))},
        "training_mean_states": sum(state_key(r) in keys for r in original_train), "evaluation_mean_states": len(test_mean),
        "predictions": {v: {label: {p: source.floor.evaluate(rows, m, p) for p in source.value.POLICIES}
            for label, rows in (("original", test_old), ("mean8", test_mean))}
            for v, m in (("cached_original", reference), ("mean8_overlay", model))},
        "critic_updates": STEPS, "actor_updates": 0, "new_native_steps": 0,
        "fit_wall_s": seconds, "evaluation_wall_s": time.monotonic()-start-seconds}
    return result, model


def result_path(run_name, mode, fold):
    return ROOT/"results"/run_name/"cells"/mode/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("prepare", "fit"), required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--fold-index", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.operation == "prepare":
        prepare(args.run_name)
        return
    if args.mode is None or args.split is None or args.fold_index is None:
        parser.error("Fit requires representation and grouped fold")
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    bank = json.loads(source.bank_path(SOURCE_RUN).read_text())
    target_file = json.loads(targets_path(args.run_name).read_text())
    if registered != manifest(bank["manifest"]) or target_file["contract"] != contract():
        raise ValueError("Expected-value worker changed its prepared targets or outer folds")
    reference = json.loads((raw_directory(source_path(ROOT/"results"/SOURCE_RUN, args.mode, fold))/"model.json").read_text())
    result, model = fit_fold(bank, target_file["targets"], args.mode, fold, reference)
    output = result_path(args.run_name, args.mode, fold)
    write_json(raw_directory(output)/"result.json", result)
    write_json(raw_directory(output)/"model.json", model)
    write_json(output, {k: v for k, v in result.items() if k not in ("fits", "predictions")})
    print(f"EXPECTED_VALUE_FIT mode={args.mode} split={args.split} fold={args.fold_index} "
        f"train_mean={result['training_mean_states']} test_mean={result['evaluation_mean_states']} fit_s={result['fit_wall_s']:.3f}", flush=True)
    print("NATIVE_EXPECTED_VALUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
