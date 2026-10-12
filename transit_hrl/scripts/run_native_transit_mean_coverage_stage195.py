#!/usr/bin/env python3
"""Expand conditional targets to forty complete development scene groups."""

import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_expected_value_stage192 as expected
from scripts.analyze_native_transit_continuation_noise_stage191 import signal
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

noise, source = expected.noise, expected.source
state_key = expected.state_key
EXPERIMENT_PROTOCOL = "native_transit_mean_coverage_stage195_v1"
ROOTS, SCENARIOS, PROBES = noise.ROOTS, noise.SCENARIOS, noise.source.PROBES
REPLICAS, DRAWS = tuple(range(4)), noise.DRAWS
MODES, SPLITS, STEPS = ("tail_summary",), source.SPLITS, source.STEPS
VARIANTS = expected.VARIANTS
SOURCE_RUN, NOISE_RUN = expected.SOURCE_RUN, expected.NOISE_RUN
PREVIOUS_RUN = "native_transit_fleet_credit_stage194_fit_20261012_r1"


def states():
    return [(r, s, i, p) for r in ROOTS for s in SCENARIOS for i in REPLICAS for p in PROBES]


def reused(key):
    _, _, replica, probe = key
    return replica == 0 and probe in noise.PROBES


def cells(*, preflight=False):
    return [(*k, d) for k in states() if not reused(k) for d in DRAWS
        if not preflight or (k[2] == 1 and k[3] == 9 and d == 0)]


def continuation_seed(root, scenario, replica, probe, draw):
    return noise.continuation_seed(root, scenario, probe, draw)+replica*1000000


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "noise_run": NOISE_RUN, "noise_protocol": noise.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "replicas": list(REPLICAS),
        "probes": list(PROBES), "draws": list(DRAWS),
        "data": "600_unchanged_inputs_240_mean8_targets_40_complete_prefix_groups_360_single_targets",
        "reuse": "30_Stage191_states_240_draws_unchanged_no_duplicate_native_collection",
        "future_seed": "Stage191_seed+replica*1000000_replica0_old_draws_identical",
        "intervention": noise.contract()["intervention"], "randomness": noise.contract()["randomness"],
        "retained": noise.contract()["retained"], "control": noise.contract()["control"],
        "preflight": "ten_replica1_probe9_draw0_cells_both_roots_all_regimes_before1670_remaining",
        "modes": list(MODES), "primary": "tail_summary:mean8_overlay:wait_feasible",
        "method": "shared_three_head_floor_absolute_proper_mse_100steps_seed178_AdamW_lr0.003_wd0.01",
        "normalization": "unchanged_training_input_scales_target_scales_recomputed_training_only",
        "reference": "reuse47_Stage188_single_target_models_same_groups_architecture_optimizer_steps",
        "folds": "40_leave_scene_out_5_leave_regime_out_2_leave_lower_root_out_all6_callbacks_grouped",
        "evaluation": "same240_heldout_states_original_and_mean8_labels_shared_fits_and_selectors",
        "policies": list(source.value.POLICIES), "variants": list(VARIANTS),
        "budget": {"collection_tasks": 1680, "new_native_episodes": 13440,
            "new_native_clock_ticks": 13440*61380, "reused_native_episodes": 1920,
            "new_prefix_scenes": 0, "new_fits": 47, "new_critic_updates": 4700,
            "reused_fits": 47, "actor_updates": 0},
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean_label_splits",
        "storage": "draw_labels_rollouts_feature_bank_mean_map_models_predictions_server_only_receipts_report_sync",
        "boundary": "40_existing_development_prefixes_future_innovations_not_new_independent_confirmation_or_full_HRL"}


def manifest(registered):
    groups = {(r, noise.source.scene_seed(r, s, i)) for r, s, i, _ in states()}
    folds = [f for f in registered["folds"] if f["split"] in SPLITS and
        (f["split"] != "scene" or tuple(f["held_out"]) in groups)]
    if len(folds) != 47:
        raise ValueError("Mean coverage requires all forty registered prefix groups")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "groups": registered["groups"],
        "folds": folds, "mean_states": [{"root": r, "scenario": s,
            "scene_seed": noise.source.scene_seed(r, s, i), "probe_index": p} for r, s, i, p in states()]}


def draw_path(run_name, root, scenario, replica, probe, draw):
    return ROOT/"results"/run_name/"draws"/scenario/f"root_{root}"/f"replica_{replica}"/f"probe_{probe}"/f"draw_{draw}"/"receipt.json"


def qualify_draw(cell):
    key = cell["root"], cell["scenario"], cell["replica"], cell["probe_index"], cell["draw"]
    if not (key in cells() and cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract()
            and cell["software_qualified"] and cell["scene_seed"] == noise.source.scene_seed(*key[:3])
            and cell["continuation_seed"] == continuation_seed(*key)
            and cell["original_source_reproductions"] == cell["exact_resampled_identities"] == 1
            and cell["prefix_state_rng_matches"] == 7 and cell["frozen_weight_qualified_episodes"] == 8
            and cell["window_ledger_qualified_episodes"] == 8 and cell["continuation_qualified"]
            and cell["native_clock_ticks"] == 8*61380 and cell["training_updates"] == 0):
        raise ValueError("Changed conditional draw, scene, prefix pairing or physical budget")
    for labels in (cell["labels"], cell["existing_labels"]):
        if [l["candidate"] for l in labels] != list(range(6)):
            raise ValueError("Missing mean-coverage candidate")
        for label in labels:
            if not (np.all(np.isfinite([label[f] for f in expected.LABEL_FIELDS]))
                    and label["future_fleet_cost"] >= 0
                    and np.isclose(label["actual_cost_delta"], label["fleet_cost_delta"]+label["nonfleet_cost_delta"], atol=1e-12, rtol=0)
                    and np.isclose(label["future_fleet_cost"]-labels[cell["original_choice"]]["future_fleet_cost"],
                        label["fleet_cost_delta"], atol=1e-12, rtol=0)
                    and np.isclose(label["physical_return_delta"], -100*label["actual_cost_delta"], atol=1e-12, rtol=0)):
                raise ValueError("Changed physical target units or fleet floor")
        if any(labels[cell["original_choice"]][f] != 0 for f in expected.LABEL_FIELDS if f != "future_fleet_cost"):
            raise ValueError("Same-tape original action is not an exact identity")


def collect(key, model, output):
    root, scenario, replica, probe, draw = key
    bank = json.loads(source.bank_path(SOURCE_RUN).read_text())
    if not (bank["protocol"] == source.EXPERIMENT_PROTOCOL and bank["contract"] == source.contract()
            and bank["software_qualified"]):
        raise ValueError("Mean collection requires the qualified unchanged bank")
    row = next(r for r in bank["training_data"] if expected.state_key(r) ==
        (root, noise.source.scene_seed(root, scenario, replica), probe))
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "replica": replica, "draw": draw,
        **noise.replay_draw(root, scenario, replica, probe, continuation_seed(*key), row, model, output)}
    result["peak_process_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    qualify_draw(result)
    return result


def build_targets(bank, new_draws, old_draws):
    if set(new_draws) != set(cells()) or set(old_draws) != set(noise.cells()):
        raise ValueError("Mean targets require every registered draw, including the reused cohort")
    for cell in new_draws.values(): qualify_draw(cell)
    for cell in old_draws.values(): noise.qualify_cell(cell)
    rows = {expected.state_key(r): r for r in bank["training_data"]}
    targets, diagnostics = [], []
    for root, scenario, replica, probe in states():
        key = root, scenario, replica, probe
        draws = [old_draws[root, scenario, probe, d] if reused(key) else new_draws[(*key, d)] for d in DRAWS]
        row = rows[root, noise.source.scene_seed(root, scenario, replica), probe]
        if any(expected.state_key(c) != expected.state_key(row) or c["original_choice"] != row["original_choice"]
                or c["existing_labels"] != row["labels"] for c in draws):
            raise ValueError("Continuation targets changed their original feature-bank state")
        labels = [{"candidate": i, **{f: float(np.mean([c["labels"][i][f] for c in draws]))
            for f in expected.LABEL_FIELDS}} for i in range(6)]
        targets.append({**{k: row[k] for k in ("root", "scenario", "scene_seed", "probe_index")}, "labels": labels})
        diagnostics.append({"root": root, "scenario": scenario, "probe_index": probe,
            **{f: signal(np.asarray([[l[f] for l in c["labels"]] for c in draws]))
                for f in ("actual_cost_delta", "wait_delta_min")}})
    return targets, diagnostics


def prepare(run_name):
    bank = json.loads(source.bank_path(SOURCE_RUN).read_text())
    registered = json.loads((ROOT/"results"/run_name/"inputs/manifest.json").read_text())
    if registered != manifest(bank["manifest"]):
        raise ValueError("Prepared coverage differs from the registered complete groups")
    new_draws = {k: json.loads((raw_directory(draw_path(run_name, *k))/"result.json").read_text()) for k in cells()}
    old_draws = {k: json.loads((raw_directory(noise.result_path(ROOT/"results"/NOISE_RUN, *k))/"result.json").read_text()) for k in noise.cells()}
    targets, diagnostics = build_targets(bank, new_draws, old_draws)
    write_json(expected.targets_path(run_name), {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "targets": targets})
    write_json(ROOT/"results"/run_name/"prepare/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "input_states": 600, "mean_states": len(targets), "prefix_groups": 40,
        "new_draws": len(new_draws), "reused_draws": len(old_draws), "new_native_episodes": len(new_draws)*8,
        "server_targets_bytes": expected.targets_path(run_name).stat().st_size,
        "peak_process_rss_mb_max": max(c["peak_process_rss_mb"] for c in new_draws.values()),
        "mean_noise": {f: {k: float(np.mean([r[f][k] for r in diagnostics])) for k in
            ("mean_pairwise_gap_variance", "mean_label_pairwise_noise_variance", "debiased_pairwise_effect_power")}
            for f in ("actual_cost_delta", "wait_delta_min")}})
    print("NATIVE_MEAN_COVERAGE_TARGETS_QUALIFIED", flush=True)


def fit_fold(bank, targets, fold, reference):
    # Refit all three heads; Stage194's nonfleet/wait weights are not frozen here.
    result, model = expected.fit_fold(bank, targets, "tail_summary", fold, reference)
    result.update(protocol=EXPERIMENT_PROTOCOL, contract=contract())
    return result, model


def result_path(run_name, mode, fold):
    return expected.result_path(run_name, mode, fold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("collect", "prepare", "fit"), required=True)
    parser.add_argument("--seed", type=int, choices=ROOTS)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--replica", type=int, choices=REPLICAS)
    parser.add_argument("--probe", type=int, choices=PROBES)
    parser.add_argument("--draw", type=int, choices=DRAWS)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--fold-index", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.operation == "prepare":
        prepare(args.run_name)
        return
    if args.operation == "collect":
        key = args.seed, args.scenario, args.replica, args.probe, args.draw
        if key not in cells(): parser.error("Collect requires a new registered state/draw")
        output = draw_path(args.run_name, *key)
        model = json.loads((ROOT/"results"/args.run_name/"inputs/models.json").read_text())[str(args.seed)]
        result = collect(key, model, output)
        write_json(raw_directory(output)/"result.json", result)
        write_json(output, {k: v for k, v in result.items() if k not in ("labels", "existing_labels")})
        print(f"NATIVE_MEAN_COVERAGE_DRAW_COMPLETE peak_process_rss_mb={result['peak_process_rss_mb']:.1f}", flush=True)
        return
    if args.split is None or args.fold_index is None: parser.error("Fit requires a grouped fold")
    bank = json.loads(source.bank_path(SOURCE_RUN).read_text())
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    target_file = json.loads(expected.targets_path(args.run_name).read_text())
    if registered != manifest(bank["manifest"]) or target_file["contract"] != contract():
        raise ValueError("Changed mean targets or registered fit groups")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    reference = json.loads((raw_directory(expected.source_path(ROOT/"results"/SOURCE_RUN, "tail_summary", fold))/"model.json").read_text())
    result, model = fit_fold(bank, target_file["targets"], fold, reference)
    output = result_path(args.run_name, "tail_summary", fold)
    write_json(raw_directory(output)/"result.json", result)
    write_json(raw_directory(output)/"model.json", model)
    write_json(output, {k: v for k, v in result.items() if k not in ("fits", "predictions")})
    print("NATIVE_MEAN_COVERAGE_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
