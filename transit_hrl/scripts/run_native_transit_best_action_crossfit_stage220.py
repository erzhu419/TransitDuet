#!/usr/bin/env python3
"""Prefix-held-out decision learning with independently held future halves."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_native_transit_best_action_stage218 as source
from scripts import analyze_native_transit_best_action_full_stage219 as previous
from freq_hrl.domains.transit import native_best_action_learning as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_best_action_crossfit_stage220_v1"
SOURCE_RUN = "native_transit_best_action_full_stage219_diagnostic_20261014_r1"
PARTITIONS, VIEWS = source.PARTITIONS, source.VIEWS
SCOPES = ("train_same", "train_other", "held_same", "held_other")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "data": "original240_states40_prefixes_same8_conditional_futures_no_additional360",
        "views": list(VIEWS), "outer": "two_complementary_folds_per_view60_train60_held10_prefixes_each_balanced_regimes_and_mixed_roots",
        "future": "first_last_and_even_odd_both_directions_train_only_four_draws_evaluate_other_four",
        "partitions": source.contract()["partitions"], "filtering": "no_half_agreement_filter_every_state_retained",
        "cost_control": "native98_feature_three_head_pair_ridge0.1_exact_closed_form_training_half_means",
        "decision": "98_feature_linear_score_all6_cross_entropy_training_half_mean_cost_argmin_with_mean_wait_le0_identity_ties",
        "wait_reference": "shared199_parameter_symmetric_draw_mass_GLM_training_half_only",
        "wait_residuals": {"linear": 199, "nonlinear_pair": 3201, "state_conditioned_pair": 3393},
        "conditioning": "same198_features_tanh_width16_plus12_absolute_causal_context_no_root_or_regime_ids",
        "optimizer": {"steps_per_branch": learning.STEPS, "lr": learning.neural.LEARNING_RATE,
            "weight_decay": learning.neural.WEIGHT_DECAY, "seed": learning.SEED, "dtype": "float64"},
        "normalization": "own_outer_training_causal_inputs_only_no_held_prefix_or_other_half_labels",
        "variants": list(learning.VARIANTS), "inference": "causal_scores_and_learned_gate_only_no_true_safe_menu",
        "evaluation": "four_scopes_physical_cost_wait_mass_loss_disputed_states_same_state_single_vs_mixed_by_root_no_CI",
        "budget": {"fit_tasks": 8, "report_tasks": 1, "training_cells": 32,
            "analytic_cost_fits": 32, "glm_fits": 32, "supervised_decision_updates": 32*learning.STEPS,
            "wait_residual_updates": 96*learning.STEPS, "new_native_episodes": 0, "rl_actor_updates": 0},
        "storage": "raw_predictions_models_remote_fit_receipts_and_compact_report_only",
        "boundary": "supervised_single_action_frozen_greedy_suffix_development_not_end_to_end_HRL_or_independent_seed_CI"}


def split_rows(rows, fold):
    if fold not in (0, 1): raise ValueError("Require one of two complementary prefix folds")
    train_groups = set()
    for regime in source.prefixes.SCENARIOS:
        for root in sorted({r["root"] for r in rows}):
            seeds = sorted({r["scene_seed"] for r in rows if r["scenario"] == regime and r["root"] == root})
            midpoint = len(seeds)//2
            train_groups.update((root, seed) for seed in seeds[fold*midpoint:(fold+1)*midpoint])
    train = [r for r in rows if (r["root"], r["scene_seed"]) in train_groups]
    held = [r for r in rows if (r["root"], r["scene_seed"]) not in train_groups]
    if len(train) != 60 or len(held) != 60 or len(train_groups) != 10:
        raise ValueError("Changed matched training size or complete-prefix split")
    return train, held


def result_path(run_name, view, fold):
    return ROOT/"results"/run_name/"cells"/view/f"fold_{fold}"/"receipt.json"


def fit_cell(train, held, panels, partition, direction, save_models):
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    targets, masses = learning.half_targets(train, panels, a)
    train_batch, held_batch = learning.neural.batches(train), learning.neural.batches(held)
    models, fit = learning.fit_heads(targets, masses, batch=train_batch)
    save_models(models)
    train_prediction = learning.predict(train, models, batch=train_batch)
    held_prediction = learning.predict(held, models, batch=held_batch)
    scopes = {"train_same": learning.records(train, panels, a, b, train_prediction),
        "train_other": learning.records(train, panels, b, a, train_prediction),
        "held_same": learning.records(held, panels, a, b, held_prediction),
        "held_other": learning.records(held, panels, b, a, held_prediction)}
    groups = lambda rows: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rows}))
    return {"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
        "training_groups": groups(train), "held_groups": groups(held), "fit": fit, "scopes": scopes}


def fit_view(bank, panels, registered, view, fold, save_models):
    rows = source.matched_views(bank["training_data"], registered)[view]
    train, held = split_rows(rows, fold)
    cells = []
    for partition in PARTITIONS:
        for direction in (0, 1):
            cell = fit_cell(train, held, panels, partition, direction,
                lambda m: save_models(partition, direction, m))
            cells.append(cell)
            print(f"BEST_ACTION_CROSSFIT_CELL {view}/{fold}/{partition}/{direction} "
                f"best_loss={cell['fit']['fits']['best']['final_loss']:.6f}", flush=True)
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "view": view, "fold": fold,
        "software_qualified": True, "cells": cells}


def qualify(result, view, fold, views):
    train, held = split_rows(views[view], fold)
    group_set = lambda rs: {(r["root"], r["scene_seed"]) for r in rs}
    if (result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract()
            or result["view"] != view or result["fold"] != fold or not result["software_qualified"]
            or {(c["partition"], c["direction"]) for c in result["cells"]} != {(p, d) for p in PARTITIONS for d in (0, 1)}
            or len(result["cells"]) != 4):
        raise ValueError("Changed crossfit registration or missing future direction")
    for cell in result["cells"]:
        partition, direction = cell["partition"], cell["direction"]
        fit = cell["fit"]
        if (cell["training_draws"] != list(PARTITIONS[partition][direction])
                or cell["evaluation_draws"] != list(PARTITIONS[partition][1-direction])
                or set(map(tuple, cell["training_groups"])) != group_set(train)
                or set(map(tuple, cell["held_groups"])) != group_set(held)
                or set(map(tuple, fit["training_groups"])) != group_set(train)
                or fit["training_states"] != 60 or fit["target_draw_count"] != [4]
                or fit["supervised_decision_updates"] != learning.STEPS
                or fit["wait_residual_updates"] != 3*learning.STEPS
                or set(cell["scopes"]) != set(SCOPES)
                or fit["new_native_episodes"] != 0 or fit["rl_actor_updates"] != 0):
            raise ValueError("Changed train-only target halves, grouping or optimizer budget")
        for name, parameters in (("best", 98), ("linear", 199), ("nonlinear_pair", 3201), ("state_conditioned_pair", 3393)):
            if fit["fits"][name]["parameter_count"] != parameters or fit["fits"][name]["steps"] != learning.STEPS:
                raise ValueError("Changed residual architecture or matched update budget")
        for scope, rows in cell["scopes"].items():
            expected = train if scope.startswith("train") else held
            if len(rows) != 60 or {source.state_key(r) for r in rows} != {source.state_key(r) for r in expected}:
                raise ValueError("Crossfit dropped a disputed state or split a prefix")
            if any(set(r["decisions"]) != set(learning.VARIANTS) for r in rows):
                raise ValueError("Missing causal selector control")


def metrics(rows):
    groups = {(r["root"], r["scene_seed"]) for r in rows}
    return {"prediction_cases": len(rows), "unique_states": len({source.state_key(r) for r in rows}),
        "unique_prefixes": len(groups), "disputed_cases": sum(r["disputed"] for r in rows),
        "best_action_cross_entropy": float(np.mean([r["best_action_cross_entropy"] for r in rows])) if rows else None,
        "variants": {name: {"mean_cost": float(np.mean([r["decisions"][name]["cost"] for r in rows])) if rows else None,
            "mean_wait_min": float(np.mean([r["decisions"][name]["wait"] for r in rows])) if rows else None,
            "changed_fraction": float(np.mean([r["decisions"][name]["changed"] for r in rows])) if rows else None,
            "unsafe_fraction": float(np.mean([r["decisions"][name]["unsafe"] for r in rows])) if rows else None,
            "best_match_fraction": float(np.mean([r["decisions"][name]["best_match"] for r in rows])) if rows else None}
            for name in learning.VARIANTS},
        "wait_heads": {gate: {"physical_mass_loss_min": float(np.mean([r["gates"][gate]["mass_loss_min"] for r in rows])) if rows else None,
            "false_safe_candidates": sum(r["gates"][gate]["false_safe"] for r in rows),
            "false_unsafe_candidates": sum(r["gates"][gate]["false_unsafe"] for r in rows)}
            for gate in ("glm", *learning.WAIT_VARIANTS)}}


def summarize(results, views):
    if set(results) != {(v, f) for v in VIEWS for f in (0, 1)}:
        raise ValueError("Complete all eight matched fit tasks before reporting")
    for (view, fold), result in results.items(): qualify(result, view, fold, views)
    output, held_records = {}, {}
    for view in VIEWS:
        cells = [cell for f in (0, 1) for cell in results[view, f]["cells"]]
        scopes = {s: [r for cell in cells for r in cell["scopes"][s]] for s in SCOPES}
        if any(len(rs) != 480 or len({source.state_key(r) for r in rs}) != 120 for rs in scopes.values()):
            raise ValueError("Changed all-state coverage or fourfold future sensitivity multiplicity")
        output[view] = {"scopes": {s: metrics(rs) for s, rs in scopes.items()},
            "held_other_by_root": {str(root): metrics([r for r in scopes["held_other"] if r["root"] == root])
                for root in sorted({r["root"] for r in views[view]})},
            "held_disputed": metrics([r for r in scopes["held_other"] if r["disputed"]]),
            "held_stable": metrics([r for r in scopes["held_other"] if not r["disputed"]]),
            "by_future_direction": {f"{p}/{d}": {s: metrics([r for c in cells if c["partition"] == p and c["direction"] == d
                for r in c["scopes"][s]]) for s in ("train_same", "held_other")} for p in PARTITIONS for d in (0, 1)}}
        held_records[view] = {(*source.state_key(r), c["partition"], c["direction"]): r
            for c in cells for r in c["scopes"]["held_other"]}
    comparisons = {}
    mixed = {k: r for v in ("mixed_a", "mixed_b") for k, r in held_records[v].items()}
    for root in source.prefixes.ROOTS:
        single = held_records[f"root_{root}"]
        same_root = {k: r for k, r in mixed.items() if r["root"] == root}
        if set(single) != set(same_root) or len(single) != 480:
            raise ValueError("Single/mixed comparisons changed their held states or evaluation futures")
        comparisons[str(root)] = {"prediction_cases": len(single), "unique_states": len({k[:3] for k in single}),
            "variants": {v: {"cost_delta": float(np.mean([same_root[k]["decisions"][v]["cost"]-r["decisions"][v]["cost"] for k, r in single.items()])),
                "wait_delta_min": float(np.mean([same_root[k]["decisions"][v]["wait"]-r["decisions"][v]["wait"] for k, r in single.items()]))}
                for v in learning.VARIANTS},
            "wait_mass_loss_delta_min": {g: float(np.mean([same_root[k]["gates"][g]["mass_loss_min"]-r["gates"][g]["mass_loss_min"]
                for k, r in single.items()])) for g in ("glm", *learning.WAIT_VARIANTS)}}
    fits = [cell["fit"] for r in results.values() for cell in r["cells"]]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "coverage_qualified": True, "views": output, "single_vs_mixed": comparisons, **contract()["budget"],
        "glm_optimizer_iterations": sum(f["glm_iterations"] for f in fits),
        "fit_loss_mean": {name: {k: float(np.mean([f["fits"][name][k] for f in fits]))
            for k in ("initial_loss", "final_loss", "parameter_update_l2")} for name in ("best", *learning.WAIT_VARIANTS)},
        "conditioning_context_weight_l2_min": min(f["fits"]["state_conditioned_pair"]["context_weight_l2"] for f in fits),
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("fit", "report"), required=True)
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != previous.EXPERIMENT_PROTOCOL or prior["contract"] != previous.contract() or not prior["linear_bounds_qualified"]:
        raise ValueError("Complete Stage219 before learned crossfit")
    bank, _, registered, _ = source.source.load_data()
    if args.operation == "fit":
        output = result_path(args.run_name, args.view, args.fold)
        raw = raw_directory(output)
        panels = source.source.wait_source.source.diagnostic.load_panels()
        result = fit_view(bank, panels, registered, args.view, args.fold,
            lambda p, d, m: write_json(raw/f"{p}_{d}_models.json", m))
        qualify(result, args.view, args.fold, source.matched_views(bank["training_data"], registered))
        write_json(raw/"result.json", result)
        write_json(output, {k: v for k, v in result.items() if k != "cells"} | {"training_cells": 4,
            "supervised_decision_updates": 4*learning.STEPS, "wait_residual_updates": 12*learning.STEPS,
            "new_native_episodes": 0, "rl_actor_updates": 0})
        print("NATIVE_BEST_ACTION_CROSSFIT_FIT_COMPLETE", flush=True)
    else:
        results = {}
        for view in VIEWS:
            for fold in (0, 1):
                receipt = result_path(args.run_name, view, fold)
                results[view, fold] = json.loads((raw_directory(receipt)/"result.json").read_text())
        result = summarize(results, source.matched_views(bank["training_data"], registered))
        write_json(ROOT/"results"/args.run_name/"report/result.json", result)
        print("NATIVE_BEST_ACTION_CROSSFIT_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
