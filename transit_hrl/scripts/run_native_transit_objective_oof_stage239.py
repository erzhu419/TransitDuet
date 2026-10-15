#!/usr/bin/env python3
"""Select objectives with clean cached prefix/future references and wait floors."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_cost_objectives_stage238 as previous
from freq_hrl.domains.transit import native_objective_oof as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_objective_oof_stage239_v1"
SOURCE_RUN = "native_transit_cost_objectives_stage238_development_20261015_r3"
REFIT_RUN = previous.SOURCE_RUN
VIEWS, PARTITIONS, KEYS = previous.VIEWS, previous.PARTITIONS, previous.KEYS
training, joint = previous.training, learning.joint
result_path = previous.result_path


def contract():
    parent = previous.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL, "refit_run": REFIT_RUN,
        "reference_run": previous.previous.REFERENCE_RUN, "model_run": previous.previous.MODEL_RUN,
        "cohort": parent["cohort"], "outer": parent["outer"], "partitions": parent["partitions"],
        "objectives": list(learning.OBJECTIVES), "filters": list(learning.FILTERS), "variants": list(learning.VARIANTS), "scopes": list(learning.SCOPES),
        "grid": learning.strength.CANDIDATES, "representation": "same198_causal_features_TSVD_basis_rank_and_all_MSE_RMS_ball_per_cached_reference",
        "inner": "10_complete_prefix_folds_x2_future_directions_train54states_2draws_validate6states_opposite2draws",
        "reference": "Stage237_changing_native_x_inner_references_and_paths_cached_no_raw_refit",
        "wait_rank": "feasible_pair_gap_weighted_logistic_ranking_of_cost/cached_cost_scale_plus_wait/cached_wait_scale_fixed_weight1_same_cost_score_offset",
        "gate": parent["gate"], "optimizer": parent["optimizer"],
        "selection": "independently_each_objective_and_joint_across_objectives_each_anchor_same17_keys_no_outer_labels",
        "pooled_safe": "per_source_cost_wait_nonharm_risk_nonincrease_pooled_worst_scaled_cost_wait_strictly_negative",
        "half_safe": "pooled_safe_plus_each_source_each_inner_future_direction_cost_wait_nonharm_and_risk_nonincrease",
        "tie_break": "minimum_pooled_worst_scaled_cost_wait_then_fixed_objective_grid_order_no_candidate_means_raw",
        "tolerance": {"physical_nonharm": learning.TOL, "strict_scaled_improvement": learning.TOL, "unsafe": 0.},
        "evaluation": "freeze_OOF_selection_then_refit_same_objectives_full60_4_under_cached_outer_reference_train_same_other_held_same_other",
        "metrics": "actual_physical_cost_wait_unsafe_own_raw_deltas_and_gate_regret_score_MSE_not_cost_calibration_for_wait_rank",
        "controls": "all640_cached_clean_OOF_raw_blocks_and_all_Stage234_allMSE_inner_curves_and6_selected_keys_reproduce",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "inner_blocks": 640, "new_raw_reference_fits": 0,
            "raw_wait_updates": 0, "residual_SVDs": 2688, "objective_solves": 16128, "new_native_episodes": 0, "rl_actor_updates": 0},
        "boundary": parent["boundary"], "storage": "remote_models_and_inner_curves_pull_small_aggregate_only",
        "limitation": "reused_development_data_OOF_selection_is_not_fresh_CI_or_per_state_safety_guarantee"}


def qualify(result, key, views):
    if result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract() or result["key"] != list(key):
        raise ValueError("Changed objective OOF protocol")
    c = result["cell"]
    train, _ = training.cohort.split_rows(views[key[0]], key[1])
    groups = {(r["root"], r["scene_seed"]) for r in train}
    a = PARTITIONS[key[2]][key[3]]
    if (len(c["blocks"]) != 20 or {(tuple(b["held_group"]), b["half"]) for b in c["blocks"]} != {(g, h) for g in groups for h in (0, 1)}
            or c["training_draws"] != list(a) or c["evaluation_draws"] != list(PARTITIONS[key[2]][1-key[3]])):
        raise ValueError("Dropped clean OOF prefix/future block")
    fits = []
    for b in c["blocks"]:
        h = b["half"]
        if (b["training_states"] != 54 or b["held_states"] != 6 or b["training_draws"] != list(a[2*h:2*h+2])
                or b["validation_draws"] != list(a[2*(1-h):2*(1-h)+2])): raise ValueError("Changed clean OOF labels")
        for kind in joint.ANCHORS:
            if set(b["curves"][kind]) != set(learning.OBJECTIVES): raise ValueError("Dropped OOF objective")
            for o in learning.OBJECTIVES:
                if set(b["curves"][kind][o]) != set(learning.strength.CANDIDATES): raise ValueError("Dropped OOF capacity")
        fits.append(b["fits"])
    fits.append(c["full_fits"])
    expected = {(o, l) for o in learning.OBJECTIVES[1:] for l in learning.strength.lowrank.LEVELS[1:]}
    for record in fits:
        if set(record) != set(joint.ANCHORS): raise ValueError("Dropped OOF anchor")
        for fs in record.values():
            if len(fs) != 12 or {(f["objective"], f["level"]) for f in fs} != expected: raise ValueError("Dropped objective fit")
            for f in fs:
                if (not f["success"] or f["parameter_count"] != f["retained_rank"] or f["optimality"] > 1e-8
                        or f["norm"] > f["radius"]+1e-12 or f["loss_after"] > f["loss_before"]+1e-12*max(1., f["loss_before"])):
                    raise ValueError("OOF capacity or optimization failed")
    for kind in joint.ANCHORS:
        if c["selection"][kind] != learning.choose(c["curves"][kind]): raise ValueError("OOF selection not frozen from curves")
    if set(c["scopes"]) != set(learning.SCOPES): raise ValueError("Dropped OOF evaluation scope")
    for stats in c["scopes"].values():
        if stats["all"]["cases"] != 60 or sum(p["cases"] for p in stats["by_root"].values()) != 60:
            raise ValueError("Dropped OOF outer state")
        if any(set(p["variants"]) != set(learning.VARIANTS) for p in (stats["all"], *stats["by_root"].values())):
            raise ValueError("Dropped OOF controller")
    r = c["reproduction"]
    if (r["oof_choice_comparisons"] != 120 or r["legacy_choice_comparisons"] != 6 or r["oof_changed_choices"] or r["legacy_changed_choices"]
            or max(r[f] for f in ("oof_max_abs_difference", "subspace_max_abs_difference", "legacy_max_abs_difference")) > 1e-12):
        raise ValueError("Cached clean controls changed")


def aggregate(cells):
    return {scope: {"all": learning.averaged([c["scopes"][scope]["all"] for c in cells]),
        "by_root": {r: learning.averaged([c["scopes"][scope]["by_root"][r] for c in cells]) for r in cells[0]["scopes"][scope]["by_root"]}}
        for scope in learning.SCOPES}


def summarize(results, views):
    if set(results) != set(KEYS): raise ValueError("Complete all32 clean OOF cells")
    for k, r in results.items(): qualify(r, k, views)
    cells = [results[k]["cell"] for k in KEYS]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "views": {v: aggregate([results[k]["cell"] for k in KEYS if k[0] == v]) for v in VIEWS},
        "mixed_pooled": aggregate([results[k]["cell"] for k in KEYS if k[0].startswith("mixed")]),
        "selections_by_cell": [{"key": list(k), "selection": results[k]["cell"]["selection"], "scopes": results[k]["cell"]["scopes"]} for k in KEYS],
        "admission": {f"{o}_{a}_{f}": {"admitted_cells": sum(c["selection"][a][o][f]["candidate"] is not None for c in cells),
            "objective_counts": {obj: sum(c["selection"][a][o][f]["objective"] == obj for c in cells) for obj in learning.OBJECTIVES}}
            for o in learning.FAMILIES for a in joint.ANCHORS for f in learning.FILTERS},
        "reproduction": {**{f: sum(c["reproduction"][f] for c in cells) for f in ("oof_choice_comparisons", "legacy_choice_comparisons", "oof_changed_choices", "legacy_changed_choices")},
            **{f: max(c["reproduction"][f] for c in cells) for f in ("oof_max_abs_difference", "subspace_max_abs_difference", "legacy_max_abs_difference")}},
        **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("fit", "report"), required=True)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--fold", type=int, choices=(0, 1))
    p.add_argument("--partition", choices=PARTITIONS)
    p.add_argument("--direction", type=int, choices=(0, 1))
    args = p.parse_args()
    bank, _, registered, _ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"], registered)
    if args.operation == "fit":
        key = args.view, args.fold, args.partition, args.direction
        source = json.loads((raw_directory(result_path(REFIT_RUN, *key))/"result.json").read_text())
        previous.previous.qualify(source, key, views)
        selection = json.loads((raw_directory(result_path(previous.previous.OUTER_RUN, *key))/"result.json").read_text())["cell"]["selection"]
        reference = json.loads((raw_directory(result_path(previous.previous.REFERENCE_RUN, *key))/"models.json").read_text())["reference"]
        paths = json.loads((raw_directory(result_path(previous.previous.MODEL_RUN, *key))/"models.json").read_text())["paths"]
        cached = json.loads((raw_directory(result_path(REFIT_RUN, *key))/"models.json").read_text())
        oof_path = raw_directory(training.precision_spec.result_path(training.binding_spec.SOURCE_RUN, *key))/"inner_oof.json"
        oof = {k: np.asarray(v) for k, v in json.loads(oof_path.read_text())["c2v4"].items()}
        train, held = training.cohort.split_rows(views[args.view], args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell, models = learning.fit_cell(train, held, panels, reference, paths, source["cell"], cached, oof, selection)
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "cell": cell}
        qualify(result, key, views)
        path = result_path(args.run_name, *key)
        write_json(raw_directory(path)/"models.json", models)
        write_json(raw_directory(path)/"result.json", result)
        write_json(path, {k: v for k, v in result.items() if k != "cell"})
        print("NATIVE_OBJECTIVE_OOF_FIT_COMPLETE", flush=True)
    else:
        report = summarize({k: json.loads((raw_directory(result_path(args.run_name, *k))/"result.json").read_text()) for k in KEYS}, views)
        write_json(ROOT/"results"/args.run_name/"report/result.json", report)
        print("NATIVE_OBJECTIVE_OOF_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
