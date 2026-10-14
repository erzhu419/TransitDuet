#!/usr/bin/env python3
"""Matched all-state cost-gap learning and training-future wait calibration."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_best_action_crossfit_stage220 as previous
from freq_hrl.domains.transit import native_gap_calibration as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_gap_calibration_stage221_v1"
SOURCE_RUN = "native_transit_best_action_crossfit_stage220_development_20261014_r1"
VIEWS, PARTITIONS, SCOPES = previous.VIEWS, previous.PARTITIONS, previous.SCOPES
PARAMETERS = {"best": 98, "gap": 98, "linear": 199, "nonlinear_pair": 3201, "state_conditioned_pair": 3393}


def contract():
    parent = previous.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cohort": parent["data"], "views": list(VIEWS), "outer": parent["outer"],
        "partitions": parent["partitions"], "filtering": "all_states_no_agreement_filter",
        "controls": "unchanged_stage220_ridge_CE_GLM_and_three_raw_wait_residuals",
        "gap_target": "softmax_of_negative_physical_mean_cost_over_training_nonidentity_RMS_cost_no_true_safe_menu",
        "gap_score": "frozen_native_ridge_cost_plus_temperature_times_zero_initialized98_feature_linear_residual",
        "gap_objective": "KL_soft_cost_target_to_predicted_soft_cost_same_temperature_600_AdamW_updates",
        "wait_calibration": "two_complementary2_train2_validate_future_splits_within_outer_training4_same_training_prefixes",
        "shrinkage": "one_alpha_per_wait_architecture_in_closed_interval0_1_minimizes_pooled_inner_OOF_physical_draw_mass_logloss",
        "normalization": "training_prefix_features_only_each_inner_GLM_uses_its_own2_future_targets",
        "parameters": PARAMETERS, "optimizer": parent["optimizer"],
        "inference": "causal_cost_and_learned_wait_logits_no_future_labels_or_true_safe_menu",
        "variants": list(learning.VARIANTS), "evaluation": "same_four_scopes_rootwise_and_future_direction_physical_cost_wait_proper_risk_no_CI",
        "budget": {"fit_tasks": 8, "report_tasks": 1, "training_cells": 32, "analytic_cost_fits": 32,
            "glm_fits": 96, "inner_wait_fits": 192, "calibration_scalar_fits": 96,
            "supervised_decision_updates": 32*(learning.base.STEPS+learning.STEPS),
            "wait_residual_updates": 32*(3*learning.base.STEPS+6*learning.STEPS),
            "new_native_episodes": 0, "rl_actor_updates": 0},
        "storage": "models_predictions_remote_only_compact_report_and_receipts",
        "boundary": "original240_states_same8_conditional_futures_not_independent_replication_or_end_to_end_HRL"}


def result_path(run_name, view, fold):
    return ROOT/"results"/run_name/"cells"/view/f"fold_{fold}"/"receipt.json"


def fit_cell(train, held, panels, partition, direction, save_models):
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    train_batch, held_batch = learning.base.neural.batches(train), learning.base.neural.batches(held)
    models, fit = learning.fit(train, panels, a, batch=train_batch)
    save_models(models)
    train_prediction = learning.predict(train, models, batch=train_batch)
    held_prediction = learning.predict(held, models, batch=held_batch)
    scopes = {name: learning.base.records(train if name.startswith("train") else held, panels,
        a if name.endswith("same") else b, b if name.endswith("same") else a,
        train_prediction if name.startswith("train") else held_prediction) for name in SCOPES}
    groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
    return {"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
        "training_groups": groups(train), "held_groups": groups(held), "fit": fit, "scopes": scopes}


def fit_view(bank, panels, registered, view, fold, save_models):
    views = previous.source.matched_views(bank["training_data"], registered)
    train, held = previous.split_rows(views[view], fold)
    cells = []
    for partition in PARTITIONS:
        for direction in (0, 1):
            cell = fit_cell(train, held, panels, partition, direction, lambda m: save_models(partition, direction, m))
            cells.append(cell)
            print(f"GAP_CALIBRATION_CELL {view}/{fold}/{partition}/{direction} "
                f"alpha={ {v: round(x['alpha'], 4) for v, x in cell['fit']['calibration'].items()} }", flush=True)
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "view": view, "fold": fold,
        "software_qualified": True, "cells": cells}


def qualify(result, view, fold, views):
    train, held = previous.split_rows(views[view], fold)
    groups = lambda rs: {(r["root"], r["scene_seed"]) for r in rs}
    if (result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract()
            or result["view"] != view or result["fold"] != fold or not result["software_qualified"]
            or len(result["cells"]) != 4
            or {(c["partition"], c["direction"]) for c in result["cells"]} != {(p, d) for p in PARTITIONS for d in (0, 1)}):
        raise ValueError("Missing registered cost-gap future cell")
    for cell in result["cells"]:
        fit = cell["fit"]
        a, b = PARTITIONS[cell["partition"]][cell["direction"]], PARTITIONS[cell["partition"]][1-cell["direction"]]
        if (cell["training_draws"] != list(a) or cell["evaluation_draws"] != list(b)
                or set(map(tuple, cell["training_groups"])) != groups(train)
                or set(map(tuple, cell["held_groups"])) != groups(held)
                or set(map(tuple, fit["training_groups"])) != groups(train)
                or fit["training_states"] != 60 or fit["target_draw_count"] != [4]
                or fit["supervised_decision_updates"] != learning.base.STEPS+learning.STEPS
                or fit["wait_residual_updates"] != 3*learning.base.STEPS+6*learning.STEPS
                or fit["glm_fits"] != 3 or fit["analytic_cost_fits"] != 1 or fit["calibration_scalar_fits"] != 3
                or fit["new_native_episodes"] != 0 or fit["rl_actor_updates"] != 0
                or len(fit["inner_fits"]) != 2 or set(cell["scopes"]) != set(SCOPES)
                or set(fit["calibration"]) != set(learning.WAIT_VARIANTS)):
            raise ValueError("Changed grouping, target halves or accounted fit budget")
        for name, count in PARAMETERS.items():
            f = fit["fits"][name]
            if f["parameter_count"] != count or f["steps"] != learning.STEPS:
                raise ValueError("Changed cost or wait architecture/updates")
        for i, inner in enumerate(fit["inner_fits"]):
            halves = (list(a[:2]), list(a[2:]))
            if inner["training_draws"] != halves[i] or inner["validation_draws"] != halves[1-i]:
                raise ValueError("Inner calibration used outer evaluation futures")
            for v in learning.WAIT_VARIANTS:
                if inner["fits"][v]["steps"] != learning.STEPS or inner["fits"][v]["parameter_count"] != PARAMETERS[v]:
                    raise ValueError("Changed inner residual update budget")
        for v in learning.WAIT_VARIANTS:
            c = fit["calibration"][v]
            if (not 0 <= c["alpha"] <= 1 or c["candidate_cases"] != 600
                    or c["shrunk_loss_min"] > min(c["reference_loss_min"], c["raw_loss_min"])+1e-10):
                raise ValueError("Invalid convex calibration solution or candidate coverage")
        for scope, rows in cell["scopes"].items():
            expected = train if scope.startswith("train") else held
            if len(rows) != 60 or {learning.base.state_key(r) for r in rows} != {learning.base.state_key(r) for r in expected}:
                raise ValueError("Dropped a held state or split a prefix")
            if any(set(r["decisions"]) != set(learning.VARIANTS) or set(r["gates"]) != set(learning.GATES) for r in rows):
                raise ValueError("Missing learned selector or calibration control")


def metrics(rows):
    mean = lambda values: float(np.mean(values)) if rows else None
    return {"prediction_cases": len(rows), "unique_states": len({learning.base.state_key(r) for r in rows}),
        "unique_prefixes": len({(r["root"], r["scene_seed"]) for r in rows}), "disputed_cases": sum(r["disputed"] for r in rows),
        "variants": {v: {"mean_cost": mean([r["decisions"][v]["cost"] for r in rows]),
            "mean_wait_min": mean([r["decisions"][v]["wait"] for r in rows]),
            "changed_fraction": mean([r["decisions"][v]["changed"] for r in rows]),
            "unsafe_fraction": mean([r["decisions"][v]["unsafe"] for r in rows])} for v in learning.VARIANTS},
        "wait_heads": {v: {"physical_mass_loss_min": mean([r["gates"][v]["mass_loss_min"] for r in rows]),
            "false_safe_candidates": sum(r["gates"][v]["false_safe"] for r in rows),
            "false_unsafe_candidates": sum(r["gates"][v]["false_unsafe"] for r in rows)} for v in learning.GATES}}


def summarize(results, views):
    if set(results) != {(v, f) for v in VIEWS for f in (0, 1)}:
        raise ValueError("Complete all eight matched tasks before reporting")
    for (view, fold), result in results.items(): qualify(result, view, fold, views)
    output, held_records = {}, {}
    for view in VIEWS:
        cells = [c for f in (0, 1) for c in results[view, f]["cells"]]
        scopes = {s: [r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        if any(len(rs) != 480 or len({learning.base.state_key(r) for r in rs}) != 120 for rs in scopes.values()):
            raise ValueError("Changed all-state future sensitivity coverage")
        output[view] = {"scopes": {s: metrics(rs) for s, rs in scopes.items()},
            "held_other_by_root": {str(root): metrics([r for r in scopes["held_other"] if r["root"] == root])
                for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction": {f"{p}/{d}": {s: metrics([r for c in cells if c["partition"] == p and c["direction"] == d
                for r in c["scopes"][s]]) for s in ("train_same", "held_other")} for p in PARTITIONS for d in (0, 1)},
            "calibration": {v: {"alpha_mean": float(np.mean([c["fit"]["calibration"][v]["alpha"] for c in cells])),
                "zero_alpha_cells": sum(c["fit"]["calibration"][v]["alpha"] == 0 for c in cells),
                "cells": [c["fit"]["calibration"][v] for c in cells]} for v in learning.WAIT_VARIANTS}}
        held_records[view] = {(*learning.base.state_key(r), c["partition"], c["direction"]): r
            for c in cells for r in c["scopes"]["held_other"]}
    mixed = {k: r for v in ("mixed_a", "mixed_b") for k, r in held_records[v].items()}
    comparisons = {}
    for root in previous.source.prefixes.ROOTS:
        single = held_records[f"root_{root}"]
        matched = {k: r for k, r in mixed.items() if r["root"] == root}
        if set(single) != set(matched) or len(single) != 480:
            raise ValueError("Changed same-state single/mixed comparison")
        comparisons[str(root)] = {"prediction_cases": len(single), "unique_states": 120,
            "variants": {v: {"cost_delta": float(np.mean([matched[k]["decisions"][v]["cost"]-r["decisions"][v]["cost"] for k, r in single.items()])),
                "wait_delta_min": float(np.mean([matched[k]["decisions"][v]["wait"]-r["decisions"][v]["wait"] for k, r in single.items()]))}
                for v in learning.VARIANTS},
            "wait_mass_loss_delta_min": {v: float(np.mean([matched[k]["gates"][v]["mass_loss_min"]-r["gates"][v]["mass_loss_min"]
                for k, r in single.items()])) for v in learning.GATES}}
    fits = [c["fit"] for r in results.values() for c in r["cells"]]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "coverage_qualified": True, "views": output, "single_vs_mixed": comparisons, **contract()["budget"],
        "glm_optimizer_iterations": sum(f["glm_iterations"]+sum(i["glm_iterations"] for i in f["inner_fits"]) for f in fits),
        "fit_loss_mean": {v: {k: float(np.mean([f["fits"][v][k] for f in fits]))
            for k in ("initial_loss", "final_loss", "parameter_update_l2")} for v in PARAMETERS},
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("fit", "report"), required=True)
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != previous.EXPERIMENT_PROTOCOL or prior["contract"] != previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage220 before cost-gap calibration")
    bank, _, registered, _ = previous.source.source.load_data()
    views = previous.source.matched_views(bank["training_data"], registered)
    if args.operation == "fit":
        output = result_path(args.run_name, args.view, args.fold)
        raw = raw_directory(output)
        panels = previous.source.source.wait_source.source.diagnostic.load_panels()
        result = fit_view(bank, panels, registered, args.view, args.fold,
            lambda p, d, m: write_json(raw/f"{p}_{d}_models.json", m))
        qualify(result, args.view, args.fold, views)
        write_json(raw/"result.json", result)
        write_json(output, {k: v for k, v in result.items() if k != "cells"} | {"training_cells": 4,
            "supervised_decision_updates": 4*(learning.base.STEPS+learning.STEPS),
            "wait_residual_updates": 4*(3*learning.base.STEPS+6*learning.STEPS), "new_native_episodes": 0, "rl_actor_updates": 0})
        print("NATIVE_GAP_CALIBRATION_FIT_COMPLETE", flush=True)
    else:
        results = {(v, f): json.loads((raw_directory(result_path(args.run_name, v, f))/"result.json").read_text())
            for v in VIEWS for f in (0, 1)}
        write_json(ROOT/"results"/args.run_name/"report/result.json", summarize(results, views))
        print("NATIVE_GAP_CALIBRATION_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
