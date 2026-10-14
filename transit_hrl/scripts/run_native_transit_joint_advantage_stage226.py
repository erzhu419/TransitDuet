#!/usr/bin/env python3
"""Train paired physical cost/wait critics with prefix/future-only selection."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_decision_relevance_stage225 as previous
from scripts import run_native_transit_refit_binding_stage224 as binding_spec
from scripts import run_native_transit_precision_stage223 as precision_spec
from scripts import run_native_transit_best_action_crossfit_stage220 as cohort
from freq_hrl.domains.transit import native_joint_advantage as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_joint_advantage_stage226_v1"
SOURCE_RUN = "native_transit_decision_relevance_stage225_development_20261014_r1"
MODEL_RUN = previous.SOURCE_RUN
VIEWS, PARTITIONS, KEYS = previous.VIEWS, previous.PARTITIONS, previous.KEYS
SCOPES = cohort.SCOPES


def contract():
    steps = max(learning.CHECKPOINTS)
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL, "model_run": MODEL_RUN,
        "cohort": previous.contract()["cohort"], "outer": previous.contract()["outer"], "partitions": previous.contract()["partitions"],
        "objective": "two_head_standardized_MSE_of_physical_actual_cost_and_mean_wait_increments",
        "anchors": list(learning.ANCHORS), "representation": "same210_causal_inputs_width16_tanh_shared_two_head_residual_over_three_head_ridge",
        "parameters_per_branch": learning.PARAMETERS, "optimizer": {"steps": steps, "lr": learning.binding.learned.neural.LEARNING_RATE,
            "weight_decay": learning.binding.learned.neural.WEIGHT_DECAY, "seed": learning.binding.learned.neural.SEED, "dtype": "float64"},
        "inner": "leave_one_complete_prefix54_train6_validation_times_both2_draw_directions_inside_outer_training4",
        "reference": "inner_raw_ridge_GLM_conditioned600_refit_excluding_validation_prefix_and_future_half_final_raw_exact_reused_final4",
        "normalization": "reference_inputs_and_own_training_nonanchor_RMS_targets_only_no_validation_scales",
        "checkpoints": list(learning.CHECKPOINTS), "selection": "MSE_or_worst_source_root_mean_normalized_cost_wait_increment_to_raw_ties_earlier",
        "hold": "joint_selection_keeps_raw_if_every_checkpoint_score_ge0_not_a_new_training_constraint",
        "final": "two60_state_four_draw_branches_run600_then_use_inner_selected_checkpoint",
        "controls": "exact_GLM_raw_conditioned_c2v4_plus_zero_last_MSE_joint_same_path_controls_for_both_anchors",
        "inference": "predict_both_increments_select_lowest_cost_among_wait_le0_with_exact_anchor_first_ties_no_truth_menu",
        "variants": list(learning.VARIANTS), "wait_gates": list(learning.CONTROLS),
        "budget": {"fit_tasks": 32, "report_tasks": 1, "training_cells": 32, "clean_inner_reference_fits": 640,
            "analytic_cost_fits": 640, "glm_fits": 640, "joint_inner_fits": 1280, "joint_final_fits": 64,
            "raw_wait_updates": 640*steps, "joint_updates": 1344*steps, "network_updates": 1984*steps,
            "reused_final4_cells": 32, "new_native_episodes": 0, "rl_actor_updates": 0},
        "compute_boundary": "original_vs_raw_anchors_same_compute_each_path600_total_method_compute_not_equal_to_frozen_controls",
        "boundary": "development240_states8_draws_supervised_single_action_greedy_suffix_not_new_seed_CI_or_end_to_end_HRL",
        "storage": "inner_reference_and_path_models_remote_only_compact_aggregate_report_pull"}


def result_path(run_name, *key):
    return previous.result_path(run_name, *key)


def records(rows, panels, indices, alternate, prediction):
    output = learning.binding.learned.records(rows, panels, indices, alternate, prediction)
    values = learning.physical_values(rows, panels, indices)
    for i, r in enumerate(output):
        r["joint_prediction_errors"] = {}
        for name, p in prediction["advantages"].items():
            mask = np.arange(6) != p["anchor"][i]
            true = values[i]-values[i, p["anchor"][i]]
            r["joint_prediction_errors"][name] = float((((p["delta"][i]-true)/p["scales"])[mask]**2).mean())
    return output


def fit_cell(train, held, panels, bundle, calibration, reused_oof, partition, direction):
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    selection, inner, reproduction = learning.collect(train, panels, a, reused_oof)
    model, final = learning.fit_final(train, panels, a, bundle, selection)
    predictions = {name: learning.predict(rows, model, bundle, calibration) for name, rows in (("train", train), ("held", held))}
    scopes = {s: records(train if s.startswith("train") else held, panels, a if s.endswith("same") else b,
        b if s.endswith("same") else a, predictions["train" if s.startswith("train") else "held"]) for s in SCOPES}
    groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
    cell = {"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
        "training_groups": groups(train), "held_groups": groups(held), "selection": selection,
        "inner_fits": inner, "final_fits": final, "inner_reference_reproduction": reproduction, "scopes": scopes}
    return cell, model


def qualify(result, key, views):
    train, held = cohort.split_rows(views[key[0]], key[1])
    groups = lambda rs: {(r["root"], r["scene_seed"]) for r in rs}
    cell = result["cell"]
    a, b = PARTITIONS[key[2]][key[3]], PARTITIONS[key[2]][1-key[3]]
    if (result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract() or result["key"] != list(key)
            or cell["partition"] != key[2] or cell["direction"] != key[3]
            or cell["training_draws"] != list(a) or cell["evaluation_draws"] != list(b)
            or set(map(tuple, cell["training_groups"])) != groups(train) or set(map(tuple, cell["held_groups"])) != groups(held)):
        raise ValueError("Changed joint-advantage outer protocol")
    ledger = cell["inner_fits"]
    if len(ledger) != 20 or {(tuple(f["held_group"]), f["direction"]) for f in ledger} != {(g,d) for g in groups(train) for d in (0,1)}:
        raise ValueError("Missing clean prefix/future reference fit")
    for f in ledger:
        expected = groups(train)-{tuple(f["held_group"])}
        d = f["direction"]
        if (f["training_draws"] != list(a[2*d:2*d+2]) or f["validation_draws"] != list(a[2*(1-d):2*(1-d)+2])
                or f["reference"]["draws"] != f["training_draws"] or f["held_states"] != 6
                or f["reference"]["training_states"] != 54 or f["reference"]["raw_steps"] != max(learning.CHECKPOINTS)
                or f["reference"]["raw_parameters"] != learning.binding.learned.neural.PARAMETERS["state_conditioned_pair"]
                or set(map(tuple,f["reference"]["training_groups"])) != expected
                or set(map(tuple,f["reference"]["wait_training_groups"])) != expected):
            raise ValueError("Inner raw reference leaked prefix/future labels or changed budget")
        for kind in learning.ANCHORS:
            fit = f["fits"][kind]
            if (fit["training_states"] != 54 or set(map(tuple,fit["training_groups"])) != expected
                    or fit["steps"] != max(learning.CHECKPOINTS) or fit["parameter_count"] != learning.PARAMETERS or fit["anchor"] != kind):
                raise ValueError("Changed inner joint branch compute")
    for kind in learning.ANCHORS:
        fit, selection = cell["final_fits"][kind], cell["selection"][kind]
        if (fit["training_states"] != 60 or set(map(tuple,fit["training_groups"])) != groups(train)
                or fit["steps"] != max(learning.CHECKPOINTS) or fit["parameter_count"] != learning.PARAMETERS
                or selection["mse_step"] not in learning.CHECKPOINTS or selection["joint_step"] not in (*learning.CHECKPOINTS,None)
                or set(selection["curves"]) != set(map(str,learning.CHECKPOINTS))):
            raise ValueError("Changed final4 fitting or inner-only checkpoint selection")
    reproduction = cell["inner_reference_reproduction"]
    if reproduction["max_abs_difference"] > 1e-12 or reproduction["changed_choices"]: raise ValueError("Inner raw controls changed")
    if set(cell["scopes"]) != set(SCOPES): raise ValueError("Missing train/held scope")
    for scope, rows in cell["scopes"].items():
        expected = train if scope.startswith("train") else held
        if (len(rows) != 60 or {learning.binding.learned.state_key(r) for r in rows} != {learning.binding.learned.state_key(r) for r in expected}
                or any(set(r["decisions"]) != set(learning.VARIANTS) or set(r["gates"]) != set(learning.CONTROLS) for r in rows)):
            raise ValueError("Dropped a state or a joint selector")


def metrics(rows):
    return {"prediction_cases": len(rows), "unique_states": len({learning.binding.learned.state_key(r) for r in rows}),
        "variants": {v: {"mean_cost": float(np.mean([r["decisions"][v]["cost"] for r in rows])),
            "mean_wait_min": float(np.mean([r["decisions"][v]["wait"] for r in rows])),
            "changed_fraction": float(np.mean([r["decisions"][v]["changed"] for r in rows])),
            "unsafe_fraction": float(np.mean([r["decisions"][v]["unsafe"] for r in rows])),
            "switch_fraction_to_raw": float(np.mean([r["decisions"][v]["selected"] != r["decisions"]["raw"]["selected"] for r in rows]))}
            for v in learning.VARIANTS},
        "wait_heads": {g: {"physical_mass_loss_min": float(np.mean([r["gates"][g]["mass_loss_min"] for r in rows]))} for g in learning.CONTROLS},
        "joint_prediction_errors": {v: {"cases": sum(v in r["joint_prediction_errors"] for r in rows),
            "normalized_mse": float(np.mean([r["joint_prediction_errors"][v] for r in rows if v in r["joint_prediction_errors"]]))
                if any(v in r["joint_prediction_errors"] for r in rows) else None} for v in learning.VARIANTS if v.startswith("pair_")}}


def summarize(results, views):
    if set(results) != set(KEYS): raise ValueError("Complete all32 joint-advantage fits")
    for key, result in results.items(): qualify(result,key,views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for key,r in results.items() if key[0] == view]
        scopes = {s: [r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        output[view] = {"scopes": {s: metrics(rs) for s,rs in scopes.items()},
            "held_other_by_root": {str(root): metrics([r for r in scopes["held_other"] if r["root"] == root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction": {f"{p}/{d}": metrics([r for c in cells if c["partition"] == p and c["direction"] == d for r in c["scopes"]["held_other"]])
                for p in PARTITIONS for d in (0,1)},
            "selection": {a: [{"mse_step": c["selection"][a]["mse_step"], "joint_step": c["selection"][a]["joint_step"],
                "curves": c["selection"][a]["curves"]} for c in cells] for a in learning.ANCHORS}}
        if any(x["prediction_cases"] != 480 or x["unique_states"] != 120 for x in output[view]["scopes"].values()):
            raise ValueError("Changed complete-state coverage")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "views": output,
        "inner_reference_max_abs_difference": max(r["cell"]["inner_reference_reproduction"]["max_abs_difference"] for r in results.values()),
        **contract()["budget"], "native_deployment_eligible": False}


def reproduce_controls(result, prior):
    mapping = {"glm": "glm", "raw": "raw_state_conditioned_pair", "c2v4": "oof_state_conditioned_pair"}
    differences = []
    for view in VIEWS:
        for scope in SCOPES:
            a,b = result["views"][view]["scopes"][scope], prior["views"][view]["scopes"][scope]
            for g,old in mapping.items():
                for m in ("mean_cost","mean_wait_min","changed_fraction","unsafe_fraction"):
                    differences.append(abs(a["variants"][g][m]-b["decisions"][old][m]))
                differences.append(abs(a["wait_heads"][g]["physical_mass_loss_min"]-b["strata"]["all"]["heads"][old]["physical_mass_loss_min"]))
    if max(differences) > 1e-12: raise ValueError("Stage225 controls failed reproduction")
    return {"comparisons": len(differences), "max_abs_difference": max(differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",required=True)
    parser.add_argument("--operation",choices=("fit","report"),required=True)
    parser.add_argument("--view",choices=VIEWS)
    parser.add_argument("--fold",type=int,choices=(0,1))
    parser.add_argument("--partition",choices=PARTITIONS)
    parser.add_argument("--direction",type=int,choices=(0,1))
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != previous.EXPERIMENT_PROTOCOL or prior["contract"] != previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage225 before joint training")
    bank,_,registered,_ = cohort.source.source.load_data()
    views = cohort.source.matched_views(bank["training_data"],registered)
    if args.operation == "fit":
        key = args.view,args.fold,args.partition,args.direction
        source = raw_directory(binding_spec.result_path(MODEL_RUN,*key))
        binding_spec.qualify(json.loads((source/"result.json").read_text()),key,views)
        reused = json.loads((source/"models.json").read_text())
        oof_path = raw_directory(precision_spec.result_path(binding_spec.SOURCE_RUN,*key))/"inner_oof.json"
        oof = {k: np.asarray(v) for k,v in json.loads(oof_path.read_text())["c2v4"].items()}
        train,held = cohort.split_rows(views[args.view],args.fold)
        panels = cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell,model = fit_cell(train,held,panels,reused["bundle"],reused["calibration"],oof,args.partition,args.direction)
        result = {"protocol": EXPERIMENT_PROTOCOL,"contract": contract(),"key": list(key),"cell": cell}
        qualify(result,key,views)
        output = result_path(args.run_name,*key)
        write_json(raw_directory(output)/"models.json",model)
        write_json(raw_directory(output)/"result.json",result)
        write_json(output,{k:v for k,v in result.items() if k != "cell"})
        print("NATIVE_JOINT_ADVANTAGE_FIT_COMPLETE",flush=True)
    else:
        result = summarize({k:json.loads((raw_directory(result_path(args.run_name,*k))/"result.json").read_text()) for k in KEYS},views)
        result["control_reproduction"] = reproduce_controls(result,prior)
        write_json(ROOT/"results"/args.run_name/"report/result.json",result)
        print("NATIVE_JOINT_ADVANTAGE_REPORT_COMPLETE",flush=True)


if __name__ == "__main__": main()
