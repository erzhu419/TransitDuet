#!/usr/bin/env python3
"""Isolate paired-critic capacity using fixed targets, budgets and physical selection."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_joint_selection_stage227 as previous
from freq_hrl.domains.transit import native_linear_joint as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

joint, training = learning.joint, previous.previous
EXPERIMENT_PROTOCOL = "native_transit_linear_joint_stage228_v1"
SOURCE_RUN = "native_transit_joint_selection_stage227_development_20261014_r1"
VIEWS, PARTITIONS, KEYS, SCOPES = previous.VIEWS, previous.PARTITIONS, previous.KEYS, previous.SCOPES


def contract():
    result = training.contract()
    parent = previous.contract()
    result.update(source_run=SOURCE_RUN, source_protocol=previous.EXPERIMENT_PROTOCOL,
        representation="linear198_engineered_causal_action_features_two_head_zero_initialized_residual_over_same_ridge",
        null_terms="omit12_action_constant_context_terms_and_bias_that_cancel_exactly_in_linear_paired_differences",
        parameters_per_branch=learning.PARAMETERS, selection=parent["selection"],
        selection_scale=parent["scale"], variants=list(learning.VARIANTS),
        controls="all11_frozen_Stage227_controllers_plus_linear_zero_last_MSE_joint_for_both_anchors",
        budget={"fit_tasks":32, "report_tasks":1, "clean_inner_reference_fits":640, "analytic_cost_fits":640,
            "glm_fits":640, "linear_inner_fits":1280, "linear_final_fits":64, "raw_wait_updates":384000,
            "linear_updates":806400, "network_updates":1190400, "reused_nonlinear_paths":64,
            "new_native_episodes":0, "rl_actor_updates":0},
        compute_boundary="linear_vs_frozen_nonlinear_same600_steps_lr_decay_targets_and_selection_not_equal_total_to_raw_controls",
        storage="clean_refits_and_new_linear_models_remote_only_pull_aggregate_report")
    result["optimizer"].pop("seed")
    result["optimizer"]["initialization"] = "zero"
    return result


def result_path(run_name, *key): return training.result_path(run_name, *key)


def fit_cell(train, held, panels, nonlinear, bundle, calibration, reused_oof, partition, direction):
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    selection, inner, reproduction = learning.collect(train, panels, a, reused_oof)
    model, final = joint.fit_final(train, panels, a, bundle, selection, path_fitter=learning.fit_path)
    predictions = {name:learning.predict(rows, model, nonlinear, bundle, calibration)
        for name,rows in (("train",train),("held",held))}
    groups = lambda rs:sorted(map(list,{(r["root"],r["scene_seed"]) for r in rs}))
    cell = {"partition":partition, "direction":direction, "training_draws":list(a), "evaluation_draws":list(b),
        "training_groups":groups(train), "held_groups":groups(held), "selection":selection,
        "inner_fits":inner, "final_fits":final, "inner_reference_reproduction":reproduction,
        "scopes":{s:training.records(train if s.startswith("train") else held, panels, a if s.endswith("same") else b,
            b if s.endswith("same") else a, predictions["train" if s.startswith("train") else "held"]) for s in SCOPES}}
    return cell, {"paths":model["paths"], "selection":selection}


def qualify(result, key, views):
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    groups = lambda rs:{(r["root"],r["scene_seed"]) for r in rs}
    c = result["cell"]
    a,b = PARTITIONS[key[2]][key[3]], PARTITIONS[key[2]][1-key[3]]
    if (result["protocol"]!=EXPERIMENT_PROTOCOL or result["contract"]!=contract() or result["key"]!=list(key)
            or c["partition"]!=key[2] or c["direction"]!=key[3] or c["training_draws"]!=list(a) or c["evaluation_draws"]!=list(b)
            or set(map(tuple,c["training_groups"]))!=groups(train) or set(map(tuple,c["held_groups"]))!=groups(held)):
        raise ValueError("Changed linear paired outer protocol")
    def fitted(fit, gs, n, kind):
        if (fit["anchor"]!=kind or fit["training_states"]!=n or set(map(tuple,fit["training_groups"]))!=gs
                or fit["parameter_count"]!=learning.PARAMETERS or fit["steps"]!=max(joint.CHECKPOINTS)):
            raise ValueError("Changed linear fitting capacity or budget")
    ledger = c["inner_fits"]
    if len(ledger)!=20 or {(tuple(f["held_group"]),f["direction"]) for f in ledger}!={(g,d) for g in groups(train) for d in (0,1)}:
        raise ValueError("Missing clean linear prefix/future fit")
    for f in ledger:
        expected, d = groups(train)-{tuple(f["held_group"])}, f["direction"]
        ref = f["reference"]
        if (f["training_draws"]!=list(a[2*d:2*d+2]) or f["validation_draws"]!=list(a[2*(1-d):2*(1-d)+2])
                or f["held_states"]!=6 or ref["draws"]!=f["training_draws"] or ref["training_states"]!=54
                or ref["raw_steps"]!=600 or ref["raw_parameters"]!=3393
                or set(map(tuple,ref["training_groups"]))!=expected or set(map(tuple,ref["wait_training_groups"]))!=expected):
            raise ValueError("Linear raw reference leaked prefix/future labels or changed budget")
        for kind in joint.ANCHORS: fitted(f["fits"][kind],expected,54,kind)
    for kind in joint.ANCHORS:
        fitted(c["final_fits"][kind],groups(train),60,kind)
        x = c["selection"][kind]
        if (x["mse_step"] not in joint.CHECKPOINTS or x["joint_step"] not in (*joint.CHECKPOINTS,None)
                or set(x["curves"])!=set(map(str,joint.CHECKPOINTS))): raise ValueError("Changed linear checkpoint set")
        if x["joint_step"] is not None:
            curve = x["curves"][str(x["joint_step"])]
            if curve["joint_score"]>=0 or any(y[k]>=0 for y in curve["by_root"].values() for k in ("mean_delta_cost","mean_delta_wait_min")):
                raise ValueError("Linear selection failed pooled physical joint improvement")
    if c["inner_reference_reproduction"]["max_abs_difference"]>1e-12 or c["inner_reference_reproduction"]["changed_choices"]:
        raise ValueError("Clean inner raw controls changed")
    if set(c["scopes"])!=set(SCOPES): raise ValueError("Missing linear scope")
    for scope,rows in c["scopes"].items():
        expected = train if scope.startswith("train") else held
        if (len(rows)!=60 or {joint.binding.learned.state_key(r) for r in rows}!={joint.binding.learned.state_key(r) for r in expected}
                or any(set(r["decisions"])!=set(learning.VARIANTS) or set(r["gates"])!=set(joint.CONTROLS) for r in rows)):
            raise ValueError("Dropped a linear state or controller")


def metrics(rows): return training.metrics(rows, variants=learning.VARIANTS)


def summarize(results, views):
    if set(results)!=set(KEYS): raise ValueError("Complete all32 linear paired fits")
    for key,r in results.items(): qualify(r,key,views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for key,r in results.items() if key[0]==view]
        scopes = {s:[r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        output[view] = {"scopes":{s:metrics(rs) for s,rs in scopes.items()},
            "held_other_by_root":{str(root):metrics([r for r in scopes["held_other"] if r["root"]==root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction":{f"{p}/{d}":metrics([r for c in cells if c["partition"]==p and c["direction"]==d for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0,1)},
            "selection":{a:[c["selection"][a] for c in cells] for a in joint.ANCHORS},
            "final_training_losses":{a:[c["final_fits"][a]["loss_ledger"] for c in cells] for a in joint.ANCHORS}}
        if any(x["prediction_cases"]!=480 or x["unique_states"]!=120 for x in output[view]["scopes"].values()):
            raise ValueError("Changed complete linear state coverage")
    return {"protocol":EXPERIMENT_PROTOCOL, "contract":contract(), "coverage_qualified":True, "views":output,
        "inner_reference_max_abs_difference":max(r["cell"]["inner_reference_reproduction"]["max_abs_difference"] for r in results.values()),
        **contract()["budget"], "native_deployment_eligible":False}


def reproduce_controls(result, prior, *, names=joint.VARIANTS):
    differences = []
    for view in VIEWS:
        for scope in SCOPES:
            a,b = result["views"][view]["scopes"][scope],prior["views"][view]["scopes"][scope]
            for name in names:
                for m in ("mean_cost","mean_wait_min","changed_fraction","unsafe_fraction","switch_fraction_to_raw"):
                    differences.append(abs(a["variants"][name][m]-b["variants"][name][m]))
                if name in joint.CONTROLS:
                    differences.append(abs(a["wait_heads"][name]["physical_mass_loss_min"]-b["wait_heads"][name]["physical_mass_loss_min"]))
                else:
                    ae,be = a["joint_prediction_errors"][name],b["joint_prediction_errors"][name]
                    if ae["cases"]!=be["cases"] or (ae["normalized_mse"] is None)!=(be["normalized_mse"] is None):
                        raise ValueError("Frozen source prediction coverage changed")
                    if ae["normalized_mse"] is not None: differences.append(abs(ae["normalized_mse"]-be["normalized_mse"]))
    if max(differences)>1e-12: raise ValueError("Frozen source controller or path changed")
    return {"comparisons":len(differences), "max_abs_difference":max(differences)}


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
    if prior["protocol"]!=previous.EXPERIMENT_PROTOCOL or prior["contract"]!=previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage227 before linear capacity test")
    bank,_,registered,_ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"],registered)
    if args.operation=="fit":
        key = args.view,args.fold,args.partition,args.direction
        source = raw_directory(previous.result_path(SOURCE_RUN,*key))
        fixed = json.loads((source/"result.json").read_text())
        previous.qualify(fixed,key,views)
        nonlinear = json.loads((raw_directory(training.result_path(previous.SOURCE_RUN,*key))/"models.json").read_text())
        nonlinear["selection"] = fixed["cell"]["selection"]
        base = json.loads((raw_directory(training.binding_spec.result_path(training.MODEL_RUN,*key))/"models.json").read_text())
        oof = json.loads((raw_directory(training.precision_spec.result_path(training.binding_spec.SOURCE_RUN,*key))/"inner_oof.json").read_text())["c2v4"]
        train,held = training.cohort.split_rows(views[args.view],args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell,model = fit_cell(train,held,panels,nonlinear,base["bundle"],base["calibration"],
            {k:np.asarray(v) for k,v in oof.items()},args.partition,args.direction)
        result = {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"key":list(key),"cell":cell}
        qualify(result,key,views)
        output = result_path(args.run_name,*key)
        write_json(raw_directory(output)/"models.json",model)
        write_json(raw_directory(output)/"result.json",result)
        write_json(output,{k:v for k,v in result.items() if k!="cell"})
        print("NATIVE_LINEAR_JOINT_FIT_COMPLETE",flush=True)
    else:
        result = summarize({k:json.loads((raw_directory(result_path(args.run_name,*k))/"result.json").read_text()) for k in KEYS},views)
        result["control_reproduction"] = reproduce_controls(result,prior)
        write_json(ROOT/"results"/args.run_name/"report/result.json",result)
        print("NATIVE_LINEAR_JOINT_REPORT_COMPLETE",flush=True)


if __name__=="__main__": main()
