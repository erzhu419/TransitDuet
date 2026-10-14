#!/usr/bin/env python3
"""Test explicit residual shrinkage and training-only feature rank."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_native_transit_linear_joint_stage228 as previous
from freq_hrl.domains.transit import native_ridge_joint as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory,write_json

joint,training = learning.joint,previous.training
EXPERIMENT_PROTOCOL = "native_transit_ridge_joint_stage229_v1"
SOURCE_RUN = "native_transit_linear_joint_stage228_development_20261014_r1"
VIEWS,PARTITIONS,KEYS,SCOPES = previous.VIEWS,previous.PARTITIONS,previous.KEYS,previous.SCOPES


def contract():
    c = previous.contract()
    c.pop("optimizer")
    c.update(source_run=SOURCE_RUN,source_protocol=previous.EXPERIMENT_PROTOCOL,
        objective="mean_pair_head_standardized_physical_residual_squared_error_plus_lambda_coefficient_Frobenius_squared_over2",
        representation="same198_linear_causal_features_and396_coefficients_solved_by_training_pair_SVD",
        ridge={"gammas":list(learning.GAMMAS),"levels":list(learning.LEVELS),
            "lambda":"gamma_times_training_Gram_trace_divided_by198","solver":"SVD","dtype":"float64","steps":0},
        diagnostics="training_only_numerical_rank_participation_effective_rank_95percent_energy_rank_effective_df_per_head_and_cost_wait_errors",
        selection="pool_physical_means_then_one_positive_training_RMS_per_source_root_MSE_or_worst_root_axis_score",
        ties="zero_residual_then_stronger_ridge_on_exact_ties_hold_raw_if_best_joint_score_ge0",
        final="two60_state_four_draw_analytic_paths_use_inner_selected_regularization",
        variants=list(learning.VARIANTS),controls="all19_frozen_Stage228_controllers_plus_ridge_zero_strong_weak_MSE_joint_both_anchors",
        budget={"fit_tasks":32,"report_tasks":1,"clean_inner_reference_fits":640,"analytic_cost_fits":640,"glm_fits":640,
            "ridge_inner_paths":1280,"ridge_final_paths":64,"residual_SVDs":1344,"positive_ridge_solutions":8064,
            "raw_wait_updates":384000,"residual_optimizer_updates":0,"network_updates":384000,
            "reused_linear_paths":64,"reused_nonlinear_paths":64,"new_native_episodes":0,"rl_actor_updates":0},
        compute_boundary="same_data_features_targets_and_internal_validation_analytic_solver_not_equal_optimizer_compute_or_grid_size",
        storage="all_models_panels_remote_only_pull_aggregate_rank_and_error_report")
    return c


def result_path(run_name,*key): return previous.result_path(run_name,*key)


def records(rows,panels,indices,alternate,prediction):
    output = training.records(rows,panels,indices,alternate,prediction)
    values = joint.physical_values(rows,panels,indices)
    for i,r in enumerate(output):
        r["ridge_head_errors"] = {}
        for name,p in prediction["advantages"].items():
            if not name.startswith("ridge_"): continue
            anchor = p["anchor"][i]
            errors = ((p["delta"][i]-(values[i]-values[i,anchor]))/p["scales"])**2
            r["ridge_head_errors"][name] = errors[np.arange(6)!=anchor].mean(axis=0).tolist()
    return output


def fit_cell(train,held,panels,frozen_linear,nonlinear,bundle,calibration,reused_oof,partition,direction):
    a,b = PARTITIONS[partition][direction],PARTITIONS[partition][1-direction]
    selection,inner,reproduction = learning.collect(train,panels,a,reused_oof)
    model,final = joint.fit_final(train,panels,a,bundle,selection,path_fitter=learning.fit_path)
    predictions = {n:learning.predict(rs,model,frozen_linear,nonlinear,bundle,calibration) for n,rs in (("train",train),("held",held))}
    groups = lambda rs:sorted(map(list,{(r["root"],r["scene_seed"]) for r in rs}))
    cell = {"partition":partition,"direction":direction,"training_draws":list(a),"evaluation_draws":list(b),
        "training_groups":groups(train),"held_groups":groups(held),"selection":selection,"inner_fits":inner,"final_fits":final,
        "inner_reference_reproduction":reproduction,"scopes":{s:records(train if s.startswith("train") else held,panels,
            a if s.endswith("same") else b,b if s.endswith("same") else a,predictions["train" if s.startswith("train") else "held"]) for s in SCOPES}}
    return cell,{"paths":model["paths"],"selection":selection}


def qualify(result,key,views):
    train,held = training.cohort.split_rows(views[key[0]],key[1])
    groups = lambda rs:{(r["root"],r["scene_seed"]) for r in rs}
    c = result["cell"]
    a,b = PARTITIONS[key[2]][key[3]],PARTITIONS[key[2]][1-key[3]]
    if (result["protocol"]!=EXPERIMENT_PROTOCOL or result["contract"]!=contract() or result["key"]!=list(key)
            or c["partition"]!=key[2] or c["direction"]!=key[3] or c["training_draws"]!=list(a) or c["evaluation_draws"]!=list(b)
            or set(map(tuple,c["training_groups"]))!=groups(train) or set(map(tuple,c["held_groups"]))!=groups(held)):
        raise ValueError("Changed ridge paired outer protocol")
    def fitted(fit,gs,n,kind):
        if (fit["anchor"]!=kind or fit["training_states"]!=n or set(map(tuple,fit["training_groups"]))!=gs
                or fit["parameter_count"]!=learning.linear.PARAMETERS or fit["steps"]!=0 or fit["solver"]!="training_pair_SVD_ridge"
                or [x["level"] for x in fit["loss_ledger"]]!=list(learning.LEVELS)
                or fit["spectrum"]["paired_rows"]!=n*5 or fit["spectrum"]["feature_count"]!=198):
            raise ValueError("Changed ridge capacity, solve grid or training rank inputs")
    ledger = c["inner_fits"]
    if len(ledger)!=20 or {(tuple(f["held_group"]),f["direction"]) for f in ledger}!={(g,d) for g in groups(train) for d in (0,1)}:
        raise ValueError("Missing clean ridge prefix/future fit")
    for f in ledger:
        expected,d = groups(train)-{tuple(f["held_group"])},f["direction"]
        ref = f["reference"]
        if (f["training_draws"]!=list(a[2*d:2*d+2]) or f["validation_draws"]!=list(a[2*(1-d):2*(1-d)+2])
                or f["held_states"]!=6 or ref["draws"]!=f["training_draws"] or ref["training_states"]!=54
                or ref["raw_steps"]!=600 or ref["raw_parameters"]!=3393
                or set(map(tuple,ref["training_groups"]))!=expected or set(map(tuple,ref["wait_training_groups"]))!=expected):
            raise ValueError("Ridge raw reference leaked prefix/future labels or changed budget")
        for kind in joint.ANCHORS: fitted(f["fits"][kind],expected,54,kind)
    for kind in joint.ANCHORS:
        fitted(c["final_fits"][kind],groups(train),60,kind)
        x = c["selection"][kind]
        if x["mse_level"] not in learning.LEVELS or x["joint_level"] not in (*learning.LEVELS,None) or set(x["curves"])!=set(learning.LEVELS):
            raise ValueError("Changed ridge internal selection grid")
        if x["joint_level"] is not None:
            curve = x["curves"][x["joint_level"]]
            if curve["joint_score"]>=0 or any(y[k]>=0 for y in curve["by_root"].values() for k in ("mean_delta_cost","mean_delta_wait_min")):
                raise ValueError("Ridge selection failed pooled physical joint improvement")
    if c["inner_reference_reproduction"]["max_abs_difference"]>1e-12 or c["inner_reference_reproduction"]["changed_choices"]:
        raise ValueError("Clean ridge inner raw controls changed")
    if set(c["scopes"])!=set(SCOPES): raise ValueError("Missing ridge scope")
    for scope,rows in c["scopes"].items():
        expected = train if scope.startswith("train") else held
        if (len(rows)!=60 or {joint.binding.learned.state_key(r) for r in rows}!={joint.binding.learned.state_key(r) for r in expected}
                or any(set(r["decisions"])!=set(learning.VARIANTS) or set(r["gates"])!=set(joint.CONTROLS) for r in rows)):
            raise ValueError("Dropped a ridge state or controller")


def metrics(rows):
    output = training.metrics(rows,variants=learning.VARIANTS)
    output["ridge_head_errors"] = {}
    for name in learning.VARIANTS:
        if not name.startswith("ridge_"): continue
        pieces = [r["ridge_head_errors"][name] for r in rows if name in r["ridge_head_errors"]]
        output["ridge_head_errors"][name] = {"cases":len(pieces),"cost_mse":float(np.mean(pieces,axis=0)[0]) if pieces else None,
            "wait_mse":float(np.mean(pieces,axis=0)[1]) if pieces else None}
    return output


def summarize(results,views):
    if set(results)!=set(KEYS): raise ValueError("Complete all32 ridge paired fits")
    for key,r in results.items(): qualify(r,key,views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for key,r in results.items() if key[0]==view]
        scopes = {s:[r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        output[view] = {"scopes":{s:metrics(rs) for s,rs in scopes.items()},
            "held_other_by_root":{str(root):metrics([r for r in scopes["held_other"] if r["root"]==root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction":{f"{p}/{d}":metrics([r for c in cells if c["partition"]==p and c["direction"]==d for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0,1)},
            "selection":{a:[c["selection"][a] for c in cells] for a in joint.ANCHORS},
            "final_training_losses":{a:[c["final_fits"][a]["loss_ledger"] for c in cells] for a in joint.ANCHORS},
            "spectrum":{a:{"inner":[f["fits"][a]["spectrum"] for c in cells for f in c["inner_fits"]],
                "final":[c["final_fits"][a]["spectrum"] for c in cells]} for a in joint.ANCHORS}}
        if any(x["prediction_cases"]!=480 or x["unique_states"]!=120 for x in output[view]["scopes"].values()):
            raise ValueError("Changed complete ridge state coverage")
    return {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"coverage_qualified":True,"views":output,
        "inner_reference_max_abs_difference":max(r["cell"]["inner_reference_reproduction"]["max_abs_difference"] for r in results.values()),
        **contract()["budget"],"native_deployment_eligible":False}


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
        raise ValueError("Complete Stage228 before explicit ridge shrinkage")
    bank,_,registered,_ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"],registered)
    if args.operation=="fit":
        key = args.view,args.fold,args.partition,args.direction
        source = raw_directory(previous.result_path(SOURCE_RUN,*key))
        previous.qualify(json.loads((source/"result.json").read_text()),key,views)
        frozen_linear = json.loads((source/"models.json").read_text())
        fixed = json.loads((raw_directory(previous.previous.result_path(previous.SOURCE_RUN,*key))/"selection.json").read_text())
        nonlinear = json.loads((raw_directory(training.result_path(previous.previous.SOURCE_RUN,*key))/"models.json").read_text())
        nonlinear["selection"] = fixed
        base = json.loads((raw_directory(training.binding_spec.result_path(training.MODEL_RUN,*key))/"models.json").read_text())
        oof = json.loads((raw_directory(training.precision_spec.result_path(training.binding_spec.SOURCE_RUN,*key))/"inner_oof.json").read_text())["c2v4"]
        train,held = training.cohort.split_rows(views[args.view],args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell,model = fit_cell(train,held,panels,frozen_linear,nonlinear,base["bundle"],base["calibration"],
            {k:np.asarray(v) for k,v in oof.items()},args.partition,args.direction)
        result = {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"key":list(key),"cell":cell}
        qualify(result,key,views)
        output = result_path(args.run_name,*key)
        write_json(raw_directory(output)/"models.json",model)
        write_json(raw_directory(output)/"result.json",result)
        write_json(output,{k:v for k,v in result.items() if k!="cell"})
        print("NATIVE_RIDGE_JOINT_FIT_COMPLETE",flush=True)
    else:
        result = summarize({k:json.loads((raw_directory(result_path(args.run_name,*k))/"result.json").read_text()) for k in KEYS},views)
        result["control_reproduction"] = previous.reproduce_controls(result,prior,names=learning.linear.VARIANTS)
        write_json(ROOT/"results"/args.run_name/"report/result.json",result)
        print("NATIVE_RIDGE_JOINT_REPORT_COMPLETE",flush=True)


if __name__=="__main__": main()
