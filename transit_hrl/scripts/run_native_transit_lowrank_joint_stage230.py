#!/usr/bin/env python3
"""Separate cost refitting from weak-mode fitting using matched low-rank arms."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_native_transit_ridge_joint_stage229 as previous
from freq_hrl.domains.transit import native_lowrank_joint as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory,write_json

joint,training = learning.joint,previous.training
EXPERIMENT_PROTOCOL = "native_transit_lowrank_joint_stage230_v1"
SOURCE_RUN = "native_transit_ridge_joint_stage229_development_20261014_r1"
VIEWS,PARTITIONS,KEYS,SCOPES = previous.VIEWS,previous.PARTITIONS,previous.KEYS,previous.SCOPES


def contract():
    c = previous.contract()
    c.pop("ridge")
    c.pop("parameters_per_branch")
    c.update(source_run=SOURCE_RUN,source_protocol=previous.EXPERIMENT_PROTOCOL,
        objective="standardized_physical_residual_least_squares_in_training_only_top_singular_subspace",
        representation="same198_causal_features_shared_TSVD_wait_coefficients_identical_across_frozen_cost_and_dual_head_arms",
        lowrank={"ranks":list(learning.RANKS),"levels":list(learning.LEVELS),"arms":list(learning.ARMS),
            "coefficient_parameters":{"wait":198,"joint":396},"degrees_of_freedom":"wait_k_or_joint_2k_per_training_design","dtype":"float64","steps":0},
        diagnostics="training_and_validation_cost_wait_errors_at_each_rank_and_held_prefix_future_error_and_control_performance",
        ties="zero_residual_then_lower_rank_on_exact_ties_hold_raw_if_best_joint_score_ge0",
        final="two60_state_four_draw_shared_TSVD_paths_each_with_two_matched_arms_use_inner_selected_rank",
        variants=list(learning.VARIANTS),controls="all29_frozen_Stage229_controllers_plus_each_rank_MSE_joint_for_two_anchors_and_two_arms",
        budget={"fit_tasks":32,"report_tasks":1,"clean_inner_reference_fits":640,"analytic_cost_fits":640,"glm_fits":640,
            "lowrank_inner_paths":1280,"lowrank_final_paths":64,"residual_SVDs":1344,"positive_rank_solutions":5376,
            "matched_arm_paths":2688,"raw_wait_updates":384000,"residual_optimizer_updates":0,"network_updates":384000,
            "reused_ridge_paths":64,"reused_linear_paths":64,"reused_nonlinear_paths":64,"new_native_episodes":0,"rl_actor_updates":0},
        compute_boundary="both_arms_share_SVD_and_exact_wait_coefficients_only_cost_residual_differs_not_equal_grid_to_frozen_controls",
        storage="models_and_panels_remote_only_pull_aggregate_report")
    return c


def result_path(run_name,*key): return previous.result_path(run_name,*key)


def records(rows,panels,indices,alternate,prediction):
    output = previous.records(rows,panels,indices,alternate,prediction)
    values = joint.physical_values(rows,panels,indices)
    for i,r in enumerate(output):
        r["lowrank_head_errors"] = {}
        for name,p in prediction["advantages"].items():
            if not name.startswith("lowrank_"): continue
            anchor = p["anchor"][i]
            e = ((p["delta"][i]-(values[i]-values[i,anchor]))/p["scales"])**2
            r["lowrank_head_errors"][name] = e[np.arange(6)!=anchor].mean(axis=0).tolist()
    return output


def fit_cell(train,held,panels,frozen_ridge,frozen_linear,nonlinear,bundle,calibration,reused_oof,partition,direction):
    a,b = PARTITIONS[partition][direction],PARTITIONS[partition][1-direction]
    selection,inner,reproduction = learning.collect(train,panels,a,reused_oof)
    model,final = joint.fit_final(train,panels,a,bundle,selection,path_fitter=learning.fit_path)
    predictions = {n:learning.predict(rs,model,frozen_ridge,frozen_linear,nonlinear,bundle,calibration) for n,rs in (("train",train),("held",held))}
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
        raise ValueError("Changed low-rank outer protocol")
    def fitted(fit,gs,n,kind):
        if (fit["anchor"]!=kind or fit["training_states"]!=n or set(map(tuple,fit["training_groups"]))!=gs
                or fit["steps"]!=0 or fit["solver"]!="training_pair_TSVD" or fit["paired_rows"]!=n*5 or fit["feature_count"]!=198
                or set(fit["arms"])!=set(learning.ARMS)):
            raise ValueError("Changed low-rank fitting inputs or shared solver")
        for arm in learning.ARMS:
            ledger = fit["arms"][arm]["loss_ledger"]
            if (fit["arms"][arm]["parameter_count"]!=(198 if arm=="wait" else 396)
                    or [x["level"] for x in ledger]!=list(learning.LEVELS)
                    or any(x["retained_rank"]!=min(k,fit["numerical_rank"]) or x["wait_df"]!=x["retained_rank"]
                        or x["cost_df"]!=(0 if arm=="wait" else x["retained_rank"]) for k,x in zip(learning.RANKS,ledger))):
                raise ValueError("Changed matched arms or rank grid")
    ledger = c["inner_fits"]
    if len(ledger)!=20 or {(tuple(f["held_group"]),f["direction"]) for f in ledger}!={(g,d) for g in groups(train) for d in (0,1)}:
        raise ValueError("Missing clean low-rank prefix/future fit")
    for f in ledger:
        expected,d = groups(train)-{tuple(f["held_group"])},f["direction"]
        ref = f["reference"]
        if (f["training_draws"]!=list(a[2*d:2*d+2]) or f["validation_draws"]!=list(a[2*(1-d):2*(1-d)+2])
                or f["held_states"]!=6 or ref["draws"]!=f["training_draws"] or ref["training_states"]!=54
                or ref["raw_steps"]!=600 or ref["raw_parameters"]!=3393
                or set(map(tuple,ref["training_groups"]))!=expected or set(map(tuple,ref["wait_training_groups"]))!=expected):
            raise ValueError("Low-rank raw reference leaked prefix/future labels or changed budget")
        for kind in joint.ANCHORS: fitted(f["fits"][kind],expected,54,kind)
    for kind in joint.ANCHORS:
        fitted(c["final_fits"][kind],groups(train),60,kind)
        if set(c["selection"][kind])!=set(learning.ARMS): raise ValueError("Missing low-rank arm selection")
        for x in c["selection"][kind].values():
            if x["mse_level"] not in learning.LEVELS or x["joint_level"] not in (*learning.LEVELS,None) or set(x["curves"])!=set(learning.LEVELS):
                raise ValueError("Changed low-rank internal selection grid")
            if x["joint_level"] is not None:
                curve = x["curves"][x["joint_level"]]
                if curve["joint_score"]>=0 or any(y[k]>=0 for y in curve["by_root"].values() for k in ("mean_delta_cost","mean_delta_wait_min")):
                    raise ValueError("Low-rank selection failed pooled physical joint improvement")
    if c["inner_reference_reproduction"]["max_abs_difference"]>1e-12 or c["inner_reference_reproduction"]["changed_choices"]:
        raise ValueError("Clean low-rank inner raw controls changed")
    if set(c["scopes"])!=set(SCOPES): raise ValueError("Missing low-rank scope")
    for scope,rows in c["scopes"].items():
        expected = train if scope.startswith("train") else held
        if (len(rows)!=60 or {joint.binding.learned.state_key(r) for r in rows}!={joint.binding.learned.state_key(r) for r in expected}
                or any(set(r["decisions"])!=set(learning.VARIANTS) or set(r["gates"])!=set(joint.CONTROLS) for r in rows)):
            raise ValueError("Dropped a low-rank state or controller")


def metrics(rows):
    output = previous.metrics(rows)
    full = training.metrics(rows,variants=learning.VARIANTS)
    output.update(variants=full["variants"],joint_prediction_errors=full["joint_prediction_errors"])
    output["lowrank_head_errors"] = {}
    for name in learning.VARIANTS:
        if not name.startswith("lowrank_"): continue
        pieces = [r["lowrank_head_errors"][name] for r in rows if name in r["lowrank_head_errors"]]
        output["lowrank_head_errors"][name] = {"cases":len(pieces),"cost_mse":float(np.mean(pieces,axis=0)[0]) if pieces else None,
            "wait_mse":float(np.mean(pieces,axis=0)[1]) if pieces else None}
    return output


def summarize(results,views):
    if set(results)!=set(KEYS): raise ValueError("Complete all32 low-rank fits")
    for key,r in results.items(): qualify(r,key,views)
    output,equivalence = {},[]
    for view in VIEWS:
        cells = [r["cell"] for key,r in results.items() if key[0]==view]
        scopes = {s:[r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        output[view] = {"scopes":{s:metrics(rs) for s,rs in scopes.items()},
            "held_other_by_root":{str(root):metrics([r for r in scopes["held_other"] if r["root"]==root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction":{f"{p}/{d}":metrics([r for c in cells if c["partition"]==p and c["direction"]==d for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0,1)},
            "selection":{a:{arm:[c["selection"][a][arm] for c in cells] for arm in learning.ARMS} for a in joint.ANCHORS},
            "final_training_losses":{a:{arm:[c["final_fits"][a]["arms"][arm]["loss_ledger"] for c in cells] for arm in learning.ARMS} for a in joint.ANCHORS}}
        for m in output[view]["scopes"].values():
            if m["prediction_cases"]!=480 or m["unique_states"]!=120: raise ValueError("Changed complete low-rank state coverage")
            for a in joint.ANCHORS:
                heads = m["lowrank_head_errors"]
                for mode in learning.MODES[:len(learning.LEVELS)]:
                    w,j = heads[f"lowrank_{a}_wait_{mode}"],heads[f"lowrank_{a}_joint_{mode}"]
                    if w["cases"]!=480 or j["cases"]!=480: raise ValueError("Dropped a fixed-rank prediction")
                    equivalence.extend((abs(w["wait_mse"]-j["wait_mse"]),abs(w["cost_mse"]-heads[f"lowrank_{a}_wait_zero"]["cost_mse"])))
    if max(equivalence)>1e-12: raise ValueError("Matched wait predictors or frozen cost changed")
    return {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"coverage_qualified":True,"views":output,
        "matched_arm_reproduction":{"comparisons":len(equivalence),"max_abs_difference":max(equivalence)},
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
        raise ValueError("Complete Stage229 before low-rank comparison")
    bank,_,registered,_ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"],registered)
    if args.operation=="fit":
        key = args.view,args.fold,args.partition,args.direction
        source = raw_directory(previous.result_path(SOURCE_RUN,*key))
        previous.qualify(json.loads((source/"result.json").read_text()),key,views)
        frozen_ridge = json.loads((source/"models.json").read_text())
        frozen_linear = json.loads((raw_directory(previous.previous.result_path(previous.SOURCE_RUN,*key))/"models.json").read_text())
        fixed = json.loads((raw_directory(previous.previous.previous.result_path(previous.previous.SOURCE_RUN,*key))/"selection.json").read_text())
        nonlinear = json.loads((raw_directory(training.result_path(previous.previous.previous.SOURCE_RUN,*key))/"models.json").read_text())
        nonlinear["selection"] = fixed
        base = json.loads((raw_directory(training.binding_spec.result_path(training.MODEL_RUN,*key))/"models.json").read_text())
        oof = json.loads((raw_directory(training.precision_spec.result_path(training.binding_spec.SOURCE_RUN,*key))/"inner_oof.json").read_text())["c2v4"]
        train,held = training.cohort.split_rows(views[args.view],args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell,model = fit_cell(train,held,panels,frozen_ridge,frozen_linear,nonlinear,base["bundle"],base["calibration"],
            {k:np.asarray(v) for k,v in oof.items()},args.partition,args.direction)
        result = {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"key":list(key),"cell":cell}
        qualify(result,key,views)
        output = result_path(args.run_name,*key)
        write_json(raw_directory(output)/"models.json",model)
        write_json(raw_directory(output)/"result.json",result)
        write_json(output,{k:v for k,v in result.items() if k!="cell"})
        print("NATIVE_LOWRANK_JOINT_FIT_COMPLETE",flush=True)
    else:
        result = summarize({k:json.loads((raw_directory(result_path(args.run_name,*k))/"result.json").read_text()) for k in KEYS},views)
        result["control_reproduction"] = previous.previous.reproduce_controls(result,prior,names=learning.ridge.VARIANTS)
        write_json(ROOT/"results"/args.run_name/"report/result.json",result)
        print("NATIVE_LOWRANK_JOINT_REPORT_COMPLETE",flush=True)


if __name__=="__main__": main()
