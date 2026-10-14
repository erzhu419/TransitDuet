#!/usr/bin/env python3
"""Select cost rank with clean raw-gate prefix/future crossvalidation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_native_transit_cost_gate_stage232 as previous
from freq_hrl.domains.transit import native_cost_rank_selection as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory,write_json

joint,training = learning.joint,previous.training
EXPERIMENT_PROTOCOL = "native_transit_cost_rank_stage233_v1"
SOURCE_RUN = "native_transit_cost_gate_stage232_development_20261014_r1"
MODEL_RUN,REFERENCE_RUN = previous.MODEL_RUN,previous.REFERENCE_RUN
VIEWS,PARTITIONS,KEYS,SCOPES,FOCUS_KEY = previous.VIEWS,previous.PARTITIONS,previous.KEYS,previous.SCOPES,previous.FOCUS_KEY


def contract():
    parent = previous.contract()
    return {"source_run":SOURCE_RUN,"source_protocol":previous.EXPERIMENT_PROTOCOL,"model_run":MODEL_RUN,"reference_run":REFERENCE_RUN,
        "cohort":parent["cohort"],"outer":parent["outer"],"partitions":parent["partitions"],
        "inner":"20_clean54_state2_draw_reference_refits_and_two_anchor_TSVD_paths_validate6_unseen_prefix_states_other2_draws_inside_outer_training4",
        "objective":"unchanged_TSVD_cost_column_raw_logits_menu_wait_residual_unused_cost_MSE_for_prediction_diagnostic",
        "normalization":"pool_physical_root_means_then_divide_own_training_RMS_same_Stage229_rule",
        "selection":"cost_MSE_or_strict_cost_wait_improvement_each_root_or_same_plus_nonincreased_wait_worsening_fraction_each_root",
        "ties":"lower_rank_exact_ties_raw_hold_if_no_strict_cost_wait_improvement_zero_exactly_raw",
        "ranks":list(learning.lowrank.RANKS),"anchors":list(joint.ANCHORS),"variants":list(learning.VARIANTS),
        "final":"frozen_Stage230_outer_paths_and_Stage226_references_only_inner_selector_changes",
        "controls":"all15_Stage232_raw_and_raw_gate_rank_MSE_inherited_selection_controllers_exact_choices_and_scores",
        "focus_key":list(FOCUS_KEY),"budget":{"fit_tasks":32,"report_tasks":1,"clean_inner_reference_fits":640,
            "analytic_cost_fits":640,"glm_fits":640,"raw_wait_updates":384000,"network_updates":384000,
            "residual_SVDs":1280,"positive_rank_solutions":5120,"residual_optimizer_updates":0,
            "reused_outer_paths":64,"reused_outer_references":32,"new_native_episodes":0,"rl_actor_updates":0},
        "boundary":"development_reused240_states8_draws_not_new_seed_CI_or_end_to_end_HRL",
        "storage":"raw_models_panels_stay_remote_write_only_new_selection_and_result_pull_aggregate"}


def result_path(run_name,*key): return previous.result_path(run_name,*key)


def fit_cell(train,held,panels,model,reference,source,reused_oof,partition,direction):
    a,b = PARTITIONS[partition][direction],PARTITIONS[partition][1-direction]
    selection,inner,reproduction = learning.collect(train,panels,a,reused_oof)
    predictions = {name:learning.predict(joint.causal_data(rows,reference),model,selection) for name,rows in (("train",train),("held",held))}
    scopes,checks = {},[]
    for scope in SCOPES:
        rows = train if scope.startswith("train") else held
        p = predictions["train" if scope.startswith("train") else "held"]
        by_key = {joint.binding.learned.state_key(r):r for r in source["scopes"][scope]}
        old = [by_key[joint.binding.learned.state_key(r)] for r in rows]
        scopes[scope],check = learning.records(rows,joint.physical_values(rows,panels,a if scope.endswith("same") else b),p,old)
        checks.append(check)
    groups = lambda rs:sorted(map(list,{(r["root"],r["scene_seed"]) for r in rs}))
    return {"partition":partition,"direction":direction,"training_draws":list(a),"evaluation_draws":list(b),
        "training_groups":groups(train),"held_groups":groups(held),"selection":selection,"inner_fits":inner,
        "inner_reference_reproduction":reproduction,"control_reproduction":{
            **{k:sum(c[k] for c in checks) for k in ("choice_comparisons","changed_choices","score_comparisons")},
            "max_abs_difference":max(c["max_abs_difference"] for c in checks)},"scopes":scopes}


def qualify(result,key,views):
    c = result["cell"]
    train,held = training.cohort.split_rows(views[key[0]],key[1])
    groups = lambda rs:{(r["root"],r["scene_seed"]) for r in rs}
    a,b = PARTITIONS[key[2]][key[3]],PARTITIONS[key[2]][1-key[3]]
    if (result["protocol"]!=EXPERIMENT_PROTOCOL or result["contract"]!=contract() or result["key"]!=list(key)
            or c["partition"]!=key[2] or c["direction"]!=key[3] or c["training_draws"]!=list(a) or c["evaluation_draws"]!=list(b)
            or set(map(tuple,c["training_groups"]))!=groups(train) or set(map(tuple,c["held_groups"]))!=groups(held)):
        raise ValueError("Changed cost-rank outer protocol")
    ledger = c["inner_fits"]
    if len(ledger)!=20 or {(tuple(f["held_group"]),f["direction"]) for f in ledger}!={(g,d) for g in groups(train) for d in (0,1)}:
        raise ValueError("Incomplete cost-rank inner prefix/future fits")
    for f in ledger:
        expected,d = groups(train)-{tuple(f["held_group"])},f["direction"]
        ref = f["reference"]
        if (f["training_draws"]!=list(a[2*d:2*d+2]) or f["validation_draws"]!=list(a[2*(1-d):2*(1-d)+2]) or f["held_states"]!=6
                or ref["draws"]!=f["training_draws"] or ref["training_states"]!=54 or ref["raw_steps"]!=600 or ref["raw_parameters"]!=3393
                or set(map(tuple,ref["training_groups"]))!=expected or set(map(tuple,ref["wait_training_groups"]))!=expected):
            raise ValueError("Cost-rank reference leaked prefix/future labels or changed budget")
        for kind in joint.ANCHORS:
            fit = f["fits"][kind]
            if (fit["anchor"]!=kind or fit["training_states"]!=54 or set(map(tuple,fit["training_groups"]))!=expected
                    or fit["steps"]!=0 or fit["solver"]!="training_pair_TSVD" or fit["paired_rows"]!=270 or fit["feature_count"]!=198
                    or [x["level"] for x in fit["arms"]["joint"]["loss_ledger"]]!=list(learning.lowrank.LEVELS)):
                raise ValueError("Changed cost-rank residual solver")
    for kind in joint.ANCHORS:
        s = c["selection"][kind]
        if s["mse_level"] not in learning.lowrank.LEVELS or set(s["curves"])!=set(learning.lowrank.LEVELS): raise ValueError("Changed cost-MSE grid")
        for selector in ("mean_level","safe_level"):
            level = s[selector]
            if level not in (*learning.lowrank.LEVELS,None): raise ValueError("Changed cost-rank selection grid")
            if level is not None:
                x = s["curves"][level]
                if x["joint_score"]>=0 or (selector=="safe_level" and any(r["mean_delta_unsafe"]>0 for r in x["by_root"].values())):
                    raise ValueError("Cost-rank selection violates its inner constraints")
    if c["inner_reference_reproduction"]["max_abs_difference"]>1e-12 or c["inner_reference_reproduction"]["changed_choices"]:
        raise ValueError("Clean inner raw controls changed")
    r = c["control_reproduction"]
    if r["choice_comparisons"]!=3600 or r["changed_choices"] or r["score_comparisons"]!=7200 or r["max_abs_difference"]>1e-12:
        raise ValueError("Frozen outer raw-gate controls incomplete")
    if set(c["scopes"])!=set(SCOPES): raise ValueError("Missing cost-rank scope")
    for scope,rows in c["scopes"].items():
        expected = train if scope.startswith("train") else held
        if len(rows)!=60 or {joint.binding.learned.state_key(r) for r in rows}!={joint.binding.learned.state_key(r) for r in expected}:
            raise ValueError("Dropped cost-rank state")
        for row in rows:
            if set(row["decisions"])!=set(learning.VARIANTS): raise ValueError("Dropped cost-rank controller")
            for kind in joint.ANCHORS:
                s = c["selection"][kind]
                levels = dict(zip(learning.NEW_MODES,(s["mse_level"],s["mean_level"],s["safe_level"])))
                if row["levels"][kind]!=levels: raise ValueError("Selected cost rank changed at inference")
                for mode,level in levels.items():
                    if level in (None,"zero") and row["decisions"][f"cost_rank_{kind}_{mode}"]!=row["decisions"]["raw"]:
                        raise ValueError("Zero or hold control no longer raw")


def metrics(rows):
    return {"prediction_cases":len(rows),"unique_states":len({joint.binding.learned.state_key(r) for r in rows}),
        "variants":{v:{"mean_cost":float(np.mean([r["decisions"][v]["cost"] for r in rows])),
            "mean_wait_min":float(np.mean([r["decisions"][v]["wait"] for r in rows])),
            "changed_fraction":float(np.mean([r["decisions"][v]["changed"] for r in rows])),
            "unsafe_fraction":float(np.mean([r["decisions"][v]["unsafe"] for r in rows])),
            "switch_fraction_to_raw":float(np.mean([r["decisions"][v]["selected"]!=r["decisions"]["raw"]["selected"] for r in rows]))} for v in learning.VARIANTS},
        "selection_counts":{a:{m:{str(l):sum(r["levels"][a][m]==l for r in rows) for l in sorted({r["levels"][a][m] for r in rows},key=str)}
            for m in learning.NEW_MODES} for a in joint.ANCHORS}}


def summarize(results,views):
    if set(results)!=set(KEYS): raise ValueError("Complete all32 cost-rank fits")
    for key,r in results.items(): qualify(r,key,views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for key,r in results.items() if key[0]==view]
        scopes = {s:[r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        output[view] = {"scopes":{s:metrics(rows) for s,rows in scopes.items()},
            "held_other_by_root":{str(root):metrics([r for r in scopes["held_other"] if r["root"]==root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction":{f"{p}/{d}":metrics([r for c in cells if c["partition"]==p and c["direction"]==d for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0,1)},
            "selection":{a:[c["selection"][a] for c in cells] for a in joint.ANCHORS}}
        if any(x["prediction_cases"]!=480 or x["unique_states"]!=120 for x in output[view]["scopes"].values()): raise ValueError("Changed cost-rank coverage")
    focus = results[FOCUS_KEY]["cell"]
    return {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"coverage_qualified":True,"views":output,
        "control_reproduction":{**{k:sum(r["cell"]["control_reproduction"][k] for r in results.values()) for k in ("choice_comparisons","changed_choices","score_comparisons")},
            "max_abs_difference":max(r["cell"]["control_reproduction"]["max_abs_difference"] for r in results.values())},
        "inner_reference_max_abs_difference":max(r["cell"]["inner_reference_reproduction"]["max_abs_difference"] for r in results.values()),
        "focus":{"key":list(FOCUS_KEY),"selection":focus["selection"],"scopes":{s:metrics(rows) for s,rows in focus["scopes"].items()},
            "held_other_by_root":{str(root):metrics([r for r in focus["scopes"]["held_other"] if r["root"]==root]) for root in (397,401)}},
        **contract()["budget"],"native_deployment_eligible":False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name",required=True)
    p.add_argument("--operation",choices=("fit","report"),required=True)
    p.add_argument("--view",choices=VIEWS)
    p.add_argument("--fold",type=int,choices=(0,1))
    p.add_argument("--partition",choices=PARTITIONS)
    p.add_argument("--direction",type=int,choices=(0,1))
    args = p.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"]!=previous.EXPERIMENT_PROTOCOL or prior["contract"]!=previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage232 before raw-gate rank selection")
    bank,_,registered,_ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"],registered)
    if args.operation=="fit":
        key = args.view,args.fold,args.partition,args.direction
        old = json.loads((raw_directory(previous.result_path(SOURCE_RUN,*key))/"result.json").read_text())
        previous.qualify(old,key,views)
        model = json.loads((raw_directory(previous.result_path(MODEL_RUN,*key))/"models.json").read_text())
        reference = json.loads((raw_directory(previous.result_path(REFERENCE_RUN,*key))/"models.json").read_text())["reference"]
        oof_path = raw_directory(training.precision_spec.result_path(training.binding_spec.SOURCE_RUN,*key))/"inner_oof.json"
        oof = {k:np.asarray(v) for k,v in json.loads(oof_path.read_text())["c2v4"].items()}
        train,held = training.cohort.split_rows(views[args.view],args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell = fit_cell(train,held,panels,model,reference,old["cell"],oof,args.partition,args.direction)
        result = {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"key":list(key),"cell":cell}
        qualify(result,key,views)
        output = result_path(args.run_name,*key)
        write_json(raw_directory(output)/"selection.json",cell["selection"])
        write_json(raw_directory(output)/"result.json",result)
        write_json(output,{k:v for k,v in result.items() if k!="cell"})
        print("NATIVE_COST_RANK_FIT_COMPLETE",flush=True)
    else:
        report = summarize({k:json.loads((raw_directory(result_path(args.run_name,*k))/"result.json").read_text()) for k in KEYS},views)
        write_json(ROOT/"results"/args.run_name/"report/result.json",report)
        print("NATIVE_COST_RANK_REPORT_COMPLETE",flush=True)


if __name__=="__main__": main()
