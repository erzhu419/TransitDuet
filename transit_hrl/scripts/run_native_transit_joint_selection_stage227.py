#!/usr/bin/env python3
"""Repair physical checkpoint aggregation without repeating joint training."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_native_transit_joint_advantage_stage226 as previous
from freq_hrl.domains.transit import native_joint_selection as selection
from freq_hrl.experiments.pointmaze_root_response import raw_directory,write_json

EXPERIMENT_PROTOCOL = "native_transit_joint_selection_stage227_v1"
SOURCE_RUN = "native_transit_joint_advantage_stage226_development_20261014_r1"
VIEWS,PARTITIONS,KEYS,SCOPES = previous.VIEWS,previous.PARTITIONS,previous.KEYS,previous.SCOPES
learning = selection.learning


def contract():
    parent = previous.contract()
    return {"source_run":SOURCE_RUN,"source_protocol":previous.EXPERIMENT_PROTOCOL,
        "cohort":parent["cohort"],"outer":parent["outer"],"partitions":parent["partitions"],
        "selection":"pool_physical_cost_wait_means_first_then_divide_by_one_positive_training_RMS_per_source_root",
        "scale":"held_case_weighted_mean_of_recorded_training_only_fold_RMS_fixed_across_checkpoints",
        "paths":"reuse_all64_final_paths_all_snapshots_weights_unchanged_MSE_selection_unchanged",
        "controls":"GLM_raw_c2v4_and_original_raw_zero_last_MSE_unchanged",
        "joint_rule":"same_worst_root_component_score_earlier_ties_keep_raw_if_score_ge0",
        "variants":list(learning.VARIANTS),"checkpoints":list(learning.CHECKPOINTS),
        "budget":{"reselection_tasks":32,"report_tasks":1,"reused_joint_paths":64,"network_updates":0,
            "new_native_episodes":0,"rl_actor_updates":0},
        "boundary":"posthoc_development_protocol_correction_same_states_futures_not_independent_confirmation",
        "storage":"source_weights_remote_save_new_selection_only_pull_aggregate_report"}


def result_path(run_name,*key): return previous.result_path(run_name,*key)


def reselect_cell(train,held,panels,source_cell,model,bundle,calibration,partition,direction):
    chosen = {a:selection.repair_selection(source_cell,a) for a in learning.ANCHORS}
    model = {**model,"selection":chosen}
    a,b = PARTITIONS[partition][direction],PARTITIONS[partition][1-direction]
    predictions = {name:learning.predict(rows,model,bundle,calibration) for name,rows in (("train",train),("held",held))}
    groups = lambda rs:sorted(map(list,{(r["root"],r["scene_seed"]) for r in rs}))
    return {"partition":partition,"direction":direction,"training_draws":list(a),"evaluation_draws":list(b),
        "training_groups":groups(train),"held_groups":groups(held),"selection":chosen,
        "source_selection":{a:{k:source_cell["selection"][a][k] for k in ("mse_step","joint_step")} for a in learning.ANCHORS},
        "scopes":{s:previous.records(train if s.startswith("train") else held,panels,a if s.endswith("same") else b,
            b if s.endswith("same") else a,predictions["train" if s.startswith("train") else "held"]) for s in SCOPES}}


def qualify(result,key,views):
    train,held = previous.cohort.split_rows(views[key[0]],key[1])
    groups = lambda rs:{(r["root"],r["scene_seed"]) for r in rs}
    c = result["cell"]
    if (result["protocol"]!=EXPERIMENT_PROTOCOL or result["contract"]!=contract() or result["key"]!=list(key)
            or c["partition"]!=key[2] or c["direction"]!=key[3]
            or c["training_draws"]!=list(PARTITIONS[key[2]][key[3]]) or c["evaluation_draws"]!=list(PARTITIONS[key[2]][1-key[3]])
            or set(map(tuple,c["training_groups"]))!=groups(train) or set(map(tuple,c["held_groups"]))!=groups(held)):
        raise ValueError("Changed physical reselection inputs")
    for a,x in c["selection"].items():
        if x["mse_step"]!=c["source_selection"][a]["mse_step"]: raise ValueError("MSE control changed")
        if x["joint_step"] is not None:
            selected = x["curves"][str(x["joint_step"])]
            if selected["joint_score"]>=0 or any(y[k]>=0 for y in selected["by_root"].values() for k in ("mean_delta_cost","mean_delta_wait_min")):
                raise ValueError("Selected checkpoint fails physical joint improvement")
    if set(c["scopes"])!=set(SCOPES): raise ValueError("Missing reselection scope")
    for scope,rows in c["scopes"].items():
        expected = train if scope.startswith("train") else held
        if len(rows)!=60 or {learning.binding.learned.state_key(r) for r in rows}!={learning.binding.learned.state_key(r) for r in expected}:
            raise ValueError("Changed complete-state coverage")


def summarize(results,views):
    if set(results)!=set(KEYS): raise ValueError("Complete all32 physical reselections")
    for key,r in results.items(): qualify(r,key,views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for key,r in results.items() if key[0]==view]
        scopes = {s:[r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        output[view] = {"scopes":{s:previous.metrics(rs) for s,rs in scopes.items()},
            "held_other_by_root":{str(root):previous.metrics([r for r in scopes["held_other"] if r["root"]==root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction":{f"{p}/{d}":previous.metrics([r for c in cells if c["partition"]==p and c["direction"]==d for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0,1)},
            "selection":{a:[c["selection"][a] for c in cells] for a in learning.ANCHORS},
            "changed_checkpoints":{a:sum(c["selection"][a]["joint_step"]!=c["source_selection"][a]["joint_step"] for c in cells) for a in learning.ANCHORS}}
    return {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"coverage_qualified":True,"views":output,
        **contract()["budget"],"native_deployment_eligible":False}


def reproduce_controls(result,prior):
    names = [v for v in learning.VARIANTS if not v.endswith("selected")]
    differences = []
    for view in VIEWS:
        for scope in SCOPES:
            a,b = result["views"][view]["scopes"][scope],prior["views"][view]["scopes"][scope]
            for v in names:
                for m in ("mean_cost","mean_wait_min","changed_fraction","unsafe_fraction"):
                    differences.append(abs(a["variants"][v][m]-b["variants"][v][m]))
                if v in learning.CONTROLS:
                    differences.append(abs(a["wait_heads"][v]["physical_mass_loss_min"]-b["wait_heads"][v]["physical_mass_loss_min"]))
                else:
                    differences.append(abs(a["joint_prediction_errors"][v]["normalized_mse"]-b["joint_prediction_errors"][v]["normalized_mse"]))
    if max(differences)>1e-12: raise ValueError("An unchanged source controller or training path changed")
    return {"comparisons":len(differences),"max_abs_difference":max(differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",required=True)
    parser.add_argument("--operation",choices=("reselect","report"),required=True)
    parser.add_argument("--view",choices=VIEWS)
    parser.add_argument("--fold",type=int,choices=(0,1))
    parser.add_argument("--partition",choices=PARTITIONS)
    parser.add_argument("--direction",type=int,choices=(0,1))
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"]!=previous.EXPERIMENT_PROTOCOL or prior["contract"]!=previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage226 before physical reselection")
    bank,_,registered,_ = previous.cohort.source.source.load_data()
    views = previous.cohort.source.matched_views(bank["training_data"],registered)
    if args.operation=="reselect":
        key = args.view,args.fold,args.partition,args.direction
        source = raw_directory(previous.result_path(SOURCE_RUN,*key))
        old = json.loads((source/"result.json").read_text())
        previous.qualify(old,key,views)
        models = json.loads((source/"models.json").read_text())
        base_path = raw_directory(previous.binding_spec.result_path(previous.MODEL_RUN,*key))/"models.json"
        base = json.loads(base_path.read_text())
        train,held = previous.cohort.split_rows(views[args.view],args.fold)
        panels = previous.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell = reselect_cell(train,held,panels,old["cell"],models,base["bundle"],base["calibration"],args.partition,args.direction)
        result = {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"key":list(key),"cell":cell}
        qualify(result,key,views)
        output = result_path(args.run_name,*key)
        write_json(raw_directory(output)/"result.json",result)
        write_json(raw_directory(output)/"selection.json",cell["selection"])
        write_json(output,{k:v for k,v in result.items() if k!="cell"})
        print("NATIVE_JOINT_SELECTION_RESELECT_COMPLETE",flush=True)
    else:
        result = summarize({k:json.loads((raw_directory(result_path(args.run_name,*k))/"result.json").read_text()) for k in KEYS},views)
        result["control_reproduction"] = reproduce_controls(result,prior)
        write_json(ROOT/"results"/args.run_name/"report/result.json",result)
        print("NATIVE_JOINT_SELECTION_REPORT_COMPLETE",flush=True)


if __name__=="__main__": main()
