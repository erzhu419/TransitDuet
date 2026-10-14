#!/usr/bin/env python3
"""Attribute frozen low-rank control changes to ranking and wait feasibility."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_native_transit_lowrank_joint_stage230 as previous
from freq_hrl.domains.transit import native_lowrank_factorial as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory,write_json

joint,training = learning.joint,previous.training
EXPERIMENT_PROTOCOL = "native_transit_factorial_stage231_v1"
SOURCE_RUN = "native_transit_lowrank_joint_stage230_development_20261014_r1"
REFERENCE_RUN = "native_transit_joint_advantage_stage226_development_20261014_r1"
VIEWS,PARTITIONS,KEYS,SCOPES = previous.VIEWS,previous.PARTITIONS,previous.KEYS,previous.SCOPES
FOCUS_KEY = ("mixed_b",1,"even_odd",0)


def contract():
    parent = previous.contract()
    return {"source_run":SOURCE_RUN,"source_protocol":previous.EXPERIMENT_PROTOCOL,"reference_run":REFERENCE_RUN,
        "cohort":parent["cohort"],"outer":parent["outer"],"partitions":parent["partitions"],
        "ranks":list(learning.lowrank.RANKS),"anchors":list(joint.ANCHORS),"arms":list(learning.ARMS),"modes":list(learning.lowrank.MODES),
        "selection":"reuse_joint_MSE_and_joint_selected_rank_exactly_for_all_four_arms_no_cost_only_reselection_raw_hold_for_all_if_None",
        "prediction":"same_frozen_cost_wait_heads_toggle_cost_and_wait_residual_independently_anchor_first_ties",
        "interventions":"base_cost_base_wait__learned_cost_base_wait__base_cost_learned_wait__learned_cost_learned_wait",
        "effects":"cost_only_minus_base_wait_only_minus_base_joint_minus_base_interaction_joint_minus_cost_only_minus_wait_only_plus_base",
        "diagnostics":"physical_cost_wait_switches_added_removed_feasible_candidates_by_source_future_and_predeclared_rank8_focus",
        "focus_key":list(FOCUS_KEY),"controls":"reproduce_frozen_raw_and_all_fixed_rank_base_wait_joint_choices_and_physical_scores_at_each_future_half",
        "budget":{"analysis_tasks":32,"report_tasks":1,"reused_lowrank_paths":64,"reused_raw_references":32,
            "network_updates":0,"residual_SVDs":0,"new_native_episodes":0,"rl_actor_updates":0},
        "boundary":"development_reused_states_futures_predictor_ablation_not_new_CI_or_end_to_end_HRL",
        "storage":"frozen_models_and_panels_remote_only_pull_compact_aggregate_no_new_weights"}


def result_path(run_name,*key): return previous.result_path(run_name,*key)


def analyze_cell(train,held,panels,model,reference,source,partition,direction):
    predictions,reproduction = {},[]
    for name,rows,scope in (("train",train,"train_same"),("held",held,"held_same")):
        by_key = {joint.binding.learned.state_key(r):r for r in source["scopes"][scope]}
        old = [by_key[joint.binding.learned.state_key(r)] for r in rows]
        predictions[name] = learning.predict(joint.causal_data(rows,reference),model)
        reproduction.append(learning.reproduce_choices(predictions[name],old))
    a,b = PARTITIONS[partition][direction],PARTITIONS[partition][1-direction]
    scopes,scores = {},[]
    for scope in SCOPES:
        name = "train" if scope.startswith("train") else "held"
        rows = train if name=="train" else held
        by_key = {joint.binding.learned.state_key(r):r for r in source["scopes"][scope]}
        old = [by_key[joint.binding.learned.state_key(r)] for r in rows]
        values = joint.physical_values(rows,panels,a if scope.endswith("same") else b)
        scores.append(learning.reproduce_scores(values,predictions[name],old))
        scopes[scope] = learning.records(rows,values,predictions[name],old)
    return {"partition":partition,"direction":direction,"training_draws":list(a),"evaluation_draws":list(b),
        "source_selection":{a:model["selection"][a]["joint"] for a in joint.ANCHORS},"levels":predictions["train"]["levels"],
        "choice_reproduction":{"comparisons":sum(x["comparisons"] for x in reproduction),"changed_choices":sum(x["changed_choices"] for x in reproduction)},
        "score_reproduction":{"comparisons":sum(x["comparisons"] for x in scores),"max_abs_difference":max(x["max_abs_difference"] for x in scores)},
        "scopes":scopes}


def qualify(result,key,views):
    c = result["cell"]
    a,b = PARTITIONS[key[2]][key[3]],PARTITIONS[key[2]][1-key[3]]
    if (result["protocol"]!=EXPERIMENT_PROTOCOL or result["contract"]!=contract() or result["key"]!=list(key)
            or c["partition"]!=key[2] or c["direction"]!=key[3] or c["training_draws"]!=list(a) or c["evaluation_draws"]!=list(b)):
        raise ValueError("Changed frozen factorial protocol")
    for kind in joint.ANCHORS:
        selected = c["source_selection"][kind]
        expected = dict(zip(learning.lowrank.MODES,(*learning.lowrank.LEVELS,selected["mse_level"],selected["joint_level"])))
        if c["levels"][kind]!=expected: raise ValueError("Frozen joint rank selection changed")
    if c["choice_reproduction"]!={"comparisons":3720,"changed_choices":0}: raise ValueError("Frozen controller reproduction incomplete")
    if c["score_reproduction"]["comparisons"]!=14880 or c["score_reproduction"]["max_abs_difference"]>1e-12:
        raise ValueError("Frozen score reproduction incomplete")
    train,held = training.cohort.split_rows(views[key[0]],key[1])
    if set(c["scopes"])!=set(SCOPES): raise ValueError("Missing factorial scope")
    for scope,rows in c["scopes"].items():
        expected = train if scope.startswith("train") else held
        if len(rows)!=60 or {joint.binding.learned.state_key(r) for r in rows}!={joint.binding.learned.state_key(r) for r in expected}:
            raise ValueError("Dropped factorial state")
        for r in rows:
            for kind in joint.ANCHORS:
                if set(r["factorial"][kind])!=set(learning.lowrank.MODES): raise ValueError("Missing factorial rank")
                for mode,x in r["factorial"][kind].items():
                    if x["level"]!=c["levels"][kind][mode] or set(x["decisions"])!=set(learning.ARMS): raise ValueError("Changed matched factorial arms")
                    if x["level"] in (None,"zero") and len({d["selected"] for d in x["decisions"].values()})!=1:
                        raise ValueError("Zero or raw-hold factorial arms differ")


def metrics(rows):
    def outcome(ds): return {"mean_cost":float(np.mean([d["cost"] for d in ds])),"mean_wait_min":float(np.mean([d["wait"] for d in ds])),
        "changed_fraction":float(np.mean([d["changed"] for d in ds])),"unsafe_fraction":float(np.mean([d["unsafe"] for d in ds]))}
    output = {"prediction_cases":len(rows),"unique_states":len({joint.binding.learned.state_key(r) for r in rows}),
        "raw":outcome([r["raw"] for r in rows]),"factorial":{}}
    for kind in joint.ANCHORS:
        output["factorial"][kind] = {}
        for mode in learning.lowrank.MODES:
            xs = [r["factorial"][kind][mode] for r in rows]
            contrasts = [learning.effects(x["decisions"]) for x in xs]
            output["factorial"][kind][mode] = {"arms":{arm:outcome([x["decisions"][arm] for x in xs]) for arm in learning.ARMS},
                "effects":{name:{"cost":float(np.mean([c[name][0] for c in contrasts])),"wait_min":float(np.mean([c[name][1] for c in contrasts]))} for name in contrasts[0]},
                "active_cases":sum(x["level"] is not None for x in xs),"nonzero_cases":sum(x["level"] not in (None,"zero") for x in xs),
                "level_counts":{str(level):sum(x["level"]==level for x in xs) for level in sorted({x["level"] for x in xs},key=str)},
                "menu_changed_fraction":float(np.mean([x["added"]+x["removed"]>0 for x in xs])),
                "mean_added":float(np.mean([x["added"] for x in xs])),"mean_removed":float(np.mean([x["removed"] for x in xs])),
                "switch_fraction":{f"{a}_vs_{b}":float(np.mean([x["decisions"][a]["selected"]!=x["decisions"][b]["selected"] for x in xs]))
                    for a,b in (("cost","base"),("wait","base"),("joint","base"),("joint","cost"),("joint","wait"))}}
    return output


def summarize(results,views):
    if set(results)!=set(KEYS): raise ValueError("Complete all32 frozen factorial analyses")
    for key,r in results.items(): qualify(r,key,views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for key,r in results.items() if key[0]==view]
        scopes = {s:[r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        output[view] = {"scopes":{s:metrics(rs) for s,rs in scopes.items()},
            "held_other_by_root":{str(root):metrics([r for r in scopes["held_other"] if r["root"]==root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction":{f"{p}/{d}":metrics([r for c in cells if c["partition"]==p and c["direction"]==d for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0,1)}}
        if any(m["prediction_cases"]!=480 or m["unique_states"]!=120 for m in output[view]["scopes"].values()): raise ValueError("Changed complete factorial coverage")
    focus = results[FOCUS_KEY]["cell"]
    return {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"coverage_qualified":True,"views":output,
        "choice_reproduction":{"comparisons":sum(r["cell"]["choice_reproduction"]["comparisons"] for r in results.values()),"changed_choices":0},
        "score_reproduction":{"comparisons":sum(r["cell"]["score_reproduction"]["comparisons"] for r in results.values()),
            "max_abs_difference":max(r["cell"]["score_reproduction"]["max_abs_difference"] for r in results.values())},
        "focus":{"key":list(FOCUS_KEY),"levels":focus["levels"],"scopes":{s:metrics(rows) for s,rows in focus["scopes"].items()},
            "held_other_by_root":{str(root):metrics([r for r in focus["scopes"]["held_other"] if r["root"]==root]) for root in (397,401)}},
        **contract()["budget"],"native_deployment_eligible":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",required=True)
    parser.add_argument("--operation",choices=("analyze","report"),required=True)
    parser.add_argument("--view",choices=VIEWS)
    parser.add_argument("--fold",type=int,choices=(0,1))
    parser.add_argument("--partition",choices=PARTITIONS)
    parser.add_argument("--direction",type=int,choices=(0,1))
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"]!=previous.EXPERIMENT_PROTOCOL or prior["contract"]!=previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage230 before factorial attribution")
    bank,_,registered,_ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"],registered)
    if args.operation=="analyze":
        key = args.view,args.fold,args.partition,args.direction
        source = raw_directory(previous.result_path(SOURCE_RUN,*key))
        old = json.loads((source/"result.json").read_text())
        previous.qualify(old,key,views)
        model = json.loads((source/"models.json").read_text())
        reference = json.loads((raw_directory(previous.result_path(REFERENCE_RUN,*key))/"models.json").read_text())["reference"]
        train,held = training.cohort.split_rows(views[args.view],args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell = analyze_cell(train,held,panels,model,reference,old["cell"],args.partition,args.direction)
        result = {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"key":list(key),"cell":cell}
        qualify(result,key,views)
        output = result_path(args.run_name,*key)
        write_json(raw_directory(output)/"result.json",result)
        write_json(output,{k:v for k,v in result.items() if k!="cell"})
        print("NATIVE_FACTORIAL_ANALYSIS_COMPLETE",flush=True)
    else:
        result = summarize({k:json.loads((raw_directory(result_path(args.run_name,*k))/"result.json").read_text()) for k in KEYS},views)
        write_json(ROOT/"results"/args.run_name/"report/result.json",result)
        print("NATIVE_FACTORIAL_REPORT_COMPLETE",flush=True)


if __name__=="__main__": main()
