#!/usr/bin/env python3
"""Cross frozen cost rankings with raw and physical wait gates."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_native_transit_factorial_stage231 as previous
from freq_hrl.domains.transit import native_cost_gate as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory,write_json

joint,training = learning.joint,previous.training
EXPERIMENT_PROTOCOL = "native_transit_cost_gate_stage232_v1"
SOURCE_RUN = "native_transit_factorial_stage231_development_20261014_r1"
MODEL_RUN,REFERENCE_RUN = previous.SOURCE_RUN,previous.REFERENCE_RUN
VIEWS,PARTITIONS,KEYS,SCOPES,FOCUS_KEY = previous.VIEWS,previous.PARTITIONS,previous.KEYS,previous.SCOPES,previous.FOCUS_KEY


def contract():
    parent = previous.contract()
    return {"source_run":SOURCE_RUN,"source_protocol":previous.EXPERIMENT_PROTOCOL,"model_run":MODEL_RUN,"reference_run":REFERENCE_RUN,
        "cohort":parent["cohort"],"outer":parent["outer"],"partitions":parent["partitions"],
        "ranks":list(learning.lowrank.RANKS),"anchors":list(joint.ANCHORS),"arms":list(learning.ARMS),"modes":list(learning.lowrank.MODES),
        "selection":"inherit_joint_MSE_and_selected_rank_for_all_four_arms_None_holds_raw_no_reselection",
        "prediction":"identical_cost_heads_crossed_with_frozen_physical_wait_or_unrebased_raw_logits_same_anchor_first_ties",
        "raw_gate":"original_action_logit_zero_raw_action_already_feasible_do_not_subtract_anchor_logit",
        "controls":"reproduce_Stage231_base_cost_all_modes_and_raw_base_all_modes_raw_cost_zero_in_all_four_scopes",
        "diagnostics":"cost_wait_unsafe_menu_switch_effects_by_source_future_and_predeclared_rank8_focus",
        "focus_key":list(FOCUS_KEY),"budget":parent["budget"],
        "boundary":"reused_development_predictor_menu_intervention_not_new_CI_or_end_to_end_HRL",
        "storage":"pull_aggregate_only_no_panels_models_or_new_weights"}


def result_path(run_name,*key): return previous.result_path(run_name,*key)


def analyze_cell(train,held,panels,model,reference,source,partition,direction):
    predictions = {name:learning.predict(joint.causal_data(rows,reference),model) for name,rows in (("train",train),("held",held))}
    a,b = PARTITIONS[partition][direction],PARTITIONS[partition][1-direction]
    scopes,checks = {},[]
    for scope in SCOPES:
        name = "train" if scope.startswith("train") else "held"
        rows = train if name=="train" else held
        by_key = {joint.binding.learned.state_key(r):r for r in source["scopes"][scope]}
        old = [by_key[joint.binding.learned.state_key(r)] for r in rows]
        values = joint.physical_values(rows,panels,a if scope.endswith("same") else b)
        checks.append(learning.reproduce(predictions[name],old,values))
        scopes[scope] = learning.records(rows,values,predictions[name],old)
    return {"partition":partition,"direction":direction,"training_draws":list(a),"evaluation_draws":list(b),
        "source_selection":{k:model["selection"][k]["joint"] for k in joint.ANCHORS},"levels":predictions["train"]["levels"],
        "reproduction":{k:sum(c[k] for c in checks) for k in ("choice_comparisons","changed_choices","score_comparisons")}
            | {"max_abs_difference":max(c["max_abs_difference"] for c in checks)},"scopes":scopes}


def qualify(result,key,views):
    c = result["cell"]
    a,b = PARTITIONS[key[2]][key[3]],PARTITIONS[key[2]][1-key[3]]
    if (result["protocol"]!=EXPERIMENT_PROTOCOL or result["contract"]!=contract() or result["key"]!=list(key)
            or c["partition"]!=key[2] or c["direction"]!=key[3] or c["training_draws"]!=list(a) or c["evaluation_draws"]!=list(b)):
        raise ValueError("Changed cost/gate protocol")
    for kind in joint.ANCHORS:
        s = c["source_selection"][kind]
        if c["levels"][kind]!=dict(zip(learning.lowrank.MODES,(*learning.lowrank.LEVELS,s["mse_level"],s["joint_level"]))):
            raise ValueError("Inherited cost/gate rank selection changed")
    r = c["reproduction"]
    if r["choice_comparisons"]!=10800 or r["changed_choices"] or r["score_comparisons"]!=21600 or r["max_abs_difference"]>1e-12:
        raise ValueError("Incomplete frozen cost/gate reproduction")
    train,held = training.cohort.split_rows(views[key[0]],key[1])
    if set(c["scopes"])!=set(SCOPES): raise ValueError("Missing cost/gate scope")
    for scope,rows in c["scopes"].items():
        expected = train if scope.startswith("train") else held
        if len(rows)!=60 or {joint.binding.learned.state_key(r) for r in rows}!={joint.binding.learned.state_key(r) for r in expected}:
            raise ValueError("Dropped cost/gate state")
        for row in rows:
            for kind in joint.ANCHORS:
                if set(row["gates"][kind])!=set(learning.lowrank.MODES): raise ValueError("Missing cost/gate rank")
                for mode,x in row["gates"][kind].items():
                    ds = x["decisions"]
                    if x["level"]!=c["levels"][kind][mode] or set(ds)!=set(learning.ARMS): raise ValueError("Changed cost/gate arms")
                    if ds["raw_base"]["selected"]!=row["raw"]["selected"]: raise ValueError("Raw baseline no longer identity")
                    if x["level"] is None and len({d["selected"] for d in ds.values()})!=1: raise ValueError("Raw hold changed")
                    if x["level"]=="zero" and (ds["physical_base"]!=ds["physical_cost"] or ds["raw_base"]!=ds["raw_cost"]):
                        raise ValueError("Zero residual changes cost ranking")


def metrics(rows):
    def outcome(ds): return {"mean_cost":float(np.mean([d["cost"] for d in ds])),"mean_wait_min":float(np.mean([d["wait"] for d in ds])),
        "changed_fraction":float(np.mean([d["changed"] for d in ds])),"unsafe_fraction":float(np.mean([d["unsafe"] for d in ds]))}
    output = {"prediction_cases":len(rows),"unique_states":len({joint.binding.learned.state_key(r) for r in rows}),
        "raw":outcome([r["raw"] for r in rows]),"gates":{}}
    for kind in joint.ANCHORS:
        output["gates"][kind] = {}
        for mode in learning.lowrank.MODES:
            xs = [r["gates"][kind][mode] for r in rows]
            es = [learning.effects(x["decisions"]) for x in xs]
            output["gates"][kind][mode] = {"arms":{a:outcome([x["decisions"][a] for x in xs]) for a in learning.ARMS},
                "effects":{name:{"cost":float(np.mean([e[name][0] for e in es])),"wait_min":float(np.mean([e[name][1] for e in es]))} for name in es[0]},
                "level_counts":{str(l):sum(x["level"]==l for x in xs) for l in sorted({x["level"] for x in xs},key=str)},
                "nonzero_cases":sum(x["level"] not in (None,"zero") for x in xs),
                "menu_changed_fraction":float(np.mean([x["added"]+x["removed"]>0 for x in xs])),
                "mean_added":float(np.mean([x["added"] for x in xs])),"mean_removed":float(np.mean([x["removed"] for x in xs])),
                "switch_fraction":{f"{a}_vs_{b}":float(np.mean([x["decisions"][a]["selected"]!=x["decisions"][b]["selected"] for x in xs]))
                    for a,b in (("physical_cost","physical_base"),("raw_cost","raw_base"),("raw_base","physical_base"),("raw_cost","physical_cost"))}}
    return output


def summarize(results,views):
    if set(results)!=set(KEYS): raise ValueError("Complete all32 cost/gate analyses")
    for key,r in results.items(): qualify(r,key,views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for key,r in results.items() if key[0]==view]
        scopes = {s:[r for c in cells for r in c["scopes"][s]] for s in SCOPES}
        output[view] = {"scopes":{s:metrics(rs) for s,rs in scopes.items()},
            "held_other_by_root":{str(root):metrics([r for r in scopes["held_other"] if r["root"]==root]) for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction":{f"{p}/{d}":metrics([r for c in cells if c["partition"]==p and c["direction"]==d for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0,1)}}
        if any(x["prediction_cases"]!=480 or x["unique_states"]!=120 for x in output[view]["scopes"].values()): raise ValueError("Changed cost/gate coverage")
    focus = results[FOCUS_KEY]["cell"]
    return {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"coverage_qualified":True,"views":output,
        "reproduction":{k:sum(r["cell"]["reproduction"][k] for r in results.values()) for k in ("choice_comparisons","changed_choices","score_comparisons")}
            | {"max_abs_difference":max(r["cell"]["reproduction"]["max_abs_difference"] for r in results.values())},
        "focus":{"key":list(FOCUS_KEY),"levels":focus["levels"],"scopes":{s:metrics(rows) for s,rows in focus["scopes"].items()},
            "held_other_by_root":{str(root):metrics([r for r in focus["scopes"]["held_other"] if r["root"]==root]) for root in (397,401)}},
        **contract()["budget"],"native_deployment_eligible":False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name",required=True)
    p.add_argument("--operation",choices=("analyze","report"),required=True)
    p.add_argument("--view",choices=VIEWS)
    p.add_argument("--fold",type=int,choices=(0,1))
    p.add_argument("--partition",choices=PARTITIONS)
    p.add_argument("--direction",type=int,choices=(0,1))
    args = p.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"]!=previous.EXPERIMENT_PROTOCOL or prior["contract"]!=previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage231 before cost/gate attribution")
    bank,_,registered,_ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"],registered)
    if args.operation=="analyze":
        key = args.view,args.fold,args.partition,args.direction
        old = json.loads((raw_directory(previous.result_path(SOURCE_RUN,*key))/"result.json").read_text())
        previous.qualify(old,key,views)
        model = json.loads((raw_directory(previous.result_path(MODEL_RUN,*key))/"models.json").read_text())
        reference = json.loads((raw_directory(previous.result_path(REFERENCE_RUN,*key))/"models.json").read_text())["reference"]
        train,held = training.cohort.split_rows(views[args.view],args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell = analyze_cell(train,held,panels,model,reference,old["cell"],args.partition,args.direction)
        result = {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"key":list(key),"cell":cell}
        qualify(result,key,views)
        output = result_path(args.run_name,*key)
        write_json(raw_directory(output)/"result.json",result)
        write_json(output,{k:v for k,v in result.items() if k!="cell"})
        print("NATIVE_COST_GATE_ANALYSIS_COMPLETE",flush=True)
    else:
        report = summarize({k:json.loads((raw_directory(result_path(args.run_name,*k))/"result.json").read_text()) for k in KEYS},views)
        write_json(ROOT/"results"/args.run_name/"report/result.json",report)
        print("NATIVE_COST_GATE_REPORT_COMPLETE",flush=True)


if __name__=="__main__": main()
