#!/usr/bin/env python3
"""Match aligned prefix-excluded inner fits to the frozen full future count."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_old_key_response_stage257 as previous
from freq_hrl.domains.transit import native_training_count_response as diagnostic
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

bundle, cached = previous.source, previous.source.source
learning, sampling = bundle.learning, cached.information
EXPERIMENT_PROTOCOL = "native_transit_training_count_stage258_v1"
SOURCE_RUN = "native_transit_old_key_response_stage257_20261016_r1"
CACHE_RUN = "native_transit_deployment_bundle_stage255_20261016_r1"
KEYS, VIEWS, PREFLIGHT, result_path = bundle.KEYS, bundle.VIEWS, bundle.PREFLIGHT, bundle.result_path
ARMS, FILTERS, CATEGORIES = previous.ARMS, previous.FILTERS, previous.CATEGORIES


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL, "cache_run": CACHE_RUN,
        "cohort": bundle.contract()["cohort"], "arms": list(ARMS), "filters": list(FILTERS), "categories": list(CATEGORIES),
        "treatment": "prefix_excluded_inner16_union_of_same_two8_training_blocks_vs_each_frozen_inner8_half; same54_fit_states_full_basis_prior_scale_RMS_cap_loss_and_penalty_rules",
        "keys": "old_Stage252_keys_frozen; Stage255_categories_frozen; no_new_selector_or_rank_strength_choice",
        "full": "cached_full_actions_and_held_outcome_anchor_verbatim; not_an_inner16_held_policy_test",
        "evaluation": "same_opposite16_Stage249_futures; first8_last8_reported; all_new_actions_and_models_written_before_evaluation_outcome_access",
        "controls": "training8_nested_in16_and_disjoint_from_evaluation; cached_state_order_raw_choices_and_training_assignment; reproduce_Stage257_inner8_full_seen_held_anchor_and_action_disagreement",
        "diagnostics": "count16_minus8_plus_full_minus16_equals_full_minus8; individual_training_halves_action_disagreement_and_mean_absolute_per_cell_full_response_gap",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "inner16_fits": 640, "inner_solves": 5120,
            "inner8_refits": 0, "new_full_critic_fits": 0, "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "new_inner16_models_all_candidate_actions_fit_ledgers_and_cell_results_server_only; receipt_and_scalar_report_pull",
        "boundary": "controlled_training_count_on_reused_development_states_futures_and_selection_conditioned_strata; banks_views_and_repeated_inner16_full_exposures_not_independent_seeds; response_agreement_not_truth_accuracy; no_default_or_deployment_change; Stage253_248_outcomes_and_169_reserves_closed"}


def evaluate_cell(info, old8, full_actions, new16, panels):
    values = lambda states, draws: np.stack([panels[tuple(st)]["values"][draws] for st in states])
    lookup = {(c["bank"],tuple(c["held_group"]),c["half"]):c for c in old8}
    pieces, maximum = [], 0.
    for bank in info["banks"]:
        number = bank["bank"]
        full = next(a for a in full_actions if a["bank"] == number and a["scope"] == "train60")
        positions = {tuple(st):i for i,st in enumerate(full["states"])}
        draws16, ev = sampling.learning.draw_assignment(number,"expanded")
        for model in learning.MODELS:
            for selector in learning.SELECTORS:
                arm = f"{model}_{selector}"
                for mode in FILTERS:
                    old, new = (bank["selection"][b][arm][mode] for b in learning.BUNDLES)
                    variant = f"prefix_refit_{arm}_{mode}"
                    parts = {p:[] for p in diagnostic.PATHS}
                    halves = [{p:[] for p in diagnostic.PATHS} for _ in (0,1)]
                    training_halves = [[],[]]
                    disagreement = dict.fromkeys(("inner8_full","inner16_full","inner16_inner8"),0)
                    cases = 0
                    for c in (c for c in new16 if c["bank"] == number):
                        if c["training_draws"] != draws16: raise ValueError("Changed inner16 training assignment")
                        st, raw = c["states"], c["raw_choices"]
                        ix = [positions[tuple(s)] for s in st]
                        if raw != [full["choices"]["raw"][i] for i in ix]: raise ValueError("Changed aligned raw actions")
                        chosen16 = c["choices"][model][old or "zero"]
                        chosen_full = [full["choices"][variant][i] for i in ix]
                        evaluation, roots = values(st,ev), [s[0] for s in st]
                        for h in (0,1):
                            before = lookup[number,tuple(c["held_group"]),h]
                            ds, _ = sampling.learning.draw_assignment(number,"expanded",h)
                            if before["states"] != st or before["raw_choices"] != raw or before["training_draws"] != ds or not set(ds) < set(draws16) or set(draws16) & set(ev):
                                raise ValueError("Changed cached inner8 state order, nesting or assignment")
                            chosen8 = before["choices"][model][old or "zero"]
                            predictions = dict(zip(diagnostic.PATHS,(chosen8,chosen16,chosen_full)))
                            for p,ch in predictions.items():
                                stat = diagnostic.response.action_stats(evaluation,ch,raw,roots)
                                parts[p].append(stat)
                                if p == "inner8": training_halves[h].append(stat)
                                for eh in (0,1): halves[eh][p].append(diagnostic.response.action_stats(evaluation[:,eh*8:(eh+1)*8],ch,raw,roots))
                            for name,a,b in (("inner8_full",chosen8,chosen_full),("inner16_full",chosen16,chosen_full),("inner16_inner8",chosen16,chosen8)):
                                disagreement[name] += int(np.count_nonzero(np.asarray(a) != b))
                            cases += len(st)
                    paths = {p:diagnostic.response.combine(ps) for p,ps in parts.items()}
                    for f in diagnostic.FIELDS:
                        maximum = max(maximum,abs(paths["full_seen"]["all"][f]-bank["matrix"]["train60"]["all"]["variants"][variant][f]))
                    status = previous.previous.learning.gate(bank["summaries"][arm][old],mode) if old is not None else None
                    pieces.append({"bank":number,"arm":arm,"filter":mode,"old_key":old,"new_key":new,
                        "category":previous.previous.learning.transition(old,new),"old_key_new_gate":status,
                        "paths":paths,"effects":diagnostic.contrasts(paths),
                        "evaluation_halves":[{p:diagnostic.response.combine(ps) for p,ps in hp.items()} for hp in halves],
                        "training_halves":[diagnostic.response.combine(ps) for ps in training_halves],
                        "action_disagreement":{f:n/cases for f,n in disagreement.items()},
                        "full_held_anchor":{f:bank["matrix"]["held60"]["all"]["variants"][variant][f] for f in diagnostic.FIELDS}})
    if maximum > 1e-12: raise ValueError("Frozen full seen outcomes changed")
    return pieces, maximum


def fit_cell(run,key,views):
    load = lambda name,file: json.loads((raw_directory(result_path(name,*key))/file).read_text())
    info, old8, full_actions = (load(CACHE_RUN,f) for f in ("aggregate.json","oof_choices.json","actions.json"))
    if info["protocol"] != bundle.EXPERIMENT_PROTOCOL or info["contract"] != bundle.contract() or not info["coverage_qualified"]:
        raise ValueError("Require qualified Stage255 cache")
    prior = load(cached.information.SOURCE_RUN,"models.json")["full"]["all_mse"]
    ledger = load(cached.fixed.SOURCE_RUN,"fits.json")["full"]
    reference = load(cached.fixed.contract()["reference_run"],"models.json")["reference"]
    train, _ = cached.training.cohort.split_rows(views[key[0]],key[1])
    geometry = learning.deployment_geometry(learning.base.causal_data(train,reference))
    panels = sampling.read_panels(cached.previous.DATA_RUN)
    models, choices = [], []
    for bank in (0,1):
        ds, _ = sampling.learning.draw_assignment(bank,"expanded")
        for group,indices,validation in learning.joint.binding.policy.prefix_folds(train):
            rows,val_rows = [train[i] for i in indices],[train[i] for i in validation]
            data,val = learning.base.causal_data(rows,reference),learning.base.causal_data(val_rows,reference)
            fitted,fit = learning.fit_inner(rows,data,learning.base.training_draws(rows,panels,ds),prior,ledger,geometry)
            paths = {m:learning.base.candidates(val,fitted[m]) for m in learning.MODELS}
            models.append({"bank":bank,"held_group":list(group),"training_draws":ds,"models":fitted,"fit":fit})
            choices.append({"bank":bank,"held_group":list(group),"training_draws":ds,
                "states":[list(sampling.old_future.learning.state_key(r)) for r in val_rows],"raw_choices":val["raw_choices"].tolist(),
                "choices":{m:{c:p["chosen"].tolist() for c,p in ps.items()} for m,ps in paths.items()}})
        print(f"TRAINING_COUNT_BANK_FROZEN {key}/{bank}",flush=True)
    path = result_path(run,*key)
    write_json(raw_directory(path)/"models.json",models)
    write_json(raw_directory(path)/"actions.json",choices)
    pieces,maximum = evaluate_cell(info,old8,full_actions,choices,panels)
    fits = [m["fit"] for m in models]
    fit_summary = {"inner16_fits":len(fits),"inner_solves":len(fits)*8,
        "subspace_max_abs_difference":max(f["subspace_max_abs_difference"] for f in fits),
        "max_kkt":max(max(x["optimality"],x["difference_optimality"],x["auxiliary_free_fit"]["optimality"]) for f in fits for x in f["fits"]),
        "radius_max_excess":max(max(x["norm"],x["auxiliary_free_fit"]["norm"])-x["old_radius"] for f in fits for x in f["fits"])}
    payload = {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"key":list(key),"coverage_qualified":True,
        "pieces":pieces,"fits":fit_summary,"reproduction_max_abs_difference":maximum}
    write_json(raw_directory(path)/"aggregate.json",payload)
    write_json(path,{k:v for k,v in payload.items() if k != "pieces"})


def report(run):
    records = [json.loads((raw_directory(result_path(run,*k))/"aggregate.json").read_text()) for k in KEYS]
    for key,r in zip(KEYS,records):
        if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) != key or not r["coverage_qualified"]:
            raise ValueError("Incomplete matched-training-count matrix")
    if any(sum(r["fits"][f] for r in records) != contract()["budget"][f] for f in ("inner16_fits","inner_solves")):
        raise ValueError("Training-count fits do not cover the registered prefix matrix")
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    groups = {"mixed_pooled":lambda k:k[0].startswith("mixed"),**{v:lambda k,v=v:k[0]==v for v in VIEWS}}
    output,maximum = {},0.
    for group,predicate in groups.items():
        pieces = [p for r in records if predicate(r["key"]) for p in r["pieces"]]
        output[group] = {}
        for label,bank in (("pooled",None),("bank0",0),("bank1",1)):
            output[group][label] = {}
            for arm in ARMS:
                for mode in FILTERS:
                    xs = [p for p in pieces if p["arm"]==arm and p["filter"]==mode and (bank is None or p["bank"]==bank)]
                    variant = f"{arm}_{mode}"
                    partitions = {cat:[p for p in xs if cat=="all" or p["category"]==cat or
                        (cat=="removed_point_harm" and p["category"]=="removed" and not p["old_key_new_gate"]["strict_gain_only"]) or
                        (cat=="removed_no_strict_gain" and p["category"]=="removed" and p["old_key_new_gate"]["strict_gain_only"])] for cat in CATEGORIES}
                    summaries = {cat:diagnostic.summarize(ps,len(xs)) for cat,ps in partitions.items()}
                    for cat,now in summaries.items():
                        before = expected["groups"][group][label][variant][cat]
                        if now["bank_cell_exposures"] != before["bank_cell_exposures"]: raise ValueError("Changed Stage257 category counts")
                        if not now["bank_cell_exposures"]: continue
                        a,b = now["conditional"],before["conditional"]
                        for p,oldp in (("inner8","inner_opposite"),("full_seen","full_seen")):
                            for f in diagnostic.FIELDS: maximum = max(maximum,abs(a["path_means"][p][f]-b["path_means"][oldp][f]))
                        for f in diagnostic.FIELDS: maximum = max(maximum,abs(a["full_held_anchor"][f]-b["path_means"]["full_held"][f]))
                        maximum = max(maximum,abs(a["action_disagreement"]["inner8_full"]-b["inner_full_action_disagreement"]))
                    output[group][label][variant] = summaries
    if maximum > 1e-12: raise ValueError("Matched-count controls failed Stage257 reproduction")
    payload = {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"coverage_qualified":True,"groups":output,
        "fits":{"inner16_fits":sum(r["fits"]["inner16_fits"] for r in records),"inner_solves":sum(r["fits"]["inner_solves"] for r in records),
            **{f:max(r["fits"][f] for r in records) for f in ("subspace_max_abs_difference","max_kkt","radius_max_excess")}},
        "reproduction":{"cell_max_abs_difference":max(r["reproduction_max_abs_difference"] for r in records),"category_max_abs_difference":maximum},
        "default_changed":False,"native_deployment_eligible":False}
    write_json(ROOT/"results"/run/"report/result.json",payload)
    print("TRAINING_COUNT_REPORT_COMPLETE",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name",required=True)
    p.add_argument("--operation",choices=("fit","report"),required=True)
    p.add_argument("--view",choices=VIEWS)
    p.add_argument("--fold",type=int,choices=(0,1))
    p.add_argument("--partition",choices=cached.fixed.PARTITIONS)
    p.add_argument("--direction",type=int,choices=(0,1))
    a = p.parse_args()
    if a.operation=="fit":
        data,_,registered,_ = cached.training.cohort.source.source.load_data()
        views = cached.training.cohort.source.matched_views(data["training_data"],registered)
        fit_cell(a.run_name,(a.view,a.fold,a.partition,a.direction),views)
    else:report(a.run_name)


if __name__=="__main__":main()
