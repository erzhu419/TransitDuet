#!/usr/bin/env python3
"""Diagnose saved fixed-calibration prefix responses without new fitting."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_native_transit_calibration_ablation_stage260 as source
from freq_hrl.domains.transit import native_curvature_response as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory,write_json

EXPERIMENT_PROTOCOL = "native_transit_curvature_stage261_v1"
SOURCE_RUN = "native_transit_calibration_ablation_stage260_20261016_r1"
cache, diagnostic = source.source,source.diagnostic
KEYS,VIEWS,PRIMARY,CATEGORIES = source.KEYS,source.VIEWS,source.PRIMARY,source.CATEGORIES


def contract():
    return {"source_run":SOURCE_RUN,"source_protocol":source.EXPERIMENT_PROTOCOL,"cohort":source.contract()["cohort"],
        "primary":list(PRIMARY),"categories":list(CATEGORIES),
        "frozen":"Stage260_full60_and_fixed_calibration_prefix54_models_old_keys_basis_prior_scale_cap_and_calibration; no_new_choices_or_parameter_selection",
        "curvature":"profiled_total_Hessian_including_cached_difference_penalty; low_block_is_numerical_null_plus_lowest_quarter_positive_modes_extended_across_ties; report_cut_and_dimension",
        "constraint":"infer_actual_full_prefix_RMS_multipliers; verify_quadratic_response_force_plus_multiplier_change_identity",
        "leverage":"removed_prefix_rows_of_SVD_hat_matrix_for_joint_service_risk_Hessian_plus_fixed_penalty_and_actual_full_KKT_dual; not_naive_subtraction_of_profiled_Hessians",
        "exposure":"inverse_KKT_curvature_and_unsupported_cost_query_design_mass; Gram_support_tolerance_dimension_times_machine_epsilon",
        "response":"low_plus_rest_prediction_squared_includes_cross_term; never_treat_RMS_ratios_as_causal_or_variance_shares",
        "controls":"reproduce_cached_full_fixed_primary_actions_and_Stage260_fit_population_cost_RMS_all_views_banks_categories; no_evaluation_panel_reads",
        "budget":{"analysis_tasks":1,"cached_cells":32,"cached_prefix_fits":640,"prefix_model_records":1280,
            "new_critic_fits":0,"new_raw_reference_fits":0,"new_native_episodes":0,"actor_updates":0},
        "storage":"models_calibrations_features_and_detailed_cell_ledgers_server_only; pull_scalar_report",
        "boundary":"conditional_development_geometry_not_accuracy_utility_seed_CI_or_nonlinearity_proof; frozen_calibration_saw_excluded_training_labels; leverage_and_dual_not_constrained_sampling_covariance; Stage253_248_169_closed; no_default_change"}


def analyze_cell(key,views):
    read = lambda run,file:source.previous.read_cell(run,key,file)
    models,calibrations,actions = (read(SOURCE_RUN,f) for f in ("models.json","calibrations.json","actions.json"))
    full_models = read(cache.cached.SOURCE_RUN,"models.json")
    prior = read(cache.cached.information.SOURCE_RUN,"models.json")["full"]["all_mse"]
    reference = read(cache.cached.fixed.contract()["reference_run"],"models.json")["reference"]
    train,held = cache.cached.training.cohort.split_rows(views[key[0]],key[1])
    data = {s:cache.learning.base.causal_data(rs,reference) for s,rs in (("train60",train),("held60",held))}
    geometry = cache.learning.deployment_geometry(data["train60"])
    x = {s:cache.learning.joint.relative(d["x"][...,:198],d["originals"]) for s,d in data.items()}
    selected = [a for a in actions if a["variant"] in PRIMARY]
    lookup = {(a["bank"],tuple(a["removed"]),a["variant"],a["scope"]):a for a in selected}
    records,maximum = [],0.
    for bank in (0,1):
        c = next(c for c in calibrations if c["bank"] == bank)
        draws,_ = cache.sampling.learning.draw_assignment(bank,"expanded")
        if c["training_draws"] != draws: raise ValueError("Changed training-bank calibration assignment")
        target,precision = np.asarray(c["target"]),np.asarray(c["precision"])
        full_contexts,full_paths = {},{}
        for fitted in (m for m in models if m["bank"] == bank):
            group = fitted["held_group"]
            if fitted["training_draws"] != draws: raise ValueError("Changed fixed prefix model training assignment")
            masks = diagnostic.regions(lookup[bank,tuple(group),PRIMARY[0],"train60"]["states"],"train60",group)
            indices,removed = np.flatnonzero(masks["retained"]),np.flatnonzero(masks["excluded"])
            if (len(indices),len(removed)) != (54,6): raise ValueError("Changed prefix deletion population")
            for variant in PRIMARY:
                a = lookup[bank,tuple(group),variant,"train60"]
                record = {f:a[f] for f in ("cell","bank","removed","variant","category","old_key","old_key_new_gate")}
                record.update(active=a["old_key"] is not None and a["old_key"] != "zero",regions={})
                model = variant.removesuffix("_expanded8_pooled_safe")
                if record["active"]:
                    option = cache.learning.strength.CANDIDATES[a["old_key"]]
                    level,alpha = option["level"],option["alpha"]
                    basis,coordinates = geometry[level]
                    z = x["train60"]@basis
                    initial = np.asarray(prior["weights"][level])@coordinates.T
                    fixed_heads = np.asarray(fitted["models"][model]["weights"][level])@coordinates.T
                    radius = next(f["old_radius"] for f in fitted["fit"]["fits"] if f["level"] == level)
                    calibration = {"target":target,"precision":precision,"penalty":c["penalties"][level]}
                    if variant not in full_contexts:
                        heads = np.asarray(full_models[str(bank)][model]["weights"][level])@coordinates.T
                        full_contexts[variant] = learning.full_context(z,calibration,data["train60"]["raw_logits"]<=0,initial,heads,model,radius)
                        full_paths[variant] = {s:cache.learning.base.candidates(d,full_models[str(bank)][model])[a["old_key"]] for s,d in data.items()}
                    context = learning.prefix_context(z,calibration,data["train60"]["raw_logits"]<=0,initial,fixed_heads,
                        indices,removed,model,radius,full_contexts[variant])
                    record["spectral"] = context["diagnostics"]
                for scope,d in data.items():
                    old = lookup[bank,tuple(group),variant,scope]
                    states = [list(cache.sampling.old_future.learning.state_key(r)) for r in (train if scope == "train60" else held)]
                    if old["states"] != states or old["raw"] != d["raw_choices"].tolist(): raise ValueError("Changed cached state order or reference")
                    region_masks = diagnostic.regions(states,scope,group)
                    feasible = (d["raw_logits"]<=0)&(np.arange(6)[None,:]!=d["originals"][:,None])
                    if record["active"]:
                        pred = cache.learning.base.candidates(d,fitted["models"][model])[a["old_key"]]
                        full = full_paths[variant][scope]
                        if (pred["chosen"].tolist()!=old["choices"]["fixed"] or full["chosen"].tolist()!=old["choices"]["full"]):
                            raise ValueError("Changed Stage260 cached primary actions")
                        zz = x[scope]@basis
                        predicted = zz@context["delta"]*(alpha*prior["scale"])
                        maximum = max(maximum,float(np.max(np.abs(predicted-(pred["cost"]-full["cost"])))))
                    else:
                        if any(old["choices"][name]!=old["raw"] for name in ("full","fixed")): raise ValueError("Inactive key changed raw actions")
                    for region,mask in region_masks.items():
                        if not mask.any(): continue
                        s = learning.region_stats(zz,mask[:,None],feasible,context,alpha*prior["scale"]) if record["active"] else {
                            "feasible_action_cases":int(feasible[mask].sum()),"squared_response":dict.fromkeys(("total","low","rest","cross"),0.),"exposure":{},"decomposition_residual":0.}
                        expected = old["function_gaps"]["fit_population"][region]
                        if s["feasible_action_cases"] != expected["feasible_action_cases"]: raise ValueError("Changed feasible-action population")
                        maximum = max(maximum,abs(s["squared_response"]["total"]-expected["squared_sums"]["cost"]))
                        s.update(state_cases=int(mask.sum()),disagreements=int(np.count_nonzero(np.asarray(old["choices"]["fixed"])[mask]!=np.asarray(old["choices"]["full"])[mask])))
                        record["regions"][region] = s
                records.append(record)
    if len(models)!=20 or len(records)!=40 or maximum>1e-12: raise ValueError("Curvature analysis failed Stage260 cell reproduction")
    print(f"CURVATURE_CELL_COMPLETE {key}",flush=True)
    return {"key":list(key),"pieces":records,"reproduction_max_abs_difference":maximum}


def summarize(pieces):
    active = [p for p in pieces if p["active"]]
    output = {"bank_cell_exposures":len({(tuple(p["cell"]),p["bank"]) for p in pieces}),
        "prefix_model_exposures":len(pieces),"active_prefix_models":len(active),"regions":{}}
    if active:
        fields = ("low_dimension_fraction","parameter_energy","low_parameter_energy","full_dual","prefix_dual")
        output["spectral"] = {f:float(np.mean([p["spectral"][f] for p in active])) for f in fields}
        energy = sum(p["spectral"]["parameter_energy"] for p in active)
        def defined_mean(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None
        output["spectral"].update(low_parameter_energy_fraction=sum(p["spectral"]["low_parameter_energy"] for p in active)/energy if energy else None,
            mean_low_parameter_fraction=defined_mean([p["spectral"]["low_parameter_fraction"] for p in active]),
            mean_low_cut_to_maximum=defined_mean([p["spectral"]["low_cut_to_maximum"] for p in active]),
            mean_minimum_positive_to_maximum=defined_mean([p["spectral"]["minimum_positive_to_maximum"] for p in active]),
            mean_full_effective_condition=defined_mean([p["spectral"]["full_effective_condition"] for p in active]),
            mean_prefix_effective_condition=defined_mean([p["spectral"]["prefix_effective_condition"] for p in active]),
            mean_block_leverage_maximum=float(np.mean([p["spectral"]["block_leverage"]["maximum"] for p in active])),
            maximum_block_leverage=max(p["spectral"]["block_leverage"]["maximum"] for p in active),
            mean_block_data_trace_fraction=defined_mean([p["spectral"]["block_leverage"]["data_trace_fraction"] for p in active]),
            full_constraint_active=sum(p["spectral"]["full_dual"]>0 for p in active),prefix_constraint_active=sum(p["spectral"]["prefix_dual"]>0 for p in active))
    for region in diagnostic.REGIONS:
        xs = [p["regions"][region] for p in pieces if region in p["regions"]]
        if not xs: continue
        cases,actions = sum(x["state_cases"] for x in xs),sum(x["feasible_action_cases"] for x in xs)
        sums = {f:sum(x["squared_response"][f] for x in xs) for f in ("total","low","rest","cross")}
        stats = {"state_model_exposures":cases,"feasible_action_exposures":actions,
            "action_disagreements":sum(x["disagreements"] for x in xs),"squared_response":sums,
            "cost_RMS":float(np.sqrt(sums["total"]/actions)) if actions else None,
            "low_RMS":float(np.sqrt(sums["low"]/actions)) if actions else None,
            "rest_RMS":float(np.sqrt(sums["rest"]/actions)) if actions else None}
        for name in ("full_factor","prefix_factor"):
            es = [x["exposure"][name] for x in xs if name in x["exposure"]]
            n = sum(x["feasible_action_cases"] for x in xs if name in x["exposure"])
            design = sum(x["design_squared_sum"] for x in es)
            stats[name] = {"mean_inverse_curvature":sum(x["inverse_curvature_sum"] for x in es)/n if n else None,
                "unsupported_design_fraction":sum(x["unsupported_squared_sum"] for x in es)/design if design else None}
        output["regions"][region] = stats
    return output


def report(records,expected):
    groups = {"mixed_pooled":lambda k:k[0].startswith("mixed"),**{v:lambda k,v=v:k[0]==v for v in VIEWS}}
    output,maximum = {},0.
    for group,predicate in groups.items():
        ps = [p for r in records if predicate(r["key"]) for p in r["pieces"]]
        output[group] = {}
        for label,bank in (("pooled",None),("bank0",0),("bank1",1)):
            output[group][label] = {}
            for variant in PRIMARY:
                output[group][label][variant] = {}
                for cat in CATEGORIES:
                    now = summarize([p for p in ps if p["variant"]==variant and (bank is None or p["bank"]==bank) and source.previous.in_category(p,cat)])
                    prior = expected["groups"][group][label][variant][cat]
                    if now["bank_cell_exposures"]!=prior["bank_cell_exposures"]: raise ValueError("Changed Stage260 category exposure")
                    for region,before in prior["comparisons"]["fit_population"].items():
                        actual = now["regions"][region]
                        if (actual["state_model_exposures"],actual["action_disagreements"])!=(before["state_model_exposures"],before["action_disagreements"]): raise ValueError("Changed Stage260 action/state counts")
                        value = before["feasible_prediction_rms"]["cost"]
                        if value is not None: maximum = max(maximum,abs(actual["cost_RMS"]-value))
                    output[group][label][variant][cat] = now
    if maximum>1e-12: raise ValueError("Curvature report failed Stage260 RMS reproduction")
    ps = [p for r in records for p in r["pieces"]]
    if len(records)!=32 or len(ps)!=1280: raise ValueError("Incomplete curvature matrix")
    return {"protocol":EXPERIMENT_PROTOCOL,"contract":contract(),"coverage_qualified":True,"groups":output,
        "reproduction":{"cached_cells":len(records),"prefix_model_records":len(ps),"cell_max_abs_difference":max(r["reproduction_max_abs_difference"] for r in records),
            "group_max_abs_difference":maximum,"influence_equation_max_residual":max(p["spectral"]["influence_equation_residual"] for p in ps if p["active"]),
            "decomposition_max_residual":max(s["decomposition_residual"] for p in ps for s in p["regions"].values())},
        "default_changed":False,"native_deployment_eligible":False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name",required=True)
    a = p.parse_args()
    data,_,registered,_ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"],registered)
    records = [analyze_cell(key,views) for key in KEYS]
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    payload = report(records,expected)
    path = ROOT/"results"/a.run_name/"report/result.json"
    write_json(raw_directory(path)/"cells.json",records)
    write_json(path,payload)
    print("CURVATURE_REPORT_COMPLETE",flush=True)


if __name__=="__main__": main()
