#!/usr/bin/env python3
"""Propagate cached joint target MC noise through frozen KKT geometry."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_head_half_forcing_stage264 as source
from scripts.run_native_transit_prefix_influence_stage259 import read_cell, in_category
from scripts.run_native_transit_kkt_response_stage262 import FIT_RUN
from freq_hrl.domains.transit import native_joint_noise as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_joint_noise_stage265_v1"
SOURCE_RUN = "native_transit_head_half_forcing_stage264_20261016_r1"
cache = source.cache
KEYS, VIEWS, PRIMARY, CATEGORIES = source.KEYS, source.VIEWS, source.PRIMARY, source.CATEGORIES
REPORT_REGIONS, FOCUS_PREFIX = source.REPORT_REGIONS, source.FOCUS_PREFIX


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "primary": list(PRIMARY), "categories": list(CATEGORIES),
        "frozen": source.contract()["frozen"],
        "noise": "per_state_service_risk_action_paired_score_covariance_centered_within_each_assigned16_bank; covariance_of_two16_mean_difference=sum_state(C_bank0/16+C_bank1/16); no_between_bank_mean_term",
        "independence": "conditional_state_future_seed_blocks_Stage249_1490000000+state_index*32+draw; sum_state_covariances_not_covariance_of_aligned_draw_index_state_sums",
        "propagation": "A=H_prefix+mu_prefix*I; C_theta=pinv(A)*C_q*pinv(A).T; preserve_head_action_and_mode_cross_covariance; exclude_retained_population_covariance_zero_under_state_seed_assumption",
        "directions": "Stage261_Hessian_low_block; supported_KKT_mode_diagonals_and_dominant_noise_covariance_direction; inverse_gain_is_coordinate_dependent_not_dimensionless_performance",
        "comparison": "observed_cached_bank_target_gap_RMS_over_plugin_two_bank_difference_noise_RMS; descriptive_not_pvalue_CI_or_noise_free_claim",
        "controls": "reproduce_Stage264_service_risk_joint_and_head_population_cross16_RMS_all_groups_banks_categories_three_query_regions_and_low_rest; direct_existing_gradient_factory_and_covariance_identities",
        "focus_prefix": list(FOCUS_PREFIX), "focus_basis": "unchanged_Stage263_designation; no_training_exclusion_or_gate",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_prefix_fits": 640, "prefix_model_records": 1280,
            "new_critic_fits": 0, "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "raw_models_labels_covariances_and_direction_profiles_server_only; pull_scalar_report",
        "boundary": "frozen_operator_plugin_noise_not_unconditional_refit_covariance_or_ground_truth_error; operator_calibration_and_keys_depend_on_anchor_bank; finite_cache_can_miss_rare_tails; no_held_outcome_evaluation_selection_or_default_change; Stage253_248_169_closed"}


def analyze_cell(key, views, panels):
    read = lambda run, file: read_cell(run, key, file)
    models, calibrations, actions = (read(FIT_RUN, f) for f in ("models.json", "calibrations.json", "actions.json"))
    cs = {c["bank"]: c for c in calibrations}
    full_models = read(cache.cached.SOURCE_RUN, "models.json")
    prior = read(cache.cached.information.SOURCE_RUN, "models.json")["full"]["all_mse"]
    reference = read(cache.cached.fixed.contract()["reference_run"], "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    data = {s: cache.learning.base.causal_data(rows, reference) for s, rows in (("train60", train), ("held60", held))}
    geometry = cache.learning.deployment_geometry(data["train60"])
    x = {s: cache.learning.joint.relative(d["x"][..., :198], d["originals"]) for s, d in data.items()}
    paired, label_error = {}, 0.
    for bank in (0, 1):
        draws, evaluation = cache.sampling.learning.draw_assignment(bank, "expanded")
        if cs[bank]["training_draws"]!=draws or len(draws)!=16 or set(draws)&set(evaluation): raise ValueError("Changed future banks")
        values = cache.learning.base.training_draws(train, panels, draws)
        residual = learning.previous.previous.residual_target(values, data["train60"], prior["scale"])
        label_error = max(label_error, float(np.max(np.abs(residual*np.asarray(cs[bank]["reliability"])-np.asarray(cs[bank]["target"])))))
        v = values[..., (2, 1)]
        paired[bank] = v-v[np.arange(len(train)), :, data["train60"]["originals"]][:, :, None]
    if label_error > 1e-12 or set(cs[0]["training_draws"])&set(cs[1]["training_draws"]): raise ValueError("Changed target reconstruction or bank overlap")
    lookup = {(a["bank"], tuple(a["removed"]), a["variant"], a["scope"]): a for a in actions if a["variant"] in PRIMARY}
    records, maximum = [], 0.
    for fitted in models:
        bank, group = fitted["bank"], fitted["held_group"]
        if fitted["training_draws"]!=cs[bank]["training_draws"]: raise ValueError("Changed cached prefix bank")
        masks = source.diagnostic.regions(lookup[bank, tuple(group), PRIMARY[0], "train60"]["states"], "train60", group)
        indices = np.flatnonzero(masks["retained"])
        if (len(indices), int(masks["excluded"].sum()))!=(54, 6): raise ValueError("Changed prefix population")
        for variant in PRIMARY:
            a = lookup[bank, tuple(group), variant, "train60"]
            record = {f: a[f] for f in ("cell", "bank", "removed", "variant", "category", "old_key", "old_key_new_gate")}
            record.update(active=a["old_key"] is not None and a["old_key"]!="zero", regions={})
            if record["active"]:
                model = variant.removesuffix("_expanded8_pooled_safe")
                option = cache.learning.strength.CANDIDATES[a["old_key"]]
                level, alpha = option["level"], option["alpha"]
                basis, coordinates = geometry[level]
                radius = next(f["old_radius"] for f in fitted["fit"]["fits"] if f["level"]==level)
                context = learning.context(x["train60"]@basis, cs[bank], paired, data["train60"]["raw_logits"]<=0,
                    np.asarray(prior["weights"][level])@coordinates.T,
                    np.asarray(full_models[str(bank)][model]["weights"][level])@coordinates.T,
                    np.asarray(fitted["models"][model]["weights"][level])@coordinates.T,
                    indices, model, radius, bank, level, prior["scale"])
                record["noise"] = context["diagnostics"]
            for scope, d in data.items():
                old = lookup[bank, tuple(group), variant, scope]
                states = [list(cache.sampling.old_future.learning.state_key(r)) for r in (train if scope=="train60" else held)]
                if old["states"]!=states or old["raw"]!=d["raw_choices"].tolist(): raise ValueError("Changed states/reference")
                feasible = (d["raw_logits"]<=0)&(np.arange(6)[None, :]!=d["originals"][:, None])
                for region, mask in source.diagnostic.regions(states, scope, group).items():
                    if region not in REPORT_REGIONS or not mask.any(): continue
                    s = learning.region_stats(x[scope]@basis, mask[:, None], feasible, context, alpha*prior["scale"]) if record["active"] else {
                        "feasible_action_cases": int(feasible[mask].sum()), "bands": {b: {"noise_squared_sum": dict.fromkeys(learning.COVARIANCES, 0.),
                            "observed_squared_sum": dict.fromkeys((*learning.POSITIVE, *learning.CROSSES, "cross_population_joint"), 0.)} for b in ("full", "low", "rest")},
                        "low_rest_noise_cross": dict.fromkeys(learning.POSITIVE, 0.), "decomposition_residual": 0.}
                    expected = old["function_gaps"]["fit_population"][region]
                    if s["feasible_action_cases"]!=expected["feasible_action_cases"]: raise ValueError("Changed feasible actions")
                    if record["active"]:
                        delta = (x[scope]@basis)[mask[:, None]&feasible]@context["delta"]*(alpha*prior["scale"])
                        maximum = max(maximum, abs(float(delta@delta)-expected["squared_sums"]["cost"]))
                    s.update(state_cases=int(mask.sum()), disagreements=int(np.count_nonzero(np.asarray(old["choices"]["fixed"])[mask]!=np.asarray(old["choices"]["full"])[mask])))
                    record["regions"][region] = s
            records.append(record)
    if len(models)!=20 or len(records)!=40 or maximum>1e-12: raise ValueError("Incomplete joint-noise cell")
    print(f"JOINT_NOISE_CELL_COMPLETE {key}", flush=True)
    return {"key": list(key), "pieces": records, "target_max_abs_difference": label_error, "reproduction_max_abs_difference": maximum}


def summarize(pieces):
    active = [p["noise"] for p in pieces if p["active"]]
    out = {"bank_cell_exposures": len({(tuple(p["cell"]), p["bank"]) for p in pieces}),
        "prefix_model_exposures": len(pieces), "active_prefix_models": len(active), "regions": {}}
    if active:
        sums = {n: sum(p[n] for p in active) for n in ("force_noise_energy", "parameter_noise_energy", "low_parameter_noise_energy", "unsupported_force_noise_energy")}
        def mean_defined(name):
            values = [p[name] for p in active if p[name] is not None]
            return float(np.mean(values)) if values else None
        out["directions"] = {**sums, "inverse_KKT_noise_energy_gain": sums["parameter_noise_energy"]/sums["force_noise_energy"] if sums["force_noise_energy"] else None,
            "low_parameter_noise_fraction": sums["low_parameter_noise_energy"]/sums["parameter_noise_energy"] if sums["parameter_noise_energy"] else None,
            "mean_low_dimension_fraction": mean_defined("low_dimension_fraction"),
            "mean_dominant_covariance_fraction": mean_defined("dominant_covariance_fraction"),
            "mean_dominant_KKT_mode_fraction": mean_defined("dominant_KKT_mode_fraction")}
    for region in REPORT_REGIONS:
        xs = [p["regions"][region] for p in pieces if region in p["regions"]]
        if not xs: continue
        n = sum(x["feasible_action_cases"] for x in xs)
        stats = {"state_model_exposures": sum(x["state_cases"] for x in xs), "feasible_action_exposures": n,
            "action_disagreements": sum(x["disagreements"] for x in xs), "bands": {},
            "low_rest_noise_cross": {f: sum(x["low_rest_noise_cross"][f] for x in xs) for f in learning.POSITIVE}}
        for band in ("full", "low", "rest"):
            ns = {f: sum(x["bands"][band]["noise_squared_sum"][f] for x in xs) for f in learning.COVARIANCES}
            os = {f: sum(x["bands"][band]["observed_squared_sum"][f] for x in xs) for f in (*learning.POSITIVE, *learning.CROSSES, "cross_population_joint")}
            stats["bands"][band] = {"noise_squared_sum": ns, "observed_squared_sum": os,
                "noise_RMS": {f: float(np.sqrt(ns[f]/n)) if n else None for f in learning.POSITIVE},
                "observed_RMS": {f: float(np.sqrt(os[f]/n)) if n else None for f in learning.POSITIVE},
                "observed_over_noise_RMS": {f: float(np.sqrt(os[f]/ns[f])) if ns[f] else None for f in learning.POSITIVE},
                "zero_noise_nonzero_gap_prefix_models": {f: sum(x["bands"][band]["noise_squared_sum"][f]==0 and x["bands"][band]["observed_squared_sum"][f]>1e-24 for x in xs) for f in learning.POSITIVE}}
        out["regions"][region] = stats
    return out


def report(records, expected):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0]==v for v in VIEWS}}
    output, maximum = {}, 0.
    for group, predicate in groups.items():
        ps = [p for r in records if predicate(r["key"]) for p in r["pieces"]]
        output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for variant in PRIMARY:
                output[group][label][variant] = {}
                for cat in CATEGORIES:
                    now = summarize([p for p in ps if p["variant"]==variant and (bank is None or p["bank"]==bank) and in_category(p, cat)])
                    prior = expected["groups"][group][label][variant][cat]
                    for f in ("bank_cell_exposures", "prefix_model_exposures", "active_prefix_models"):
                        if now[f]!=prior[f]: raise ValueError("Changed Stage264 category exposure")
                    for region, before in prior["regions"].items():
                        actual = now["regions"][region]
                        for f in ("state_model_exposures", "feasible_action_exposures", "action_disagreements"):
                            if actual[f]!=before[f]: raise ValueError("Changed Stage264 region exposure")
                        for band, b in before["bands"].items():
                            for field, old in learning.OLD_NAMES.items():
                                if b["RMS"][old] is not None: maximum = max(maximum, abs(actual["bands"][band]["observed_RMS"][field]-b["RMS"][old]))
                    output[group][label][variant][cat] = now
    ps = [p for r in records for p in r["pieces"]]
    if len(records)!=32 or len(ps)!=1280 or maximum>1e-12: raise ValueError("Incomplete Stage265 reproduction")
    focus = {v: summarize([p for p in ps if p["cell"][0].startswith("mixed") and p["variant"]==v
        and tuple(p["removed"])==FOCUS_PREFIX and in_category(p, "removed")]) for v in PRIMARY}
    active = [p["noise"] for p in ps if p["active"]]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "focus_mixed_removed": focus, "reproduction": {"cached_cells": len(records), "prefix_model_records": len(ps), "group_max_abs_difference": maximum,
            "target_max_abs_difference": max(r["target_max_abs_difference"] for r in records),
            "cell_max_abs_difference": max(r["reproduction_max_abs_difference"] for r in records),
            **{name: max(p[name] for p in active) for name in ("reference_max_abs_difference", "covariance_identity_residual", "spectral_propagation_residual", "influence_equation_residual")},
            "prediction_decomposition_max_residual": max(x["decomposition_residual"] for p in ps for x in p["regions"].values())},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    records = [analyze_cell(key, views, panels) for key in KEYS]
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    payload = report(records, expected)
    path = ROOT/"results"/a.run_name/"report/result.json"
    write_json(raw_directory(path)/"cells.json", records)
    write_json(path, payload)
    print("JOINT_NOISE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
