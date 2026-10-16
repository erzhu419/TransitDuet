#!/usr/bin/env python3
"""Compare fixed-multiplier and active-RMS tangent noise on frozen caches."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_joint_noise_stage265 as source
from freq_hrl.domains.transit import native_tangent_noise as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_tangent_noise_stage266_v1"
SOURCE_RUN = "native_transit_joint_noise_stage265_20261016_r1"
cache, KEYS, VIEWS, PRIMARY = source.cache, source.KEYS, source.VIEWS, source.PRIMARY
CATEGORIES, REPORT_REGIONS, FOCUS_PREFIX = source.CATEGORIES, source.REPORT_REGIONS, source.FOCUS_PREFIX
OPS, HEADS = ("fixed", "tangent", "radial"), ("joint", "service", "risk")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "primary": list(PRIMARY), "categories": list(CATEGORIES),
        "frozen": source.contract()["frozen"], "noise": source.contract()["noise"], "independence": source.contract()["independence"],
        "operators": "B=pinv(H_prefix+mu_prefix*I); strictly_active_T=B-Bu*Bu.T/(u.T*Bu); interior_T=B; du=-T*dg; dmu=-u.T*B*dg/(u.T*Bu); u=cached_prefix_total",
        "comparison": "same_Stage265_prefix_minus_full_gradient_force_covariance_and_finite_bank_contrast_under_two_operators; not_simultaneous_full_prefix_refit_covariance; fixed_radius_and_active_set_local_derivative",
        "decomposition": "B_response=tangent+radial; C_fixed=C_tangent+C_radial+signed_cross; retain_service_risk_population_and_low_rest_cross_covariance; query_noise_need_not_decrease",
        "controls": "reproduce_Stage265_all_group_bank_category_region_band_scalar_sums_counts_and_RMS; verify_KKT_stationarity_tangency_and_covariance_identities; no_fits_or_action_changes",
        "focus_prefix": list(FOCUS_PREFIX), "focus_basis": "unchanged_Stage263_designation; separate_contexts_and_constraint_activity_not_matched_rank_ablation",
        "budget": source.contract()["budget"], "storage": "all_labels_models_matrices_and_cell_ledgers_server_only; pull_compact_scalar_report",
        "boundary": "conditional_plugin_local_noise_and_finite_forcing_response_not_finite_refit_error_ground_truth_performance_pvalues_or_seed_CI; frozen_anchor_calibration_and_keys; no_selection_or_default_change; Stage253_248_169_closed"}


def analyze_cell(key, views, panels):
    read = lambda run, file: source.read_cell(run, key, file)
    models, calibrations, actions = (read(source.FIT_RUN, f) for f in ("models.json", "calibrations.json", "actions.json"))
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
        residual = source.learning.previous.previous.residual_target(values, data["train60"], prior["scale"])
        label_error = max(label_error, float(np.max(np.abs(residual*np.asarray(cs[bank]["reliability"])-np.asarray(cs[bank]["target"])))))
        v = values[..., (2, 1)]
        paired[bank] = v-v[np.arange(len(train)), :, data["train60"]["originals"]][:, :, None]
    if label_error>1e-12 or set(cs[0]["training_draws"])&set(cs[1]["training_draws"]): raise ValueError("Changed target/bank reconstruction")
    lookup = {(a["bank"], tuple(a["removed"]), a["variant"], a["scope"]): a for a in actions if a["variant"] in PRIMARY}
    records, maximum = [], 0.
    for fitted in models:
        bank, group = fitted["bank"], fitted["held_group"]
        if fitted["training_draws"]!=cs[bank]["training_draws"]: raise ValueError("Changed prefix bank")
        masks = source.source.diagnostic.regions(lookup[bank, tuple(group), PRIMARY[0], "train60"]["states"], "train60", group)
        indices = np.flatnonzero(masks["retained"])
        if (len(indices), int(masks["excluded"].sum()))!=(54, 6): raise ValueError("Changed prefix population")
        for variant in PRIMARY:
            a = lookup[bank, tuple(group), variant, "train60"]
            record = {f: a[f] for f in ("cell", "bank", "removed", "variant", "category", "old_key", "old_key_new_gate")}
            record.update(active=a["old_key"] not in (None, "zero"), regions={})
            if record["active"]:
                model = variant.removesuffix("_expanded8_pooled_safe")
                option = cache.learning.strength.CANDIDATES[a["old_key"]]
                level, alpha = option["level"], option["alpha"]
                basis, coordinates = geometry[level]
                radius = next(f["old_radius"] for f in fitted["fit"]["fits"] if f["level"]==level)
                contexts = learning.context(x["train60"]@basis, cs[bank], paired, data["train60"]["raw_logits"]<=0,
                    np.asarray(prior["weights"][level])@coordinates.T,
                    np.asarray(full_models[str(bank)][model]["weights"][level])@coordinates.T,
                    np.asarray(fitted["models"][model]["weights"][level])@coordinates.T,
                    indices, model, radius, bank, level, prior["scale"])
                record.update(noise=contexts["fixed"]["diagnostics"], constraint=contexts["diagnostics"])
            for scope, d in data.items():
                old = lookup[bank, tuple(group), variant, scope]
                states = [list(cache.sampling.old_future.learning.state_key(r)) for r in (train if scope=="train60" else held)]
                if old["states"]!=states or old["raw"]!=d["raw_choices"].tolist(): raise ValueError("Changed states/reference")
                feasible = (d["raw_logits"]<=0)&(np.arange(6)[None, :]!=d["originals"][:, None])
                for region, mask in source.source.diagnostic.regions(states, scope, group).items():
                    if region not in REPORT_REGIONS or not mask.any(): continue
                    if record["active"]:
                        s = learning.region_stats(x[scope]@basis, mask[:, None], feasible, contexts, alpha*prior["scale"])
                        delta = (x[scope]@basis)[mask[:, None]&feasible]@contexts["fixed"]["delta"]*(alpha*prior["scale"])
                        maximum = max(maximum, abs(float(delta@delta)-old["function_gaps"]["fit_population"][region]["squared_sums"]["cost"]))
                    else:
                        zero = {"feasible_action_cases": int(feasible[mask].sum()), "bands": {b: {
                            "noise_squared_sum": dict.fromkeys(source.learning.COVARIANCES, 0.),
                            "observed_squared_sum": dict.fromkeys((*source.learning.POSITIVE, *source.learning.CROSSES, "cross_population_joint"), 0.)} for b in ("full", "low", "rest")},
                            "low_rest_noise_cross": dict.fromkeys(source.learning.POSITIVE, 0.), "decomposition_residual": 0.}
                        s = {"operators": {op: zero for op in OPS}, "cross": {b: dict.fromkeys(source.learning.COVARIANCES, 0.) for b in ("full", "low", "rest")}, "decomposition_residual": 0.}
                    for op in OPS:
                        st = s["operators"][op]
                        if st["feasible_action_cases"]!=old["function_gaps"]["fit_population"][region]["feasible_action_cases"]: raise ValueError("Changed feasible actions")
                        st.update(state_cases=int(mask.sum()), disagreements=int(np.count_nonzero(np.asarray(old["choices"]["fixed"])[mask]!=np.asarray(old["choices"]["full"])[mask])))
                    record["regions"][region] = s
            records.append(record)
    if len(models)!=20 or len(records)!=40 or maximum>1e-12: raise ValueError("Incomplete Stage266 cell")
    print(f"TANGENT_NOISE_CELL_COMPLETE {key}", flush=True)
    return {"key": list(key), "pieces": records, "target_max_abs_difference": label_error, "reproduction_max_abs_difference": maximum}


def baseline_records(records):
    return [{**r, "pieces": [{**p, "regions": {n: s["operators"]["fixed"] for n, s in p["regions"].items()}} for p in r["pieces"]]} for r in records]


def summarize(pieces):
    active = [p["constraint"] for p in pieces if p["active"]]
    sums = {n: sum(p[n] for p in active) for n in ("fixed_parameter_noise_energy", "tangent_parameter_noise_energy", "radial_parameter_noise_energy", "parameter_noise_cross", "dual_noise_variance")}
    out = {"prefix_model_exposures": len(pieces), "constraint_counts": dict(Counter(p["constraint_state"] for p in active)),
        "inactive_keys": sum(not p["active"] for p in pieces), "directions": sums, "regions": {}}
    for region in REPORT_REGIONS:
        xs = [p["regions"][region] for p in pieces if region in p["regions"]]
        if not xs: continue
        n = sum(x["operators"]["fixed"]["feasible_action_cases"] for x in xs)
        stats = {"feasible_action_exposures": n, "operators": {}, "cross": {b: {h: sum(x["cross"][b][h] for x in xs) for h in HEADS} for b in ("full", "low", "rest")}}
        for op in OPS:
            bands = {}
            for band in ("full", "low", "rest"):
                ns = {h: sum(x["operators"][op]["bands"][band]["noise_squared_sum"][h] for x in xs) for h in source.learning.COVARIANCES}
                os = {h: sum(x["operators"][op]["bands"][band]["observed_squared_sum"][h] for x in xs) for h in HEADS}
                bands[band] = {"noise_squared_sum": ns, "observed_squared_sum": os,
                    "noise_RMS": {h: float(np.sqrt(ns[h]/n)) if n else None for h in HEADS},
                    "observed_RMS": {h: float(np.sqrt(os[h]/n)) if n else None for h in HEADS},
                    "observed_over_noise_RMS": {h: float(np.sqrt(os[h]/ns[h])) if ns[h] else None for h in HEADS}}
            stats["operators"][op] = {"bands": bands,
                "low_rest_noise_cross": {h: sum(x["operators"][op]["low_rest_noise_cross"][h] for x in xs) for h in HEADS}}
        stats["tangent_over_fixed_variance"] = {h: stats["operators"]["tangent"]["bands"]["full"]["noise_squared_sum"][h]/stats["operators"]["fixed"]["bands"]["full"]["noise_squared_sum"][h]
            if stats["operators"]["fixed"]["bands"]["full"]["noise_squared_sum"][h] else None for h in HEADS}
        stats["query_variance_increased_prefix_models"] = {h: sum(x["operators"]["tangent"]["bands"]["full"]["noise_squared_sum"][h]>x["operators"]["fixed"]["bands"]["full"]["noise_squared_sum"][h]*(1+1e-8) for x in xs) for h in HEADS}
        out["regions"][region] = stats
    return out


def report(records, expected):
    base = baseline_records(records)
    maximum = 0.
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0]==v for v in VIEWS}}
    output = {}
    for group, predicate in groups.items():
        ps = [p for r in records if predicate(r["key"]) for p in r["pieces"]]
        bs = [p for r in base if predicate(r["key"]) for p in r["pieces"]]
        output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for variant in PRIMARY:
                output[group][label][variant] = {}
                for cat in CATEGORIES:
                    selected = lambda p: p["variant"]==variant and (bank is None or p["bank"]==bank) and source.in_category(p, cat)
                    before = expected["groups"][group][label][variant][cat]
                    now = source.summarize([p for p in bs if selected(p)])
                    def compare(a, b):
                        nonlocal maximum
                        if isinstance(b, dict):
                            if a.keys()!=b.keys(): raise ValueError("Changed Stage265 scalar structure")
                            for k in b: compare(a[k], b[k])
                        elif b is None:
                            if a is not None: raise ValueError("Changed undefined Stage265 scalar")
                        else:
                            error = abs(a-b)/max(1., abs(b));maximum = max(maximum, error)
                            if error>1e-10: raise ValueError("Changed Stage265 baseline scalar")
                    compare(now, before)
                    output[group][label][variant][cat] = summarize([p for p in ps if selected(p)])
    ps = [p for r in records for p in r["pieces"]]
    if len(records)!=32 or len(ps)!=1280: raise ValueError("Incomplete Stage266 matrix")
    focus = [p for p in ps if p["cell"][0].startswith("mixed") and tuple(p["removed"])==FOCUS_PREFIX and source.in_category(p, "removed")]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "focus_mixed_removed": {v: summarize([p for p in focus if p["variant"]==v]) for v in PRIMARY},
        "focus_contexts": [{**{k: p[k] for k in ("cell", "bank", "removed", "variant", "old_key", "constraint")}, "summary": summarize([p])} for p in focus if p["active"]],
        "activity": {state: {v: summarize([p for p in ps if p["variant"]==v and p["active"] and p["constraint"]["constraint_state"]==state]) for v in PRIMARY} for state in ("interior", "strictly_active")},
        "reproduction": {"cached_cells": len(records), "prefix_model_records": len(ps), "baseline_max_relative_difference": maximum,
            "target_max_abs_difference": max(r["target_max_abs_difference"] for r in records), "cell_max_abs_difference": max(r["reproduction_max_abs_difference"] for r in records),
            **{k: max(p["constraint"][k] for p in ps if p["active"]) for k in ("stationarity_residual", "tangency_residual", "covariance_residual")},
            "query_decomposition_residual": max(s["decomposition_residual"] for p in ps for s in p["regions"].values())},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    records = [analyze_cell(key, views, panels) for key in KEYS]
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    payload = report(records, expected)
    path = ROOT/"results"/a.run_name/"report/result.json"
    write_json(raw_directory(path)/"cells.json", records)
    write_json(path, payload)
    print("TANGENT_NOISE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
