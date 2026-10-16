#!/usr/bin/env python3
"""Joint full/prefix local response with shared cached future noise, no fitting."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_tangent_noise_stage266 as source
from freq_hrl.domains.transit import native_coupled_noise as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_coupled_noise_stage267_v1"
SOURCE_RUN = "native_transit_tangent_noise_stage266_20261016_r1"
cache, KEYS, VIEWS, PRIMARY = source.cache, source.KEYS, source.VIEWS, source.PRIMARY
CATEGORIES, REPORT_REGIONS, FOCUS_PREFIX = source.CATEGORIES, source.REPORT_REGIONS, source.FOCUS_PREFIX
HEADS = ("joint", "service", "risk")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "primary": list(PRIMARY), "categories": list(CATEGORIES),
        "frozen": source.contract()["frozen"], "noise": source.contract()["noise"], "independence": source.contract()["independence"],
        "response": "delta=u_prefix-u_full; d_delta=-T_prefix*d_g_prefix+T_full*d_g_full; each_T_uses_own_cached_H_mu_u_and_same_fixed_radius",
        "terms": "prefix=-Tp*gp; minus_full=+Tf*gf; coupled=prefix+minus_full; one_ended=-Tp*(gp-gf); operator_change=(Tf-Tp)*gf; coupled=one_ended+operator_change",
        "covariance": "shared_state_future_label_scores_propagated_before_statewise_within_bank_covariance; direct_Gram_of_sum; also_direct_signed_endpoint_and_operator_change_cross_terms; never_independent_endpoint_sum",
        "query": "unchanged_Stage266_feasible_queries_and_prefix_H_low_basis; retain_head_population_and_low_rest_cross_terms; scales_are_cost_function_response_not_outcome_improvement",
        "controls": "reproduce_Stage266_one_ended_all_cells_regions_bands_component_covariance_sums_observed_sums_and_exposure_counts; direct_existing_gradient_factory; each_endpoint_KKT_and_both_decompositions",
        "focus_prefix": list(FOCUS_PREFIX), "focus_basis": source.contract()["focus_basis"],
        "budget": source.contract()["budget"], "storage": "raw_labels_models_covariances_cell_ledgers_and_full_band_aggregate_server_only; pull_full_query_scalar_report_with_signed_cross_terms",
        "boundary": "conditional_frozen_calibration_joint_local_refit_response_not_finite_refit_error_ground_truth_accuracy_performance_pvalues_or_seed_CI; report_response_over_radius_not_assume_small_noise; Stage253_248_169_closed; no_selection_or_default_change"}


def scalar_difference(actual, expected):
    if isinstance(expected, dict):
        if actual.keys()!=expected.keys(): raise ValueError("Changed cached scalar structure")
        return max((scalar_difference(actual[k], expected[k]) for k in expected), default=0.)
    if expected is None:
        if actual is not None: raise ValueError("Changed undefined scalar")
        return 0.
    return abs(actual-expected)/max(1., abs(expected))


def analyze_cell(key, views, panels, before):
    read = lambda run, file: source.source.read_cell(run, key, file)
    models, calibrations, actions = (read(source.source.FIT_RUN, f) for f in ("models.json", "calibrations.json", "actions.json"))
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
        residual = source.source.learning.previous.previous.residual_target(values, data["train60"], prior["scale"])
        label_error = max(label_error, float(np.max(np.abs(residual*np.asarray(cs[bank]["reliability"])-np.asarray(cs[bank]["target"])))))
        v = values[..., (2, 1)];paired[bank] = v-v[np.arange(len(train)), :, data["train60"]["originals"]][:, :, None]
    if label_error>1e-12 or set(cs[0]["training_draws"])&set(cs[1]["training_draws"]): raise ValueError("Changed cached target reconstruction")
    fits = {(m["bank"], tuple(m["held_group"])): m for m in models}
    lookup = {(a["bank"], tuple(a["removed"]), a["variant"], a["scope"]): a for a in actions if a["variant"] in PRIMARY}
    records, maximum = [], 0.
    for old_record in before["pieces"]:
        bank, group, variant = old_record["bank"], old_record["removed"], old_record["variant"]
        fitted = fits[bank, tuple(group)]
        if fitted["training_draws"]!=cs[bank]["training_draws"]: raise ValueError("Changed prefix bank")
        a = lookup[bank, tuple(group), variant, "train60"]
        meta = {k: a[k] for k in ("cell", "bank", "removed", "variant", "category", "old_key", "old_key_new_gate")}
        if any(meta[k]!=old_record[k] for k in meta): raise ValueError("Changed Stage266 record")
        record = {**meta, "active": a["old_key"] not in (None, "zero"), "regions": {}}
        if record["active"]!=old_record["active"]: raise ValueError("Changed activity")
        if record["active"]:
            masks = source.source.source.diagnostic.regions(a["states"], "train60", group)
            indices = np.flatnonzero(masks["retained"])
            if (len(indices), int(masks["excluded"].sum()))!=(54, 6): raise ValueError("Changed prefix population")
            model = variant.removesuffix("_expanded8_pooled_safe")
            option = cache.learning.strength.CANDIDATES[a["old_key"]];level, alpha = option["level"], option["alpha"]
            basis, coordinates = geometry[level]
            radius = next(f["old_radius"] for f in fitted["fit"]["fits"] if f["level"]==level)
            ctx = learning.context(x["train60"]@basis, cs[bank], paired, data["train60"]["raw_logits"]<=0,
                np.asarray(prior["weights"][level])@coordinates.T,
                np.asarray(full_models[str(bank)][model]["weights"][level])@coordinates.T,
                np.asarray(fitted["models"][model]["weights"][level])@coordinates.T,
                indices, model, radius, bank, level, prior["scale"])
            record["coupled"] = ctx["diagnostics"]
        for scope, d in data.items():
            old = lookup[bank, tuple(group), variant, scope]
            states = [list(cache.sampling.old_future.learning.state_key(r)) for r in (train if scope=="train60" else held)]
            if old["states"]!=states or old["raw"]!=d["raw_choices"].tolist(): raise ValueError("Changed states/reference")
            feasible = (d["raw_logits"]<=0)&(np.arange(6)[None, :]!=d["originals"][:, None])
            for region, mask in source.source.source.diagnostic.regions(states, scope, group).items():
                if region not in REPORT_REGIONS or not mask.any(): continue
                prior_stats = old_record["regions"][region]["operators"]["tangent"]
                if record["active"]:
                    s = learning.region_stats(x[scope]@basis, mask[:, None], feasible, ctx, alpha*prior["scale"])
                else:
                    zero = {k: v for k, v in prior_stats.items() if k not in ("state_cases", "disagreements")}
                    s = {"terms": {t: zero.copy() for t in learning.TERMS}, "cross": {b: {t: {"noise_squared_sum": dict.fromkeys(learning.noise.COVARIANCES, 0.), "observed_squared_sum": dict.fromkeys(learning.noise.POSITIVE, 0.)} for t in learning.PAIRS} for b in ("full", "low", "rest")}, "decomposition_residual": 0.}
                for term in s["terms"].values():
                    if term["feasible_action_cases"]!=int(feasible[mask].sum()): raise ValueError("Changed feasible-action denominator")
                    term.update(state_cases=int(mask.sum()), disagreements=int(np.count_nonzero(np.asarray(old["choices"]["fixed"])[mask]!=np.asarray(old["choices"]["full"])[mask])))
                maximum = max(maximum, scalar_difference(s["terms"]["one_ended"], prior_stats))
                record["regions"][region] = s
        records.append(record)
    if len(models)!=20 or len(records)!=40 or maximum>1e-9: raise ValueError("Stage267 failed cached one-ended reproduction")
    print(f"COUPLED_NOISE_CELL_COMPLETE {key}", flush=True)
    return {"key": list(key), "pieces": records, "target_max_abs_difference": label_error, "baseline_max_relative_difference": maximum}


def summarize(pieces):
    active = [p["coupled"] for p in pieces if p["active"]]
    energies = {t: sum(c["parameter_noise_energy"][t] for c in active) for t in learning.TERMS}
    radii = sum(c["radius"]**2 for c in active)
    out = {"prefix_model_exposures": len(pieces), "inactive_keys": sum(not p["active"] for p in pieces),
        "activity_pairs": dict(Counter(c["full"]["constraint_state"]+"/"+c["prefix"]["constraint_state"] for c in active)),
        "parameter_noise_energy": energies, "parameter_noise_cross": {k: sum(c["parameter_noise_cross"][k] for c in active) for k in learning.PAIRS},
        "noise_norm_RMS_over_radius_RMS": {t: float(np.sqrt(e/radii)) if radii else None for t, e in energies.items()},
        "above_radius_prefix_models": {t: sum(c["noise_norm_RMS_over_radius"][t]>1 for c in active) for t in learning.TERMS}, "regions": {}}
    for region in REPORT_REGIONS:
        xs = [p["regions"][region] for p in pieces if region in p["regions"]]
        if not xs: continue
        n = sum(x["terms"]["coupled"]["feasible_action_cases"] for x in xs)
        stats = {"feasible_action_exposures": n, "terms": {}, "cross": {}}
        for term in learning.TERMS:
            stats["terms"][term] = {"bands": {}, "low_rest_noise_cross": {h: sum(x["terms"][term]["low_rest_noise_cross"][h] for x in xs) for h in HEADS}}
            for band in ("full", "low", "rest"):
                ns = {h: sum(x["terms"][term]["bands"][band]["noise_squared_sum"][h] for x in xs) for h in learning.noise.COVARIANCES}
                os = {h: sum(x["terms"][term]["bands"][band]["observed_squared_sum"][h] for x in xs) for h in HEADS}
                stats["terms"][term]["bands"][band] = {"noise_squared_sum": ns, "observed_squared_sum": os,
                    "noise_RMS": {h: float(np.sqrt(ns[h]/n)) if n else None for h in HEADS},
                    "observed_RMS": {h: float(np.sqrt(os[h]/n)) if n else None for h in HEADS},
                    "observed_over_noise_RMS": {h: float(np.sqrt(os[h]/ns[h])) if ns[h] else None for h in HEADS}}
        for band in ("full", "low", "rest"):
            stats["cross"][band] = {t: {f: {h: sum(x["cross"][band][t][f][h] for x in xs) for h in HEADS} for f in ("noise_squared_sum", "observed_squared_sum")} for t in learning.PAIRS}
        q = lambda t, h: stats["terms"][t]["bands"]["full"]["noise_squared_sum"][h]
        stats["coupled_over_one_ended_variance"] = {h: q("coupled", h)/q("one_ended", h) if q("one_ended", h) else None for h in HEADS}
        stats["coupled_over_independent_endpoint_variance"] = {h: q("coupled", h)/(q("prefix", h)+q("minus_full", h)) if q("prefix", h)+q("minus_full", h) else None for h in HEADS}
        out["regions"][region] = stats
    return out


def report(records, expected):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0]==v for v in VIEWS}}
    output, maximum = {}, 0.
    for group, predicate in groups.items():
        ps = [p for r in records if predicate(r["key"]) for p in r["pieces"]];output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for variant in PRIMARY:
                output[group][label][variant] = {}
                for cat in CATEGORIES:
                    now = summarize([p for p in ps if p["variant"]==variant and (bank is None or p["bank"]==bank) and source.source.in_category(p, cat)])
                    before = expected["groups"][group][label][variant][cat]
                    if (now["prefix_model_exposures"], now["inactive_keys"])!=(before["prefix_model_exposures"], before["inactive_keys"]): raise ValueError("Changed category exposure")
                    for region, old in before["regions"].items():
                        actual = now["regions"][region]
                        if actual["feasible_action_exposures"]!=old["feasible_action_exposures"]: raise ValueError("Changed query exposure")
                        maximum = max(maximum, scalar_difference(actual["terms"]["one_ended"], old["operators"]["tangent"]))
                    output[group][label][variant][cat] = now
    ps = [p for r in records for p in r["pieces"]]
    if len(records)!=32 or len(ps)!=1280 or maximum>1e-9: raise ValueError("Incomplete Stage267 reproduction")
    focus = [p for p in ps if p["cell"][0].startswith("mixed") and tuple(p["removed"])==FOCUS_PREFIX and source.source.in_category(p, "removed")]
    active = [p["coupled"] for p in ps if p["active"]]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "focus_mixed_removed": {v: summarize([p for p in focus if p["variant"]==v]) for v in PRIMARY},
        "focus_contexts": [{**{k: p[k] for k in ("cell", "bank", "removed", "variant", "old_key", "coupled")}, "summary": summarize([p])} for p in focus if p["active"]],
        "reproduction": {"cached_cells": len(records), "prefix_model_records": len(ps), "baseline_cell_max_relative_difference": max(r["baseline_max_relative_difference"] for r in records),
            "baseline_group_max_relative_difference": maximum, "target_max_abs_difference": max(r["target_max_abs_difference"] for r in records),
            **{k: max(c[k] for c in active) for k in ("endpoint_optimality_residual", "covariance_residual", "response_identity_residual", "direct_gradient_residual")},
            "tangent_equation_max_residual": max(c[e][k] for c in active for e in ("full", "prefix") for k in ("stationarity_residual", "tangency_residual")),
            "query_decomposition_residual": max(s["decomposition_residual"] for p in ps for s in p["regions"].values())},
        "default_changed": False, "native_deployment_eligible": False}


def compact_report(payload):
    def summary(s):
        return {**s, "regions": {r: {**v, "terms": {t: {**a, "bands": {"full": a["bands"]["full"]}} for t, a in v["terms"].items()},
            "cross": {"full": v["cross"]["full"]}} for r, v in s["regions"].items()}}
    return {**payload, "groups": {g: {b: {v: {c: summary(s) for c, s in cats.items()} for v, cats in variants.items()} for b, variants in banks.items()} for g, banks in payload["groups"].items()},
        "focus_mixed_removed": {v: summary(s) for v, s in payload["focus_mixed_removed"].items()},
        "focus_contexts": [{**c, "summary": summary(c["summary"])} for c in payload["focus_contexts"]]}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report_raw/cells.json").read_text())
    lookup = {tuple(r["key"]): r for r in before}
    if set(lookup)!=set(KEYS) or len(before)!=32: raise ValueError("Incomplete Stage266 cell ledger")
    records = [analyze_cell(key, views, panels, lookup[key]) for key in KEYS]
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    payload = report(records, expected)
    path = ROOT/"results"/a.run_name/"report/result.json"
    write_json(raw_directory(path)/"cells.json", records)
    write_json(raw_directory(path)/"aggregate.json", payload)
    write_json(path, compact_report(payload))
    print("COUPLED_NOISE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
