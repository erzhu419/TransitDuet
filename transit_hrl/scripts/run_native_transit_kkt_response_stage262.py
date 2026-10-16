#!/usr/bin/env python3
"""Separate saved data-population and RMS-multiplier responses without fits."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_curvature_stage261 as source
from freq_hrl.domains.transit import native_kkt_response as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_kkt_response_stage262_v1"
SOURCE_RUN = "native_transit_curvature_stage261_20261016_r1"
FIT_RUN = source.SOURCE_RUN
cache, diagnostic = source.cache, source.diagnostic
KEYS, VIEWS, PRIMARY, CATEGORIES = source.KEYS, source.VIEWS, source.PRIMARY, source.CATEGORIES


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL, "fit_run": FIT_RUN,
        "cohort": source.contract()["cohort"], "primary": list(PRIMARY), "categories": list(CATEGORIES),
        "frozen": "Stage261_full60_fixed54_models_keys_basis_prior_scale_cap_and_full_calibration",
        "identity": "A=H_prefix+mu_prefix*I; q=grad_prefix(u_full)-grad_full(u_full); d=(mu_prefix-mu_full)*u_full; A*delta+q+d=0",
        "split": "data=-pinv(A)*q; dual=-pinv(A)*d; null=(I-pinv(A)*A)*delta; residual=delta-data-dual-null",
        "support": "symmetric_eigenvalue_tolerance_dimension_times_machine_epsilon_times_maximum_absolute_eigenvalue",
        "bands": source.contract()["curvature"],
        "aggregation": "all_four_component_squares_and_six_signed_cross_terms; full_low_rest_reported_separately; RMS_not_additive_or_causal_shares",
        "controls": "reproduce_full_fixed_cached_primary_actions_and_Stage261_all_groups_banks_categories_counts_cost_low_rest_RMS",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_prefix_fits": 640, "prefix_model_records": 1280,
            "new_critic_fits": 0, "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "detailed_cell_ledgers_and_models_server_only; pull_scalar_report",
        "boundary": "conditional_algebraic_path_attribution_not_independent_interventions_accuracy_or_utility; fixed_calibration_saw_excluded_labels; no_evaluation_panels_or_selection; Stage253_248_169_closed; no_default_change"}


def analyze_cell(key, views):
    read = lambda run, file: source.source.previous.read_cell(run, key, file)
    models, calibrations, actions = (read(FIT_RUN, f) for f in ("models.json", "calibrations.json", "actions.json"))
    full_models = read(cache.cached.SOURCE_RUN, "models.json")
    prior = read(cache.cached.information.SOURCE_RUN, "models.json")["full"]["all_mse"]
    reference = read(cache.cached.fixed.contract()["reference_run"], "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    data = {s: cache.learning.base.causal_data(rows, reference) for s, rows in (("train60", train), ("held60", held))}
    geometry = cache.learning.deployment_geometry(data["train60"])
    x = {s: cache.learning.joint.relative(d["x"][..., :198], d["originals"]) for s, d in data.items()}
    lookup = {(a["bank"], tuple(a["removed"]), a["variant"], a["scope"]): a for a in actions if a["variant"] in PRIMARY}
    records, maximum = [], 0.
    for bank in (0, 1):
        c = next(c for c in calibrations if c["bank"] == bank)
        draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
        if c["training_draws"] != draws: raise ValueError("Changed training calibration bank")
        target, precision = np.asarray(c["target"]), np.asarray(c["precision"])
        full_contexts, full_paths = {}, {}
        for fitted in (m for m in models if m["bank"] == bank):
            group = fitted["held_group"]
            if fitted["training_draws"] != draws: raise ValueError("Changed prefix training bank")
            masks = diagnostic.regions(lookup[bank, tuple(group), PRIMARY[0], "train60"]["states"], "train60", group)
            indices = np.flatnonzero(masks["retained"])
            if (len(indices), int(masks["excluded"].sum())) != (54, 6): raise ValueError("Changed deletion population")
            for variant in PRIMARY:
                a = lookup[bank, tuple(group), variant, "train60"]
                record = {f: a[f] for f in ("cell", "bank", "removed", "variant", "category", "old_key", "old_key_new_gate")}
                record.update(active=a["old_key"] is not None and a["old_key"] != "zero", regions={})
                model = variant.removesuffix("_expanded8_pooled_safe")
                if record["active"]:
                    option = cache.learning.strength.CANDIDATES[a["old_key"]]
                    level, alpha = option["level"], option["alpha"]
                    basis, coordinates = geometry[level]
                    z = x["train60"]@basis
                    initial = np.asarray(prior["weights"][level])@coordinates.T
                    fixed_heads = np.asarray(fitted["models"][model]["weights"][level])@coordinates.T
                    radius = next(f["old_radius"] for f in fitted["fit"]["fits"] if f["level"] == level)
                    calibration = {"target": target, "precision": precision, "penalty": c["penalties"][level]}
                    if variant not in full_contexts:
                        heads = np.asarray(full_models[str(bank)][model]["weights"][level])@coordinates.T
                        full_contexts[variant] = learning.full_context(z, calibration, data["train60"]["raw_logits"] <= 0,
                            initial, heads, model, radius)
                        full_paths[variant] = {s: cache.learning.base.candidates(d, full_models[str(bank)][model])[a["old_key"]] for s, d in data.items()}
                    context = learning.prefix_context(z, calibration, data["train60"]["raw_logits"] <= 0,
                        initial, fixed_heads, indices, model, radius, full_contexts[variant])
                    record["kkt"] = context["diagnostics"]
                for scope, d in data.items():
                    old = lookup[bank, tuple(group), variant, scope]
                    states = [list(cache.sampling.old_future.learning.state_key(r)) for r in (train if scope == "train60" else held)]
                    if old["states"] != states or old["raw"] != d["raw_choices"].tolist(): raise ValueError("Changed cached states or reference")
                    masks = diagnostic.regions(states, scope, group)
                    feasible = (d["raw_logits"] <= 0)&(np.arange(6)[None, :] != d["originals"][:, None])
                    if record["active"]:
                        pred = cache.learning.base.candidates(d, fitted["models"][model])[a["old_key"]]
                        full = full_paths[variant][scope]
                        if pred["chosen"].tolist() != old["choices"]["fixed"] or full["chosen"].tolist() != old["choices"]["full"]:
                            raise ValueError("Changed cached primary actions")
                        zz = x[scope]@basis
                        predicted = zz@context["delta"]*(alpha*prior["scale"])
                        maximum = max(maximum, float(np.max(np.abs(predicted-(pred["cost"]-full["cost"])))))
                    elif any(old["choices"][name] != old["raw"] for name in ("full", "fixed")):
                        raise ValueError("Inactive key changed raw actions")
                    for region, mask in masks.items():
                        if not mask.any(): continue
                        s = learning.region_stats(zz, mask[:, None], feasible, context, alpha*prior["scale"]) if record["active"] else {
                            "feasible_action_cases": int(feasible[mask].sum()), "squared_response": dict.fromkeys(learning.FIELDS, 0.),
                            "bands": {b: dict.fromkeys(learning.FIELDS, 0.) for b in ("low", "rest")}, "decomposition_residual": 0.}
                        expected = old["function_gaps"]["fit_population"][region]
                        if s["feasible_action_cases"] != expected["feasible_action_cases"]: raise ValueError("Changed feasible actions")
                        maximum = max(maximum, abs(s["squared_response"]["total"]-expected["squared_sums"]["cost"]))
                        s.update(state_cases=int(mask.sum()), disagreements=int(np.count_nonzero(
                            np.asarray(old["choices"]["fixed"])[mask] != np.asarray(old["choices"]["full"])[mask])))
                        record["regions"][region] = s
                records.append(record)
    if len(models) != 20 or len(records) != 40 or maximum > 1e-12: raise ValueError("Failed cached cell reproduction")
    print(f"KKT_RESPONSE_CELL_COMPLETE {key}", flush=True)
    return {"key": list(key), "pieces": records, "reproduction_max_abs_difference": maximum}


def sum_squared(records):
    return {f: sum(r[f] for r in records) for f in learning.FIELDS}


def response_summary(sums, n):
    return {"squared_response": sums,
        "RMS": {f: float(np.sqrt(sums[f]/n)) if n else None for f in ("total", *learning.PARTS)}}


def summarize(pieces):
    active = [p["kkt"] for p in pieces if p["active"]]
    out = {"bank_cell_exposures": len({(tuple(p["cell"]), p["bank"]) for p in pieces}),
        "prefix_model_exposures": len(pieces), "active_prefix_models": len(active), "regions": {}}
    if active:
        out["kkt"] = {"mean": {f: float(np.mean([p[f] for p in active])) for f in (
            "full_dual", "prefix_dual", "dual_change", "data_force_norm", "dual_force_norm", "low_dimension_fraction")},
            "full_constraint_active": sum(p["full_dual"] > 0 for p in active),
            "prefix_constraint_active": sum(p["prefix_dual"] > 0 for p in active),
            "dual_increased": sum(p["dual_change"] > 1e-12 for p in active),
            "dual_decreased": sum(p["dual_change"] < -1e-12 for p in active),
            "parameter_squared_response": sum_squared([p["parameter_squared_response"] for p in active])}
    for region in diagnostic.REGIONS:
        xs = [p["regions"][region] for p in pieces if region in p["regions"]]
        if not xs: continue
        n = sum(x["feasible_action_cases"] for x in xs)
        out["regions"][region] = {"state_model_exposures": sum(x["state_cases"] for x in xs),
            "feasible_action_exposures": n, "action_disagreements": sum(x["disagreements"] for x in xs),
            **response_summary(sum_squared([x["squared_response"] for x in xs]), n),
            "bands": {b: response_summary(sum_squared([x["bands"][b] for x in xs]), n) for b in ("low", "rest")}}
    return out


def report(records, expected):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, maximum = {}, 0.
    for group, predicate in groups.items():
        ps = [p for r in records if predicate(r["key"]) for p in r["pieces"]]
        output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for variant in PRIMARY:
                output[group][label][variant] = {}
                for cat in CATEGORIES:
                    now = summarize([p for p in ps if p["variant"] == variant and (bank is None or p["bank"] == bank)
                        and source.source.previous.in_category(p, cat)])
                    prior = expected["groups"][group][label][variant][cat]
                    for f in ("bank_cell_exposures", "prefix_model_exposures", "active_prefix_models"):
                        if now[f] != prior[f]: raise ValueError("Changed Stage261 category exposure")
                    for region, before in prior["regions"].items():
                        actual = now["regions"][region]
                        for f in ("state_model_exposures", "feasible_action_exposures", "action_disagreements"):
                            if actual[f] != before[f]: raise ValueError("Changed Stage261 action/state counts")
                        for band, value in ((None, before["cost_RMS"]), ("low", before["low_RMS"]), ("rest", before["rest_RMS"])):
                            measured = actual["RMS"]["total"] if band is None else actual["bands"][band]["RMS"]["total"]
                            if value is not None: maximum = max(maximum, abs(measured-value))
                    output[group][label][variant][cat] = now
    if maximum > 1e-12: raise ValueError("Stage262 failed Stage261 response reproduction")
    ps = [p for r in records for p in r["pieces"]]
    if len(records) != 32 or len(ps) != 1280: raise ValueError("Incomplete cached matrix")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"cached_cells": len(records), "prefix_model_records": len(ps),
            "cell_max_abs_difference": max(r["reproduction_max_abs_difference"] for r in records), "group_max_abs_difference": maximum,
            "influence_equation_max_residual": max(p["kkt"]["influence_equation_residual"] for p in ps if p["active"]),
            "parameter_decomposition_max_residual": max(p["kkt"]["decomposition_residual"] for p in ps if p["active"]),
            "prediction_decomposition_max_residual": max(s["decomposition_residual"] for p in ps for s in p["regions"].values())},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    records = [analyze_cell(key, views) for key in KEYS]
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    payload = report(records, expected)
    path = ROOT/"results"/a.run_name/"report/result.json"
    write_json(raw_directory(path)/"cells.json", records)
    write_json(path, payload)
    print("KKT_RESPONSE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
