#!/usr/bin/env python3
"""Attribute cached future instability by target head, population and half."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_future_forcing_stage263 as source
from scripts import run_native_transit_prefix_influence_stage259 as cache_io
from freq_hrl.domains.transit import native_head_half_forcing as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_head_half_forcing_stage264_v1"
SOURCE_RUN = "native_transit_future_forcing_stage263_20261016_r1"
cache, diagnostic = source.cache, source.diagnostic
KEYS, VIEWS, PRIMARY, CATEGORIES = source.KEYS, source.VIEWS, source.PRIMARY, source.CATEGORIES
REPORT_REGIONS = ("excluded", "retained", "held")
FOCUS_PREFIX = (397, 913970203)


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "primary": list(PRIMARY), "categories": list(CATEGORIES),
        "frozen": source.contract()["frozen"], "target_heads": "service_nonfleet_and_risk_fleet_target_perturbations_of_total_cost_response_not_observed_outcome_components",
        "population": "split_target_changes_on_excluded6_and_retained54_states; all_four_head_population_parts_and_six_signed_cross_terms",
        "halves": "assigned16_draws_first8_last8_in_each_bank; cross16=(first8+last8)/2; first8-last8=within_anchor-within_other; fixed_reliability_precision_lambda_mu_operator_for_all",
        "gradient": "exact_linear_target_part_of_existing_profiled_quadratic_gradient_with_cached_weights; check_against_Stage263_direct_factory_response; no_parameter_optimization",
        "controls": "reproduce_all_Stage263_all_removed_counts_and_target_swap_RMS_full_low_rest_all_seven_regions; report_three_main_query_regions",
        "focus_prefix": list(FOCUS_PREFIX), "focus_basis": "frozen_Stage263_posthoc_concentration; not_a_training_exclusion_or_policy_gate",
        "labels": "cached_Stage249_training_labels_only; raw_risk_nonzero_counts_are_paired_draw_state_feasible_action_exposures_not_independent_fleet_events",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_prefix_fits": 640, "prefix_model_records": 1280,
            "new_critic_fits": 0, "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "raw_models_labels_and_all_region_cell_ledgers_server_only; pull_scalar_report",
        "boundary": "conditional_linear_response_not_new_policy_outcome_accuracy_noise_variance_or_seed_CI; anchor_dependencies_and_reused_halves_preclude_independent_replication_claim; Stage253_248_169_closed; no_selection_or_default_change"}


def analyze_cell(key, views, panels):
    read = lambda run, file: cache_io.read_cell(run, key, file)
    models, calibrations, actions = (read(source.source.FIT_RUN, f) for f in ("models.json", "calibrations.json", "actions.json"))
    cs = {c["bank"]: c for c in calibrations}
    full_models = read(cache.cached.SOURCE_RUN, "models.json")
    prior = read(cache.cached.information.SOURCE_RUN, "models.json")["full"]["all_mse"]
    reference = read(cache.cached.fixed.contract()["reference_run"], "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    data = {s: cache.learning.base.causal_data(rows, reference) for s, rows in (("train60", train), ("held60", held))}
    geometry = cache.learning.deployment_geometry(data["train60"])
    x = {s: cache.learning.joint.relative(d["x"][..., :198], d["originals"]) for s, d in data.items()}
    residuals, halves, paired, label_error = {}, {}, {}, 0.
    for bank in (0, 1):
        draws, evaluation = cache.sampling.learning.draw_assignment(bank, "expanded")
        if cs[bank]["training_draws"] != draws or len(draws) != 16 or set(draws)&set(evaluation): raise ValueError("Changed disjoint training banks")
        values = cache.learning.base.training_draws(train, panels, draws)
        residuals[bank] = learning.previous.residual_target(values, data["train60"], prior["scale"])
        for half in (0, 1): halves[bank, half] = learning.previous.residual_target(values[:, 8*half:8*(half+1)], data["train60"], prior["scale"])
        v = values[..., (2, 1)]
        paired[bank] = v-v[np.arange(len(train)), :, data["train60"]["originals"]][:, :, None]
        label_error = max(label_error, float(np.max(np.abs(residuals[bank]*np.asarray(cs[bank]["reliability"])-np.asarray(cs[bank]["target"])))))
    if label_error > 1e-12 or set(cs[0]["training_draws"])&set(cs[1]["training_draws"]): raise ValueError("Changed cached targets or bank overlap")
    lookup = {(a["bank"], tuple(a["removed"]), a["variant"], a["scope"]): a for a in actions if a["variant"] in PRIMARY}
    records, labels, maximum = [], [], 0.
    for fitted in models:
        bank, group = fitted["bank"], fitted["held_group"]
        if fitted["training_draws"] != cs[bank]["training_draws"]: raise ValueError("Changed cached prefix bank")
        masks = diagnostic.regions(lookup[bank, tuple(group), PRIMARY[0], "train60"]["states"], "train60", group)
        indices = np.flatnonzero(masks["retained"])
        if (len(indices), int(masks["excluded"].sum())) != (54, 6): raise ValueError("Changed prefix population")
        d = data["train60"]
        feasible = (d["raw_logits"]<=0)&(np.arange(6)[None, :] != d["originals"][:, None])
        labels.append({"cell": list(key), "bank": bank, "removed": group,
            "reliability_mean": np.asarray(cs[bank]["reliability"])[masks["excluded"]].mean(axis=(0, 1)).tolist(),
            "populations": {pop: {f"bank{b}_half{half}": learning.raw_label_stats(paired[b][:, 8*half:8*(half+1)], feasible, masks[pop])
                for b in (0, 1) for half in (0, 1)} for pop in ("excluded", "retained")}})
        for variant in PRIMARY:
            a = lookup[bank, tuple(group), variant, "train60"]
            record = {f: a[f] for f in ("cell", "bank", "removed", "variant", "category", "old_key", "old_key_new_gate")}
            record.update(active=a["old_key"] is not None and a["old_key"] != "zero", regions={})
            if record["active"]:
                model = variant.removesuffix("_expanded8_pooled_safe")
                option = cache.learning.strength.CANDIDATES[a["old_key"]]
                level, alpha = option["level"], option["alpha"]
                basis, coordinates = geometry[level]
                radius = next(f["old_radius"] for f in fitted["fit"]["fits"] if f["level"] == level)
                context = learning.context(x["train60"]@basis, cs[bank], residuals, halves, data["train60"]["raw_logits"]<=0,
                    np.asarray(prior["weights"][level])@coordinates.T,
                    np.asarray(full_models[str(bank)][model]["weights"][level])@coordinates.T,
                    np.asarray(fitted["models"][model]["weights"][level])@coordinates.T,
                    indices, model, radius, bank, level)
                record["forcing"] = context["diagnostics"]
            for scope, d in data.items():
                old = lookup[bank, tuple(group), variant, scope]
                states = [list(cache.sampling.old_future.learning.state_key(r)) for r in (train if scope == "train60" else held)]
                if old["states"] != states or old["raw"] != d["raw_choices"].tolist(): raise ValueError("Changed cached states/reference")
                feasible = (d["raw_logits"]<=0)&(np.arange(6)[None, :] != d["originals"][:, None])
                for region, mask in diagnostic.regions(states, scope, group).items():
                    if not mask.any(): continue
                    s = learning.region_stats(x[scope]@basis, mask[:, None], feasible, context, alpha*prior["scale"]) if record["active"] else {
                        "feasible_action_cases": int(feasible[mask].sum()), "bands": {b: dict.fromkeys(learning.FIELDS, 0.) for b in ("full", "low", "rest")}, "decomposition_residual": 0.}
                    expected = old["function_gaps"]["fit_population"][region]
                    if s["feasible_action_cases"] != expected["feasible_action_cases"]: raise ValueError("Changed feasible actions")
                    if record["active"]:
                        delta = (x[scope]@basis)[mask[:, None]&feasible]@context["delta"]*(alpha*prior["scale"])
                        maximum = max(maximum, abs(float(delta@delta)-expected["squared_sums"]["cost"]))
                    s.update(state_cases=int(mask.sum()), disagreements=int(np.count_nonzero(np.asarray(old["choices"]["fixed"])[mask] != np.asarray(old["choices"]["full"])[mask])))
                    record["regions"][region] = s
            records.append(record)
    if len(models) != 20 or len(records) != 40 or maximum > 1e-12: raise ValueError("Incomplete cached head/half analysis")
    print(f"HEAD_HALF_CELL_COMPLETE {key}", flush=True)
    return {"key": list(key), "pieces": records, "labels": labels, "target_max_abs_difference": label_error, "reproduction_max_abs_difference": maximum}


def summarize(pieces):
    out = {"bank_cell_exposures": len({(tuple(p["cell"]), p["bank"]) for p in pieces}),
        "prefix_model_exposures": len(pieces), "active_prefix_models": sum(p["active"] for p in pieces), "regions": {}}
    for region in diagnostic.REGIONS:
        xs = [p["regions"][region] for p in pieces if region in p["regions"]]
        if not xs: continue
        n = sum(x["feasible_action_cases"] for x in xs)
        stats = {"state_model_exposures": sum(x["state_cases"] for x in xs), "feasible_action_exposures": n,
            "action_disagreements": sum(x["disagreements"] for x in xs), "bands": {}}
        for band in ("full", "low", "rest"):
            sums = {f: sum(x["bands"][band][f] for x in xs) for f in learning.FIELDS}
            halves = {}
            for h in learning.HEADS:
                denominator = np.sqrt(sums[f"first8_{h}"]*sums[f"last8_{h}"])
                defined = [x["bands"][band] for x in xs if x["bands"][band][f"first8_{h}"]>0 and x["bands"][band][f"last8_{h}"]>0]
                halves[h] = {"cosine": sums[f"first_last_dot_{h}"]/denominator if denominator else None,
                    "defined_prefix_models": len(defined), "positive_dot_prefix_models": sum(v[f"first_last_dot_{h}"]>0 for v in defined)}
            stats["bands"][band] = {"squared_response": sums,
                "RMS": {f: float(np.sqrt(sums[f]/n)) if n else None for f in learning.NAMES}, "half_alignment": halves}
        out["regions"][region] = stats
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
                    now = summarize([p for p in ps if p["variant"]==variant and (bank is None or p["bank"]==bank) and cache_io.in_category(p, cat)])
                    prior = expected["groups"][group][label][variant][cat]
                    for f in ("bank_cell_exposures", "prefix_model_exposures", "active_prefix_models"):
                        if now[f] != prior[f]: raise ValueError("Changed Stage263 category exposure")
                    for region, before in prior["regions"].items():
                        actual = now["regions"][region]
                        for f in ("state_model_exposures", "feasible_action_exposures", "action_disagreements"):
                            if actual[f] != before[f]: raise ValueError("Changed Stage263 region exposure")
                        for band, b in before["bands"].items():
                            if b["RMS"]["target_swap"] is not None:
                                maximum = max(maximum, abs(actual["bands"][band]["RMS"]["cross16_joint"]-b["RMS"]["target_swap"]))
                    now["regions"] = {r: s for r, s in now["regions"].items() if r in REPORT_REGIONS}
                    output[group][label][variant][cat] = now
    ps = [p for r in records for p in r["pieces"]]
    if len(records) != 32 or len(ps) != 1280 or maximum > 1e-12: raise ValueError("Incomplete Stage264 reproduction")
    focus = {v: summarize([p for p in ps if p["cell"][0].startswith("mixed") and p["variant"]==v
        and tuple(p["removed"])==FOCUS_PREFIX and cache_io.in_category(p, "removed")]) for v in PRIMARY}
    for s in focus.values(): s["regions"] = {r: v for r, v in s["regions"].items() if r in REPORT_REGIONS}
    active = [p["forcing"] for p in ps if p["active"]]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "focus_mixed_removed": focus,
        "focus_label_diagnostics": [label for r in records for label in r["labels"] if tuple(label["removed"])==FOCUS_PREFIX],
        "reproduction": {"cached_cells": len(records), "prefix_model_records": len(ps), "group_max_abs_difference": maximum,
            "target_max_abs_difference": max(r["target_max_abs_difference"] for r in records),
            "cell_max_abs_difference": max(r["reproduction_max_abs_difference"] for r in records),
            "reference_max_abs_difference": max(p["reference_max_abs_difference"] for p in active),
            "identity_max_residual": max(p["identity_max_residual"] for p in active),
            "influence_equation_max_residual": max(p["influence_equation_residual"] for p in active),
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
    print("HEAD_HALF_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
