#!/usr/bin/env python3
"""Compare cached future targets and GLS weights without fitting or selection."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_kkt_response_stage262 as source
from freq_hrl.domains.transit import native_future_forcing as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_future_forcing_stage263_v1"
SOURCE_RUN = "native_transit_kkt_response_stage262_20261016_r1"
cache, diagnostic = source.cache, source.diagnostic
KEYS, VIEWS, PRIMARY = source.KEYS, source.VIEWS, source.PRIMARY
CATEGORIES = ("all", "removed")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "primary": list(PRIMARY), "categories": list(CATEGORIES),
        "frozen": "anchor_bank_full_coefficients_keys_basis_prior_scale_reliability_lambda_prefix_mu_and_cap; same_prefix_and_fixed_A_for_all_contrasts",
        "targets": "reconstruct_unshrunk_residual_means_from_cached_Stage249_training_state_futures; apply_anchor_reliability_to_both_disjoint16_banks; reproduce_cached_anchor_target",
        "factorial": "2x2_future_target_bank_and_cached_GLS_precision_bank; target_swap_plus_precision_swap_plus_interaction_equals_joint_swap_at_same_full_point_and_inverse_KKT_operator",
        "split": "anchor_data_response=zero_target_design_prior+two_bank_target_mean+two_bank_target_half_contrast; preserve_three_cross_terms",
        "controls": "reproduce_Stage262_anchor_data_full_low_rest_RMS_and_all_removed_exposure_counts; verify_cached_target_and_Stage260_full_fixed_cost_gap",
        "panels": "server_decompresses_existing_Stage249_label_archive_once; only_each_cell_training_state_labels_used; held_state_causal_features_only; no_held_outcome_evaluation",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_prefix_fits": 640, "prefix_model_records": 1280,
            "new_critic_fits": 0, "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "models_labels_vectors_and_cell_ledgers_server_only; pull_scalar_report",
        "boundary": "conditional_directional_stability_not_ground_truth_signal_noise_estimation_accuracy_utility_or_cap_counterfactual; anchor_model_keys_calibration_depend_on_anchor_bank; Stage253_248_169_closed; no_selection_or_default_change"}


def analyze_cell(key, views, panels):
    read = lambda run, file: source.source.source.previous.read_cell(run, key, file)
    models, calibrations, actions = (read(source.FIT_RUN, f) for f in ("models.json", "calibrations.json", "actions.json"))
    cs = {c["bank"]: c for c in calibrations}
    full_models = read(cache.cached.SOURCE_RUN, "models.json")
    prior = read(cache.cached.information.SOURCE_RUN, "models.json")["full"]["all_mse"]
    reference = read(cache.cached.fixed.contract()["reference_run"], "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in (("train60", train), ("held60", held))}
    geometry = cache.learning.deployment_geometry(data["train60"])
    x = {s: cache.learning.joint.relative(d["x"][..., :198], d["originals"]) for s, d in data.items()}
    residuals, maximum, label_error, records = {}, 0., 0., []
    for bank in (0, 1):
        draws, evaluation = cache.sampling.learning.draw_assignment(bank, "expanded")
        if cs[bank]["training_draws"] != draws or set(draws)&set(evaluation): raise ValueError("Changed disjoint future banks")
        residuals[bank] = learning.residual_target(cache.learning.base.training_draws(train, panels, draws), data["train60"], prior["scale"])
        label_error = max(label_error, float(np.max(np.abs(residuals[bank]*np.asarray(cs[bank]["reliability"])-np.asarray(cs[bank]["target"])))))
    if label_error > 1e-12: raise ValueError("Unshrunk labels do not reproduce cached targets")
    if set(cs[0]["training_draws"])&set(cs[1]["training_draws"]): raise ValueError("Future banks overlap")
    lookup = {(a["bank"], tuple(a["removed"]), a["variant"], a["scope"]): a for a in actions if a["variant"] in PRIMARY}
    for fitted in models:
        bank, group = fitted["bank"], fitted["held_group"]
        if fitted["training_draws"] != cs[bank]["training_draws"]: raise ValueError("Changed cached prefix fit bank")
        masks = diagnostic.regions(lookup[bank, tuple(group), PRIMARY[0], "train60"]["states"], "train60", group)
        indices = np.flatnonzero(masks["retained"])
        if (len(indices), int(masks["excluded"].sum())) != (54, 6): raise ValueError("Changed deletion population")
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
                context = learning.context(x["train60"]@basis, cs, residuals, data["train60"]["raw_logits"] <= 0,
                    np.asarray(prior["weights"][level])@coordinates.T,
                    np.asarray(full_models[str(bank)][model]["weights"][level])@coordinates.T,
                    np.asarray(fitted["models"][model]["weights"][level])@coordinates.T,
                    indices, model, radius, bank, level)
                record["forcing"] = context["diagnostics"]
            for scope, d in data.items():
                old = lookup[bank, tuple(group), variant, scope]
                states = [list(cache.sampling.old_future.learning.state_key(r)) for r in (train if scope == "train60" else held)]
                if states != old["states"] or d["raw_choices"].tolist() != old["raw"]: raise ValueError("Changed state order or reference")
                feasible = (d["raw_logits"] <= 0)&(np.arange(6)[None, :] != d["originals"][:, None])
                for region, mask in diagnostic.regions(states, scope, group).items():
                    if not mask.any(): continue
                    s = learning.region_stats(x[scope]@basis, mask[:, None], feasible, context, alpha*prior["scale"]) if record["active"] else {
                        "feasible_action_cases": int(feasible[mask].sum()), "bands": {b: dict.fromkeys(learning.FIELDS, 0.) for b in ("full", "low", "rest")},
                        "decomposition_residual": 0.}
                    expected = old["function_gaps"]["fit_population"][region]
                    if s["feasible_action_cases"] != expected["feasible_action_cases"]: raise ValueError("Changed feasible cases")
                    if record["active"]:
                        delta = (x[scope]@basis)[mask[:, None]&feasible]@context["delta"]*(alpha*prior["scale"])
                        maximum = max(maximum, abs(float(delta@delta)-expected["squared_sums"]["cost"]))
                    s.update(state_cases=int(mask.sum()), disagreements=int(np.count_nonzero(np.asarray(old["choices"]["fixed"])[mask] != np.asarray(old["choices"]["full"])[mask])))
                    record["regions"][region] = s
            records.append(record)
    if len(models) != 20 or len(records) != 40 or maximum > 1e-12: raise ValueError("Cached model response failed reproduction")
    print(f"FUTURE_FORCING_CELL_COMPLETE {key}", flush=True)
    return {"key": list(key), "pieces": records, "target_max_abs_difference": label_error, "reproduction_max_abs_difference": maximum}


def summarize(pieces):
    active = [p["forcing"] for p in pieces if p["active"]]
    out = {"bank_cell_exposures": len({(tuple(p["cell"]), p["bank"]) for p in pieces}),
        "prefix_model_exposures": len(pieces), "active_prefix_models": len(active), "regions": {}}
    for name in ("force_alignment", "target_force_alignment"):
        xs = [p[name] for p in active]
        sums = {f: sum(x[f] for x in xs) for f in ("anchor_squared", "other_squared", "dot")}
        denominator = np.sqrt(sums["anchor_squared"]*sums["other_squared"])
        out[name] = {**sums, "cosine": sums["dot"]/denominator if denominator else None,
            "defined_prefix_models": sum(x["cosine"] is not None for x in xs),
            "positive_dot_prefix_models": sum(x["cosine"] is not None and x["dot"] > 0 for x in xs)}
    for region in diagnostic.REGIONS:
        xs = [p["regions"][region] for p in pieces if region in p["regions"]]
        if not xs: continue
        n = sum(x["feasible_action_cases"] for x in xs)
        stats = {"state_model_exposures": sum(x["state_cases"] for x in xs), "feasible_action_exposures": n,
            "action_disagreements": sum(x["disagreements"] for x in xs), "bands": {}}
        for band in ("full", "low", "rest"):
            sums = {f: sum(x["bands"][band][f] for x in xs) for f in learning.FIELDS}
            denominator = 2*np.sqrt(sums["anchor"]*sums["other"])
            stats["bands"][band] = {"squared_response": sums,
                "RMS": {f: float(np.sqrt(sums[f]/n)) if n else None for f in learning.NAMES},
                "anchor_other_cosine": sums["cross_anchor_other"]/denominator if denominator else None}
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
                    now = summarize([p for p in ps if p["variant"] == variant and (bank is None or p["bank"] == bank)
                        and source.source.source.previous.in_category(p, cat)])
                    prior = expected["groups"][group][label][variant][cat]
                    for f in ("bank_cell_exposures", "prefix_model_exposures", "active_prefix_models"):
                        if now[f] != prior[f]: raise ValueError("Changed Stage262 category exposure")
                    for region, before in prior["regions"].items():
                        actual = now["regions"][region]
                        for f in ("state_model_exposures", "feasible_action_exposures", "action_disagreements"):
                            if actual[f] != before[f]: raise ValueError("Changed Stage262 state/action exposures")
                        for band, target in (("full", before), *before["bands"].items()):
                            value = target["RMS"]["data"]
                            if value is not None: maximum = max(maximum, abs(actual["bands"][band]["RMS"]["anchor"]-value))
                    output[group][label][variant][cat] = now
    ps = [p for r in records for p in r["pieces"]]
    if len(records) != 32 or len(ps) != 1280 or maximum > 1e-12: raise ValueError("Incomplete Stage263 reproduction")
    active = [p["forcing"] for p in ps if p["active"]]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"cached_cells": len(records), "prefix_model_records": len(ps), "group_max_abs_difference": maximum,
            "target_max_abs_difference": max(r["target_max_abs_difference"] for r in records),
            "cell_max_abs_difference": max(r["reproduction_max_abs_difference"] for r in records),
            "influence_equation_max_residual": max(p["influence_equation_residual"] for p in active),
            "force_response_max_abs_difference": max(p["response_reproduction_error"] for p in active),
            "force_split_max_residual": max(p["split_error"] for p in active),
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
    print("FUTURE_FORCING_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
