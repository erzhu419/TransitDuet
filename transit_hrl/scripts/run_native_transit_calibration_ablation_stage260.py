#!/usr/bin/env python3
"""Separate prefix-fit population changes from GLS/shrinkage recalibration."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_prefix_influence_stage259 as previous
from freq_hrl.domains.transit import native_calibration_ablation as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

source, diagnostic = previous.source, previous.diagnostic
KEYS, VIEWS, PREFLIGHT, result_path = source.KEYS, source.VIEWS, source.PREFLIGHT, source.result_path
SOURCE_RUN = "native_transit_prefix_influence_stage259_20261016_r1"
EXPERIMENT_PROTOCOL = "native_transit_calibration_ablation_stage260_v1"
PRIMARY, CATEGORIES = previous.PRIMARY, previous.CATEGORIES


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "primary": list(PRIMARY), "categories": list(CATEGORIES),
        "path": "cached_full60 -> prefix54_with_full60_calibration -> cached_prefix54_with_refit_calibration",
        "calibration": "full60_assigned_training16_reliability_state_balanced_pair_GLS_precision_and_cached_full_rankwise_penalties_fixed_and_sliced; no_subset_renormalization",
        "frozen": "same16_training_futures_198_features_full_basis_historical_prior_scale_RMS_cap_old_keys_Stage255_categories; no_selection_or_full_raw_fits",
        "controls": "cached_full_KKT_and_profiled_head_function_without_solving_full; cached_refit_full_actions_and_all_Stage259_region_outcomes_reproduce",
        "evaluation": "same_opposite16_first8_last8; all_new_models_and_actions_written_before_evaluation_draw_use",
        "diagnostics": "fit_population_plus_recalibration_equals_total_for_physical_effects; action_transition_partition_and_function_RMS_not_additive; same_other_root_retained_and_held",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "prefix54_fits": 640, "inner_solves": 5120,
            "full_KKT_checks": 512, "new_full_critic_fits": 0, "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "server_models_actions_calibration_arrays_and_cell_ledgers_only; pull_receipt_and_scalar_report",
        "boundary": "diagnostic_frozen_calibration_uses_excluded_prefix_training_labels_not_strict_OOF_or_deployable_policy; development_and_repeated_exposures_not_seed_CI; ordered_path_not_unique_causal_share_or_individual_weight_lambda_attribution; Stage253_248_169_closed; no_default_change"}


def fit_cell(run, key, views):
    read = lambda name, file: previous.read_cell(name, key, file)
    before = read(previous.SOURCE_RUN, "aggregate.json")
    if before["protocol"] != source.EXPERIMENT_PROTOCOL or before["contract"] != source.contract() or not before["coverage_qualified"]:
        raise ValueError("Require completed Stage258 cell")
    adaptive = read(previous.SOURCE_RUN, "models.json")
    old_actions = read(SOURCE_RUN, "actions.json")
    full_models = read(source.cached.SOURCE_RUN, "models.json")
    full_info = read(source.cached.SOURCE_RUN, "aggregate.json")
    prior = read(source.cached.information.SOURCE_RUN, "models.json")["full"]["all_mse"]
    cap = read(source.cached.fixed.SOURCE_RUN, "fits.json")["full"]
    reference = read(source.cached.fixed.contract()["reference_run"], "models.json")["reference"]
    train, held = source.cached.training.cohort.split_rows(views[key[0]], key[1])
    data = {s: source.learning.base.causal_data(rows, reference) for s, rows in (("train60", train), ("held60", held))}
    geometry = source.learning.deployment_geometry(data["train60"])
    panels = source.sampling.read_panels(source.cached.previous.DATA_RUN)
    new_models, actions, calibrations, checks = [], [], [], []
    lookup = {(c["bank"], tuple(c["removed"]), c["scope"], c["variant"]): c for c in old_actions}
    for bank in (0, 1):
        draws, evaluation = source.sampling.learning.draw_assignment(bank, "expanded")
        if set(draws) & set(evaluation): raise ValueError("Training and evaluation futures overlap")
        ledger = next(b for b in full_info["banks"] if b["bank"] == bank)["full_fit"]
        calibration = learning.full_calibration(train, data["train60"],
            source.learning.base.training_draws(train, panels, draws), prior, ledger)
        checks.append(learning.check_cached_full(data["train60"], calibration, prior, cap, geometry, full_models[str(bank)]))
        calibrations.append({"bank": bank, "training_draws": draws, **calibration})
        full_paths = {s: {m: source.learning.base.candidates(d, full_models[str(bank)][m]) for m in source.learning.MODELS} for s, d in data.items()}
        pieces = [p for p in before["pieces"] if p["bank"] == bank]
        for group, indices, _ in source.learning.joint.binding.policy.prefix_folds(train):
            models, fit = learning.fit_fixed(data["train60"], calibration, indices, prior, cap, geometry)
            new_models.append({"bank": bank, "held_group": list(group), "training_draws": draws, "models": models, "fit": fit})
            refit = next(m for m in adaptive if m["bank"] == bank and m["held_group"] == list(group))
            if refit["training_draws"] != draws: raise ValueError("Changed cached refit future assignment")
            for scope, d in data.items():
                paths = {name: {m: source.learning.base.candidates(d, ms[m]) for m in source.learning.MODELS}
                    for name, ms in (("fixed", models), ("refit", refit["models"]))}
                paths["full"] = full_paths[scope]
                for p in pieces:
                    model, chosen = p["arm"].rsplit("_", 1)[0], p["old_key"] or "zero"
                    variant = f"{p['arm']}_{p['filter']}"
                    old = lookup[bank, tuple(group), scope, variant]
                    pred = {name: ms[model][chosen] for name, ms in paths.items()}
                    if (pred["full"]["chosen"].tolist() != old["full"] or pred["refit"]["chosen"].tolist() != old["inner"]
                            or d["raw_choices"].tolist() != old["raw"]): raise ValueError("Changed Stage259 frozen actions")
                    masks = diagnostic.regions(old["states"], scope, group)
                    actions.append({**{f: old[f] for f in ("cell", "bank", "removed", "variant", "category", "old_key", "old_key_new_gate", "scope", "states", "raw")},
                        "choices": {name: ps["chosen"].tolist() for name, ps in pred.items()},
                        "function_gaps": {contrast: {region: diagnostic.function_gap(d, pred[a], pred[b], mask)
                            for region, mask in masks.items() if mask.any()} for contrast, (a, b) in learning.PAIRS.items()}})
        print(f"CALIBRATION_ABLATION_BANK_FROZEN {key}/{bank}", flush=True)
    if len(new_models) != 20: raise ValueError("Incomplete prefix54 fitting coverage")
    path = result_path(run, *key)
    write_json(raw_directory(path)/"models.json", new_models)
    write_json(raw_directory(path)/"calibrations.json", calibrations)
    write_json(raw_directory(path)/"actions.json", actions)
    pieces, error, telescope = evaluate_cell(actions, panels, key)
    fits = [m["fit"] for m in new_models]
    summary = {"prefix54_fits": len(fits), "inner_solves": len(fits)*8, "full_KKT_checks": sum(c["full_KKT_checks"] for c in checks),
        "max_kkt": max([c["max_kkt"] for c in checks]+[max(f["optimality"], f["difference_optimality"], f["auxiliary_free_fit"]["optimality"], f["auxiliary_free_fit"]["difference_optimality"]) for fit in fits for f in fit["fits"]]),
        "full_profiled_head_max_abs_difference": max(c["profiled_head_max_abs_difference"] for c in checks),
        "subspace_max_abs_difference": max([c["subspace_max_abs_difference"] for c in checks]+[f["subspace_max_abs_difference"] for f in fits]),
        "radius_max_excess": max(max(f["norm"], f["auxiliary_free_fit"]["norm"])-f["old_radius"] for fit in fits for f in fit["fits"])}
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "key": list(key),
        "pieces": pieces, "fits": summary, "reproduction_max_abs_difference": error, "telescope_max_abs_difference": telescope}
    write_json(raw_directory(path)/"aggregate.json", payload)
    write_json(path, {k: v for k, v in payload.items() if k != "pieces"})


def evaluate_cell(actions, panels, key):
    records, maximum, telescope = {}, 0., 0.
    old = previous.evaluate_cell(SOURCE_RUN, key, panels)
    lookup = {(p["bank"], tuple(p["removed"]), p["variant"]): p for p in old["pieces"]}
    for c in actions:
        _, ev = source.sampling.learning.draw_assignment(c["bank"], "expanded")
        values = np.stack([panels[tuple(st)]["values"][ev] for st in c["states"]])
        ident = c["bank"], tuple(c["removed"]), c["variant"]
        record = records.setdefault(ident, {f: c[f] for f in ("cell", "bank", "removed", "variant", "category", "old_key", "old_key_new_gate")})
        comparisons = record.setdefault("comparisons", {name: {} for name in learning.PAIRS})
        transitions = record.setdefault("action_transitions", {})
        for region, mask in diagnostic.regions(c["states"], c["scope"], c["removed"]).items():
            if not mask.any(): continue
            choices = {name: np.asarray(ch)[mask] for name, ch in c["choices"].items()}
            v, raw = values[mask], np.asarray(c["raw"])[mask]
            for name, (a, b) in learning.PAIRS.items():
                comparisons[name][region] = {label: diagnostic.outcomes(vv, choices[a], choices[b], raw)
                    for label, vv in (("whole", v), ("first8", v[:, :8]), ("last8", v[:, 8:]))}
                comparisons[name][region]["function_gap"] = c["function_gaps"][name][region]
            transitions[region] = learning.action_transitions(choices["full"], choices["fixed"], choices["refit"])
            for label in ("whole", "first8", "last8"):
                telescope = max(telescope, learning.check_telescope({name: regions[region][label]["effect"] for name, regions in comparisons.items()}, diagnostic.FIELDS))
                expected, actual = lookup[ident]["regions"][region][label], comparisons["total"][region][label]
                if (actual["cases"], actual["disagreements"]) != (expected["cases"], expected["disagreements"]): raise ValueError("Changed Stage259 region counts")
                for part in ("inner", "full", "effect"):
                    for f in diagnostic.FIELDS: maximum = max(maximum, abs(actual[part][f]-expected[part][f]))
            expected = lookup[ident]["regions"][region]["function_gap"]["squared_sums"]
            for f, value in comparisons["total"][region]["function_gap"]["squared_sums"].items(): maximum = max(maximum, abs(value-expected[f]))
    if len(records) != len(lookup) or maximum > 1e-12: raise ValueError("Stage260 failed Stage259 region reproduction")
    return list(records.values()), max(maximum, old["reproduction_max_abs_difference"]), telescope


def summarize(pieces):
    cs = {name: diagnostic.summarize([{**p, "regions": p["comparisons"][name]} for p in pieces]) for name in learning.PAIRS}
    regions = {r for p in pieces for r in p["action_transitions"]}
    counts = {r: {name: sum(p["action_transitions"].get(r, {}).get(name, 0) for p in pieces)
        for name in ("unchanged", "fit_only", "calibration_only", "cancelled", "both_distinct")} for r in sorted(regions)}
    for region, values in counts.items():
        if sum(values.values()) != cs["total"]["regions"][region]["state_model_exposures"]: raise ValueError("Action transition partition dropped cases")
    return {"bank_cell_exposures": cs["total"]["bank_cell_exposures"],
        "comparisons": {name: c["regions"] for name, c in cs.items()}, "action_transitions": counts}


def report(run):
    records = [previous.read_cell(run, key, "aggregate.json") for key in KEYS]
    for key, r in zip(KEYS, records):
        if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) != key or not r["coverage_qualified"]:
            raise ValueError("Incomplete calibration-ablation matrix")
    budget = contract()["budget"]
    if any(sum(r["fits"][f] for r in records) != budget[f] for f in ("prefix54_fits", "inner_solves", "full_KKT_checks")):
        raise ValueError("Calibration-ablation budget coverage changed")
    output, maximum = {}, 0.
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    for group, predicate in groups.items():
        ps = [p for r in records if predicate(r["key"]) for p in r["pieces"]]
        output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for variant in PRIMARY:
                output[group][label][variant] = {}
                for cat in CATEGORIES:
                    actual = summarize([p for p in ps if p["variant"] == variant and (bank is None or p["bank"] == bank) and previous.in_category(p, cat)])
                    before = expected["groups"][group][label][variant][cat]
                    if actual["bank_cell_exposures"] != before["bank_cell_exposures"]: raise ValueError("Changed frozen category exposure counts")
                    for region, stats in before["regions"].items():
                        now = actual["comparisons"]["total"][region]
                        for part in ("inner", "full", "effect", "absolute_state_effect", "absolute_prefix_mean_effect", "feasible_prediction_rms"):
                            for f, value in stats[part].items():
                                if value is None:
                                    if now[part][f] is not None: raise ValueError("Changed empty feasible-action group")
                                else: maximum = max(maximum, abs(now[part][f]-value))
                    output[group][label][variant][cat] = actual
    if maximum > 1e-12: raise ValueError("Stage260 failed Stage259 group reproduction")
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "fits": {f: sum(r["fits"][f] for r in records) for f in ("prefix54_fits", "inner_solves", "full_KKT_checks")},
        "reproduction": {"cell_max_abs_difference": max(r["reproduction_max_abs_difference"] for r in records), "group_max_abs_difference": maximum,
            "telescope_max_abs_difference": max(r["telescope_max_abs_difference"] for r in records)},
        "fit_checks": {f: max(r["fits"][f] for r in records) for f in ("max_kkt", "full_profiled_head_max_abs_difference", "subspace_max_abs_difference", "radius_max_excess")},
        "default_changed": False, "native_deployment_eligible": False}
    write_json(ROOT/"results"/run/"report/result.json", payload)
    print("CALIBRATION_ABLATION_REPORT_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("fit", "report"), required=True)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--fold", type=int, choices=(0, 1))
    p.add_argument("--partition", choices=source.cached.fixed.PARTITIONS)
    p.add_argument("--direction", type=int, choices=(0, 1))
    args = p.parse_args()
    if args.operation == "fit":
        data, _, registered, _ = source.cached.training.cohort.source.source.load_data()
        views = source.cached.training.cohort.source.matched_views(data["training_data"], registered)
        fit_cell(args.run_name, (args.view, args.fold, args.partition, args.direction), views)
    else: report(args.run_name)


if __name__ == "__main__": main()
