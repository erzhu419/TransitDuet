#!/usr/bin/env python3
"""Align incremental OOF learning to the cached full deployment bundle."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_selector_bridge_stage254 as bridge_source
from freq_hrl.domains.transit import native_deployment_bundle as learning
from freq_hrl.domains.transit import native_selector_bridge as bridge
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

source = bridge_source.source
EXPERIMENT_PROTOCOL = "native_transit_deployment_bundle_stage255_v1"
SOURCE_RUN, BRIDGE_RUN = bridge_source.SOURCE_RUN, "native_transit_selector_bridge_stage254_20261016_r1"
KEYS, VIEWS, PREFLIGHT, result_path = source.KEYS, source.VIEWS, source.PREFLIGHT, source.result_path


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL, "bridge_run": BRIDGE_RUN,
        "cohort": source.contract()["cohort"], "bundles": list(learning.BUNDLES), "models": list(learning.MODELS),
        "selectors": list(learning.SELECTORS), "variants": list(learning.VARIANTS),
        "contrast": "fixed_bundle_minus_prefix_refit_separately_for_each_model_selector_filter_no_winner_selection",
        "alignment": "full_past_reference_feature_coordinates_TSVD_basis_historical_all_MSE_initialization_scale_and_full_RMS_cap_fixed_for_every_inner_fold",
        "inner": "exclude_current_prefix_and_use_same8_new_training_futures; no_new_raw_fit_or_full_critic_fit",
        "validation": "new2_and_new8_only_other_half_of_training_bank; no_old_A2_because_past_bundle_saw_old_A_labels",
        "conditioning": "incremental_crossfit_conditional_on_pretrained_bundle_not_OOF_validation_of_entire_learner; outer_held_prefixes_unseen_by_bundle",
        "full": "Stage251_models_verbatim_for_both_bundles; only_selected_candidate_key_can_change_full_actions",
        "frozen": "same198_features_ranks17_candidates_GLS_reliability_penalty_rules; unchanged_selector_point_nonharm_half_safe_and_historical_fold_score_scales",
        "evaluation": "opposite16_Stage249_development_futures; freeze_all_full_actions_before_outcome_access; both_banks_four_views_seen_and_held",
        "diagnostics": "validation_to_inner_independent_to_full_independent_bridge; aligned_raw_gate_disagreement_must_be_zero; selector_key_coverage_and_fleet_service_wait",
        "controls": "Stage252_new2_new8_full_actions_and_all_outcome_fields_reproduce; Stage254_old_bridge_verbatim",
        "label_boundary": "no_Stage253_Stage248_outcomes_or_Stage169_reserved_prefixes; no_evaluation_selection",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "inner_solves": 10240, "new_full_critic_fits": 0,
            "new_native_episodes": 0, "new_raw_reference_fits": 0, "actor_updates": 0, "new_prefixes": 0},
        "storage": "server_cached_panels_models_inner_actions_and_fit_ledgers; only_receipts_and_scalar_report_pulled",
        "boundary": "bundled_alignment_not_reference_only_causal_effect; reused_development_means_not_fresh_confirmation_seed_CI_or_end_to_end_HRL; no_default_change"}


def fit_cell(run, key, views):
    load = lambda name, file: json.loads((raw_directory(result_path(name, *key))/file).read_text())
    old, old_actions = load(SOURCE_RUN, "aggregate.json"), load(SOURCE_RUN, "actions.json")
    if old["protocol"] != source.EXPERIMENT_PROTOCOL or old["contract"] != source.contract() or not old["coverage_qualified"]:
        raise ValueError("Require qualified Stage252 controls")
    full_models = load(source.SOURCE_RUN, "models.json")
    cached = load(source.information.SOURCE_RUN, "models.json")["full"]["all_mse"]
    ledger = load(source.fixed.SOURCE_RUN, "fits.json")
    scales = {(tuple(b["held_group"]), b["half"]): b["scales"] for b in ledger["blocks"]}
    reference = load(source.fixed.contract()["reference_run"], "models.json")["reference"]
    train, held = source.training.cohort.split_rows(views[key[0]], key[1])
    base = learning.base
    full_data = base.causal_data(train, reference)
    geometry = learning.deployment_geometry(full_data)
    panels = source.information.read_panels(source.previous.DATA_RUN)
    banks, inner_cache, fit_cache, frozen, actions = [], [], [], [], []
    path = result_path(run, *key)
    for bank in (0, 1):
        blocks = []
        for group, indices, validation in learning.joint.binding.policy.prefix_folds(train):
            rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
            data, val = base.causal_data(rows, reference), base.causal_data(val_rows, reference)
            for half in (0, 1):
                ds, _ = source.information.learning.draw_assignment(bank, "expanded", half)
                models, fit = learning.fit_inner(rows, data, base.training_draws(rows, panels, ds), cached, ledger["full"], geometry)
                paths = {m: base.candidates(val, models[m]) for m in learning.MODELS}
                truths = {s: base.training_draws(val_rows, panels, source.learning.validation_draws(bank, half, s)) for s in learning.SELECTORS}
                curves = {f"{m}_{s}": {c: base.curve(val, truths[s], p) for c, p in paths[m].items()}
                    for m in learning.MODELS for s in learning.SELECTORS}
                blocks.append({"held_group": list(group), "half": half, "scales": scales[tuple(group), half],
                    "curves": {"all_mse": curves["free_old_matched2"], **curves}})
                inner_cache.append({"bank": bank, "half": half, "held_group": list(group), "training_draws": ds,
                    "states": [list(source.information.old_future.learning.state_key(r)) for r in val_rows],
                    "raw_choices": val["raw_choices"].tolist(), "gate": (val["raw_logits"] <= 0).tolist(),
                    "choices": {m: {c: p["chosen"].tolist() for c, p in ps.items()} for m, ps in paths.items()}})
                fit_cache.append({"bank": bank, "half": half, "held_group": list(group), **fit})
        selection, summaries = learning.select(blocks)
        prior = next(b for b in old["banks"] if b["bank"] == bank)
        selections = {"fixed_bundle": selection, "prefix_refit": {a: prior["selection"][a] for a in learning.ARMS}}
        banks.append({"bank": bank, "selection": selections, "summaries": summaries, "matrix": {}})
        for scope, rows in (("train60", train), ("held60", held)):
            data = full_data if scope == "train60" else base.causal_data(rows, reference)
            paths = {m: base.candidates(data, full_models[str(bank)][m]) for m in learning.MODELS}
            pred = learning.predict(paths, selections)
            actions.append({"bank": bank, "scope": scope,
                "states": [list(source.information.old_future.learning.state_key(r)) for r in rows],
                "choices": {v: p["chosen"].tolist() for v, p in pred.items()}})
            frozen.append((bank, scope, rows, data, pred))
        print(f"DEPLOYMENT_BUNDLE_BANK_FROZEN {key}/{bank}", flush=True)
    comparisons, maximum = 0, 0.
    for now, before in zip(actions, old_actions):
        if (now["bank"], now["scope"], now["states"]) != (before["bank"], before["scope"], before["states"]):
            raise ValueError("Changed frozen control state order")
        for v, choices in now["choices"].items():
            if v.startswith("fixed_bundle_"): continue
            alias = v.removeprefix("prefix_refit_")
            if choices != before["choices"][alias]: raise ValueError("Changed frozen control actions")
            comparisons += len(choices)
    write_json(raw_directory(path)/"actions.json", actions)
    write_json(raw_directory(path)/"oof_choices.json", inner_cache)
    write_json(raw_directory(path)/"fits.json", fit_cache)
    for bank, scope, rows, data, pred in frozen:
        _, evaluation = source.information.learning.draw_assignment(bank, "expanded")
        matrix = source.learning.previous.balanced.metrics(data, base.training_draws(rows, panels, evaluation), pred)
        banks[bank]["matrix"][scope] = matrix
        prior = next(b for b in old["banks"] if b["bank"] == bank)["matrix"][scope]
        for actual, expected in [(matrix["all"], prior["all"]), *((matrix["by_root"][r], p) for r, p in prior["by_root"].items())]:
            for v, stats in actual["variants"].items():
                if v.startswith("fixed_bundle_"): continue
                for f, value in stats.items(): maximum = max(maximum, abs(value-expected["variants"][v.removeprefix("prefix_refit_")][f]))
    if maximum > 1e-12: raise ValueError("Changed frozen control outcomes")
    bridges = []
    row_map = {source.information.old_future.learning.state_key(r): r for r in train}
    for c in inner_cache:
        bank, half = c["bank"], c["half"]
        rows = [row_map[tuple(st)] for st in c["states"]]
        data = base.causal_data(rows, reference)
        full = next(a for a in actions if a["bank"] == bank and a["scope"] == "train60")
        positions = {tuple(st): i for i, st in enumerate(full["states"])}
        ix = [positions[tuple(st)] for st in c["states"]]
        _, evaluation = source.information.learning.draw_assignment(bank, "expanded")
        ev = base.training_draws(rows, panels, evaluation)
        truths = {s: base.training_draws(rows, panels, source.learning.validation_draws(bank, half, s)) for s in learning.SELECTORS}
        pieces = {}
        for m in learning.MODELS:
            for s in learning.SELECTORS:
                arm = f"{m}_{s}"
                for f in learning.FILTERS:
                    variant = f"{arm}_{f}"
                    selected = banks[bank]["selection"]["fixed_bundle"][arm][f] or "zero"
                    p = bridge.compare(truths[s], ev, c["choices"][m][selected],
                        [full["choices"][f"fixed_bundle_{variant}"][i] for i in ix], c["raw_choices"],
                        [full["choices"]["raw"][i] for i in ix], np.asarray(c["gate"]), data["raw_logits"] <= 0)
                    if p["strata"]["all"]["counts"]["raw_action_disagreement"] or p["strata"]["all"]["counts"]["feasible_mask_disagreement"]:
                        raise ValueError("Fixed deployment raw or gate differs across folds")
                    pieces[variant] = p
        bridges.append({"bank": bank, "prefix": c["held_group"], "half": half, "scenario": rows[0]["scenario"], "pieces": pieces})
    fit_summary = {"inner_solves": len(fit_cache)*8,
        "subspace_max_abs_difference": max(f["subspace_max_abs_difference"] for f in fit_cache),
        "max_kkt": max(max(x["optimality"], x["difference_optimality"], x["auxiliary_free_fit"]["optimality"]) for f in fit_cache for x in f["fits"]),
        "radius_max_excess": max(max(x["norm"], x["auxiliary_free_fit"]["norm"])-x["old_radius"] for f in fit_cache for x in f["fits"])}
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "coverage_qualified": True,
        "banks": banks, "bridge": bridges, "fits": fit_summary,
        "reproduction": {"action_exposures": comparisons, "max_abs_difference": maximum, "mismatches": 0}}
    write_json(raw_directory(path)/"aggregate.json", result)
    write_json(path, {k: v for k, v in result.items() if k not in ("banks", "bridge")})


def compact(records, scope, bank):
    bs = [b for r in records for b in r["banks"] if bank is None or b["bank"] == bank]
    matrix = learning.aggregate([b["matrix"][scope] for b in bs])
    outcomes = matrix["all"]["variants"]
    return {"model_bank_exposures": len(bs), "outcomes": outcomes, "outcomes_by_root": matrix["by_root"],
        "alignment_contrasts": {f"{a}_{f}": {field: outcomes[f"fixed_bundle_{a}_{f}"][field]-outcomes[f"prefix_refit_{a}_{f}"][field]
            for field in ("delta_cost", "delta_wait_min", "delta_risk", "delta_service")} for a in learning.ARMS for f in learning.FILTERS},
        "admitted": {bundle: {a: {f: sum(b["selection"][bundle][a][f] is not None for b in bs) for f in learning.FILTERS}
            for a in learning.ARMS} for bundle in learning.BUNDLES},
        "changed_keys": {a: {f: sum(b["selection"]["fixed_bundle"][a][f] != b["selection"]["prefix_refit"][a][f] for b in bs)
            for f in learning.FILTERS} for a in learning.ARMS}}


def report(run):
    records = [json.loads((raw_directory(result_path(run, *k))/"aggregate.json").read_text()) for k in KEYS]
    for key, r in zip(KEYS, records):
        if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) != key or not r["coverage_qualified"]:
            raise ValueError("Incomplete deployment-bundle matrix")
    old_bridge = json.loads((ROOT/"results"/BRIDGE_RUN/"report/result.json").read_text())
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output = {}
    for name, predicate in groups.items():
        rs = [r for r in records if predicate(r["key"])]
        output[name] = {scope: {label: compact(rs, scope, bank) for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1))}
            for scope in ("train60", "held60")}
        output[name]["bridge"] = {label: {"fixed_bundle": {f"{a}_{f}": bridge.combine([b["pieces"][f"{a}_{f}"] for r in rs for b in r["bridge"]
            if bank is None or b["bank"] == bank])["all"] for a in learning.ARMS for f in learning.FILTERS},
            "prefix_refit": {f"{a}_{f}": old_bridge["groups"][name][label][f"{a}_{f}"]["all"] for a in learning.ARMS for f in learning.FILTERS}}
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1))}
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "fits": {"inner_solves": sum(r["fits"]["inner_solves"] for r in records),
            **{f: max(r["fits"][f] for r in records) for f in ("subspace_max_abs_difference", "max_kkt", "radius_max_excess")}},
        "reproduction": {"action_exposures": sum(r["reproduction"]["action_exposures"] for r in records), "mismatches": 0,
            "max_abs_difference": max(r["reproduction"]["max_abs_difference"] for r in records)},
        "default_changed": False, "native_deployment_eligible": False}
    write_json(ROOT/"results"/run/"report/result.json", payload)
    print("DEPLOYMENT_BUNDLE_REPORT_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("fit", "report"), required=True)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--fold", type=int, choices=(0, 1))
    p.add_argument("--partition", choices=source.fixed.PARTITIONS)
    p.add_argument("--direction", type=int, choices=(0, 1))
    args = p.parse_args()
    if args.operation == "fit":
        data, _, registered, _ = source.training.cohort.source.source.load_data()
        views = source.training.cohort.source.matched_views(data["training_data"], registered)
        fit_cell(args.run_name, (args.view, args.fold, args.partition, args.direction), views)
    else: report(args.run_name)


if __name__ == "__main__": main()
