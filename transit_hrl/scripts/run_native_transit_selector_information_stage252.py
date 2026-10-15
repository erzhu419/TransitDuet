#!/usr/bin/env python3
"""Isolate OOF validation source and count under cached full-model controls."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_head_shrinkage_stage251 as previous
from freq_hrl.domains.transit import native_selector_information as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_selector_information_stage252_v1"
SOURCE_RUN = "native_transit_head_shrinkage_stage251_20261015_r1"
KEYS, VIEWS, training, result_path, PREFLIGHT = previous.KEYS, previous.VIEWS, previous.training, previous.result_path, previous.PREFLIGHT
information, fixed, average = previous.information, previous.fixed, previous.average


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL, "data_run": previous.DATA_RUN,
        "cohort": previous.contract()["cohort"], "models": list(learning.MODELS), "selectors": list(learning.SELECTORS), "variants": list(learning.VARIANTS),
        "primary": "expanded8_minus_matched2_separately_for_free_and_shrunk_pooled_safe_no_model_or_selector_winner_selection",
        "secondary": "matched2_minus_old_A2_and_expanded8_minus_old_A2_both_banks_four_views",
        "inner": "reconstruct_identical_Stage250_free_and_Stage251_shrunk_8_future_inner_fits_no_training_rule_change",
        "full": "all_full_models_read_Stage251_verbatim_no_full_critic_fit_or_raw_reference_fit",
        "validation": "old_A2_verbatim_past_labels; matched2_first2_of_other_training_bank_half; expanded8_all_other_half; prefix_also_excluded_from_inner_fit",
        "nesting": "matched2_nested_in_expanded8; all_new_validation_draw_indices_disjoint_from_inner8_and_opposite16_evaluation",
        "frozen": "198_features_TSVD_ranks_scales_old_RMS_caps_penalty_formula_candidate17_grid_OOF_score_nonharm_and_half_safe_rules",
        "controls": "old_A2_keys_and_all_candidate_summaries_reproduce_Stage250_free_Stage251_shrunk; cached_old_and_frozen_difference_controls",
        "evaluation": previous.contract()["evaluation"], "action_order": "freeze_all_actions_before_opposite16_outcome_access",
        "diagnostics": "selection_coverage_key_histograms_paired_cost_wait_service_fleet_and_selected_predictor_errors; selection_changes_not_new_model_accuracy",
        "storage": "server_cached_full_models; persist_inner_candidate_actions_for_followups; raw_panels_actions_and_OOF_cache_not_pulled",
        "label_boundary": "Stage249_development_futures_only; no_Stage248_reads_no_Stage169_reserve_opening_no_evaluation_bank_selection",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "inner_reconstruction_solves": 10240,
            "new_full_critic_fits": 0, "new_native_episodes": 0, "new_raw_reference_fits": 0, "actor_updates": 0, "new_prefixes": 0},
        "boundary": "reused_development_validation_information_test_not_fresh_confirmation_or_seed_CI_no_default_change"}


def fit_cell(run, key, views):
    load = lambda source, name: json.loads((raw_directory(result_path(source, *key))/name).read_text())
    info, models = load(SOURCE_RUN, "aggregate.json"), load(SOURCE_RUN, "models.json")
    if info["protocol"] != previous.EXPERIMENT_PROTOCOL or info["contract"] != previous.contract() or not info["coverage_qualified"]:
        raise ValueError("Require qualified Stage251 cell")
    free_source = load(previous.SOURCE_RUN, "aggregate.json")
    cached, ledger = load(information.SOURCE_RUN, "models.json"), load(fixed.SOURCE_RUN, "fits.json")
    old_blocks = {(tuple(b["held_group"]), b["half"]): b for b in ledger["blocks"]}
    reference = load(fixed.contract()["reference_run"], "models.json")["reference"]
    refs = load(fixed.contract()["refit_run"], "models.json")["references"]
    trials = load(fixed.contract()["refit_run"], "result.json")["cell"]["trials"]
    lookup = {(tuple(t["removed_group"]), t["half"]): str(i) for i, t in enumerate(trials)
        if t["removed_group"] and t["half"] is not None}
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    panels, past = information.read_panels(previous.DATA_RUN), fixed.panel_cache()[0]
    a, b = fixed.PARTITIONS[key[2]][key[3]], fixed.PARTITIONS[key[2]][1-key[3]]
    actions, output, oof_cache, maximum = [], [], [], 0.
    for bank in (0, 1):
        blocks = []
        for group, indices, validation in learning.joint.binding.policy.prefix_folds(train):
            rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
            for half in (0, 1):
                index = lookup[tuple(group), half]
                data, val = learning.base.causal_data(rows, refs[index]), learning.base.causal_data(val_rows, refs[index])
                ds, _ = information.learning.draw_assignment(bank, "expanded", half)
                inner, _ = learning.rebuild_inner(rows, data, learning.base.training_draws(rows, panels, ds),
                    cached[index]["all_mse"], old_blocks[tuple(group), half]["fits"])
                paths = {m: learning.base.candidates(val, inner[m]) for m in learning.MODELS}
                _, _, dv = fixed.fixed.draw_assignment(a, b, half)
                truths = {"old_A2": learning.base.training_draws(val_rows, past, dv),
                    **{s: learning.base.training_draws(val_rows, panels, learning.validation_draws(bank, half, s)) for s in learning.SELECTORS[1:]}}
                curves = {f"{m}_{s}": {c: learning.base.curve(val, truths[s], p) for c, p in paths[m].items()}
                    for m in learning.MODELS for s in learning.SELECTORS}
                blocks.append({"held_group": list(group), "half": half, "scales": old_blocks[tuple(group), half]["scales"],
                    "curves": {"all_mse": curves['free_old_old_A2'], **curves}})
                oof_cache.append({"bank": bank, "half": half, "held_group": list(group), "training_draws": ds,
                    "new_validation_draws": {s: learning.validation_draws(bank, half, s) for s in learning.SELECTORS[1:]},
                    "states": [list(information.old_future.learning.state_key(r)) for r in val_rows],
                    "raw_choices": val["raw_choices"].tolist(), "roots": val["roots"].tolist(),
                    "choices": {m: {c: p["chosen"].tolist() for c, p in ps.items()} for m, ps in paths.items()}})
        selected, summaries = learning.select(blocks)
        old = next(x for x in info["banks"] if x["bank"] == bank)
        free = next(x for x in free_source["banks"] if x["bank"] == bank)
        for m, source in (("free_old", free), ("shrunk", old)):
            if selected[f"{m}_old_A2"] != old["selection"][m]: raise ValueError("Reconstructed old A2 key differs")
            maximum = max(maximum, learning.summary_difference(summaries[f"{m}_old_A2"], source["summaries"][m]))
        if maximum > 1e-12: raise ValueError("Reconstructed old A2 curves differ")
        selected.update({m: old["selection"][m] for m in ("old", "frozen_old")})
        frozen = []
        for scope, rows in (("train60", train), ("held60", held)):
            d = learning.base.causal_data(rows, reference)
            paths = {m: learning.base.candidates(d, models[str(bank)][m]) for m in ("old", "frozen_old", *learning.MODELS)}
            pred = learning.predict(paths, selected)
            actions.append({"bank": bank, "scope": scope, "states": [list(information.old_future.learning.state_key(r)) for r in rows],
                "choices": {v: p["chosen"].tolist() for v, p in pred.items()}})
            frozen.append((scope, rows, d, pred))
        write_json(raw_directory(result_path(run, *key))/"actions.json", actions)
        _, evaluation = information.learning.draw_assignment(bank, "expanded")
        matrix, diagnostics = {}, {}
        for scope, rows, d, pred in frozen:
            values = learning.base.training_draws(rows, panels, evaluation)
            matrix[scope] = learning.previous.balanced.metrics(d, values, pred)
            ds, _ = information.learning.draw_assignment(bank, "expanded")
            diagnostics[scope] = information.learning.information_error(learning.base.training_draws(rows, panels, ds), values, d, pred)
        output.append({"bank": bank, "selection": selected, "summaries": summaries, "matrix": matrix, "diagnostics": diagnostics})
        print(f"SELECTOR_INFORMATION_BANK_COMPLETE {key}/{bank}", flush=True)
    expected = load(SOURCE_RUN, "actions.json")
    aliases = {"raw": "raw", **{f"{m}_{f}": f"{m}_{f}" for m in ("old", "frozen_old") for f in learning.FILTERS},
        **{f"{m}_old_A2_{f}": f"{m}_{f}" for m in learning.MODELS for f in learning.FILTERS}}
    comparisons = 0
    if len(actions) != len(expected): raise ValueError("Dropped cached control scope")
    for now, before in zip(actions, expected):
        if (now["bank"], now["scope"], now["states"]) != (before["bank"], before["scope"], before["states"]):
            raise ValueError("Changed cached control state order")
        for name, alias in aliases.items():
            if now["choices"][name] != before["choices"][alias]: raise ValueError("Changed cached control actions")
            comparisons += len(now["states"])
    path = result_path(run, *key)
    write_json(raw_directory(path)/"oof_choices.json", oof_cache)
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "coverage_qualified": True,
        "banks": output, "reproduction": {"action_exposures_compared": comparisons, "mismatches": 0, "summary_max_abs_difference": maximum}}
    write_json(raw_directory(path)/"aggregate.json", result)
    write_json(path, {k: v for k, v in result.items() if k != "banks"})


def compact(records, scope, bank):
    bs = [b for r in records for b in r["banks"] if bank is None or b["bank"] == bank]
    matrix = learning.aggregate([b["matrix"][scope] for b in bs])
    outcomes = matrix["all"]["variants"]
    return {"model_bank_exposures": len(bs), "outcomes": outcomes, "outcomes_by_root": matrix["by_root"],
        "contrasts": {f"{m}_{a}-{b}": {f: outcomes[f"{m}_{a}_pooled_safe"][f]-outcomes[f"{m}_{b}_pooled_safe"][f]
            for f in ("delta_cost", "delta_wait_min", "delta_risk", "delta_service")}
            for m in learning.MODELS for a, b in (("expanded8", "matched2"), ("matched2", "old_A2"), ("expanded8", "old_A2"))},
        "predictor_error": average([b["diagnostics"][scope]["predictor_error"] for b in bs]),
        "admitted": {arm: sum(b["selection"][arm]["pooled_safe"] is not None for b in bs) for arm in learning.ARMS},
        "selection_counts": {arm: {mode: {c: sum((b["selection"][arm][mode] or "zero") == c for b in bs)
            for c in learning.strength.CANDIDATES} for mode in learning.FILTERS} for arm in learning.ARMS}}


def report(run):
    records = [json.loads((raw_directory(result_path(run, *k))/"aggregate.json").read_text()) for k in KEYS]
    for key, r in zip(KEYS, records):
        if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) != key or not r["coverage_qualified"]:
            raise ValueError("Incomplete selector information matrix")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "groups": {name: {scope: {label: compact([r for r in records if predicate(r["key"])], scope, bank)
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1))} for scope in ("train60", "held60")}
            for name, predicate in groups.items()},
        "reproduction": {"action_exposures_compared": sum(r["reproduction"]["action_exposures_compared"] for r in records),
            "mismatches": 0, "summary_max_abs_difference": max(r["reproduction"]["summary_max_abs_difference"] for r in records)},
        "interpretation": "conditional crossfit development means, not independent seed CI; selected-error differences reflect changed keys, not retraining",
        "default_changed": False, "native_deployment_eligible": False}
    write_json(ROOT/"results"/run/"report/result.json", payload)
    print("SELECTOR_INFORMATION_REPORT_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("fit", "report"), required=True)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--fold", type=int, choices=(0, 1))
    p.add_argument("--partition", choices=fixed.PARTITIONS)
    p.add_argument("--direction", type=int, choices=(0, 1))
    args = p.parse_args()
    if args.operation == "fit":
        bank, _, registered, _ = training.cohort.source.source.load_data()
        views = training.cohort.source.matched_views(bank["training_data"], registered)
        fit_cell(args.run_name, (args.view, args.fold, args.partition, args.direction), views)
    else: report(args.run_name)


if __name__ == "__main__": main()
