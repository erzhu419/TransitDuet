#!/usr/bin/env python3
"""Fixed-future head/budget factorial; reuse every raw reference and control."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_training_information_stage249 as previous
from scripts.analyze_native_transit_training_information_stage249 import average
from freq_hrl.domains.transit import native_head_budget as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_head_budget_stage250_v1"
SOURCE_RUN = "native_transit_training_information_stage249_20261015_r1"
KEYS, VIEWS, training, result_path = previous.KEYS, previous.VIEWS, previous.training, previous.result_path
PREFLIGHT = previous.previous.PREFLIGHT


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cohort": previous.contract()["cohort"], "arms": list(learning.ARMS), "variants": list(learning.VARIANTS),
        "training": "Stage249_expanded_only_8_inner_16_full_same_banks_same_state_folds",
        "evaluation": "opposite16_bank_train60_and_held60; both_reciprocal_directions_and_four_views",
        "frozen": "raw_references_198_features_TSVD_ranks_scale_state_GLS_shrinkage_precision_A2_OOF_selector_17_candidates_actions_suffix",
        "head_factor": "cached_difference_vs_free_difference_profiled_in_same_subspace_total_df_k_component_df_k_vs_2k",
        "budget_factor": "cached_A_radius_vs_max_A_radius_norm_of_OLS_projection_of_shrunk_training_head_sum_no_validation_or_eval_calibration",
        "controls": "Stage249_expanded_frozen_old_and_Stage247_old_cached_models_selections_actions_no_control_refits",
        "contrasts": ["free_old-frozen_old", "frozen_train-frozen_old", "free_train-frozen_train",
            "free_train-free_old", "free_train-free_old-frozen_train+frozen_old"],
        "primary_diagnostic": "held60_mixed_pooled_fleet_nonanchor_MSE_and_feasible_pair_MSE_each_bank_separate_no_winner_selection",
        "policy": "pooled_safe_primary_half_safe_secondary_old_A2_OOF_only_no_arm_or_scenario_gate",
        "rank_diagnostic": "all_four_registered_ranks_at_alpha1_training_bank_seen_prefix_new_future_and_held_prefix_new_future",
        "label_boundary": "Stage248_not_read; Stage249_is_now_development_data_not_fresh_confirmation; Stage169_reserved_cohorts_closed",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "new_quadratic_solves": 16128,
            "new_native_episodes": 0, "new_raw_reference_fits": 0, "actor_updates": 0, "new_prefixes": 0},
        "storage": "server_only_panels_models_actions_aggregates_small_fit_receipts_and_compact_report_pull",
        "boundary": "development_factorial_not_new_prefix_confirmation_not_end_to_end_HRL_no_default_change"}


def fit_cell(run, key, views):
    load = lambda source, name: json.loads((raw_directory(result_path(source, *key))/name).read_text())
    info, frozen_models = load(SOURCE_RUN, "aggregate.json"), load(SOURCE_RUN, "models.json")
    if info["protocol"] != previous.EXPERIMENT_PROTOCOL or info["contract"] != previous.contract() or not info["coverage_qualified"]:
        raise ValueError("Require qualified Stage249 cell")
    old = previous.previous
    cached = load(previous.SOURCE_RUN, "models.json")
    ledger = load(old.SOURCE_RUN, "fits.json")
    old_blocks = {(tuple(b["held_group"]), b["half"]): b for b in ledger["blocks"]}
    reference = load(old.contract()["reference_run"], "models.json")["reference"]
    refs = load(old.contract()["refit_run"], "models.json")["references"]
    trials = load(old.contract()["refit_run"], "result.json")["cell"]["trials"]
    lookup = {(tuple(t["removed_group"]), t["half"]): str(i) for i, t in enumerate(trials)
        if t["removed_group"] and t["half"] is not None}
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    panels, past = previous.read_panels(SOURCE_RUN), old.panel_cache()[0]
    a, b = old.PARTITIONS[key[2]][key[3]], old.PARTITIONS[key[2]][1-key[3]]
    actions, output, model_cache = [], [], {}
    for bank in (0, 1):
        blocks = []
        for group, indices, validation in learning.joint.binding.policy.prefix_folds(train):
            rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
            for half in (0, 1):
                index = lookup[tuple(group), half]
                data, val = learning.base.causal_data(rows, refs[index]), learning.base.causal_data(val_rows, refs[index])
                ds, _ = previous.learning.draw_assignment(bank, "expanded", half)
                models, fit = learning.fit_paths(rows, data, learning.base.training_draws(rows, panels, ds),
                    cached[index]["all_mse"], old_blocks[tuple(group), half]["fits"])
                learning.qualify_fit(fit)
                _, _, dv = old.fixed.draw_assignment(a, b, half)
                truth = learning.base.training_draws(val_rows, past, dv)
                blocks.append({"held_group": list(group), "half": half, "scales": old_blocks[tuple(group), half]["scales"],
                    "curves": {arm: {c: learning.base.curve(val, truth, p) for c, p in learning.base.candidates(val, m).items()}
                        for arm, m in models.items()}})
        # The generic selector reads this alias only to establish roots/scales.
        selected, summaries = learning.select([{**x, "curves": {"all_mse": x["curves"]["free_old"], **x["curves"]}} for x in blocks])
        source_bank = next(x for x in info["banks"] if x["bank"] == bank)
        selected.update(old=source_bank["selection"]["old"], frozen_old=source_bank["selection"]["expanded"])
        data = learning.base.causal_data(train, reference)
        ds, evaluation = previous.learning.draw_assignment(bank, "expanded")
        models, full_fit = learning.fit_paths(train, data, learning.base.training_draws(train, panels, ds),
            cached["full"]["all_mse"], ledger["full"])
        learning.qualify_fit(full_fit)
        models.update(old=frozen_models[str(bank)]["old"], frozen_old=frozen_models[str(bank)]["expanded"])
        model_cache[str(bank)] = models
        frozen = []
        for scope, rows in (("train60", train), ("held60", held)):
            d = data if scope == "train60" else learning.base.causal_data(rows, reference)
            paths = {arm: learning.base.candidates(d, m) for arm, m in models.items()}
            pred = learning.predict(paths, selected)
            actions.append({"bank": bank, "scope": scope, "states": [list(previous.old_future.learning.state_key(r)) for r in rows],
                "choices": {v: p["chosen"].tolist() for v, p in pred.items()}})
            frozen.append((scope, rows, d, pred, paths))
        write_json(raw_directory(result_path(run, *key))/"actions.json", actions)
        matrix, diagnostics = {}, {}
        for scope, rows, d, pred, paths in frozen:
            values = learning.base.training_draws(rows, panels, evaluation)
            train_values = learning.base.training_draws(rows, panels, ds)
            matrix[scope] = learning.previous.metrics(d, values, pred)
            diagnostics[scope] = previous.learning.information_error(train_values, values, d, pred)
            diagnostics[scope]["ranks"] = {arm: {k: v for k, v in previous.learning.information_error(
                train_values, values, d, previous.learning.rank_predictions(paths[arm])).items()
                if k in ("predictor_error", "predictor_error_on_training_bank")} for arm in learning.ARMS}
        output.append({"bank": bank, "selection": selected, "summaries": summaries, "matrix": matrix,
            "diagnostics": diagnostics, "full_fit": full_fit})
        print(f"HEAD_BUDGET_BANK_COMPLETE {key}/{bank}", flush=True)
    expected = load(SOURCE_RUN, "actions.json")
    aliases = {"raw": "raw", **{f"{a}_{f}": f"{s}_{f}" for a, s in (("old", "old"), ("frozen_old", "expanded")) for f in learning.FILTERS}}
    comparisons = 0
    for now, before in zip(actions, expected):
        if (now["bank"], now["scope"], now["states"]) != (before["bank"], before["scope"], before["states"]):
            raise ValueError("Changed frozen control state order")
        for name, alias in aliases.items():
            if now["choices"][name] != before["choices"][alias]: raise ValueError("Changed frozen control actions")
            comparisons += len(now["states"])
    if len(actions) != len(expected): raise ValueError("Dropped frozen control bank/scope")
    path = result_path(run, *key)
    write_json(raw_directory(path)/"models.json", model_cache)
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "coverage_qualified": True,
        "banks": output, "frozen_reproduction": {"action_exposures_compared": comparisons, "mismatches": 0},
        "new_native_episodes": 0, "new_raw_reference_fits": 0}
    write_json(raw_directory(path)/"aggregate.json", result)
    write_json(path, {k: v for k, v in result.items() if k != "banks"})


def compact(records, scope, bank):
    bs = [b for r in records for b in r["banks"] if bank is None or b["bank"] == bank]
    ds = [b["diagnostics"][scope] for b in bs]
    matrix = learning.aggregate([b["matrix"][scope] for b in bs])
    outcomes = matrix["all"]["variants"]
    contrasts = {f"{a}-{b}": {field: outcomes[f"{a}_pooled_safe"][field]-outcomes[f"{b}_pooled_safe"][field]
        for field in ("delta_cost", "delta_wait_min", "delta_risk", "delta_service")}
        for a, b in (("free_old", "frozen_old"), ("frozen_train", "frozen_old"), ("free_train", "frozen_train"), ("free_train", "free_old"))}
    interaction = "free_train-free_old-frozen_train+frozen_old"
    contrasts[interaction] = {field: contrasts['free_train-free_old'][field]-contrasts['frozen_train-frozen_old'][field]
        for field in contrasts['free_train-free_old']}
    errors = {a: average([d["predictor_error"][f"{a}_pooled_safe"] for d in ds]) for a in learning.ARMS}
    error_contrasts = {name: {head: {metric: errors[a][head][metric]-errors[b][head][metric]
        for metric in errors[a][head]} for head in errors[a]}
        for name, (a, b) in zip(list(contrasts)[:4], (("free_old", "frozen_old"), ("frozen_train", "frozen_old"),
            ("free_train", "frozen_train"), ("free_train", "free_old")))}
    error_contrasts[interaction] = {head: {metric: error_contrasts['free_train-free_old'][head][metric]-error_contrasts['frozen_train-frozen_old'][head][metric]
        for metric in errors['free_train'][head]} for head in errors['free_train']}
    return {"model_bank_exposures": len(bs), "outcomes": outcomes, "outcomes_by_root": matrix["by_root"],
        "contrasts": contrasts, "error_contrasts": error_contrasts, "evaluation_mc_noise": average([d["evaluation_mc_noise"] for d in ds]),
        "zero_prediction_error": average([d["zero_prediction_error"] for d in ds]),
        "models": {v: {"evaluation": average([d["predictor_error"][v] for d in ds]),
            "training_bank": average([d["predictor_error_on_training_bank"][v] for d in ds])} for v in learning.VARIANTS},
        "ranks": {a: average([d["ranks"][a] for d in ds]) for a in learning.ARMS},
        "admitted": {a: sum(b["selection"][a]["pooled_safe"] is not None for b in bs) for a in ("old", *learning.ARMS)},
        "full_fits": {a: {level: average([{f: x[f] for f in ("radius", "old_radius", "norm", "loss_before", "loss_after", "difference_change_rms", "optimality")}
            for b in bs for x in b["full_fit"]["fits"] if x["arm"] == a and x["level"] == level])
            for level in learning.strength.lowrank.LEVELS[1:]} for a in learning.NEW_ARMS}}


def report(run):
    records = [json.loads((raw_directory(result_path(run, *k))/"aggregate.json").read_text()) for k in KEYS]
    for key, r in zip(KEYS, records):
        if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) != key or not r["coverage_qualified"]:
            raise ValueError("Incomplete head/budget factorial")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "groups": {name: {scope: {label: compact([r for r in records if predicate(r["key"])], scope, bank)
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1))} for scope in ("train60", "held60")}
            for name, predicate in groups.items()},
        "frozen_reproduction": {"action_exposures_compared": sum(r["frozen_reproduction"]["action_exposures_compared"] for r in records), "mismatches": 0},
        "interpretation": "existing-development-prefix reciprocal crossfit, not independent seed CI; training_bank fit-exposed on train60 for factorial arms, not Stage247 old",
        "default_changed": False, "native_deployment_eligible": False}
    write_json(ROOT/"results"/run/"report/result.json", payload)
    print("HEAD_BUDGET_REPORT_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("fit", "report"), required=True)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--fold", type=int, choices=(0, 1))
    p.add_argument("--partition", choices=previous.previous.PARTITIONS)
    p.add_argument("--direction", type=int, choices=(0, 1))
    args = p.parse_args()
    if args.operation == "fit":
        bank, _, registered, _ = training.cohort.source.source.load_data()
        views = training.cohort.source.matched_views(bank["training_data"], registered)
        fit_cell(args.run_name, (args.view, args.fold, args.partition, args.direction), views)
    else: report(args.run_name)


if __name__ == "__main__": main()
