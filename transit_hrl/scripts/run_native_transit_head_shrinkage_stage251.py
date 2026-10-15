#!/usr/bin/env python3
"""Train difference-shrunk critics under the frozen old total RMS cap."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_head_budget_stage250 as previous
from freq_hrl.domains.transit import native_head_shrinkage as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_head_shrinkage_stage251_v1"
SOURCE_RUN = "native_transit_head_budget_stage250_20261015_r1"
DATA_RUN = previous.SOURCE_RUN
information, fixed = previous.previous, previous.previous.previous
KEYS, VIEWS, training, result_path, PREFLIGHT = previous.KEYS, previous.VIEWS, previous.training, previous.result_path, previous.PREFLIGHT
average = previous.average


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "data_run": DATA_RUN, "cohort": previous.contract()["cohort"], "arms": list(learning.ARMS), "variants": list(learning.VARIANTS),
        "training": previous.contract()["training"], "evaluation": previous.contract()["evaluation"],
        "frozen": previous.contract()["frozen"], "total_cap": "cached_A_radius_only_no_training_expansion",
        "new_fit": "state_GLS_plus_half_lambda_squared_norm_of_new_head_difference_around_cached_difference",
        "noise": "sum_state_covariance_of_per_future_paired_service_minus_risk_scores_divided_by_D; hold_training_precision_reliability_and_total_fixed; H_pinv_cov_H_pinv",
        "penalty": "signal=max_squared_auxiliary_free_difference_increment_minus_trace_noise_0; lambda=trace_H_noise_over_signal; no_positive_signal_freezes_difference",
        "interpretation": "plugin_conditional_noise_not_full_fit_covariance_or_empirical_Bayes_guarantee; retain_service_risk_and_action_CRN_covariance",
        "capacity": "same198_features_TSVD_ranks_total_df_k_component_df_max2k_no_extra_rank_or_strength_grid",
        "controls": "Stage250_old_frozen_old_free_old_models_selections_and_actions_verbatim; auxiliary_free_solves_only_for_penalty_estimation",
        "contrasts": ["shrunk-frozen_old", "shrunk-free_old", "shrunk-old"],
        "primary_diagnostic": "mixed_pooled_held60_fleet_feasible_pair_MSE_both_banks; nonanchor_MSE_and_head_difference_MSE_also_reported",
        "policy": previous.contract()["policy"], "rank_diagnostic": previous.contract()["rank_diagnostic"],
        "label_boundary": "Stage249_futures_reused_development; no_Stage248_reads_no_Stage169_reserved_prefixes_no_arm_winner_selection",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "auxiliary_free_solves": 5376, "new_shrunk_solves": 5376,
            "new_native_episodes": 0, "new_raw_reference_fits": 0, "actor_updates": 0, "new_prefixes": 0},
        "storage": previous.contract()["storage"], "boundary": "development_diagnostic_not_fresh_confirmation_or_end_to_end_HRL_no_default_change"}


def fit_cell(run, key, views):
    load = lambda source, name: json.loads((raw_directory(result_path(source, *key))/name).read_text())
    info, controls = load(SOURCE_RUN, "aggregate.json"), load(SOURCE_RUN, "models.json")
    if info["protocol"] != previous.EXPERIMENT_PROTOCOL or info["contract"] != previous.contract() or not info["coverage_qualified"]:
        raise ValueError("Require qualified Stage250 cell")
    cached, ledger = load(information.SOURCE_RUN, "models.json"), load(fixed.SOURCE_RUN, "fits.json")
    old_blocks = {(tuple(b["held_group"]), b["half"]): b for b in ledger["blocks"]}
    reference = load(fixed.contract()["reference_run"], "models.json")["reference"]
    refs = load(fixed.contract()["refit_run"], "models.json")["references"]
    trials = load(fixed.contract()["refit_run"], "result.json")["cell"]["trials"]
    lookup = {(tuple(t["removed_group"]), t["half"]): str(i) for i, t in enumerate(trials)
        if t["removed_group"] and t["half"] is not None}
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    panels, past = information.read_panels(DATA_RUN), fixed.panel_cache()[0]
    a, b = fixed.PARTITIONS[key[2]][key[3]], fixed.PARTITIONS[key[2]][1-key[3]]
    actions, output, model_cache = [], [], {}
    for bank in (0, 1):
        blocks = []
        for group, indices, validation in learning.joint.binding.policy.prefix_folds(train):
            rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
            for half in (0, 1):
                index = lookup[tuple(group), half]
                data, val = learning.base.causal_data(rows, refs[index]), learning.base.causal_data(val_rows, refs[index])
                ds, _ = information.learning.draw_assignment(bank, "expanded", half)
                models, fit = learning.fit_paths(rows, data, learning.base.training_draws(rows, panels, ds),
                    cached[index]["all_mse"], old_blocks[tuple(group), half]["fits"])
                learning.qualify_fit(fit)
                _, _, dv = fixed.fixed.draw_assignment(a, b, half)
                truth = learning.base.training_draws(val_rows, past, dv)
                curves = {c: learning.base.curve(val, truth, p) for c, p in learning.base.candidates(val, models["shrunk"]).items()}
                blocks.append({"held_group": list(group), "half": half, "scales": old_blocks[tuple(group), half]["scales"],
                    "curves": {"shrunk": curves, "all_mse": curves}})
        selected, summaries = learning.select(blocks)
        source_bank = next(x for x in info["banks"] if x["bank"] == bank)
        selected.update({arm: source_bank["selection"][arm] for arm in ("old", "frozen_old", "free_old")})
        data = learning.base.causal_data(train, reference)
        ds, evaluation = information.learning.draw_assignment(bank, "expanded")
        models, full_fit = learning.fit_paths(train, data, learning.base.training_draws(train, panels, ds),
            cached["full"]["all_mse"], ledger["full"])
        learning.qualify_fit(full_fit)
        models.update({arm: controls[str(bank)][arm] for arm in ("old", "frozen_old", "free_old")})
        model_cache[str(bank)] = models
        frozen = []
        for scope, rows in (("train60", train), ("held60", held)):
            d = data if scope == "train60" else learning.base.causal_data(rows, reference)
            paths = {arm: learning.base.candidates(d, m) for arm, m in models.items()}
            pred = learning.predict(paths, selected)
            actions.append({"bank": bank, "scope": scope, "states": [list(information.old_future.learning.state_key(r)) for r in rows],
                "choices": {v: p["chosen"].tolist() for v, p in pred.items()}})
            frozen.append((scope, rows, d, pred, paths))
        write_json(raw_directory(result_path(run, *key))/"actions.json", actions)
        matrix, diagnostics = {}, {}
        for scope, rows, d, pred, paths in frozen:
            values, training_values = learning.base.training_draws(rows, panels, evaluation), learning.base.training_draws(rows, panels, ds)
            matrix[scope] = learning.balanced.metrics(d, values, pred)
            diagnostics[scope] = information.learning.information_error(training_values, values, d, pred)
            diagnostics[scope]["ranks"] = {arm: {k: v for k, v in information.learning.information_error(
                training_values, values, d, information.learning.rank_predictions(paths[arm])).items()
                if k in ("predictor_error", "predictor_error_on_training_bank")} for arm in learning.ARMS}
        output.append({"bank": bank, "selection": selected, "summaries": summaries, "matrix": matrix,
            "diagnostics": diagnostics, "full_fit": full_fit})
        print(f"HEAD_SHRINKAGE_BANK_COMPLETE {key}/{bank}", flush=True)
    expected = load(SOURCE_RUN, "actions.json")
    comparisons = 0
    if len(actions) != len(expected): raise ValueError("Dropped frozen bank/scope")
    for now, before in zip(actions, expected):
        if (now["bank"], now["scope"], now["states"]) != (before["bank"], before["scope"], before["states"]):
            raise ValueError("Changed frozen control state order")
        for name in ("raw", *(f"{a}_{f}" for a in ("old", "frozen_old", "free_old") for f in learning.FILTERS)):
            if now["choices"][name] != before["choices"][name]: raise ValueError("Changed frozen control actions")
            comparisons += len(now["states"])
    path = result_path(run, *key)
    write_json(raw_directory(path)/"models.json", model_cache)
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "coverage_qualified": True,
        "banks": output, "frozen_reproduction": {"action_exposures_compared": comparisons, "mismatches": 0},
        "new_native_episodes": 0, "new_raw_reference_fits": 0}
    write_json(raw_directory(path)/"aggregate.json", result)
    write_json(path, {k: v for k, v in result.items() if k != "banks"})


def difference_error(errors):
    return {m: 2*(errors["service"][m]+errors["fleet"][m])-errors["cost"][m]
        for m in ("nonanchor_mse", "feasible_pair_mse")}


def compact(records, scope, bank):
    bs = [b for r in records for b in r["banks"] if bank is None or b["bank"] == bank]
    ds = [b["diagnostics"][scope] for b in bs]
    matrix = learning.aggregate([b["matrix"][scope] for b in bs])
    outcomes = matrix["all"]["variants"]
    errors = {v: average([d["predictor_error"][v] for d in ds]) for v in learning.VARIANTS}
    fits = {level: [x for b in bs for x in b["full_fit"]["fits"] if x["level"] == level]
        for level in learning.strength.lowrank.LEVELS[1:]}
    return {"model_bank_exposures": len(bs), "outcomes": outcomes, "outcomes_by_root": matrix["by_root"],
        "contrasts": {f"shrunk-{a}": {f: outcomes['shrunk_pooled_safe'][f]-outcomes[f'{a}_pooled_safe'][f]
            for f in ("delta_cost", "delta_wait_min", "delta_risk", "delta_service")} for a in ("old", "frozen_old", "free_old")},
        "error_contrasts": {f"shrunk-{a}": {h: {m: errors['shrunk_pooled_safe'][h][m]-errors[f'{a}_pooled_safe'][h][m]
            for m in errors['shrunk_pooled_safe'][h]} for h in errors['shrunk_pooled_safe']} for a in ("old", "frozen_old", "free_old")},
        "models": {v: {"evaluation": e, "difference_error": difference_error(e),
            "training_bank": average([d["predictor_error_on_training_bank"][v] for d in ds])} for v, e in errors.items()},
        "ranks": {a: average([d["ranks"][a] for d in ds]) for a in learning.ARMS},
        "evaluation_mc_noise": average([d["evaluation_mc_noise"] for d in ds]),
        "zero_prediction_error": average([d["zero_prediction_error"] for d in ds]),
        "admitted": {a: sum(b["selection"][a]["pooled_safe"] is not None for b in bs) for a in ("old", *learning.ARMS)},
        "full_fits": {level: {"fits": len(xs), "frozen_difference": sum(x["penalty"] is None for x in xs),
            "mean": average([{f: x[f] for f in ("radius", "norm", "noise_energy", "weighted_noise_energy", "signal_energy",
                "free_difference_change_rms", "difference_change_rms", "data_loss_after", "free_data_loss_after")} for x in xs]),
            "max_kkt": max(max(x["optimality"], x["difference_optimality"]) for x in xs),
            "finite_penalties": [x["penalty"] for x in xs if x["penalty"] is not None]} for level, xs in fits.items()}}


def report(run):
    records = [json.loads((raw_directory(result_path(run, *k))/"aggregate.json").read_text()) for k in KEYS]
    for key, r in zip(KEYS, records):
        if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) != key or not r["coverage_qualified"]:
            raise ValueError("Incomplete difference-shrinkage matrix")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "groups": {name: {scope: {label: compact([r for r in records if predicate(r["key"])], scope, bank)
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1))} for scope in ("train60", "held60")}
            for name, predicate in groups.items()},
        "frozen_reproduction": {"action_exposures_compared": sum(r["frozen_reproduction"]["action_exposures_compared"] for r in records), "mismatches": 0},
        "interpretation": "reused development-future crossfit, not independent seed CI; training_bank fit-exposed on train60 except Stage247 old",
        "default_changed": False, "native_deployment_eligible": False}
    write_json(ROOT/"results"/run/"report/result.json", payload)
    print("HEAD_SHRINKAGE_REPORT_COMPLETE", flush=True)


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
