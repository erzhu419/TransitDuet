#!/usr/bin/env python3
"""Select wait shrinkage by inner held-out decisions, not prediction error alone."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_nested_wait_stage203 as source
from freq_hrl.domains.transit import native_pair_ridge as cost
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_decision_wait_stage204_v1"
SOURCE_RUN = "native_transit_nested_wait_stage203_development_20261013_r1"
VARIANTS = (*source.VARIANTS, "decision_selected_wait")
INTERACTION_RIDGES, SPLITS = source.INTERACTION_RIDGES, source.SPLITS
wait, mean, coverage, paired, state_key = source.wait, source.mean, source.coverage, source.paired, source.state_key
credit = source.source.source


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "same600_inputs_240_mean8_targets_360_single_targets_47_outer_folds",
        "variants": list(VARIANTS), "primary": "decision_selected_wait",
        "cost": "outer_Stage199_cost_heads_and_all_four_Stage203_controls_frozen",
        "inner_cost": "one_train_only98_feature_three_head_pair_ridge0.1_per_inner_regime_wait_head_unused",
        "wait": "unchanged198_features_base102_ridge0.1_interaction_grid0.1_1_10_100",
        "inner": "outer_training_regime_holdout_all_scene_callbacks_and_candidates_grouped_mean8_only_validation",
        "criterion": "min_positive_worst_root_regime_actual_wait_then_worst_root_equal_regime_cost_then_equal_root_cost_then_worst_pair_MSE_then_stronger_penalty",
        "infeasible": "least_wait_violation_no_threshold_relaxation_or_claim_of_guaranteed_safety",
        "normalization": "inner_cost_and_wait_own_inner_train_scales_final_wait_own_outer_train_scales",
        "weights": "training_only_mean8_weight8_single_weight1_all15_action_pairs",
        "selection": "zero_margin_wait_nonpositive_then_cost_min_identity_exact0_and_tie_retained",
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean_label_splits",
        "budget": {"new_closed_form_fits": 1197, "inner_cost_fits": 230, "inner_wait_fits": 920,
            "final_wait_fits": 47, "reused_outer_cost_models": 47, "reused_outer_wait_models": 141,
            "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_targets_models_inner_decision_ledgers_predictions_server_only_small_report_sync",
        "boundary": "nested_supervised_single_action_development_not_independent_CI_or_full_HRL_validation"}


def manifest(previous):
    if previous["protocol"] != source.EXPERIMENT_PROTOCOL or previous["contract"] != source.contract():
        raise ValueError("Decision-level selection requires the exact Stage203 outer cohort")
    return {**previous, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def centered(raw, rows, model):
    x = (raw-model["center"])/model["scale"]
    x -= x[np.arange(len(rows)), [r["original_choice"] for r in rows]][:, None].copy()
    return x


def cost_predictions(raw, rows, model):
    return ((centered(raw[..., :98], rows, model)@np.asarray(model["coefficients"]))*model["target_scale"])[..., :2]


def decision_metrics(rows, heads, waits):
    observed = []
    for row, prediction, predicted_wait in zip(rows, heads, waits, strict=True):
        selected = paired.choice_from_cost(np.where(predicted_wait <= 0, prediction.sum(axis=1), np.inf), row["original_choice"])
        label = next(l for l in row["labels"] if l["candidate"] == selected)
        observed.append({"root": row["root"], "selected": selected,
            "changed": selected != row["original_choice"], "cost": label["actual_cost_delta"], "wait": label["wait_delta_min"]})
    def summary(records):
        return {"states": len(records), "changed_choices": sum(r["changed"] for r in records),
            "mean_actual_cost_delta": float(np.mean([r["cost"] for r in records])),
            "mean_wait_delta_min": float(np.mean([r["wait"] for r in records]))}
    return {"overall": summary(observed), "by_root": {str(root): summary([r for r in observed if r["root"] == root])
        for root in sorted({r["root"] for r in rows})}}


def combine_scores(inner):
    scores = []
    for penalty in INTERACTION_RIDGES:
        costs, waits = defaultdict(list), []
        for regime in inner:
            for root, metrics in regime["decisions"][str(penalty)]["by_root"].items():
                costs[root].append(metrics["mean_actual_cost_delta"])
                waits.append(metrics["mean_wait_delta_min"])
        root_costs = [float(np.mean(values)) for values in costs.values()]
        scores.append({"interaction_ridge": penalty, "wait_violation_min": max(0., max(waits)),
            "worst_root_equal_regime_cost": max(root_costs), "equal_root_cost": float(np.mean(root_costs)),
            "worst_regime_pair_mse_min2": max(r["pair_mse_min2"][str(penalty)] for r in inner)})
    return scores


def choose_penalty(scores):
    return min(scores, key=lambda r: (r["wait_violation_min"], r["worst_root_equal_regime_cost"],
        r["equal_root_cost"], r["worst_regime_pair_mse_min2"], -r["interaction_ridge"]))["interaction_ridge"]


def select_within_training(rows, mean_keys):
    inner = []
    for regime in sorted({r["scenario"] for r in rows}):
        train = [r for r in rows if r["scenario"] != regime]
        validation = [r for r in rows if r["scenario"] == regime and state_key(r) in mean_keys]
        weights = [8 if state_key(r) in mean_keys else 1 for r in train]
        price = cost.fit_value(train, sample_weights=weights)
        models = wait.fit_wait_path(train, "conditional_wait", sample_weights=weights, interaction_ridges=INTERACTION_RIDGES)
        raw = np.stack([wait.features(r, "conditional_wait") for r in validation])
        heads = cost_predictions(raw, validation, price)
        decisions = {str(p): decision_metrics(validation, heads,
            (centered(raw, validation, m)@np.asarray(m["coefficients"]))*m["target_scale_min"])
            for p, m in zip(INTERACTION_RIDGES, models, strict=True)}
        inner.append({"held_regime": regime, "training_groups": models[0]["training_groups"],
            "evaluation_groups": sorted(map(list, {paired.scene_group(r) for r in validation})),
            "training_states": len(train), "training_mean_states": sum(state_key(r) in mean_keys for r in train),
            "evaluation_mean_states": len(validation), "new_closed_form_fits": 5,
            "normalization": {k: models[0][k] for k in ("center", "scale", "target_scale_min")},
            "cost_fit": {k: price[k] for k in ("training_groups", "training_states", "parameter_count", "steps",
                "ridge", "sample_weight_sum", "center", "scale", "target_scale")},
            "decisions": decisions,
            "pair_mse_min2": {str(p): source.heldout_error(raw, validation, m) for p, m in zip(INTERACTION_RIDGES, models, strict=True)}})
    scores = combine_scores(inner)
    selected = choose_penalty(scores)
    return {"inner": inner, "scores": scores, "selected_ridge": selected,
        "inner_wait_feasible": next(r["wait_violation_min"] for r in scores if r["interaction_ridge"] == selected) == 0.}


def fit_fold(bank, targets, fold, prior):
    train, test, _ = coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {state_key(r) for r in targets}
    train = mean.expected.replace_targets(train, targets)
    test = mean.expected.replace_targets([r for r in test if state_key(r) in keys], targets)
    if prior["frozen_cost_fit"]["training_groups"] != sorted(map(list, {paired.scene_group(r) for r in train})):
        raise ValueError("Decision selection changed outer cost training membership")
    start = time.monotonic()
    selection = select_within_training(train, keys)
    model = wait.fit_wait(train, "conditional_wait", sample_weights=[8 if state_key(r) in keys else 1 for r in train],
        interaction_ridge=selection["selected_ridge"])
    predictions = dict(prior["predictions"])
    frozen = {state_key(r): r for r in predictions["frozen_wait"]}
    predictions["decision_selected_wait"] = source.source.selection.evaluate(test, {state_key(r): {**frozen[state_key(r)],
        "predicted_wait_delta_min": wait.predict_wait(r, model).tolist()} for r in test}, 0.)
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "frozen_cost_fit": prior["frozen_cost_fit"], "fit": {k: model[k] for k in source.FIT_FIELDS},
        "selection": selection, "predictions": predictions, "new_closed_form_fits": 5*len(selection["inner"])+1,
        "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0,
        "fit_and_evaluation_wall_s": time.monotonic()-start}, model


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    mean_states = [r for r in registered["mean_states"] if paired.scene_group(r) in {tuple(g) for g in membership}]
    test_keys = {state_key(r) for r in registered["mean_states"] if credit.group_key(r, fold["split"]) == fold["held_out"]}
    selection, model = cell["selection"], cell["fit"]
    regimes = sorted({g["scenario"] for g in groups})
    scores = combine_scores(selection["inner"])
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["new_closed_form_fits"] == 5*len(regimes)+1
            and cell["critic_optimizer_updates"] == cell["actor_updates"] == cell["new_native_episodes"] == 0
            and cell["frozen_cost_fit"]["training_groups"] == membership
            and model["training_groups"] == membership and model["training_states"] == 6*len(groups)
            and model["parameter_count"] == 198 and model["steps"] == 0 and model["ridge"] == .1
            and model["sample_weight_sum"] == 6*len(groups)+7*len(mean_states)
            and selection["scores"] == scores and selection["selected_ridge"] == choose_penalty(scores)
            and model["interaction_ridge"] == selection["selected_ridge"]
            and selection["inner_wait_feasible"] == (next(r["wait_violation_min"] for r in scores if r["interaction_ridge"] == selection["selected_ridge"]) == 0.)
            and [r["held_regime"] for r in selection["inner"]] == regimes):
        raise ValueError("Changed decision-level selection, outer training population or budget")
    for inner in selection["inner"]:
        train = [g for g in groups if g["scenario"] != inner["held_regime"]]
        evaluation = [r for r in mean_states if r["scenario"] == inner["held_regime"]]
        expected = sorted([[g["root"], g["scene_seed"]] for g in train])
        price = inner["cost_fit"]
        if not (inner["training_groups"] == price["training_groups"] == expected
                and inner["training_states"] == price["training_states"] == 6*len(train)
                and inner["evaluation_groups"] == sorted(map(list, {paired.scene_group(r) for r in evaluation}))
                and inner["evaluation_mean_states"] == len(evaluation)
                and inner["training_mean_states"] == len(mean_states)-len(evaluation)
                and price["parameter_count"] == 294 and price["steps"] == 0 and price["ridge"] == .1
                and price["sample_weight_sum"] == 6*len(train)+7*(len(mean_states)-len(evaluation))
                and inner["new_closed_form_fits"] == 5 and set(inner["decisions"]) == set(map(str, INTERACTION_RIDGES))):
            raise ValueError("Inner cost/wait fitting leaked a validation regime or changed training weights")
        for metrics in inner["decisions"].values():
            if (metrics["overall"]["states"] != len(evaluation) or
                    {r: m["states"] for r, m in metrics["by_root"].items()} != dict(Counter(str(r["root"]) for r in evaluation))):
                raise ValueError("Inner decision criterion lost a held-out root or state")
    frozen = cell["predictions"]["frozen_wait"]
    for variant in VARIANTS:
        rows = cell["predictions"][variant]
        if len(rows) != len(test_keys) or {state_key(r) for r in rows} != test_keys:
            raise ValueError("Decision selection changed its outer evaluation cohort")
        for old, row in zip(frozen, rows, strict=True):
            original = row["original_choice"]
            if (state_key(row) != state_key(old) or row["predicted_heads"] != old["predicted_heads"]
                    or row["predicted_wait_delta_min"][original] != 0. or source.source.selection.choose(row, original, 0.) != row["selected"]):
                raise ValueError("Decision selection changed frozen outer costs, zero margin or identity")


def summarize(cells, registered, targets):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete all47 decision-level folds before reporting")
    for fold in registered["folds"]: qualify(cells[fold["split"], fold["index"]], fold, registered)
    by_key = {state_key(r): r for r in targets}
    variants = {}
    for variant in VARIANTS:
        validation = {}
        for split in SPLITS:
            rows = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][variant]]
            if len(rows) != 240 or {state_key(r) for r in rows} != set(by_key): raise ValueError("Lost a registered mean8 state")
            errors = credit.diagnostic.pair_records([by_key[state_key(r)] for r in rows], rows)
            validation[split] = {"overall": credit.choice_summary(rows),
                "by_root": {str(root): credit.choice_summary([r for r in rows if r["root"] == root]) for root in mean.ROOTS},
                "wait_pair_error": credit.diagnostic.summary(errors)["pair_metrics"]["wait_delta_min"]}
        gates = {"cost": all(x["mean_actual_cost_delta"] < 0 for s in validation.values() for x in s["by_root"].values()),
            "wait": all(x["mean_wait_delta_min"] <= 0 for s in validation.values() for x in s["by_root"].values())}
        variants[variant] = {"validation": validation, "gates": gates}
    if sum(c["new_closed_form_fits"] for c in cells.values()) != 1197: raise ValueError("Changed registered fit count")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "evaluation_states": 240, "evaluation_prefix_groups": 40, "variants": variants,
        "selected_penalties": {s: dict(Counter(str(c["selection"]["selected_ridge"]) for (split, _), c in cells.items() if split == s)) for s in SPLITS},
        "inner_wait_feasible_folds": {s: sum(c["selection"]["inner_wait_feasible"] for (split, _), c in cells.items() if split == s) for s in SPLITS},
        "development_gate": all(variants[contract()["primary"]]["gates"].values()),
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("fit", "report"), required=True)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--fold-index", type=int)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    previous = json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text())
    if registered != manifest(previous): raise ValueError("Changed decision-level selection protocol")
    targets = json.loads(mean.expected.targets_path(credit.source.SOURCE_RUN).read_text())["targets"]
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered, targets)
        write_json(directory/"report/result.json", result)
        print(json.dumps({v: d["gates"] for v, d in result["variants"].items()}, indent=2), flush=True)
        print("NATIVE_DECISION_WAIT_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None: parser.error("Fit requires split and fold index")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    prior = json.loads((raw_directory(source.result_path(SOURCE_RUN, fold))/"result.json").read_text())
    source.qualify(prior, fold, previous)
    bank = json.loads(coverage.bank_path(mean.SOURCE_RUN).read_text())
    result, model = fit_fold(bank, targets, fold, prior)
    qualify(result, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", result)
    write_json(raw_directory(output)/"decision_selected_wait.json", model)
    write_json(output, {k: v for k, v in result.items() if k not in ("fit", "selection", "predictions", "frozen_cost_fit")})
    print(f"DECISION_WAIT_FIT split={args.split} fold={args.fold_index} penalty={result['selection']['selected_ridge']} feasible={result['selection']['inner_wait_feasible']} fit_s={result['fit_and_evaluation_wall_s']:.3f}", flush=True)
    print("NATIVE_DECISION_WAIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
