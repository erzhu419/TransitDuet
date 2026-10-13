#!/usr/bin/env python3
"""Select wait interaction shrinkage entirely inside each outer training split."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_conditional_wait_stage202 as source
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_nested_wait_stage203_v1"
SOURCE_RUN = "native_transit_conditional_wait_stage202_development_20261013_r1"
INTERACTION_RIDGES = (.1, 1., 10., 100.)
VARIANTS = (*source.VARIANTS, "regime_selected_wait")
SPLITS, state_key, wait = source.SPLITS, source.state_key, source.wait
coverage, mean, paired = source.source.diagnostic.coverage, source.source.diagnostic.mean, source.source.source.paired
FIT_FIELDS = (*source.FIT_FIELDS, "interaction_ridge")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "same600_inputs_240_mean8_targets_360_single_targets_47_outer_folds",
        "variants": list(VARIANTS), "primary": "regime_selected_wait",
        "cost": "exact_frozen_Stage199_two_cost_heads_reuse_all_Stage202_controls",
        "wait": "198_features_base102_ridge0.1_last96_interaction_penalty_selected_from_fixed_grid",
        "interaction_ridges": list(INTERACTION_RIDGES),
        "inner": "leave_each_outer_training_regime_out_all_scenes_callbacks_and_candidates_together",
        "selection": "min_worst_regime_mean8_physical_wait_pair_MSE_then_equal_regime_mean_then_stronger_penalty",
        "normalization": "each_inner_model_own_training_input_std_and_wait_pair_RMS_final_own_outer_train_scales",
        "weights": "training_only_mean8_weight8_single_weight1_all15_pairs_per_state",
        "execution": "reuse_each_inner_feature_matrix_and_normal_equations_for_all_four_penalties",
        "action": "zero_margin_wait_nonpositive_then_unchanged_cost_ranking_identity_exact0",
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean_label_splits",
        "budget": {"new_closed_form_fits": 967, "inner_wait_fits": 920, "final_wait_fits": 47,
            "reused_wait_models": 94, "reused_cost_models": 47,
            "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_targets_wait_models_inner_ledgers_predictions_server_only_small_reports_sync",
        "boundary": "nested_development_single_action_screen_not_independent_CI_or_full_HRL_validation"}


def manifest(previous):
    if previous["protocol"] != source.EXPERIMENT_PROTOCOL or previous["contract"] != source.contract():
        raise ValueError("Nested wait requires the exact Stage202 outer cohort")
    return {**previous, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def combine_scores(inner):
    return [{"interaction_ridge": penalty,
        "worst_regime_pair_mse_min2": max(r["pair_mse_min2"][str(penalty)] for r in inner),
        "equal_regime_mean_pair_mse_min2": float(np.mean([r["pair_mse_min2"][str(penalty)] for r in inner]))}
        for penalty in INTERACTION_RIDGES]


def choose_penalty(scores):
    return min(scores, key=lambda r: (r["worst_regime_pair_mse_min2"],
        r["equal_regime_mean_pair_mse_min2"], -r["interaction_ridge"]))["interaction_ridge"]


def heldout_error(raw, rows, model):
    x = (raw-model["center"])/model["scale"]
    original = [r["original_choice"] for r in rows]
    x -= x[np.arange(len(rows)), original][:, None].copy()
    predictions = (x@np.asarray(model["coefficients"]))*model["target_scale_min"]
    labels = [{l["candidate"]: l["wait_delta_min"] for l in row["labels"]} for row in rows]
    targets = np.asarray([[r[i] for i in range(6)] for r in labels])
    a, b = np.triu_indices(6, 1)
    return float(np.mean((predictions[:, a]-predictions[:, b]-(targets[:, a]-targets[:, b]))**2))


def select_within_training(rows, mean_keys):
    inner = []
    for regime in sorted({r["scenario"] for r in rows}):
        train = [r for r in rows if r["scenario"] != regime]
        validation = [r for r in rows if r["scenario"] == regime and state_key(r) in mean_keys]
        models = wait.fit_wait_path(train, "conditional_wait",
            sample_weights=[8 if state_key(r) in mean_keys else 1 for r in train], interaction_ridges=INTERACTION_RIDGES)
        raw = np.stack([wait.features(r, "conditional_wait") for r in validation])
        inner.append({"held_regime": regime, "training_groups": models[0]["training_groups"],
            "evaluation_groups": sorted(map(list, {paired.scene_group(r) for r in validation})),
            "training_states": len(train), "training_mean_states": sum(state_key(r) in mean_keys for r in train),
            "evaluation_mean_states": len(validation), "new_closed_form_fits": len(models),
            "normalization": {k: models[0][k] for k in ("center", "scale", "target_scale_min")},
            "pair_mse_min2": {str(p): heldout_error(raw, validation, m) for p, m in zip(INTERACTION_RIDGES, models, strict=True)}})
    scores = combine_scores(inner)
    return {"inner": inner, "scores": scores, "selected_ridge": choose_penalty(scores)}


def fit_fold(bank, targets, fold, prior):
    train, test, _ = coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {state_key(r) for r in targets}
    train = mean.expected.replace_targets(train, targets)
    test = mean.expected.replace_targets([r for r in test if state_key(r) in keys], targets)
    membership = sorted(map(list, {paired.scene_group(r) for r in train}))
    if prior["frozen_cost_fit"]["training_groups"] != membership:
        raise ValueError("Nested wait changed frozen cost training membership")
    start = time.monotonic()
    selection = select_within_training(train, keys)
    model = wait.fit_wait(train, "conditional_wait", sample_weights=[8 if state_key(r) in keys else 1 for r in train],
        interaction_ridge=selection["selected_ridge"])
    predictions = dict(prior["predictions"])
    frozen = {state_key(r): r for r in predictions["frozen_wait"]}
    predictions["regime_selected_wait"] = source.selection.evaluate(test, {state_key(r): {**frozen[state_key(r)],
        "predicted_wait_delta_min": wait.predict_wait(r, model).tolist()} for r in test}, 0.)
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "frozen_cost_fit": prior["frozen_cost_fit"], "control_fits": prior["fits"],
        "fit": {k: model[k] for k in FIT_FIELDS}, "selection": selection, "predictions": predictions,
        "new_closed_form_fits": 4*len(selection["inner"])+1,
        "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0,
        "fit_and_evaluation_wall_s": time.monotonic()-start}, model


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if source.source.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    regimes = sorted({g["scenario"] for g in groups})
    mean_states = [r for r in registered["mean_states"] if paired.scene_group(r) in {tuple(g) for g in membership}]
    test_keys = {state_key(r) for r in registered["mean_states"] if source.source.group_key(r, fold["split"]) == fold["held_out"]}
    model, selection = cell["fit"], cell["selection"]
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["new_closed_form_fits"] == 4*len(regimes)+1
            and cell["critic_optimizer_updates"] == cell["actor_updates"] == cell["new_native_episodes"] == 0
            and cell["frozen_cost_fit"]["training_groups"] == membership
            and model["training_groups"] == membership and model["training_states"] == 6*len(groups)
            and model["parameter_count"] == 198 and model["steps"] == 0 and model["ridge"] == .1
            and model["sample_weight_sum"] == 6*len(groups)+7*len(mean_states)
            and model["interaction_ridge"] == selection["selected_ridge"]
            and selection["scores"] == combine_scores(selection["inner"])
            and selection["selected_ridge"] == choose_penalty(selection["scores"])
            and [r["held_regime"] for r in selection["inner"]] == regimes):
        raise ValueError("Changed nested selection, outer fitting population or budget")
    for inner in selection["inner"]:
        regime = inner["held_regime"]
        train = [g for g in groups if g["scenario"] != regime]
        evaluation = [r for r in mean_states if r["scenario"] == regime]
        if not (inner["training_groups"] == sorted([[g["root"], g["scene_seed"]] for g in train])
                and inner["evaluation_groups"] == sorted(map(list, {paired.scene_group(r) for r in evaluation}))
                and inner["training_states"] == 6*len(train) and inner["evaluation_mean_states"] == len(evaluation)
                and inner["training_mean_states"] == len(mean_states)-len(evaluation)
                and inner["new_closed_form_fits"] == 4 and set(inner["pair_mse_min2"]) == set(map(str, INTERACTION_RIDGES))):
            raise ValueError("Inner regime selection leaked a held-out population or changed the grid")
    frozen = cell["predictions"]["frozen_wait"]
    for variant in VARIANTS:
        rows = cell["predictions"][variant]
        if len(rows) != len(test_keys) or {state_key(r) for r in rows} != test_keys:
            raise ValueError("Nested wait changed its outer evaluation cohort")
        for old, row in zip(frozen, rows, strict=True):
            original = row["original_choice"]
            if (state_key(old) != state_key(row) or old["predicted_heads"] != row["predicted_heads"]
                    or row["predicted_wait_delta_min"][original] != 0. or source.selection.choose(row, original, 0.) != row["selected"]):
                raise ValueError("Nested wait changed frozen costs, zero-margin selection or identity")


def summarize(cells, registered, targets):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete all47 nested wait folds before reporting")
    for fold in registered["folds"]: qualify(cells[fold["split"], fold["index"]], fold, registered)
    by_key = {state_key(r): r for r in targets}
    variants = {}
    for variant in VARIANTS:
        validation = {}
        for split in SPLITS:
            rows = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][variant]]
            if len(rows) != 240 or {state_key(r) for r in rows} != set(by_key):
                raise ValueError("Nested wait summary lost a registered mean8 state")
            errors = source.source.diagnostic.pair_records([by_key[state_key(r)] for r in rows], rows)
            validation[split] = {"overall": source.source.choice_summary(rows),
                "by_root": {str(root): source.source.choice_summary([r for r in rows if r["root"] == root]) for root in mean.ROOTS},
                "wait_pair_error": source.source.diagnostic.summary(errors)["pair_metrics"]["wait_delta_min"]}
        gates = {"cost": all(x["mean_actual_cost_delta"] < 0 for s in validation.values() for x in s["by_root"].values()),
            "wait": all(x["mean_wait_delta_min"] <= 0 for s in validation.values() for x in s["by_root"].values())}
        variants[variant] = {"validation": validation, "gates": gates}
    actual_fits = sum(c["new_closed_form_fits"] for c in cells.values())
    if actual_fits != contract()["budget"]["new_closed_form_fits"]: raise ValueError("Changed registered nested fit count")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "evaluation_states": 240, "evaluation_prefix_groups": 40, "variants": variants,
        "selected_penalties": {s: dict(Counter(str(c["selection"]["selected_ridge"]) for (split, _), c in cells.items() if split == s)) for s in SPLITS},
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
    if registered != manifest(previous): raise ValueError("Changed registered nested wait protocol")
    targets = json.loads(mean.expected.targets_path(source.source.source.SOURCE_RUN).read_text())["targets"]
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered, targets)
        write_json(directory/"report/result.json", result)
        print(json.dumps({v: d["gates"] for v, d in result["variants"].items()}, indent=2), flush=True)
        print("NATIVE_NESTED_WAIT_REPORT_COMPLETE", flush=True)
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
    write_json(raw_directory(output)/"regime_selected_wait.json", model)
    write_json(output, {k: v for k, v in result.items() if k not in ("fit", "selection", "predictions", "control_fits", "frozen_cost_fit")})
    print(f"NESTED_WAIT_FIT split={args.split} fold={args.fold_index} penalty={result['selection']['selected_ridge']} fit_s={result['fit_and_evaluation_wall_s']:.3f}", flush=True)
    print("NATIVE_NESTED_WAIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
