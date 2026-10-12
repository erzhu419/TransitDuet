#!/usr/bin/env python3
"""Replace only wait credit over the frozen Stage199 physical-cost rankings."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_pair_ridge_stage199 as source
from freq_hrl.domains.transit import native_conditional_wait as wait
from freq_hrl.domains.transit import native_wait_calibration as selection
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_conditional_wait_stage202_v1"
SOURCE_RUN = "native_transit_pair_ridge_stage199_development_20261013_r1"
DIAGNOSTIC_RUN = "native_transit_wait_selection_stage201_diagnostic_20261013_r1"
VARIANTS = ("frozen_wait", *wait.REPRESENTATIONS)
SPLITS, state_key = source.SPLITS, source.source.state_key
FIT_FIELDS = ("method", "representation", "training_groups", "training_states", "sample_weight_sum",
    "parameter_count", "steps", "ridge", "center", "scale", "target_scale_min", "final_pair_loss")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "diagnostic_run": DIAGNOSTIC_RUN, "data": "same600_inputs_240_mean8_targets_360_single_targets_47_outer_folds",
        "variants": list(VARIANTS), "primary": "conditional_wait",
        "cost": "exact_frozen_Stage199_count_weighted_ridge_two_cost_heads_and_ranking",
        "wait": "two_wait_only_weighted_pair_ridge_fits_all15_pairs_per_training_state_ridge0.1",
        "features": {"base": 98, "geometry_wait": 102, "conditional_wait": 198,
            "geometry": list(wait.GEOMETRY_NAMES), "conditions": list(wait.STATE_NAMES)},
        "weights": "same_mean8_weight8_single_weight1_all_training_candidates_not_selected_only",
        "normalization": "own_unweighted_training_only_raw_input_std_and_wait_pair_RMS",
        "selection": "zero_margin_predicted_wait_nonpositive_then_frozen_cost_min_identity_exact_zero_and_tie_retained",
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean_label_splits",
        "budget": {"new_closed_form_fits": 94, "reused_cost_models": 47,
            "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_targets_wait_models_predictions_server_only_small_receipts_and_report_sync",
        "boundary": "development_supervised_single_action_screen_not_matched_capacity_CI_or_full_learned_HRL"}


def manifest(previous):
    if previous["protocol"] != source.EXPERIMENT_PROTOCOL or previous["contract"] != source.contract():
        raise ValueError("Conditional wait requires the exact Stage199 cohort")
    return {**previous, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def fit_fold(bank, targets, fold, prior):
    train, test, _ = source.diagnostic.coverage.split_rows(bank, fold["split"], fold["index"])
    mean_keys = {state_key(r) for r in targets}
    train = source.diagnostic.mean.expected.replace_targets(train, targets)
    test = source.diagnostic.mean.expected.replace_targets([r for r in test if state_key(r) in mean_keys], targets)
    cost_fit = prior["fits"]["ridge_count_weighted"]
    if cost_fit["training_groups"] != sorted(map(list, {source.source.paired.scene_group(r) for r in train})):
        raise ValueError("Conditional wait changed frozen cost training membership")
    reference = prior["predictions"]["ridge_count_weighted"]["wait_feasible"]
    frozen = {state_key(r): r for r in reference}
    models, predictions = {}, {"frozen_wait": reference}
    start = time.monotonic()
    for variant in wait.REPRESENTATIONS:
        model = wait.fit_wait(train, variant, sample_weights=source.source.sample_counts(train, targets))
        predictions[variant] = selection.evaluate(test, {state_key(r): {**frozen[state_key(r)],
            "predicted_wait_delta_min": wait.predict_wait(r, model).tolist()} for r in test}, 0.)
        models[variant] = model
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "frozen_cost_fit": {k: cost_fit[k] for k in source.FIT_FIELDS},
        "fits": {v: {k: m[k] for k in FIT_FIELDS} for v, m in models.items()}, "predictions": predictions,
        "new_closed_form_fits": 2, "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0,
        "fit_and_evaluation_wall_s": time.monotonic()-start}, models


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if source.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    keys = {state_key(r) for r in registered["mean_states"] if source.group_key(r, fold["split"]) == fold["held_out"]}
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["new_closed_form_fits"] == 2
            and cell["critic_optimizer_updates"] == cell["actor_updates"] == cell["new_native_episodes"] == 0
            and cell["frozen_cost_fit"]["training_groups"] == membership
            and cell["frozen_cost_fit"]["training_states"] == 6*len(groups)
            and cell["frozen_cost_fit"]["parameter_count"] == 294):
        raise ValueError("Changed frozen cost predictor, training population or fitting budget")
    for variant in wait.REPRESENTATIONS:
        fit = cell["fits"][variant]
        if not (fit["representation"] == variant and fit["training_groups"] == membership
                and fit["training_states"] == 6*len(groups) and fit["steps"] == 0 and fit["ridge"] == .1
                and fit["parameter_count"] == wait.FEATURE_COUNTS[variant]
                and fit["sample_weight_sum"] == 6*len(groups)+7*(240-len(keys))):
            raise ValueError("Changed conditional wait normalization population, features or weights")
    frozen = cell["predictions"]["frozen_wait"]
    for variant in VARIANTS:
        rows = cell["predictions"][variant]
        if len(rows) != len(keys) or {state_key(r) for r in rows} != keys:
            raise ValueError("Conditional wait changed its held-out target population")
        for old, row in zip(frozen, rows, strict=True):
            original = row["original_choice"]
            if (state_key(row) != state_key(old) or row["predicted_heads"] != old["predicted_heads"]
                    or row["predicted_wait_delta_min"][original] != 0. or selection.choose(row, original, 0.) != row["selected"]):
                raise ValueError("Conditional wait changed frozen costs, zero-margin filter or paired identity")


def summarize(cells, registered, targets):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete all47 conditional wait folds before aggregation")
    for fold in registered["folds"]: qualify(cells[fold["split"], fold["index"]], fold, registered)
    by_key = {state_key(r): r for r in targets}
    variants = {}
    for variant in VARIANTS:
        validation = {}
        for split in SPLITS:
            rows = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][variant]]
            if len(rows) != 240 or {state_key(r) for r in rows} != set(by_key):
                raise ValueError("Conditional wait summary lost a registered mean8 state")
            errors = source.diagnostic.pair_records([by_key[state_key(r)] for r in rows], rows)
            validation[split] = {"overall": source.choice_summary(rows),
                "by_root": {str(root): source.choice_summary([r for r in rows if r["root"] == root]) for root in source.diagnostic.mean.ROOTS},
                "wait_pair_error": source.diagnostic.summary(errors)["pair_metrics"]["wait_delta_min"]}
        gates = {"cost": all(x["mean_actual_cost_delta"] < 0 for s in validation.values() for x in s["by_root"].values()),
            "wait": all(x["mean_wait_delta_min"] <= 0 for s in validation.values() for x in s["by_root"].values())}
        variants[variant] = {"validation": validation, "gates": gates}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "evaluation_states": 240, "evaluation_prefix_groups": 40, "variants": variants,
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
    if registered != manifest(previous): raise ValueError("Changed registered conditional wait cohort")
    targets = json.loads(source.diagnostic.mean.expected.targets_path(source.source.SOURCE_RUN).read_text())["targets"]
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered, targets)
        write_json(directory/"report/result.json", result)
        print(json.dumps({v: d["gates"] for v, d in result["variants"].items()}, indent=2), flush=True)
        print("NATIVE_CONDITIONAL_WAIT_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None: parser.error("Fit requires split and fold index")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    bank = json.loads(source.diagnostic.coverage.bank_path(source.diagnostic.mean.SOURCE_RUN).read_text())
    prior = json.loads((raw_directory(source.result_path(SOURCE_RUN, fold))/"result.json").read_text())
    result, models = fit_fold(bank, targets, fold, prior)
    qualify(result, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", result)
    for variant, model in models.items(): write_json(raw_directory(output)/f"{variant}.json", model)
    write_json(output, {k: v for k, v in result.items() if k not in ("fits", "predictions", "frozen_cost_fit")})
    print(f"CONDITIONAL_WAIT_FIT split={args.split} fold={args.fold_index} fit_s={result['fit_and_evaluation_wall_s']:.3f}", flush=True)
    print("NATIVE_CONDITIONAL_WAIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
