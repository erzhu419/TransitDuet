#!/usr/bin/env python3
"""Test a lower-capacity paired critic after the frozen-model transfer audit."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_weighted_credit_stage197 as source
from scripts import analyze_native_transit_fit_transfer_stage198 as diagnostic
from scripts.analyze_native_transit_weighted_credit_stage197 import choice_summary
from scripts.analyze_native_transit_coverage_value_stage188 import group_key
from freq_hrl.domains.transit import native_pair_ridge as ridge
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_pair_ridge_stage199_v1"
SOURCE_RUN = diagnostic.SOURCE_RUN
DIAGNOSTIC_RUN = "native_transit_fit_transfer_stage198_diagnostic_20261013_r1"
VARIANTS = ("neural_uniform", "neural_count_weighted", "ridge_uniform", "ridge_count_weighted")
SPLITS = source.SPLITS
FIT_FIELDS = ("method", "training_groups", "training_states", "steps", "parameter_count", "center", "scale", "target_scale")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "diagnostic_run": DIAGNOSTIC_RUN, "data": "unchanged600_inputs_240_mean8_targets_360_single_targets_47_folds",
        "variants": list(VARIANTS), "primary": "ridge_count_weighted:wait_feasible",
        "ridge": "existing_native_action_value_RIDGE0.1_three_heads_98_features_294_coefficients_no_intercept",
        "normalization": "exact_Stage197_neural_advantage_training_input_and_target_scales_within_each_fold",
        "weights": "uniform_or_mean8_weight8_single_weight1_training_states_only",
        "reference": "reuse94_frozen_neural_advantage_models_and_exact_heldout_predictions_1635_parameters_100updates",
        "evaluation": "same240_mean8_states_both_roots_all_three_grouped_splits_both_fixed_selectors",
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean_label_splits",
        "budget": {"new_closed_form_fits": 94, "reused_neural_models": 94,
            "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "features_mean_map_models_predictions_server_only_compact_receipts_report_sync",
        "boundary": "lower_capacity_regularized_baseline_not_matched_capacity_or_optimizer_budget_or_HRL_proof"}


def manifest(previous):
    if previous["protocol"] != source.EXPERIMENT_PROTOCOL or previous["contract"] != source.contract():
        raise ValueError("Pair-ridge baseline requires the exact Stage197 grouped cohort")
    return {**previous, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def fit_fold(bank, targets, fold, references, prior):
    train, test, _ = diagnostic.coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {source.state_key(r) for r in targets}
    train = diagnostic.mean.expected.replace_targets(train, targets)
    test = diagnostic.mean.expected.replace_targets([r for r in test if source.state_key(r) in keys], targets)
    models, predictions = {}, {}
    start = time.monotonic()
    for weighting in ("uniform", "count_weighted"):
        old = f"advantage_{weighting}"
        reference = references[old]
        models[f"neural_{weighting}"] = reference
        model = ridge.fit_value(train, reference,
            sample_weights=source.sample_counts(train, targets) if weighting == "count_weighted" else None)
        if any(model[k] != reference[k] for k in ("training_groups", "training_states", "center", "scale", "target_scale")):
            raise ValueError("Pair-ridge comparison changed data membership or normalization")
        models[f"ridge_{weighting}"] = model
        predictions[f"neural_{weighting}"] = prior["predictions"][old]["mean8"]
        predictions[f"ridge_{weighting}"] = {p: ridge.evaluate(test, model, p) for p in source.value.POLICIES}
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "fits": {v: {**{k: m[k] for k in FIT_FIELDS}, "sample_weight_sum": m.get("sample_weight_sum", len(train))}
            for v, m in models.items()}, "training_mean_states": sum(source.state_key(r) in keys for r in train),
        "evaluation_mean_states": len(test), "predictions": predictions,
        "new_closed_form_fits": 2, "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0,
        "fit_and_evaluation_wall_s": time.monotonic()-start}
    return result, models


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    keys = {source.state_key(r) for r in registered["mean_states"] if group_key(r, fold["split"]) == fold["held_out"]}
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["new_closed_form_fits"] == 2 and cell["critic_optimizer_updates"] == 0
            and cell["actor_updates"] == cell["new_native_episodes"] == 0):
        raise ValueError("Changed pair-ridge baseline or training budget")
    for variant in VARIANTS:
        fit = cell["fits"][variant]
        reference = cell["fits"]["neural_uniform"]
        if not (fit["training_groups"] == membership and fit["training_states"] == 6*len(groups)
                and fit["parameter_count"] == (294 if variant.startswith("ridge") else 1635)
                and fit["steps"] == (0 if variant.startswith("ridge") else 100)
                and fit["sample_weight_sum"] == 6*len(groups)+(7*(240-len(keys)) if variant.endswith("count_weighted") else 0)
                and all(fit[k] == reference[k] for k in ("center", "scale", "target_scale"))):
            raise ValueError("Changed pair-ridge groups, scales, capacity or weights")
        for policy, rows in cell["predictions"][variant].items():
            if len(rows) != len(keys) or {source.state_key(r) for r in rows} != keys:
                raise ValueError("Changed held-out pair-ridge evaluation cohort")
            for row in rows:
                heads, waits = np.asarray(row["predicted_heads"]), np.asarray(row["predicted_wait_delta_min"])
                cost, original = heads.sum(axis=1), row["original_choice"]
                if policy == "wait_feasible": cost = np.where(waits <= 0, cost, np.inf)
                if row["selected"] != source.paired.choice_from_cost(cost, original) or not np.all(heads[original] == 0) or waits[original] != 0:
                    raise ValueError("Changed fixed selector or paired identity")


def summarize(cells, registered, targets, previous):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete all47 pair-ridge folds before aggregation")
    for fold in registered["folds"]: qualify(cells[fold["split"], fold["index"]], fold, registered)
    by_key = {source.state_key(r): r for r in targets}
    variants = {}
    for variant in VARIANTS:
        for policy in source.value.POLICIES:
            splits = {}
            for split in SPLITS:
                rows = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][variant][policy]]
                error = diagnostic.summary(diagnostic.pair_records([by_key[source.state_key(r)] for r in rows], rows))
                noise = previous["variants"]["advantage_uniform:wait_feasible"]["pair_error"][split]["overall"]["pair_metrics"]
                for field, metrics in error["pair_metrics"].items():
                    metrics.update(mean_label_noise=noise[field]["mean_label_noise"],
                        noise_corrected_pair_mse=metrics["observed_pair_mse"]-noise[field]["mean_label_noise"])
                splits[split] = {"overall": choice_summary(rows),
                    "by_root": {str(root): choice_summary([r for r in rows if r["root"] == root]) for root in diagnostic.mean.ROOTS},
                    "pair_errors": error}
            gates = {"cost": all(x["mean_actual_cost_delta"] < 0 for s in splits.values() for x in s["by_root"].values()),
                "wait": all(x["mean_wait_delta_min"] <= 0 for s in splits.values() for x in s["by_root"].values())}
            variants[f"{variant}:{policy}"] = {"validation": splits, "gates": gates}
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
    if registered != manifest(previous): raise ValueError("Changed registered pair-ridge cohort")
    targets = json.loads(diagnostic.mean.expected.targets_path(source.SOURCE_RUN).read_text())["targets"]
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered, targets, json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text()))
        write_json(directory/"report/result.json", result)
        print(json.dumps({k: v["gates"] for k, v in result["variants"].items()}, indent=2), flush=True)
        print("NATIVE_PAIR_RIDGE_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None: parser.error("Fit requires split and fold index")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    path = raw_directory(source.result_path(SOURCE_RUN, fold))
    references = {v: json.loads((path/f"{v}.json").read_text()) for v in ("advantage_uniform", "advantage_count_weighted")}
    bank = json.loads(diagnostic.coverage.bank_path(diagnostic.mean.SOURCE_RUN).read_text())
    result, models = fit_fold(bank, targets, fold, references, json.loads((path/"result.json").read_text()))
    qualify(result, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", result)
    for variant in ("ridge_uniform", "ridge_count_weighted"): write_json(raw_directory(output)/f"{variant}.json", models[variant])
    write_json(output, {k: v for k, v in result.items() if k not in ("fits", "predictions")})
    print(f"PAIR_RIDGE_FIT split={args.split} fold={args.fold_index} wall_s={result['fit_and_evaluation_wall_s']:.3f}", flush=True)
    print("NATIVE_PAIR_RIDGE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
