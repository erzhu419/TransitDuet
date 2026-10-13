#!/usr/bin/env python3
"""Refit paired cost credit with mean8 labels and frozen learned wait masks."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_mean_wait_stage213 as previous
from scripts import run_native_transit_decision_wait_stage204 as base
from scripts import analyze_native_transit_bound_ranking_stage214 as diagnosis
from scripts.run_native_transit_nonlinear_wait_stage208 import evaluate, half_summary
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_mean_cost_stage215_v1"
SOURCE_RUN = diagnosis.SOURCE_RUN
DIAGNOSTIC_RUN = "native_transit_bound_ranking_stage214_diagnostic_20261013_r1"
WAIT_HEADS = {"glm": previous.NEW_GLM, "shrunk": previous.PRIMARY}
VARIANTS = tuple(f"{price}_{wait}" for price in ("old_cost", "mean_cost") for wait in WAIT_HEADS)
PRIMARY = "mean_cost_glm"
SPLITS, state_key = previous.SPLITS, previous.state_key


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL, "diagnostic_run": DIAGNOSTIC_RUN,
        "data": "same600_inputs_all600_mean8_cost_targets_original240_evaluation_states_47_folds",
        "variants": list(VARIANTS), "primary": PRIMARY,
        "cost": "same98_features294_coefficient_three_rhs_pair_ridge0.1_original_training_scales_fixed_unused_wait_rhs_ignored",
        "control": "reproduce_old240_mean8_360_single_target_pair_ridge_and_archived_cost_heads_before_comparison",
        "weights": previous.collection.contract()["weights"],
        "wait": "exact_Stage213_GLM_and_shrunk_harm_shares_scores_masks_bound_and_threshold0.5_frozen",
        "gate": "negative_cost_nonpositive_wait_both_roots_all_three_mean_label_splits",
        "future_halves": previous.source.diagnostic.contract()["partitions"],
        "budget": {"new_cost_fits": 47, "control_cost_refits": 47, "closed_form_solves": 94,
            "operational_wait_fits": 0, "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_targets_cost_models_and_predictions_remote_small_report_only",
        "boundary": "supervised_development_target_precision_comparison_not_independent_CI_or_full_learned_HRL"}


def manifest(old):
    if old["protocol"] != previous.EXPERIMENT_PROTOCOL or old["contract"] != previous.contract():
        raise ValueError("Mean cost requires the original Stage213 folds and wait controls")
    return {**old, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def fit_fold(bank, old_targets, all_targets, masses, fold, prior):
    train, test, _ = base.coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {state_key(r) for r in old_targets}
    test = base.mean.expected.replace_targets([r for r in test if state_key(r) in keys], old_targets)
    old_train, mean_train = [base.mean.expected.replace_targets(train, t) for t in (old_targets, all_targets)]
    weights = previous.source.feasibility.sample_weights(train, masses)
    reference = prior["frozen_cost_fit"]
    if reference["training_groups"] != sorted(map(list, {base.paired.scene_group(r) for r in train})):
        raise ValueError("Cost refit changed its original training population")
    start = time.monotonic()
    models = {"old_cost": base.cost.fit_value(old_train, reference, sample_weights=weights),
        "mean_cost": base.cost.fit_value(mean_train, reference, sample_weights=weights)}
    heads = {v: {state_key(r): base.cost.predict(r, m)["predicted_heads"] for r in test} for v, m in models.items()}
    frozen = {state_key(r): r for r in prior["predictions"][previous.NEW_GLM]}
    if any(not np.allclose(heads["old_cost"][state_key(r)], frozen[state_key(r)]["predicted_heads"], atol=1e-12, rtol=0) for r in test):
        raise ValueError("Old-label analytic fit failed to reproduce the archived cost predictor")
    predictions = {}
    for name, wait_variant in WAIT_HEADS.items():
        old = prior["predictions"][wait_variant]
        predictions[f"old_cost_{name}"] = old
        by_key = {state_key(r): {**r, "predicted_heads": heads["mean_cost"][state_key(r)]} for r in old}
        predictions[f"mean_cost_{name}"] = evaluate(test, by_key,
            [{k: by_key[state_key(r)][k] for k in ("predicted_harm_mass_share", "predicted_feasibility_score")} for r in test])
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "reference": reference, "fits": {v: {k: x for k, x in m.items() if k != "coefficients"} for v, m in models.items()},
        "old_cost_reproduced": True, "predictions": predictions, "closed_form_solves": 2,
        "operational_wait_fits": 0, "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0,
        "fit_and_evaluation_wall_s": time.monotonic()-start}, models


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if base.credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    keys = {state_key(r) for r in registered["mean_states"] if base.credit.group_key(r, fold["split"]) == fold["held_out"]}
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["old_cost_reproduced"] and cell["closed_form_solves"] == 2
            and cell["operational_wait_fits"] == cell["critic_optimizer_updates"] == cell["actor_updates"] == cell["new_native_episodes"] == 0):
        raise ValueError("Changed the paired cost-refit protocol or budget")
    for fit in cell["fits"].values():
        if not (fit["training_groups"] == membership and fit["training_states"] == 6*len(groups)
                and fit["parameter_count"] == 294 and fit["steps"] == 0 and fit["ridge"] == base.paired.RIDGE
                and fit["sample_weight_sum"] == 6*len(groups)+7*(240-len(keys))
                and all(fit[k] == cell["reference"][k] for k in ("center", "scale", "target_scale"))):
            raise ValueError("Changed critic capacity, regularization, scales or original weights")
    for name in WAIT_HEADS:
        old = {state_key(r): r for r in cell["predictions"][f"old_cost_{name}"]}
        for price in ("old_cost", "mean_cost"):
            rows = cell["predictions"][f"{price}_{name}"]
            if len(rows) != len(keys) or {state_key(r) for r in rows} != keys:
                raise ValueError("Lost an original evaluation state")
            for row in rows:
                if (any(row[k] != old[state_key(row)][k] for k in ("predicted_harm_mass_share", "predicted_feasibility_score"))
                        or row["predicted_heads"][row["original_choice"]] != [0., 0.]
                        or previous.source.choose(row, row["original_choice"]) != row["selected"]):
                    raise ValueError("Changed the frozen wait mask, identity or fixed selector")


def summarize(cells, registered, panels):
    if len(cells) != 47 or set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete exactly47 paired cost folds before reporting")
    for fold in registered["folds"]: qualify(cells[fold["split"], fold["index"]], fold, registered)
    keys = {state_key(r) for r in registered["mean_states"]}
    if set(panels) != keys: raise ValueError("Changed the original future evaluation panels")
    variants = {}
    for variant in VARIANTS:
        validation, halves = {}, {}
        for split in SPLITS:
            rows = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][variant]]
            if len(rows) != 240 or {state_key(r) for r in rows} != keys: raise ValueError("Lost one complete evaluation split")
            errors = {p: [] for p in diagnosis.PARTITIONS}
            for row in rows:
                values = panels[state_key(row)]["values"]
                observed = values[:, row["selected"]][:, [0, 3]].mean(axis=0)
                if not np.allclose(observed, [row["actual_cost_delta"], row["wait_delta_min"]], atol=1e-12, rtol=0):
                    raise ValueError("Cost refit changed its held-out physical labels")
                cost = np.asarray(row["predicted_heads"]).sum(axis=1)
                for p in errors:
                    e = diagnosis.factorial.panel(values, cost, np.asarray(row["predicted_feasibility_score"]), row["original_choice"], p)["errors"]
                    errors[p].append(e)
            validation[split] = {"overall": previous.source.summary(rows),
                "by_root": {str(r): previous.source.summary([x for x in rows if x["root"] == r]) for r in previous.collection.ROOTS},
                "by_regime": {s: previous.source.summary([x for x in rows if x["scenario"] == s]) for s in previous.collection.SCENARIOS},
                "stable_ranking": {p: {k: sum(e[k] for e in es) for k in ("stable_feasible_cost_pairs", "stable_cost_inversions", "stable_cost_prediction_ties")}
                    for p, es in errors.items()}}
            halves[split] = {p: {str(r): {"A": half_summary([x for x in rows if x["root"] == r], panels, a),
                "B": half_summary([x for x in rows if x["root"] == r], panels, b)} for r in previous.collection.ROOTS}
                for p, (a, b) in diagnosis.PARTITIONS.items()}
        variants[variant] = {"validation": validation, "future_halves": halves,
            "gates": {"cost": all(m["mean_actual_cost_delta"] < 0 for s in validation.values() for m in s["by_root"].values()),
                "wait": all(m["mean_wait_delta_min"] <= 0 for s in validation.values() for m in s["by_root"].values())},
            "half_gates": {p: {"cost": all(m["mean_cost"] < 0 for s in halves.values() for r in s[p].values() for m in r.values()),
                "wait": all(m["mean_wait_min"] <= 0 for s in halves.values() for r in s[p].values() for m in r.values())} for p in diagnosis.PARTITIONS}}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "variants": variants, "development_gate": all(variants[PRIMARY]["gates"].values()),
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("fit", "report"), required=True)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--fold-index", type=int)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    old = json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text())
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    if registered != manifest(old): raise ValueError("Changed the mean-cost comparison")
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered, previous.source.diagnostic.load_panels())
        control = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
        for name, v in WAIT_HEADS.items():
            for s in SPLITS:
                for scope in ("overall", "by_root", "by_regime"):
                    if result["variants"][f"old_cost_{name}"]["validation"][s][scope] != control["variants"][v]["validation"][s][scope]:
                        raise ValueError("Original wait/cost controls failed to reproduce")
        write_json(directory/"report/result.json", result)
        print(json.dumps({v: r["gates"] for v, r in result["variants"].items()}), flush=True)
        print("NATIVE_MEAN_COST_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None: parser.error("Fit requires a registered outer fold")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    bank = json.loads(previous.collection.mean_source.source.bank_path(previous.collection.mean_source.SOURCE_RUN).read_text())
    prepared = json.loads((raw_directory(ROOT/"results"/previous.SOURCE_RUN/"prepare/receipt.json")/"targets.json").read_text())
    original = json.loads(previous.source.mass_path(previous.collection.MASS_RUN).read_text())
    masses = previous.checked_masses(bank, prepared, original, old)
    targets = json.loads(previous.collection.mean_source.expected.targets_path(previous.collection.MODEL_RUN).read_text())["targets"]
    prior = json.loads((raw_directory(previous.result_path(SOURCE_RUN, fold))/"result.json").read_text())
    previous.qualify(prior, fold, old)
    cell, models = fit_fold(bank, targets, [*targets, *prepared["new_mean_labels"]], masses, fold, prior)
    qualify(cell, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", cell)
    for v, m in models.items(): write_json(raw_directory(output)/f"{v}.json", m)
    write_json(output, {k: v for k, v in cell.items() if k not in ("reference", "fits", "predictions")})
    print("NATIVE_MEAN_COST_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
