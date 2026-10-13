#!/usr/bin/env python3
"""Train wait-feasibility losses without changing outer costs or candidate plans."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_decision_wait_stage204 as source
from scripts import analyze_native_transit_oracle_factorial_stage205 as diagnostic
from freq_hrl.domains.transit import native_wait_feasibility as feasibility
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_wait_feasibility_stage206_v1"
SOURCE_RUN = diagnostic.SOURCE_RUN
DIAGNOSTIC_RUN = "native_transit_oracle_factorial_stage205_diagnostic_20261013_r1"
VARIANTS = ("frozen_decision_wait", *feasibility.FALSE_SAFE_COSTS)
SPLITS, state_key = source.SPLITS, source.state_key


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "diagnostic_run": DIAGNOSTIC_RUN, "data": "same600_inputs_240_mean8_targets_360_single_targets_47_folds",
        "variants": list(VARIANTS), "primary": "asymmetric_feasibility",
        "cost": "exact_Stage204_outer_cost_heads_and_candidate_menu_frozen",
        "target": "per_action_mean_positive_and_negative_wait_magnitudes_from_each_future_draw",
        "loss": "lambda_positive_mass_softplus_minus_logit_plus_negative_mass_softplus_logit",
        "false_safe_costs": feasibility.FALSE_SAFE_COSTS,
        "features": "same198_causal_conditional_wait_features_identity_centered_plus_intercept_199_parameters",
        "scales": "outer_train_only_feature_std_and_draw_count_weighted_mean_absolute_wait",
        "weights": "mean8_count8_single_count1_five_nonidentity_candidates_each",
        "optimizer": "scipy_LBFGSB_200_max_iterations_ridge0.1_all_coefficients_fixed_no_tuning",
        "selection": "fixed_harm_mass_share_at_most0.5_then_frozen_cost_min_identity_exact_zero_and_ties_retained",
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_mean_label_splits",
        "future_halves": diagnostic.contract()["partitions"],
        "budget": {"new_wait_fits": 94, "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "draw_masses_features_models_predictions_server_only_small_receipts_and_report",
        "boundary": "supervised_single_action_development_not_independent_CI_or_full_learned_HRL"}


def manifest(previous):
    if previous["protocol"] != source.EXPERIMENT_PROTOCOL or previous["contract"] != source.contract():
        raise ValueError("Feasibility learning requires the exact Stage204 frozen cohort")
    return {**previous, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def mass_path(run_name):
    return raw_directory(ROOT/"results"/run_name/"prepare/receipt.json")/"targets.json"


def build_masses(rows, panels):
    result = []
    for row in rows:
        key, original = state_key(row), row["original_choice"]
        if key in panels:
            p = panels[key]
            if original != p["original_choice"]:
                raise ValueError("Draw targets changed the original candidate")
            values = p["values"][..., 3]
        else:
            labels = {l["candidate"]: l["wait_delta_min"] for l in row["labels"]}
            values = np.asarray([[labels[i] for i in range(6)]])
        if np.any(values[:, original] != 0):
            raise ValueError("Wait feasibility requires paired identity labels exactly zero")
        positive, negative = feasibility.wait_masses(values)
        result.append({**{k: row[k] for k in ("root", "scenario", "scene_seed", "probe_index", "original_choice")},
            "draw_count": len(values), "positive_wait_min": positive.tolist(), "negative_wait_min": negative.tolist()})
    return result


def choose(prediction, original):
    cost = np.asarray(prediction["predicted_heads"]).sum(axis=1)
    scores = np.asarray(prediction["predicted_feasibility_score"])
    return source.paired.choice_from_cost(np.where(scores <= 0, cost, np.inf), original)


def evaluate(rows, frozen, model):
    scores = {state_key(r): {**frozen[state_key(r)], **feasibility.predict(r, model)} for r in rows}
    records = source.paired.evaluate_action_value(rows, model,
        predictor=lambda r, m: np.asarray(frozen[state_key(r)]["predicted_heads"]),
        selector=lambda r, m: choose(scores[state_key(r)], r["original_choice"]))
    for row in records:
        row.update({k: scores[state_key(row)][k] for k in ("predicted_harm_mass_share", "predicted_feasibility_score")})
    return records


def fit_fold(bank, targets, masses, fold, prior):
    train, test, _ = source.coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {state_key(r) for r in targets}
    test = source.mean.expected.replace_targets([r for r in test if state_key(r) in keys], targets)
    membership = sorted(map(list, {source.paired.scene_group(r) for r in train}))
    if prior["frozen_cost_fit"]["training_groups"] != membership:
        raise ValueError("Wait feasibility changed frozen outer cost membership")
    start = time.monotonic()
    baseline = prior["predictions"][source.contract()["primary"]]
    frozen = {state_key(r): r for r in baseline}
    models = {v: feasibility.fit_wait(train, masses, v) for v in feasibility.FALSE_SAFE_COSTS}
    predictions = {"frozen_decision_wait": [{**r, "predicted_feasibility_score": r["predicted_wait_delta_min"]} for r in baseline],
        **{v: evaluate(test, frozen, m) for v, m in models.items()}}
    fits = {v: {k: x for k, x in m.items() if k not in ("coefficients", "center", "scale")} for v, m in models.items()}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "frozen_cost_fit": prior["frozen_cost_fit"], "fits": fits, "predictions": predictions, "new_wait_fits": 2,
        "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0,
        "wait_optimizer_iterations": sum(m["optimizer_iterations"] for m in models.values()),
        "fit_and_evaluation_wall_s": time.monotonic()-start}, models


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if source.credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    keys = {state_key(r) for r in registered["mean_states"] if source.credit.group_key(r, fold["split"]) == fold["held_out"]}
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["new_wait_fits"] == 2 and cell["new_cost_fits"] == cell["actor_updates"] == cell["new_native_episodes"] == 0
            and cell["frozen_cost_fit"]["training_groups"] == membership):
        raise ValueError("Changed wait-feasibility cohort or training budget")
    for variant, cost in feasibility.FALSE_SAFE_COSTS.items():
        fit = cell["fits"][variant]
        if not (fit["training_groups"] == membership and fit["training_states"] == 6*len(groups)
                and fit["training_mean_states"] == 240-len(keys)
                and fit["sample_weight_sum"] == 6*len(groups)+7*(240-len(keys))
                and fit["parameter_count"] == 199 and fit["false_safe_cost"] == cost and fit["ridge"] == feasibility.RIDGE
                and fit["max_iterations"] == feasibility.MAX_ITERATIONS and fit["solver_success"]
                and 0 <= fit["optimizer_iterations"] <= feasibility.MAX_ITERATIONS):
            raise ValueError("Changed train-only wait masses, normalization population or solver")
    if cell["wait_optimizer_iterations"] != sum(m["optimizer_iterations"] for m in cell["fits"].values()):
        raise ValueError("Wait optimizer iteration accounting changed")
    frozen = {state_key(r): r for r in cell["predictions"]["frozen_decision_wait"]}
    for variant in VARIANTS:
        rows = cell["predictions"][variant]
        if len(rows) != len(keys) or {state_key(r) for r in rows} != keys:
            raise ValueError("Lost a held-out feasibility state")
        for row in rows:
            original = row["original_choice"]
            if (row["predicted_heads"] != frozen[state_key(row)]["predicted_heads"] or
                    row["predicted_feasibility_score"][original] != 0 or choose(row, original) != row["selected"]):
                raise ValueError("Changed frozen cost predictions, threshold or identity")
            if variant != "frozen_decision_wait":
                share = np.asarray(row["predicted_harm_mass_share"])
                score = np.where(np.arange(6) == original, 0., share-.5)
                if share[original] != 0. or not np.array_equal(score, row["predicted_feasibility_score"]):
                    raise ValueError("Changed magnitude share into a different operating threshold")


def summary(rows):
    return {**source.paired.summarize_choices(rows),
        "false_feasible_choices": sum(r["selected"] != r["original_choice"] and r["wait_delta_min"] > 0 for r in rows)}


def summarize(cells, registered, panels):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]} or len(cells) != 47:
        raise ValueError("Complete exactly47 feasibility fits before reporting")
    for fold in registered["folds"]: qualify(cells[fold["split"], fold["index"]], fold, registered)
    keys = {state_key(r) for r in registered["mean_states"]}
    if set(panels) != keys: raise ValueError("Changed future-half evaluation states")
    variants = {}
    for variant in VARIANTS:
        validation, halves = {}, {p: {} for p in diagnostic.PARTITIONS}
        for split in SPLITS:
            rows = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][variant]]
            if len(rows) != 240 or {state_key(r) for r in rows} != keys: raise ValueError("Lost one240-state split")
            validation[split] = {"overall": summary(rows),
                "by_root": {str(root): summary([r for r in rows if r["root"] == root]) for root in source.mean.ROOTS},
                "by_regime": {s: summary([r for r in rows if r["scenario"] == s]) for s in source.mean.SCENARIOS}}
            for partition in diagnostic.PARTITIONS:
                records = []
                for row in rows:
                    p = panels[state_key(row)]
                    values = p["values"][:, row["selected"], [0, 3]].mean(axis=0)
                    if not np.allclose(values, [row["actual_cost_delta"], row["wait_delta_min"]], atol=1e-12, rtol=0):
                        raise ValueError("Draw halves changed the mean8 selected-action labels")
                    records.append({**{k: row[k] for k in ("root", "scenario", "scene_seed", "probe_index")},
                        **diagnostic.panel(p["values"], np.asarray(row["predicted_heads"]).sum(axis=1),
                            np.asarray(row["predicted_feasibility_score"]), row["original_choice"], partition)})
                halves[partition][split] = diagnostic.scopes(records, variant == contract()["primary"])
        gates = {"cost": all(m["mean_actual_cost_delta"] < 0 for v in validation.values() for m in v["by_root"].values()),
            "wait": all(m["mean_wait_delta_min"] <= 0 for v in validation.values() for m in v["by_root"].values())}
        variants[variant] = {"validation": validation, "gates": gates,
            "future_halves": {p: {"validation": v, "gates": diagnostic.gates(v)} for p, v in halves.items()}}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "wait_optimizer_iterations": sum(c["wait_optimizer_iterations"] for c in cells.values()),
        "evaluation_states": 240, "evaluation_prefix_groups": 40, "variants": variants,
        "development_gate": all(variants[contract()["primary"]]["gates"].values()),
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("prepare", "fit", "report"), required=True)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--fold-index", type=int)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    previous = json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text())
    if registered != manifest(previous): raise ValueError("Changed registered feasibility protocol")
    if args.operation == "prepare":
        bank = json.loads(source.coverage.bank_path(source.mean.SOURCE_RUN).read_text())
        panels = diagnostic.load_panels()
        if set(panels) != {state_key(r) for r in registered["mean_states"]} or len(bank["training_data"]) != 600:
            raise ValueError("Changed registered mass-target inputs")
        masses = build_masses(bank["training_data"], panels)
        write_json(mass_path(args.run_name), {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "targets": masses})
        write_json(directory/"prepare/receipt.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
            "software_qualified": True, "states": len(masses), "mean8_states": sum(m["draw_count"] == 8 for m in masses)})
        print("NATIVE_WAIT_FEASIBILITY_PREPARE_COMPLETE", flush=True)
        return
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered, diagnostic.load_panels())
        write_json(directory/"report/result.json", result)
        print(json.dumps({v: d["gates"] for v, d in result["variants"].items()}, indent=2), flush=True)
        print("NATIVE_WAIT_FEASIBILITY_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None: parser.error("Fit requires split and fold index")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    prior = json.loads((raw_directory(source.result_path(SOURCE_RUN, fold))/"result.json").read_text())
    source.qualify(prior, fold, previous)
    bank = json.loads(source.coverage.bank_path(source.mean.SOURCE_RUN).read_text())
    targets = json.loads(source.mean.expected.targets_path(source.credit.source.SOURCE_RUN).read_text())["targets"]
    prepared = json.loads(mass_path(args.run_name).read_text())
    if prepared["contract"] != contract(): raise ValueError("Changed prepared feasibility targets")
    masses = {state_key(r): r for r in prepared["targets"]}
    cell, models = fit_fold(bank, targets, masses, fold, prior)
    qualify(cell, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", cell)
    for variant, model in models.items(): write_json(raw_directory(output)/f"{variant}.json", model)
    write_json(output, {k: v for k, v in cell.items() if k not in ("fits", "predictions", "frozen_cost_fit")})
    print(f"WAIT_FEASIBILITY_FIT split={args.split} fold={args.fold_index} iterations={cell['wait_optimizer_iterations']} fit_s={cell['fit_and_evaluation_wall_s']:.3f}", flush=True)
    print("NATIVE_WAIT_FEASIBILITY_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
