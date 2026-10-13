#!/usr/bin/env python3
"""Test nonlinear and absolute-state wait heads with frozen costs and GLM controls."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_wait_feasibility_stage206 as source
from scripts import analyze_native_transit_wait_transfer_stage207 as transfer
from freq_hrl.domains.transit import native_nonlinear_wait as neural
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_nonlinear_wait_stage208_v1"
SOURCE_RUN = transfer.SOURCE_RUN
DIAGNOSTIC_RUN = "native_transit_wait_transfer_stage207_diagnostic_20261013_r1"
VARIANTS = (*transfer.VARIANTS, *neural.VARIANTS)
SPLITS, state_key = source.SPLITS, source.state_key


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL, "diagnostic_run": DIAGNOSTIC_RUN,
        "data": "same600_inputs_240_mean8_targets_360_single_targets_47_folds",
        "variants": list(VARIANTS), "primary": "state_conditioned_pair",
        "cost": "exact_Stage206_cost_heads_candidate_menu_and_both_GLMS_frozen",
        "target_loss": "unchanged_symmetric_positive_negative_mass_loss_draw_count8_or1_train_only_scale",
        "architecture": "frozen_symmetric_GLM_logit_plus_learned16_tanh_residual_zero_output_initialization",
        "inputs": "same198_paired_features_or_plus12_absolute_causal_conditions_no_root_regime_or_future_features",
        "normalization": "exact_frozen_GLM_train_scales_plus_own_training_only_context_mean_std",
        "parameters": neural.PARAMETERS, "optimizer": {"steps": neural.STEPS, "seed": neural.SEED,
            "lr": neural.LEARNING_RATE, "weight_decay": neural.WEIGHT_DECAY, "dtype": "float64", "method": "AdamW"},
        "selection": "fixed_share_at_most0.5_then_frozen_cost_min_identity_zero_ties_retained",
        "checkpoint": "fixed_final_step_no_heldout_selection_or_threshold_tuning",
        "gate": "negative_mean_cost_nonpositive_wait_both_roots_all_three_splits",
        "training_fit_gate": "lower_mean8_excess_mass_loss_and_boundary_error_all_splits_vs_frozen_symmetric_GLM_scene_matched_root_regime",
        "future_halves": source.diagnostic.contract()["partitions"],
        "budget": {"new_wait_fits": 94, "wait_optimizer_steps": 94*neural.STEPS, "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_masses_models_and_predictions_server_only_small_receipts_report",
        "boundary": "capacity_optimizer_regularization_differ_from_GLM_supervised_development_not_CI_or_full_HRL"}


def manifest(previous):
    if previous["protocol"] != source.EXPERIMENT_PROTOCOL or previous["contract"] != source.contract():
        raise ValueError("Nonlinear wait comparison requires the exact Stage206 cohort")
    return {**previous, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def evaluate(rows, frozen, predictions):
    by_key = {state_key(r): {**frozen[state_key(r)], **p} for r, p in zip(rows, predictions, strict=True)}
    records = source.source.paired.evaluate_action_value(rows, None,
        predictor=lambda r, m: np.asarray(frozen[state_key(r)]["predicted_heads"]),
        selector=lambda r, m: source.choose(by_key[state_key(r)], r["original_choice"]))
    for row in records:
        row.update({k: by_key[state_key(row)][k] for k in ("predicted_harm_mass_share", "predicted_feasibility_score")})
    return records


def fit_fold(bank, targets, masses, fold, prior, reference):
    train, test, _ = source.source.coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {state_key(r) for r in targets}
    test = source.source.mean.expected.replace_targets([r for r in test if state_key(r) in keys], targets)
    membership = sorted(map(list, {source.source.paired.scene_group(r) for r in train}))
    if prior["frozen_cost_fit"]["training_groups"] != membership:
        raise ValueError("Nonlinear wait changed frozen outer cost training membership")
    if {k: reference[k] for k in prior["fits"]["symmetric_feasibility"]} != prior["fits"]["symmetric_feasibility"]:
        raise ValueError("Nonlinear head changed its frozen symmetric GLM reference")
    start = time.monotonic()
    batch = neural.batches(train)
    models = {v: neural.fit_wait(train, masses, reference, v, batch=batch) for v in neural.VARIANTS}
    predictions = {v: prior["predictions"][v] for v in transfer.VARIANTS}
    frozen = {state_key(r): r for r in predictions["symmetric_feasibility"]}
    test_batch = neural.batches(test)
    predictions.update({v: evaluate(test, frozen, neural.predict_many(test, m, batch=test_batch)) for v, m in models.items()})
    fits = {v: {k: x for k, x in m.items() if k not in ("reference", "weights", "context_center", "context_scale")} for v, m in models.items()}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "frozen_cost_fit": prior["frozen_cost_fit"], "frozen_glm_fits": prior["fits"], "fits": fits,
        "predictions": predictions, "new_wait_fits": 2, "wait_optimizer_steps": 2*neural.STEPS,
        "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0, "fit_and_evaluation_wall_s": time.monotonic()-start}, models


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    keys = {state_key(r) for r in registered["mean_states"] if source.source.credit.group_key(r, fold["split"]) == fold["held_out"]}
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["frozen_cost_fit"]["training_groups"] == membership
            and cell["new_wait_fits"] == 2 and cell["wait_optimizer_steps"] == 2*neural.STEPS
            and cell["new_cost_fits"] == cell["actor_updates"] == cell["new_native_episodes"] == 0):
        raise ValueError("Changed nonlinear wait population or training budget")
    for variant in neural.VARIANTS:
        fit = cell["fits"][variant]
        if not (fit["training_groups"] == membership and fit["training_states"] == 6*len(groups)
                and fit["training_mean_states"] == 240-len(keys) and fit["sample_weight_sum"] == 6*len(groups)+7*(240-len(keys))
                and fit["parameter_count"] == neural.PARAMETERS[variant] and fit["steps"] == neural.STEPS
                and fit["initialization_seed"] == neural.SEED and fit["false_safe_cost"] == 1.
                and fit["frozen_parameters"] == 199 and fit["hidden_gradient_l2_max"] > 0
                and fit["parameter_update_l2"] > 0 and np.isfinite(fit["final_mass_loss"])):
            raise ValueError("Nonlinear branch did not train or changed its inputs/budget")
    frozen = {state_key(r): r for r in cell["predictions"]["symmetric_feasibility"]}
    for variant in VARIANTS:
        rows = cell["predictions"][variant]
        if len(rows) != len(keys) or {state_key(r) for r in rows} != keys: raise ValueError("Lost a held-out nonlinear state")
        for row in rows:
            original = row["original_choice"]
            share = np.asarray(row["predicted_harm_mass_share"])
            score = np.where(np.arange(6) == original, 0., share-.5)
            if (row["predicted_heads"] != frozen[state_key(row)]["predicted_heads"] or share[original] != 0.
                    or not np.array_equal(score, row["predicted_feasibility_score"]) or source.choose(row, original) != row["selected"]):
                raise ValueError("Changed frozen costs, fixed threshold or identity")


def half_summary(rows, panels, indices):
    outcomes = [panels[state_key(r)]["values"][list(indices), r["selected"]][:, [0, 3]].mean(axis=0) for r in rows]
    return {"states": len(rows), "mean_cost": float(np.mean([v[0] for v in outcomes])),
        "mean_wait_min": float(np.mean([v[1] for v in outcomes]))}


def summarize(cells, registered, bank, masses, panels, load_models):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]} or len(cells) != 47:
        raise ValueError("Complete exactly47 nonlinear wait folds before reporting")
    for fold in registered["folds"]: qualify(cells[fold["split"], fold["index"]], fold, registered)
    rows, keys = bank["training_data"], {state_key(r) for r in registered["mean_states"]}
    if set(panels) != keys: raise ValueError("Changed future-half diagnostic cohort")
    batch = neural.batches(rows)
    originals = np.asarray([r["original_choice"] for r in rows])
    mean = np.asarray([state_key(r) in keys for r in rows])
    training = {v: {s: [] for s in SPLITS} for v in VARIANTS}
    held_diagnostics = {v: {s: [] for s in SPLITS} for v in VARIANTS}
    for fold in registered["folds"]:
        models = load_models(fold)
        cell = cells[fold["split"], fold["index"]]
        held = np.asarray([source.source.credit.group_key(r, fold["split"]) == fold["held_out"] for r in rows])
        train_mask = ~held & mean
        if fold["split"] == "scene":
            exemplar = rows[int(np.flatnonzero(held & mean)[0])]
            train_mask &= np.asarray([(r["root"], r["scenario"]) == (exemplar["root"], exemplar["scenario"]) for r in rows])
        for variant in VARIANTS:
            model = models[variant]
            if variant in neural.VARIANTS:
                if {k: model[k] for k in cell["fits"][variant]} != cell["fits"][variant]:
                    raise ValueError("New model differs from its fitting receipt")
                logits = neural.logits_many(rows, model, batch=batch)
                predicted = neural.predict_many(rows, model, batch=batch)
                scores = np.asarray([p["predicted_feasibility_score"] for p in predicted])
            else:
                logits, _, scores = transfer.model_scores(batch[0], originals, model)
            frozen = {state_key(r): r for r in cell["predictions"][variant]}
            for i in np.flatnonzero(held & mean):
                old = frozen[state_key(rows[i])]["predicted_feasibility_score"]
                if not np.allclose(scores[i], old, atol=1e-12, rtol=0): raise ValueError("Changed frozen held-out scores during inference")
                scores[i] = old
            diagnostics = transfer.records(rows, masses, logits, scores, model["false_safe_cost"], panels)
            training[variant][fold["split"]].extend(diagnostics[i] for i in np.flatnonzero(train_mask))
            held_diagnostics[variant][fold["split"]].extend(diagnostics[i] for i in np.flatnonzero(held & mean))
    variants = {}
    for variant in VARIANTS:
        validation, halves = {}, {}
        for split in SPLITS:
            records = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][variant]]
            if len(records) != 240 or {state_key(r) for r in records} != keys: raise ValueError("Lost a240-state split")
            for row in records:
                p = panels[state_key(row)]
                observed = p["values"][:, row["selected"]][:, [0, 3]].mean(axis=0)
                if p["original_choice"] != row["original_choice"] or not np.allclose(observed,
                        [row["actual_cost_delta"], row["wait_delta_min"]], atol=1e-12, rtol=0):
                    raise ValueError("Future-half labels do not match the archived mean8 action outcomes")
            validation[split] = {"overall": source.summary(records),
                "by_root": {str(root): source.summary([r for r in records if r["root"] == root]) for root in source.source.mean.ROOTS},
                "by_regime": {s: source.summary([r for r in records if r["scenario"] == s]) for s in source.source.mean.SCENARIOS},
                "training_mean8": transfer.summary(training[variant][split]),
                "held_out_diagnostics": transfer.scopes(held_diagnostics[variant][split])}
            halves[split] = {p: {str(root): {"A": half_summary([r for r in records if r["root"] == root], panels, a),
                "B": half_summary([r for r in records if r["root"] == root], panels, b)} for root in source.source.mean.ROOTS}
                for p, (a, b) in source.diagnostic.PARTITIONS.items()}
        gates = {"cost": all(m["mean_actual_cost_delta"] < 0 for v in validation.values() for m in v["by_root"].values()),
            "wait": all(m["mean_wait_delta_min"] <= 0 for v in validation.values() for m in v["by_root"].values())}
        variants[variant] = {"validation": validation, "gates": gates, "future_halves": halves,
            "half_gates": {p: {"cost": all(m["mean_cost"] < 0 for split in halves.values() for root in split[p].values() for m in root.values()),
                "wait": all(m["mean_wait_min"] <= 0 for split in halves.values() for root in split[p].values() for m in root.values())}
                for p in source.diagnostic.PARTITIONS}}
    for variant in neural.VARIANTS:
        variants[variant]["training_fit_gate"] = all(
            variants[variant]["validation"][s]["training_mean8"][m] < variants["symmetric_feasibility"]["validation"][s]["training_mean8"][m]
            for s in SPLITS for m in ("mass_weighted_excess_nats", "own_loss_error_rate_informative"))
        variants[variant]["relative_to_symmetric"] = {s: {root: {"cost_delta": v["mean_actual_cost_delta"]-variants["symmetric_feasibility"]["validation"][s]["by_root"][root]["mean_actual_cost_delta"],
            "wait_delta_min": v["mean_wait_delta_min"]-variants["symmetric_feasibility"]["validation"][s]["by_root"][root]["mean_wait_delta_min"]}
            for root, v in variants[variant]["validation"][s]["by_root"].items()} for s in SPLITS}
    primary = variants[contract()["primary"]]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "evaluation_states": 240, "evaluation_prefix_groups": 40, "variants": variants,
        "development_gate": primary["training_fit_gate"] and all(primary["gates"].values()),
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
    if registered != manifest(previous): raise ValueError("Changed registered nonlinear wait protocol")
    bank = json.loads(source.source.coverage.bank_path(source.source.mean.SOURCE_RUN).read_text())
    prepared = json.loads(source.mass_path(SOURCE_RUN).read_text())
    if prepared["contract"] != source.contract(): raise ValueError("Changed frozen wait targets")
    masses = {state_key(r): r for r in prepared["targets"]}
    def old_models(fold):
        d = raw_directory(source.result_path(SOURCE_RUN, fold))
        return {v: json.loads((d/f"{v}.json").read_text()) for v in transfer.VARIANTS}
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        def load_models(fold):
            d = raw_directory(result_path(args.run_name, fold))
            return {**old_models(fold), **{v: json.loads((d/f"{v}.json").read_text()) for v in neural.VARIANTS}}
        result = summarize(cells, registered, bank, masses, source.diagnostic.load_panels(), load_models)
        write_json(directory/"report/result.json", result)
        print(json.dumps({v: {"gates": d["gates"], "training_fit_gate": d.get("training_fit_gate")} for v, d in result["variants"].items()}, indent=2), flush=True)
        print("NATIVE_NONLINEAR_WAIT_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None: parser.error("Fit requires split and fold index")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    prior = json.loads((raw_directory(source.result_path(SOURCE_RUN, fold))/"result.json").read_text())
    source.qualify(prior, fold, previous)
    targets = json.loads(source.source.mean.expected.targets_path(source.source.credit.source.SOURCE_RUN).read_text())["targets"]
    cell, models = fit_fold(bank, targets, masses, fold, prior, old_models(fold)["symmetric_feasibility"])
    qualify(cell, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", cell)
    for v, model in models.items(): write_json(raw_directory(output)/f"{v}.json", model)
    write_json(output, {k: v for k, v in cell.items() if k not in ("fits", "predictions", "frozen_cost_fit", "frozen_glm_fits")})
    print(f"NONLINEAR_WAIT_FIT split={args.split} fold={args.fold_index} steps={cell['wait_optimizer_steps']} fit_s={cell['fit_and_evaluation_wall_s']:.3f}", flush=True)
    print("NATIVE_NONLINEAR_WAIT_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
