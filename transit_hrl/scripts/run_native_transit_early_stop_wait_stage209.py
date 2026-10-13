#!/usr/bin/env python3
"""Select wait-head checkpoints on inner scenes, then evaluate frozen outer folds."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_nonlinear_wait_stage208 as previous
from freq_hrl.domains.transit import native_early_stop_wait as stopping
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_early_stop_wait_stage209_v1"
SOURCE_RUN = "native_transit_nonlinear_wait_stage208_development_20261013_r1"
PRIMARY = "inner_scene_early_stop"
VARIANTS = ("symmetric_feasibility", "state_conditioned_pair", PRIMARY)
source, transfer, neural = previous.source, previous.transfer, previous.neural
SPLITS, state_key = source.SPLITS, source.state_key


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "data": "same600_inputs_240_mean8_targets_360_single_targets_47_outer_folds",
        "variants": list(VARIANTS), "primary": PRIMARY,
        "inner_folds": stopping.grouping.MEMBERS, "partition_seed": stopping.grouping.PARTITION_SEED,
        "partition": "outer_training_scenes_stratified_by_root_regime_mean8_membership_all_callbacks_grouped",
        "inner_reference": "refit_symmetric_GLM_and_all_normalizations_on_inner_train_only",
        "checkpoint_grid": list(stopping.CHECKPOINTS),
        "selection": "pooled_inner_mean8_mass_weighted_crossentropy_minimum_exact_ties_earliest_including_GLM_step0",
        "refit": "same210_input16_tanh_residual_seed208_AdamW_lr0.003_decay0.01_float64_on_all_outer_train_at_selected_step",
        "outer": "frozen_Stage206_GLM_costs_menu_and_threshold0.5_exact_identity_ties_Stage208_fixed600_control",
        "gate": "negative_cost_nonpositive_wait_both_roots_all_three_splits_and_lower_held_excess_loss_than_fixed600_all_splits",
        "future_halves": source.diagnostic.contract()["partitions"],
        "budget": {"new_inner_glm_fits": 188, "inner_neural_paths": 188, "outer_model_assemblies": 47,
            "max_adamw_steps": 141000, "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "raw_inputs_inner_ledger_and_final_models_remote_small_report_only",
        "boundary": "supervised_single_action_development_on_existing_prefixes_not_independent_CI_or_full_HRL"}


def manifest(old):
    if old["protocol"] != previous.EXPERIMENT_PROTOCOL or old["contract"] != previous.contract():
        raise ValueError("Early stopping requires the exact Stage208 cohort")
    return {**old, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def fit_fold(bank, targets, masses, fold, prior, reference, registered):
    train, test, _ = source.source.coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {state_key(r) for r in targets}
    test = source.source.mean.expected.replace_targets([r for r in test if state_key(r) in keys], targets)
    groups = [g for g in registered["groups"] if source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    if reference["training_groups"] != membership or prior["frozen_cost_fit"]["training_groups"] != membership:
        raise ValueError("Early stopping changed the frozen outer training membership")
    start = time.monotonic()
    model, selection = stopping.fit_wait(train, masses, reference, groups, keys)
    predictions = {v: prior["predictions"][v] for v in VARIANTS[:-1]}
    frozen = {state_key(r): r for r in predictions["symmetric_feasibility"]}
    predictions[PRIMARY] = previous.evaluate(test, frozen, neural.predict_many(test, model))
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "frozen_cost_fit": prior["frozen_cost_fit"], "fit": stopping.metadata(model), "selection": selection,
        "predictions": predictions, "inner_glm_fits": 4, "inner_neural_paths": 4, "outer_model_assemblies": 1,
        "adamw_steps": selection["inner_adamw_steps"]+selection["outer_adamw_steps"],
        "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0,
        "fit_and_evaluation_wall_s": time.monotonic()-start}, model


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    keys = {state_key(r) for r in registered["mean_states"] if source.source.credit.group_key(r, fold["split"]) == fold["held_out"]}
    fit, selection = cell["fit"], cell["selection"]
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["frozen_cost_fit"]["training_groups"] == membership
            and fit["training_groups"] == membership and fit["training_states"] == len(groups)*6
            and fit["training_mean_states"] == 240-len(keys) and fit["sample_weight_sum"] == len(groups)*6+7*(240-len(keys))
            and fit["parameter_count"] == 3393 and fit["initialization_seed"] == neural.SEED and fit["false_safe_cost"] == 1.
            and fit["steps"] == selection["selected_step"] in stopping.CHECKPOINTS
            and cell["inner_glm_fits"] == cell["inner_neural_paths"] == len(selection["inner_fits"]) == 4
            and cell["outer_model_assemblies"] == 1 and cell["adamw_steps"] == 4*max(stopping.CHECKPOINTS)+fit["steps"]
            and cell["new_cost_fits"] == cell["actor_updates"] == cell["new_native_episodes"] == 0):
        raise ValueError("Changed early-stop cohort, checkpoint or training accounting")
    flattened = [g for bucket in selection["partition"] for g in bucket]
    if len(flattened) != len(membership) or sorted(flattened) != membership:
        raise ValueError("Inner partition lost or repeated an outer training scene")
    for bucket, inner in zip(selection["partition"], selection["inner_fits"], strict=True):
        expected = [g for g in membership if g not in bucket]
        if inner["held_groups"] != bucket or any(inner[k]["training_groups"] != expected for k in ("glm_fit", "neural_fit")):
            raise ValueError("Inner GLM or nonlinear normalizations included validation scenes")
    chosen, curve = stopping.select_checkpoint([i["curve"] for i in selection["inner_fits"]])
    if chosen != fit["steps"] or curve != selection["pooled_curve"]:
        raise ValueError("Chosen checkpoint differs from the registered inner criterion")
    if fit["steps"] == 0:
        if fit["parameter_update_l2"] != 0 or fit["residual_logit_rms"] != 0:
            raise ValueError("Step0 did not preserve the GLM fallback")
    elif fit["parameter_update_l2"] <= 0 or fit["hidden_gradient_l2_max"] <= 0:
        raise ValueError("Selected nonlinear head did not train")
    frozen = {state_key(r): r for r in cell["predictions"]["symmetric_feasibility"]}
    for variant in VARIANTS:
        rows = cell["predictions"][variant]
        if len(rows) != len(keys) or {state_key(r) for r in rows} != keys:
            raise ValueError("Lost a held-out early-stop state")
        for row in rows:
            original, share = row["original_choice"], np.asarray(row["predicted_harm_mass_share"])
            if (row["predicted_heads"] != frozen[state_key(row)]["predicted_heads"] or share[original] != 0
                    or not np.array_equal(np.where(np.arange(6) == original, 0., share-.5), row["predicted_feasibility_score"])
                    or source.choose(row, original) != row["selected"]):
                raise ValueError("Changed frozen costs, feasibility threshold or identity tie")


def summarize(cells, registered, bank, masses, panels, load_models):
    if len(cells) != 47 or set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete exactly47 early-stop fits before reporting")
    rows, mean_keys = bank["training_data"], {state_key(r) for r in registered["mean_states"]}
    if set(panels) != mean_keys:
        raise ValueError("Changed the mean8 future-half population")
    batch = neural.batches(rows)
    originals = np.asarray([r["original_choice"] for r in rows])
    training = {v: {s: [] for s in SPLITS} for v in VARIANTS}
    held_diagnostics = {v: {s: [] for s in SPLITS} for v in VARIANTS}
    for fold in registered["folds"]:
        cell = cells[fold["split"], fold["index"]]
        qualify(cell, fold, registered)
        held = np.asarray([source.source.credit.group_key(r, fold["split"]) == fold["held_out"] for r in rows])
        mean = np.asarray([state_key(r) in mean_keys for r in rows])
        train_mask = ~held & mean
        if fold["split"] == "scene":
            exemplar = rows[int(np.flatnonzero(held & mean)[0])]
            train_mask &= np.asarray([(r["root"], r["scenario"]) == (exemplar["root"], exemplar["scenario"]) for r in rows])
        models = load_models(fold)
        if stopping.metadata(models[PRIMARY]) != cell["fit"]:
            raise ValueError("Refitted model differs from its receipt")
        for v, model in models.items():
            if v == "symmetric_feasibility":
                logits, _, scores = transfer.model_scores(batch[0], originals, model)
            else:
                logits = neural.logits_many(rows, model, batch=batch)
                scores = np.asarray([p["predicted_feasibility_score"] for p in neural.predict_many(rows, model, batch=batch)])
            frozen = {state_key(r): r for r in cell["predictions"][v]}
            for i in np.flatnonzero(held & mean):
                old = frozen[state_key(rows[i])]["predicted_feasibility_score"]
                if not np.allclose(scores[i], old, atol=1e-12, rtol=0):
                    raise ValueError("Model no longer reproduces the archived held-out choices")
                scores[i] = old
            diagnostics = transfer.records(rows, masses, logits, scores, 1., panels)
            training[v][fold["split"]].extend(diagnostics[i] for i in np.flatnonzero(train_mask))
            held_diagnostics[v][fold["split"]].extend(diagnostics[i] for i in np.flatnonzero(held & mean))
    variants = {}
    for v in VARIANTS:
        validation, halves = {}, {}
        for split in SPLITS:
            predictions = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][v]]
            if len(predictions) != 240 or {state_key(r) for r in predictions} != mean_keys:
                raise ValueError("Lost or duplicated a240-state outer split")
            for row in predictions:
                observed = panels[state_key(row)]["values"][:, row["selected"]][:, [0, 3]].mean(axis=0)
                if not np.allclose(observed, [row["actual_cost_delta"], row["wait_delta_min"]], atol=1e-12, rtol=0):
                    raise ValueError("Future panels differ from the archived action outcomes")
            validation[split] = {"overall": source.summary(predictions),
                "by_root": {str(root): source.summary([r for r in predictions if r["root"] == root]) for root in source.source.mean.ROOTS},
                "by_regime": {s: source.summary([r for r in predictions if r["scenario"] == s]) for s in source.source.mean.SCENARIOS},
                "training_mean8": transfer.summary(training[v][split]),
                "held_out_diagnostics": transfer.scopes(held_diagnostics[v][split])}
            halves[split] = {p: {str(root): {"A": previous.half_summary([r for r in predictions if r["root"] == root], panels, a),
                "B": previous.half_summary([r for r in predictions if r["root"] == root], panels, b)} for root in source.source.mean.ROOTS}
                for p, (a, b) in source.diagnostic.PARTITIONS.items()}
        variants[v] = {"validation": validation, "future_halves": halves,
            "gates": {"cost": all(m["mean_actual_cost_delta"] < 0 for s in validation.values() for m in s["by_root"].values()),
                "wait": all(m["mean_wait_delta_min"] <= 0 for s in validation.values() for m in s["by_root"].values())},
            "half_gates": {p: {"cost": all(m["mean_cost"] < 0 for s in halves.values() for r in s[p].values() for m in r.values()),
                "wait": all(m["mean_wait_min"] <= 0 for s in halves.values() for r in s[p].values() for m in r.values())}
                for p in source.diagnostic.PARTITIONS}}
    primary = variants[PRIMARY]
    primary["held_excess_improves_fixed600"] = all(primary["validation"][s]["held_out_diagnostics"]["overall"]["mass_weighted_excess_nats"] <
        variants["state_conditioned_pair"]["validation"][s]["held_out_diagnostics"]["overall"]["mass_weighted_excess_nats"] for s in SPLITS)
    selections = [{"fold": c["fold"], "selected_step": c["selection"]["selected_step"],
        "pooled_curve": c["selection"]["pooled_curve"], "validation_mean_states": c["selection"]["validation_mean_states"]} for c in cells.values()]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "adamw_steps": sum(c["adamw_steps"] for c in cells.values()),
        "inner_glm_iterations": sum(c["selection"]["inner_glm_iterations"] for c in cells.values()),
        "outer_optimized_fits": sum(c["fit"]["steps"] > 0 for c in cells.values()),
        "checkpoint_counts": dict(sorted(Counter(str(s["selected_step"]) for s in selections).items())), "selections": selections,
        "evaluation_states": 240, "evaluation_prefix_groups": 40, "variants": variants,
        "development_gate": all(primary["gates"].values()) and primary["held_excess_improves_fixed600"],
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("fit", "report"), required=True)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--fold-index", type=int)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    old_manifest = json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text())
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    if registered != manifest(old_manifest):
        raise ValueError("Changed the registered early-stop protocol")
    bank = json.loads(source.source.coverage.bank_path(source.source.mean.SOURCE_RUN).read_text())
    prepared = json.loads(source.mass_path(previous.SOURCE_RUN).read_text())
    if prepared["contract"] != source.contract():
        raise ValueError("Changed the frozen wait masses")
    masses = {state_key(r): r for r in prepared["targets"]}
    def load_models(fold):
        old = raw_directory(previous.result_path(SOURCE_RUN, fold))
        glm = raw_directory(source.result_path(previous.SOURCE_RUN, fold))
        new = raw_directory(result_path(args.run_name, fold))
        return {"symmetric_feasibility": json.loads((glm/"symmetric_feasibility.json").read_text()),
            "state_conditioned_pair": json.loads((old/"state_conditioned_pair.json").read_text()),
            PRIMARY: json.loads((new/f"{PRIMARY}.json").read_text())}
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered, bank, masses, source.diagnostic.load_panels(), load_models)
        write_json(directory/"report/result.json", result)
        print(json.dumps({"checkpoints": result["checkpoint_counts"], "gates": result["variants"][PRIMARY]["gates"]}), flush=True)
        print("NATIVE_EARLY_STOP_WAIT_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None:
        parser.error("Fit requires split and fold index")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    prior = json.loads((raw_directory(previous.result_path(SOURCE_RUN, fold))/"result.json").read_text())
    previous.qualify(prior, fold, old_manifest)
    glm = raw_directory(source.result_path(previous.SOURCE_RUN, fold))/"symmetric_feasibility.json"
    reference = json.loads(glm.read_text())
    targets = json.loads(source.source.mean.expected.targets_path(source.source.credit.source.SOURCE_RUN).read_text())["targets"]
    cell, model = fit_fold(bank, targets, masses, fold, prior, reference, registered)
    qualify(cell, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", cell)
    write_json(raw_directory(output)/f"{PRIMARY}.json", model)
    write_json(output, {k: v for k, v in cell.items() if k not in ("fit", "selection", "predictions", "frozen_cost_fit")})
    print(f"EARLY_STOP_WAIT_FIT split={args.split} fold={args.fold_index} selected={model['steps']} adamw={cell['adamw_steps']} fit_s={cell['fit_and_evaluation_wall_s']:.3f}", flush=True)
    print("NATIVE_EARLY_STOP_WAIT_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
