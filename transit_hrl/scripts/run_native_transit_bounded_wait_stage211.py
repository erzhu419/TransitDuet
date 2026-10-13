#!/usr/bin/env python3
"""Evaluate bounded and functionally-shrunk wait corrections on frozen native labels."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_early_stop_wait_stage209 as previous
from scripts import run_native_transit_future_transfer_stage210 as diagnosis
from freq_hrl.domains.transit import native_bounded_wait as bounded
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_bounded_wait_stage211_v1"
SOURCE_RUN = "native_transit_early_stop_wait_stage209_development_20261013_r1"
DIAGNOSTIC_RUN = "native_transit_future_transfer_stage210_diagnostic_20261013_r1"
PRIMARY = "shrunk_bounded"
VARIANTS = (*previous.VARIANTS, *bounded.FUNCTIONAL_RIDGES)
source, transfer, neural, stopping = previous.source, previous.transfer, previous.neural, previous.stopping
SPLITS, state_key = source.SPLITS, source.state_key


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL, "diagnostic_run": DIAGNOSTIC_RUN,
        "data": "original600_inputs_240_mean8_targets_360_single_targets_47_outer_folds",
        "variants": list(VARIANTS), "primary": PRIMARY, "residual_bound": bounded.BOUND, "functional_ridges": bounded.FUNCTIONAL_RIDGES,
        "correction": "GLM_logit_plus_1_tanh_raw_residual_shared3393_parameter_state_conditioned_net",
        "penalty": "half_ridge_times_train_wait_mass_and_draw_count_weighted_mean_squared_bounded_residual",
        "optimizer": "same_seed208_float64_AdamW_lr0.003_decay0.01_symmetric_mass_loss",
        "checkpoint_grid": list(stopping.CHECKPOINTS), "inner_folds": 4, "partition_seed": stopping.grouping.PARTITION_SEED,
        "selection": "same_grouped_inner_mean8_crossentropy_ties_earliest_including_step0_GLM_no_bound_ridge_or_threshold_search",
        "inner_reference": "one_new_inner_symmetric_GLM_and_training_only_scales_shared_by_two_heads_per_fold",
        "outer": "exact_Stage206_GLM_costs_menu_threshold0.5_identity_ties_and_Stage208_209_controls_frozen",
        "gate": "negative_cost_nonpositive_wait_both_roots_all_three_splits_and_lower_held_excess_than_GLM_all_splits",
        "future_halves": source.diagnostic.contract()["partitions"],
        "budget": {"new_inner_glm_fits": 188, "inner_neural_paths": 376, "outer_model_assemblies": 94,
            "max_adamw_steps": 282000, "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "raw_labels_models_and_fold_predictions_remote_small_report_only",
        "boundary": "single_intervention_supervised_development_not_independent_CI_or_full_learned_HRL"}


def manifest(old):
    if old["protocol"] != previous.EXPERIMENT_PROTOCOL or old["contract"] != previous.contract():
        raise ValueError("Bounded residuals require the exact Stage209 cohort")
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
        raise ValueError("Bounded residual changed the frozen outer training population")
    start = time.monotonic()
    models, selections = bounded.fit_wait(train, masses, reference, groups, keys)
    predictions = {v: prior["predictions"][v] for v in previous.VARIANTS}
    frozen = {state_key(r): r for r in predictions["symmetric_feasibility"]}
    batch = neural.batches(test)
    predictions.update({v: previous.previous.evaluate(test, frozen, bounded.predict_many(test, m, batch=batch)) for v, m in models.items()})
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "frozen_cost_fit": prior["frozen_cost_fit"], "fits": {v: stopping.metadata(m) for v, m in models.items()},
        "selections": selections, "predictions": predictions, "inner_glm_fits": 4, "inner_neural_paths": 8, "outer_model_assemblies": 2,
        "adamw_steps": sum(s["inner_adamw_steps"]+s["outer_adamw_steps"] for s in selections.values()),
        "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0,
        "fit_and_evaluation_wall_s": time.monotonic()-start}, models


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    keys = {state_key(r) for r in registered["mean_states"] if source.source.credit.group_key(r, fold["split"]) == fold["held_out"]}
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["frozen_cost_fit"]["training_groups"] == membership
            and cell["inner_glm_fits"] == 4 and cell["inner_neural_paths"] == 8 and cell["outer_model_assemblies"] == 2
            and cell["adamw_steps"] == 8*max(stopping.CHECKPOINTS)+sum(f["steps"] for f in cell["fits"].values())
            and cell["new_cost_fits"] == cell["actor_updates"] == cell["new_native_episodes"] == 0):
        raise ValueError("Changed bounded residual cohort or training budget")
    for variant, ridge in bounded.FUNCTIONAL_RIDGES.items():
        fit, selection = cell["fits"][variant], cell["selections"][variant]
        if not (fit["training_groups"] == membership and fit["training_states"] == 6*len(groups)
                and fit["training_mean_states"] == 240-len(keys) and fit["sample_weight_sum"] == 6*len(groups)+7*(240-len(keys))
                and fit["correction"] == variant and fit["parameter_count"] == 3393 and fit["false_safe_cost"] == 1.
                and fit["residual_bound"] == bounded.BOUND and fit["functional_ridge"] == ridge
                and fit["steps"] == selection["selected_step"] in stopping.CHECKPOINTS
                and fit["initialization_seed"] == neural.SEED and fit["residual_logit_max_abs"] <= bounded.BOUND
                and np.isfinite(fit["final_objective"])):
            raise ValueError("Changed bound, shrinkage, target depth or fitted model")
        flat = [g for bucket in selection["partition"] for g in bucket]
        if len(flat) != len(membership) or sorted(flat) != membership or len(selection["inner_fits"]) != 4:
            raise ValueError("Bounded inner folds lost or duplicated scenes")
        for bucket, inner in zip(selection["partition"], selection["inner_fits"], strict=True):
            expected = [g for g in membership if g not in bucket]
            if inner["held_groups"] != bucket or any(inner[k]["training_groups"] != expected for k in ("glm_fit", "neural_fit")):
                raise ValueError("Bounded inner initialization included validation scenes")
        step, curve = stopping.select_checkpoint([i["curve"] for i in selection["inner_fits"]])
        if step != fit["steps"] or curve != selection["pooled_curve"]:
            raise ValueError("Bounded checkpoint was not selected by the inner criterion")
        if fit["steps"] == 0:
            if fit["parameter_update_l2"] != 0 or fit["residual_logit_rms"] != 0:
                raise ValueError("Bounded step0 did not preserve the GLM")
        elif fit["parameter_update_l2"] <= 0 or fit["hidden_gradient_l2_max"] <= 0:
            raise ValueError("Chosen bounded correction did not train")
    a, b = [cell["selections"][v] for v in bounded.FUNCTIONAL_RIDGES]
    if a["partition"] != b["partition"] or [i["glm_fit"] for i in a["inner_fits"]] != [i["glm_fit"] for i in b["inner_fits"]]:
        raise ValueError("Paired heads did not share the same four inner GLM fits")
    frozen = {state_key(r): r for r in cell["predictions"]["symmetric_feasibility"]}
    for variant in VARIANTS:
        rows = cell["predictions"][variant]
        if len(rows) != len(keys) or {state_key(r) for r in rows} != keys:
            raise ValueError("Lost or duplicated a held-out bounded state")
        for row in rows:
            original, share = row["original_choice"], np.asarray(row["predicted_harm_mass_share"])
            if (row["predicted_heads"] != frozen[state_key(row)]["predicted_heads"] or share[original] != 0
                    or not np.array_equal(np.where(np.arange(6) == original, 0., share-.5), row["predicted_feasibility_score"])
                    or source.choose(row, original) != row["selected"]):
                raise ValueError("Changed frozen costs, safety threshold or identity tie")


def summarize(cells, registered, bank, masses, panels, load_models):
    if len(cells) != 47 or set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete exactly47 bounded wait folds before reporting")
    rows, keys = bank["training_data"], {state_key(r) for r in registered["mean_states"]}
    if set(panels) != keys:
        raise ValueError("Changed the mean8 future-half population")
    batch = neural.batches(rows)
    originals = np.asarray([r["original_choice"] for r in rows])
    mean = np.asarray([state_key(r) in keys for r in rows])
    training = {v: {s: [] for s in SPLITS} for v in VARIANTS}
    held_diagnostics = {v: {s: [] for s in SPLITS} for v in VARIANTS}
    for fold in registered["folds"]:
        cell = cells[fold["split"], fold["index"]]
        qualify(cell, fold, registered)
        held = np.asarray([source.source.credit.group_key(r, fold["split"]) == fold["held_out"] for r in rows])
        train = ~held & mean
        if fold["split"] == "scene":
            exemplar = rows[int(np.flatnonzero(held & mean)[0])]
            train &= np.asarray([(r["root"], r["scenario"]) == (exemplar["root"], exemplar["scenario"]) for r in rows])
        models = load_models(fold)
        for v, model in models.items():
            if v == "symmetric_feasibility":
                logits, _, scores = transfer.model_scores(batch[0], originals, model)
            else:
                api = bounded if v in bounded.FUNCTIONAL_RIDGES else neural
                if v in bounded.FUNCTIONAL_RIDGES and stopping.metadata(model) != cell["fits"][v]:
                    raise ValueError("Bounded model differs from its fitting receipt")
                logits = api.logits_many(rows, model, batch=batch)
                scores = np.asarray([p["predicted_feasibility_score"] for p in api.predict_many(rows, model, batch=batch)])
            frozen = {state_key(r): r for r in cell["predictions"][v]}
            for i in np.flatnonzero(held & mean):
                old = frozen[state_key(rows[i])]["predicted_feasibility_score"]
                if not np.allclose(scores[i], old, atol=1e-12, rtol=0):
                    raise ValueError("Inference changed an archived held-out bounded score")
                scores[i] = old
            records = transfer.records(rows, masses, logits, scores, 1., panels)
            training[v][fold["split"]].extend(records[i] for i in np.flatnonzero(train))
            held_diagnostics[v][fold["split"]].extend(records[i] for i in np.flatnonzero(held & mean))
    variants = {}
    for v in VARIANTS:
        validation, halves = {}, {}
        for split in SPLITS:
            predictions = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][v]]
            if len(predictions) != 240 or {state_key(r) for r in predictions} != keys:
                raise ValueError("Lost an exact240-state outer split")
            for row in predictions:
                observed = panels[state_key(row)]["values"][:, row["selected"]][:, [0, 3]].mean(axis=0)
                if not np.allclose(observed, [row["actual_cost_delta"], row["wait_delta_min"]], atol=1e-12, rtol=0):
                    raise ValueError("Future labels differ from the archived selected action")
            validation[split] = {"overall": source.summary(predictions),
                "by_root": {str(root): source.summary([r for r in predictions if r["root"] == root]) for root in source.source.mean.ROOTS},
                "by_regime": {s: source.summary([r for r in predictions if r["scenario"] == s]) for s in source.source.mean.SCENARIOS},
                "training_mean8": transfer.summary(training[v][split]), "held_out_diagnostics": transfer.scopes(held_diagnostics[v][split])}
            halves[split] = {p: {str(root): {"A": previous.previous.half_summary([r for r in predictions if r["root"] == root], panels, a),
                "B": previous.previous.half_summary([r for r in predictions if r["root"] == root], panels, b)} for root in source.source.mean.ROOTS}
                for p, (a, b) in source.diagnostic.PARTITIONS.items()}
        variants[v] = {"validation": validation, "future_halves": halves,
            "gates": {"cost": all(m["mean_actual_cost_delta"] < 0 for s in validation.values() for m in s["by_root"].values()),
                "wait": all(m["mean_wait_delta_min"] <= 0 for s in validation.values() for m in s["by_root"].values())},
            "half_gates": {p: {"cost": all(m["mean_cost"] < 0 for s in halves.values() for r in s[p].values() for m in r.values()),
                "wait": all(m["mean_wait_min"] <= 0 for s in halves.values() for r in s[p].values() for m in r.values())} for p in source.diagnostic.PARTITIONS}}
    for v in bounded.FUNCTIONAL_RIDGES:
        variants[v]["held_excess_improves_GLM"] = all(variants[v]["validation"][s]["held_out_diagnostics"]["overall"]["mass_weighted_excess_nats"] <
            variants["symmetric_feasibility"]["validation"][s]["held_out_diagnostics"]["overall"]["mass_weighted_excess_nats"] for s in SPLITS)
    selections = {v: [{"fold": c["fold"], "selected_step": c["selections"][v]["selected_step"],
        "pooled_curve": c["selections"][v]["pooled_curve"]} for c in cells.values()] for v in bounded.FUNCTIONAL_RIDGES}
    primary = variants[PRIMARY]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "adamw_steps": sum(c["adamw_steps"] for c in cells.values()),
        "inner_glm_iterations": sum(c["selections"][PRIMARY]["inner_glm_iterations"] for c in cells.values()),
        "outer_optimized_fits": {v: sum(c["fits"][v]["steps"] > 0 for c in cells.values()) for v in bounded.FUNCTIONAL_RIDGES},
        "checkpoint_counts": {v: dict(sorted(Counter(str(s["selected_step"]) for s in rs).items())) for v, rs in selections.items()},
        "selections": selections, "evaluation_states": 240, "evaluation_prefix_groups": 40, "variants": variants,
        "development_gate": all(primary["gates"].values()) and primary["held_excess_improves_GLM"],
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
    if registered != manifest(old):
        raise ValueError("Changed the registered bounded wait protocol")
    bank = json.loads(source.source.coverage.bank_path(source.source.mean.SOURCE_RUN).read_text())
    prepared = json.loads(source.mass_path(previous.previous.SOURCE_RUN).read_text())
    if prepared["contract"] != source.contract():
        raise ValueError("Changed the original mean8 wait masses")
    masses = {state_key(r): r for r in prepared["targets"]}
    def load_models(fold):
        glm = raw_directory(source.result_path(previous.previous.SOURCE_RUN, fold))
        fixed = raw_directory(previous.previous.result_path(previous.SOURCE_RUN, fold))
        early = raw_directory(previous.result_path(SOURCE_RUN, fold))
        new = raw_directory(result_path(args.run_name, fold))
        return {"symmetric_feasibility": json.loads((glm/"symmetric_feasibility.json").read_text()),
            "state_conditioned_pair": json.loads((fixed/"state_conditioned_pair.json").read_text()),
            previous.PRIMARY: json.loads((early/f"{previous.PRIMARY}.json").read_text()),
            **{v: json.loads((new/f"{v}.json").read_text()) for v in bounded.FUNCTIONAL_RIDGES}}
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered, bank, masses, source.diagnostic.load_panels(), load_models)
        write_json(directory/"report/result.json", result)
        print(json.dumps({"checkpoints": result["checkpoint_counts"], "primary_gates": result["variants"][PRIMARY]["gates"]}), flush=True)
        print("NATIVE_BOUNDED_WAIT_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None:
        parser.error("Fit requires split and fold index")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    prior = json.loads((raw_directory(previous.result_path(SOURCE_RUN, fold))/"result.json").read_text())
    previous.qualify(prior, fold, old)
    glm = raw_directory(source.result_path(previous.previous.SOURCE_RUN, fold))/"symmetric_feasibility.json"
    targets = json.loads(source.source.mean.expected.targets_path(source.source.credit.source.SOURCE_RUN).read_text())["targets"]
    cell, models = fit_fold(bank, targets, masses, fold, prior, json.loads(glm.read_text()), registered)
    qualify(cell, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", cell)
    for v, m in models.items():
        write_json(raw_directory(output)/f"{v}.json", m)
    write_json(output, {k: v for k, v in cell.items() if k not in ("fits", "selections", "predictions", "frozen_cost_fit")})
    print(f"BOUNDED_WAIT_FIT split={args.split} fold={args.fold_index} steps={[m['steps'] for m in models.values()]} adamw={cell['adamw_steps']}", flush=True)
    print("NATIVE_BOUNDED_WAIT_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
