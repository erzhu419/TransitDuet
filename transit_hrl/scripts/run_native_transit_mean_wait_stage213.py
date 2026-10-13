#!/usr/bin/env python3
"""Isolate training-label precision with fixed costs, weights and wait heads."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_training_means_stage212 as collection
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

previous = collection.previous
source, transfer, bounded, stopping, neural = previous.source, previous.transfer, previous.bounded, previous.stopping, previous.neural
EXPERIMENT_PROTOCOL = "native_transit_mean_wait_stage213_v1"
SOURCE_RUN = "native_transit_training_means_stage212_development_20261013_r1"
CONTROL_RUN = collection.SOURCE_RUN
CONTROLS = {"old_glm": "symmetric_feasibility", "old_bounded": "bounded_only", "old_shrunk": "shrunk_bounded"}
NEW_GLM = "mean8_glm"
CORRECTIONS = {"mean8_bounded": "bounded_only", "mean8_shrunk": "shrunk_bounded"}
PRIMARY = "mean8_shrunk"
VARIANTS = (*CONTROLS, NEW_GLM, *CORRECTIONS)
SPLITS, state_key = previous.SPLITS, previous.state_key


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": collection.EXPERIMENT_PROTOCOL, "control_run": CONTROL_RUN,
        "data": "same600_inputs_all600_mean8_training_masses_original240_evaluation_states_47_outer_folds",
        "weights": collection.contract()["weights"], "variants": list(VARIANTS), "primary": PRIMARY,
        "costs": "exact_Stage211_cost_heads_menu_threshold0.5_identity_ties_frozen",
        "reference": "new_outer_train_only_symmetric_GLM_and_four_shared_inner_train_only_GLMS",
        "residual_bound": bounded.BOUND, "functional_ridges": bounded.FUNCTIONAL_RIDGES,
        "optimizer": "same3393_parameters_seed208_float64_AdamW_lr0.003_wd0.01",
        "inner_selection": "unchanged_four_scene_partitions_validation_original240_only_pooled_crossentropy_ties_earliest",
        "partition_seed": stopping.grouping.PARTITION_SEED, "checkpoint_grid": list(stopping.CHECKPOINTS),
        "gate": "negative_cost_nonpositive_wait_both_roots_all_three_splits_and_lower_held_excess_than_new_GLM_all_splits",
        "future_halves": source.diagnostic.contract()["partitions"],
        "budget": {"new_outer_glm_fits": 47, "new_inner_glm_fits": 188, "inner_neural_paths": 376,
            "outer_model_assemblies": 94, "max_adamw_steps": 282000, "new_cost_fits": 0,
            "actor_updates": 0, "new_native_episodes": 0},
        "storage": "mean_targets_models_predictions_remote_small_report_only",
        "boundary": "single_action_supervised_development_not_independent_CI_or_full_learned_HRL"}


def manifest(old):
    if old["protocol"] != collection.EXPERIMENT_PROTOCOL or old["contract"] != collection.contract():
        raise ValueError("Mean-wait comparison requires the completed Stage212 cohort")
    return {**old, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def checked_masses(bank, prepared, original, registered):
    masses = {state_key(m): m for m in prepared["targets"]}
    old = {state_key(m): m for m in original["targets"]}
    keys = {state_key(r) for r in bank["training_data"]}
    evaluation = {state_key(r) for r in registered["mean_states"]}
    upgraded = {state_key(r) for r in registered["training_states"]}
    if not (prepared["protocol"] == collection.EXPERIMENT_PROTOCOL and prepared["contract"] == collection.contract()
            and original["protocol"] == source.EXPERIMENT_PROTOCOL and original["contract"] == source.contract()
            and len(masses) == len(prepared["targets"]) == len(keys) == 600 and set(masses) == set(old) == keys
            and len(evaluation) == 240 and len(upgraded) == 360 and not evaluation & upgraded
            and evaluation | upgraded == keys
            and {state_key(r) for r in prepared["new_mean_labels"]} == upgraded):
        raise ValueError("Changed training/evaluation target membership")
    for key, mass in masses.items():
        if mass["draw_count"] != 8 or mass.get("sample_weight", mass["draw_count"]) != old[key]["draw_count"]:
            raise ValueError("Changed optimization weights while increasing label precision")
        if key in evaluation and mass != old[key]:
            raise ValueError("Changed an original mean8 evaluation target")
    return masses


def fit_fold(bank, targets, masses, fold, prior, registered):
    train, test, _ = source.source.coverage.split_rows(bank, fold["split"], fold["index"])
    keys = {state_key(r) for r in registered["mean_states"]}
    test = source.source.mean.expected.replace_targets([r for r in test if state_key(r) in keys], targets)
    groups = [g for g in registered["groups"] if source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    if prior["frozen_cost_fit"]["training_groups"] != membership:
        raise ValueError("Mean labels changed the frozen cost training population")
    start = time.monotonic()
    glm = source.feasibility.fit_wait(train, masses, "symmetric_feasibility")
    corrections, selections = bounded.fit_wait(train, masses, glm, groups, keys)
    for variant in CORRECTIONS.values():
        if selections[variant]["partition"] != prior["selections"][variant]["partition"]:
            raise ValueError("Mean labels changed the original inner validation partition")
    models = {NEW_GLM: glm, **{v: corrections[k] for v, k in CORRECTIONS.items()}}
    predictions = {v: prior["predictions"][k] for v, k in CONTROLS.items()}
    frozen = {state_key(r): r for r in predictions["old_glm"]}
    predictions[NEW_GLM] = source.evaluate(test, frozen, glm)
    batch = neural.batches(test)
    predictions.update({v: previous.previous.previous.evaluate(test, frozen, bounded.predict_many(test, models[v], batch=batch)) for v in CORRECTIONS})
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "frozen_cost_fit": prior["frozen_cost_fit"], "fits": {v: stopping.metadata(m) for v, m in models.items()},
        "selections": {v: selections[k] for v, k in CORRECTIONS.items()}, "predictions": predictions,
        "outer_glm_fits": 1, "inner_glm_fits": 4, "inner_neural_paths": 8, "outer_model_assemblies": 2,
        "adamw_steps": sum(s["inner_adamw_steps"]+s["outer_adamw_steps"] for s in selections.values()),
        "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0,
        "fit_and_evaluation_wall_s": time.monotonic()-start}, models


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    keys = {state_key(r) for r in registered["mean_states"] if source.source.credit.group_key(r, fold["split"]) == fold["held_out"]}
    mean_groups = {(r["root"], r["scene_seed"]) for r in registered["mean_states"]}
    partition = stopping.grouping.partition_groups(groups, mean_groups)
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["frozen_cost_fit"]["training_groups"] == membership
            and (cell["outer_glm_fits"], cell["inner_glm_fits"], cell["inner_neural_paths"], cell["outer_model_assemblies"]) == (1, 4, 8, 2)
            and cell["adamw_steps"] == 8*max(stopping.CHECKPOINTS)+sum(cell["fits"][v]["steps"] for v in CORRECTIONS)
            and cell["new_cost_fits"] == cell["actor_updates"] == cell["new_native_episodes"] == 0):
        raise ValueError("Changed mean-wait fitting cohort or budget")
    for variant, fit in cell["fits"].items():
        if not (fit["training_groups"] == membership and fit["training_states"] == fit["training_mean_states"] == 6*len(groups)
                and fit["sample_weight_sum"] == 6*len(groups)+7*(240-len(keys)) and fit["false_safe_cost"] == 1.):
            raise ValueError("Changed mean target depth, groups or retained sample weights")
        if variant == NEW_GLM:
            if not (fit["parameter_count"] == 199 and fit["ridge"] == source.feasibility.RIDGE
                    and fit["max_iterations"] == source.feasibility.MAX_ITERATIONS and fit["solver_success"]):
                raise ValueError("Changed the training-only symmetric GLM")
            continue
        suffix, selection = CORRECTIONS[variant], cell["selections"][variant]
        if not (fit["parameter_count"] == 3393 and fit["correction"] == suffix and fit["residual_bound"] == bounded.BOUND
                and fit["functional_ridge"] == bounded.FUNCTIONAL_RIDGES[suffix] and fit["initialization_seed"] == neural.SEED
                and fit["steps"] == selection["selected_step"] in stopping.CHECKPOINTS
                and fit["residual_logit_max_abs"] <= bounded.BOUND and np.isfinite(fit["final_objective"])
                and selection["partition"] == partition and len(selection["inner_fits"]) == 4):
            raise ValueError("Changed bounded head, original inner partition or chosen step")
        for bucket, inner in zip(partition, selection["inner_fits"], strict=True):
            expected = [g for g in membership if g not in bucket]
            mean_states = [r for r in registered["mean_states"] if [r["root"], r["scene_seed"]] in expected]
            validation_count = sum([r["root"], r["scene_seed"]] in bucket for r in registered["mean_states"])
            if inner["held_groups"] != bucket or inner["validation_mean_states"] != validation_count:
                raise ValueError("Inner validation admitted a newly averaged training state")
            for name in ("glm_fit", "neural_fit"):
                model = inner[name]
                if not (model["training_groups"] == expected and model["training_states"] == model["training_mean_states"] == 6*len(expected)
                        and model["sample_weight_sum"] == 6*len(expected)+7*len(mean_states)):
                    raise ValueError("Inner training included validation scenes or changed weights")
        step, curve = stopping.select_checkpoint([i["curve"] for i in selection["inner_fits"]])
        if step != fit["steps"] or curve != selection["pooled_curve"]:
            raise ValueError("Checkpoint was not chosen by the unchanged inner criterion")
        if fit["steps"] == 0:
            if fit["parameter_update_l2"] != 0 or fit["residual_logit_rms"] != 0:
                raise ValueError("Step0 no longer preserves the new GLM")
        elif fit["parameter_update_l2"] <= 0 or fit["hidden_gradient_l2_max"] <= 0:
            raise ValueError("Selected bounded correction did not train")
    a, b = [cell["selections"][v] for v in CORRECTIONS]
    if [i["glm_fit"] for i in a["inner_fits"]] != [i["glm_fit"] for i in b["inner_fits"]]:
        raise ValueError("Paired corrections did not share four inner GLMs")
    frozen = {state_key(r): r for r in cell["predictions"]["old_glm"]}
    for variant in VARIANTS:
        rows = cell["predictions"][variant]
        if len(rows) != len(keys) or {state_key(r) for r in rows} != keys:
            raise ValueError("Lost or repeated an original evaluation state")
        for row in rows:
            share, original = np.asarray(row["predicted_harm_mass_share"]), row["original_choice"]
            if (row["predicted_heads"] != frozen[state_key(row)]["predicted_heads"] or share[original] != 0
                    or not np.array_equal(np.where(np.arange(6) == original, 0., share-.5), row["predicted_feasibility_score"])
                    or source.choose(row, original) != row["selected"]):
                raise ValueError("Changed frozen costs, threshold or identity ties")


def variant_summary(cells, registered, diagnostics, panels, variant):
    keys = {state_key(r) for r in registered["mean_states"]}
    validation, halves = {}, {}
    for split in SPLITS:
        rows = [r for f in registered["folds"] if f["split"] == split for r in cells[split, f["index"]]["predictions"][variant]]
        if len(rows) != 240 or {state_key(r) for r in rows} != keys:
            raise ValueError("Lost one complete240-state split")
        for row in rows:
            panel = panels[state_key(row)]
            observed = panel["values"][:, row["selected"]][:, [0, 3]].mean(axis=0)
            if panel["original_choice"] != row["original_choice"] or not np.allclose(observed,
                    [row["actual_cost_delta"], row["wait_delta_min"]], atol=1e-12, rtol=0):
                raise ValueError("Changed the original held-out future outcomes")
        validation[split] = {"overall": source.summary(rows),
            "by_root": {str(r): source.summary([x for x in rows if x["root"] == r]) for r in collection.ROOTS},
            "by_regime": {s: source.summary([x for x in rows if x["scenario"] == s]) for s in collection.SCENARIOS},
            "training_mean8": transfer.summary(diagnostics["training"][variant][split]),
            "held_out_diagnostics": transfer.scopes(diagnostics["held"][variant][split])}
        halves[split] = {p: {str(r): {"A": previous.previous.previous.half_summary([x for x in rows if x["root"] == r], panels, a),
            "B": previous.previous.previous.half_summary([x for x in rows if x["root"] == r], panels, b)} for r in collection.ROOTS}
            for p, (a, b) in source.diagnostic.PARTITIONS.items()}
    return {"validation": validation, "future_halves": halves,
        "gates": {"cost": all(m["mean_actual_cost_delta"] < 0 for s in validation.values() for m in s["by_root"].values()),
            "wait": all(m["mean_wait_delta_min"] <= 0 for s in validation.values() for m in s["by_root"].values())},
        "half_gates": {p: {"cost": all(m["mean_cost"] < 0 for s in halves.values() for r in s[p].values() for m in r.values()),
            "wait": all(m["mean_wait_min"] <= 0 for s in halves.values() for r in s[p].values() for m in r.values())} for p in source.diagnostic.PARTITIONS}}


def summarize(cells, registered, bank, masses, panels, load_models):
    if set(cells) != {(f["split"], f["index"]) for f in registered["folds"]} or len(cells) != 47:
        raise ValueError("Complete all47 mean-wait folds before reporting")
    keys = {state_key(r) for r in registered["mean_states"]}
    if set(panels) != keys:
        raise ValueError("Changed the original evaluation future panels")
    rows = [r for r in bank["training_data"] if state_key(r) in keys]
    batch = neural.batches(rows)
    diagnostics = {scope: {v: {s: [] for s in SPLITS} for v in VARIANTS} for scope in ("training", "held")}
    for fold in registered["folds"]:
        cell = cells[fold["split"], fold["index"]]
        qualify(cell, fold, registered)
        held = np.asarray([source.source.credit.group_key(r, fold["split"]) == fold["held_out"] for r in rows])
        train = ~held
        if fold["split"] == "scene":
            exemplar = rows[int(np.flatnonzero(held)[0])]
            train &= np.asarray([(r["root"], r["scenario"]) == (exemplar["root"], exemplar["scenario"]) for r in rows])
        for v, model in load_models(fold).items():
            if v in ("old_glm", NEW_GLM):
                logits, _, scores = transfer.model_scores(batch[0], [r["original_choice"] for r in rows], model)
            else:
                logits = bounded.logits_many(rows, model, batch=batch)
                scores = np.asarray([p["predicted_feasibility_score"] for p in bounded.predict_many(rows, model, batch=batch)])
            if v not in CONTROLS and stopping.metadata(model) != cell["fits"][v]:
                raise ValueError("Mean-wait model differs from its fitting receipt")
            frozen = {state_key(r): r for r in cell["predictions"][v]}
            for i in np.flatnonzero(held):
                old = frozen[state_key(rows[i])]["predicted_feasibility_score"]
                if not np.allclose(scores[i], old, atol=1e-12, rtol=0):
                    raise ValueError("Model no longer reproduces its held-out scores")
                scores[i] = old
            records = transfer.records(rows, masses, logits, scores, 1., panels)
            for scope, mask in (("training", train), ("held", held)):
                diagnostics[scope][v][fold["split"]].extend(records[i] for i in np.flatnonzero(mask))
    variants = {v: variant_summary(cells, registered, diagnostics, panels, v) for v in VARIANTS}
    for v in CORRECTIONS:
        variants[v]["held_excess_improves_GLM"] = all(variants[v]["validation"][s]["held_out_diagnostics"]["overall"]["mass_weighted_excess_nats"] <
            variants[NEW_GLM]["validation"][s]["held_out_diagnostics"]["overall"]["mass_weighted_excess_nats"] for s in SPLITS)
    comparisons = {NEW_GLM: "old_glm", "mean8_bounded": "old_bounded", PRIMARY: "old_shrunk"}
    for v, old in comparisons.items():
        variants[v]["relative_to_old_targets"] = {s: {r: {"cost_delta": m["mean_actual_cost_delta"]-variants[old]["validation"][s]["by_root"][r]["mean_actual_cost_delta"],
            "wait_delta_min": m["mean_wait_delta_min"]-variants[old]["validation"][s]["by_root"][r]["mean_wait_delta_min"]}
            for r, m in variants[v]["validation"][s]["by_root"].items()} for s in SPLITS}
    primary = variants[PRIMARY]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "evaluation_states": 240, "evaluation_prefix_groups": 40, "variants": variants,
        "adamw_steps": sum(c["adamw_steps"] for c in cells.values()),
        "outer_glm_iterations": sum(c["fits"][NEW_GLM]["optimizer_iterations"] for c in cells.values()),
        "inner_glm_iterations": sum(c["selections"][PRIMARY]["inner_glm_iterations"] for c in cells.values()),
        "checkpoint_counts": {v: dict(sorted(Counter(str(c["fits"][v]["steps"]) for c in cells.values()).items())) for v in CORRECTIONS},
        "development_gate": all(primary["gates"].values()) and primary["held_excess_improves_GLM"],
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("fit", "report"), required=True)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--fold-index", type=int)
    args = parser.parse_args()
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    old = json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text())
    if registered != manifest(old):
        raise ValueError("Changed the registered mean-wait protocol")
    bank = json.loads(collection.mean_source.source.bank_path(collection.mean_source.SOURCE_RUN).read_text())
    prepared = json.loads((raw_directory(ROOT/"results"/SOURCE_RUN/"prepare/receipt.json")/"targets.json").read_text())
    original = json.loads(source.mass_path(collection.MASS_RUN).read_text())
    masses = checked_masses(bank, prepared, original, registered)
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        def load_models(fold):
            old_glm = raw_directory(source.result_path(collection.MASS_RUN, fold))
            old_heads, new = raw_directory(previous.result_path(CONTROL_RUN, fold)), raw_directory(result_path(args.run_name, fold))
            return {"old_glm": json.loads((old_glm/"symmetric_feasibility.json").read_text()),
                **{v: json.loads((old_heads/f"{k}.json").read_text()) for v, k in CONTROLS.items() if v != "old_glm"},
                **{v: json.loads((new/f"{v}.json").read_text()) for v in (NEW_GLM, *CORRECTIONS)}}
        result = summarize(cells, registered, bank, masses, source.diagnostic.load_panels(), load_models)
        control = json.loads((ROOT/"results"/CONTROL_RUN/"report/result.json").read_text())
        for v, k in CONTROLS.items():
            for s in SPLITS:
                for scope in ("overall", "by_root", "by_regime"):
                    if result["variants"][v]["validation"][s][scope] != control["variants"][k]["validation"][s][scope]:
                        raise ValueError("Archived Stage211 control outcomes did not reproduce")
            if result["variants"][v]["future_halves"] != control["variants"][k]["future_halves"]:
                raise ValueError("Archived Stage211 future-half controls did not reproduce")
        result["controls_reproduced"] = True
        write_json(ROOT/"results"/args.run_name/"report/result.json", result)
        print(json.dumps({"checkpoints": result["checkpoint_counts"], "gate": result["development_gate"]}), flush=True)
        print("NATIVE_MEAN_WAIT_REPORT_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None: parser.error("Fit requires a registered grouped fold")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    prior = json.loads((raw_directory(previous.result_path(CONTROL_RUN, fold))/"result.json").read_text())
    previous.qualify(prior, fold, json.loads((ROOT/"results"/CONTROL_RUN/"inputs/manifest.json").read_text()))
    targets = json.loads(collection.mean_source.expected.targets_path(collection.MODEL_RUN).read_text())["targets"]
    cell, models = fit_fold(bank, targets, masses, fold, prior, registered)
    qualify(cell, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", cell)
    for v, m in models.items(): write_json(raw_directory(output)/f"{v}.json", m)
    write_json(output, {k: v for k, v in cell.items() if k not in ("fits", "selections", "predictions", "frozen_cost_fit")})
    print(f"MEAN_WAIT_FIT {fold['split']}/{fold['index']} adamw={cell['adamw_steps']}", flush=True)
    print("NATIVE_MEAN_WAIT_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
