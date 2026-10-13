#!/usr/bin/env python3
"""Separate same-prefix future transfer from held-prefix wait-head transfer."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_early_stop_wait_stage209 as previous
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_future_transfer_stage210_v1"
SOURCE_RUN = "native_transit_early_stop_wait_stage209_development_20261013_r1"
source, transfer, neural, stopping = previous.source, previous.transfer, previous.neural, previous.stopping
SPLITS, state_key, PARTITIONS = source.SPLITS, source.state_key, source.diagnostic.PARTITIONS
POPULATIONS = ("same_prefix_train_future", "same_prefix_other_future", "held_prefix_train_future", "held_prefix_other_future")
METRICS = ("mass_weighted_excess_nats", "own_loss_error_rate_informative")
SUMMARY_FIELDS = ("state_model_evaluations", "unique_states", "unique_prefix_groups", "candidate_evaluations",
    "informative_candidates", "zero_signal_candidates", "own_loss_error_rate_informative", "false_safe_rate_informative",
    "mass_weighted_excess_nats", "mean_expected_boundary_regret_min", "half_sign_disagreement_candidates",
    "both_halves_false_safe", "both_halves_false_unsafe", "weighted_loss_sum_min", "weighted_oracle_loss_sum_min", "weighted_mass_sum_min")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "data": "same600_causal_inputs_240_mean8_states_360_single_targets_47_grouped_outer_folds",
        "partitions": {k: [list(a), list(b)] for k, (a, b) in PARTITIONS.items()}, "directions": ["A", "B"],
        "target": "only_training_half_positive_negative_wait_masses_count4_mean_states_count1_unchanged_single_states",
        "reference": "new_symmetric_GLM_and_all_scales_fit_on_outer_train_and_training_half_only",
        "neural": "unchanged_state_conditioned210_input16_tanh_residual_seed208_lr0.003_decay0.01_float64",
        "checkpoints": list(stopping.CHECKPOINTS), "checkpoint_selection": "none_report_all_primary600_step0_exact_GLM",
        "populations": list(POPULATIONS), "scene_reference": "same_root_regime_training_mean_states",
        "contrasts": "neural_minus_own_GLM_on_same_prefix_other_future_and_held_prefix_other_future",
        "controls": "same_fit_predictions_on_both_label_halves_no_cost_fit_or_policy_selection",
        "budget": {"new_wait_glm_fits": 188, "neural_paths": 188, "adamw_steps": 188*max(stopping.CHECKPOINTS),
            "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "bank_half_targets_and_state_diagnostics_remote_small_receipts_report_only",
        "boundary": "conditional_future_and_prefix_transfer_diagnostic_not_CI_native_reward_or_isolated_lower_policy_effect"}


def manifest(old):
    if old["protocol"] != previous.EXPERIMENT_PROTOCOL or old["contract"] != previous.contract():
        raise ValueError("Future-transfer diagnosis requires the exact Stage209 cohort")
    return {**old, "protocol": EXPERIMENT_PROTOCOL, "contract": contract()}


def result_path(run_name, fold):
    return ROOT/"results"/run_name/"cells"/fold["split"]/f"fold_{fold['index']:03}"/"receipt.json"


def mass_path(run_name):
    return raw_directory(ROOT/"results"/run_name/"prepare/receipt.json")/"halves.json"


def half_masses(rows, full_masses, panels):
    halves = {}
    for partition, indices in PARTITIONS.items():
        halves[partition] = {}
        for half, positions in zip(("A", "B"), indices, strict=True):
            subset = {k: {**p, "values": p["values"][list(positions)]} for k, p in panels.items()}
            halves[partition][half] = source.build_masses(rows, subset)
        a, b = [{state_key(r): r for r in halves[partition][h]} for h in ("A", "B")]
        for row in rows:
            k = state_key(row)
            for field in ("positive_wait_min", "negative_wait_min"):
                if not np.allclose((np.asarray(a[k][field])+b[k][field])/2, full_masses[k][field], atol=1e-12, rtol=0):
                    raise ValueError("Future halves do not reproduce the archived mean8 or singleton target")
    return halves


def population_keys(rows, fold):
    held = [r for r in rows if source.source.credit.group_key(r, fold["split"]) == fold["held_out"]]
    train = [r for r in rows if source.source.credit.group_key(r, fold["split"]) != fold["held_out"]]
    if fold["split"] == "scene":
        root, regime = held[0]["root"], held[0]["scenario"]
        train = [r for r in train if (r["root"], r["scenario"]) == (root, regime)]
    return {state_key(r) for r in train}, {state_key(r) for r in held}


def fit_record(model, masses, rows):
    result = stopping.metadata(model)
    result["training_mean8_states"] = result.pop("training_mean_states")
    result["training_half4_states"] = sum(masses[state_key(r)]["draw_count"] == 4 for r in rows)
    return result


def fit_case(train, mean_rows, half_tables, fold, train_half, *, train_batch=None, mean_batch=None):
    start = time.monotonic()
    other_half = "B" if train_half == "A" else "A"
    masses = half_tables[train_half]
    glm = source.feasibility.fit_wait(train, masses, "symmetric_feasibility")
    mean_batch = neural.batches(mean_rows) if mean_batch is None else mean_batch
    originals = np.asarray([r["original_choice"] for r in mean_rows])
    mask = np.arange(6)[None, :] != originals[:, None]
    train_keys, held_keys = population_keys(mean_rows, fold)
    a, b = [np.asarray([np.asarray(half_tables[h][state_key(r)]["positive_wait_min"])-
        half_tables[h][state_key(r)]["negative_wait_min"] for r in mean_rows]) for h in ("A", "B")]
    signal = np.asarray([sum(np.asarray(half_tables[h][state_key(r)]["positive_wait_min"])+
        half_tables[h][state_key(r)]["negative_wait_min"] for h in ("A", "B")) for r in mean_rows]) > 0
    checkpoints = {}
    def observe(model):
        logits = neural.logits_many(mean_rows, model, batch=mean_batch)
        scores = expit(logits)-.5
        scores[np.arange(len(mean_rows)), originals] = 0.
        if model["steps"] == 0:
            reference_logits, _, _ = transfer.model_scores(mean_batch[0], originals, glm)
            if not np.array_equal(logits, reference_logits):
                raise ValueError("Step0 no longer reproduces its own half-trained GLM")
        safe = scores <= 0
        stable_safe_error = (mask & safe & (a > 0) & (b > 0)).sum(axis=1)
        stable_unsafe_error = (mask & ~safe & (a <= 0) & (b <= 0) & signal).sum(axis=1)
        disagreement = (mask & ((a > 0) != (b > 0))).sum(axis=1)
        populations = {}
        for half, suffix in ((train_half, "train_future"), (other_half, "other_future")):
            records = transfer.records(mean_rows, half_tables[half], logits, scores, 1., {})
            for i, r in enumerate(records):
                r.update(half_sign_disagreement_candidates=int(disagreement[i]), both_halves_false_safe=int(stable_safe_error[i]),
                    both_halves_false_unsafe=int(stable_unsafe_error[i]))
            populations["same_prefix_"+suffix] = [r for r in records if state_key(r) in train_keys]
            populations["held_prefix_"+suffix] = [r for r in records if state_key(r) in held_keys]
        checkpoints[str(model["steps"])] = populations
    model, _ = stopping.fit_path(train, masses, glm, stopping.CHECKPOINTS, batch=train_batch, on_checkpoint=observe)
    return {"train_half": train_half, "evaluation_half": other_half, "glm_fit": fit_record(glm, masses, train),
        "neural_fit": fit_record(model, masses, train), "checkpoints": checkpoints,
        "fit_and_diagnostic_wall_s": time.monotonic()-start}


def fit_fold(bank, tables, fold, registered):
    train, _, _ = source.source.coverage.split_rows(bank, fold["split"], fold["index"])
    mean_keys = {state_key(r) for r in registered["mean_states"]}
    mean_rows = [r for r in bank["training_data"] if state_key(r) in mean_keys]
    train_batch, mean_batch = neural.batches(train), neural.batches(mean_rows)
    cases = {p+":"+h: fit_case(train, mean_rows, tables[p], fold, h, train_batch=train_batch, mean_batch=mean_batch)
        for p in PARTITIONS for h in ("A", "B")}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "fold": fold,
        "cases": cases, "new_wait_glm_fits": 4, "neural_paths": 4, "adamw_steps": 4*max(stopping.CHECKPOINTS),
        "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0}


def qualify(cell, fold, registered):
    groups = [g for g in registered["groups"] if source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
    membership = sorted([[g["root"], g["scene_seed"]] for g in groups])
    train_keys, held_keys = population_keys(registered["mean_states"], fold)
    mean_train = 240-len(held_keys)
    if not (cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["fold"] == fold and cell["new_wait_glm_fits"] == cell["neural_paths"] == 4
            and cell["adamw_steps"] == 4*max(stopping.CHECKPOINTS)
            and cell["new_cost_fits"] == cell["actor_updates"] == cell["new_native_episodes"] == 0
            and set(cell["cases"]) == {p+":"+h for p in PARTITIONS for h in ("A", "B")}):
        raise ValueError("Changed half-future cohort or fitting budget")
    for name, case in cell["cases"].items():
        if not name.endswith(":"+case["train_half"]) or case["train_half"] == case["evaluation_half"]:
            raise ValueError("Future validation uses the training half")
        for kind in ("glm_fit", "neural_fit"):
            m = case[kind]
            if not (m["training_groups"] == membership and m["training_states"] == 6*len(groups)
                    and m["training_half4_states"] == mean_train and m["training_mean8_states"] == 0
                    and m["sample_weight_sum"] == 6*len(groups)+3*mean_train and m["false_safe_cost"] == 1.):
                raise ValueError("Half-trained head included outer-held scenes or changed weights")
        glm, model = case["glm_fit"], case["neural_fit"]
        if not (glm["parameter_count"] == 199 and glm["solver_success"] and model["parameter_count"] == 3393
                and model["steps"] == max(stopping.CHECKPOINTS) and model["initialization_seed"] == neural.SEED
                and model["parameter_update_l2"] > 0 and model["hidden_gradient_l2_max"] > 0):
            raise ValueError("Half-trained GLM or nonlinear optimizer did not run")
        if set(case["checkpoints"]) != set(map(str, stopping.CHECKPOINTS)):
            raise ValueError("Lost a registered checkpoint")
        for populations in case["checkpoints"].values():
            if set(populations) != set(POPULATIONS):
                raise ValueError("Lost a same-prefix or held-prefix future comparison")
            for population, rows in populations.items():
                expected = train_keys if population.startswith("same_prefix") else held_keys
                if len(rows) != len(expected) or {state_key(r) for r in rows} != expected:
                    raise ValueError("Diagnostic duplicated or lost a mean state")


def summary(records):
    measured = transfer.summary(records)
    return {k: measured[k] for k in SUMMARY_FIELDS}


def scopes(records):
    return {"by_root": {str(r): summary([v for v in records if v["root"] == r]) for r in sorted({v["root"] for v in records})},
        "by_regime": {s: summary([v for v in records if v["scenario"] == s]) for s in sorted({v["scenario"] for v in records})}}


def contrasts(current, reference):
    result = {}
    for metric in METRICS:
        effect = {p: current[p][metric]-reference[p][metric] for p in POPULATIONS}
        result[metric] = {"neural_minus_GLM": effect,
            "future_gap_change_vs_GLM": effect["same_prefix_other_future"]-effect["same_prefix_train_future"],
            "held_prefix_gap_change_vs_GLM": effect["held_prefix_other_future"]-effect["same_prefix_other_future"]}
    return result


def summarize(cells, registered):
    if len(cells) != 47 or set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Complete exactly47 future-transfer folds before reporting")
    for fold in registered["folds"]:
        qualify(cells[fold["split"], fold["index"]], fold, registered)
    validation = {}
    for partition in PARTITIONS:
        validation[partition] = {}
        for split in SPLITS:
            validation[partition][split] = {}
            for half in ("A", "B"):
                steps, details, baseline = {}, {}, None
                for step in stopping.CHECKPOINTS:
                    pooled = {p: [r for f in registered["folds"] if f["split"] == split
                        for r in cells[split, f["index"]]["cases"][partition+":"+half]["checkpoints"][str(step)][p]] for p in POPULATIONS}
                    for p in POPULATIONS[2:]:
                        if len(pooled[p]) != 240 or len({state_key(r) for r in pooled[p]}) != 240:
                            raise ValueError("Lost the exact240-state held-prefix split")
                    current = {p: summary(rs) for p, rs in pooled.items()}
                    if step == 0:
                        baseline = current
                    steps[str(step)] = {"populations": current, "contrasts": contrasts(current, baseline)}
                    if step in (0, max(stopping.CHECKPOINTS)):
                        details[str(step)] = {p: scopes(rs) for p, rs in pooled.items()}
                validation[partition][split][half+"_to_"+("B" if half == "A" else "A")] = {"steps": steps, "details": details}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, **contract()["budget"],
        "glm_optimizer_iterations": sum(case["glm_fit"]["optimizer_iterations"] for c in cells.values() for case in c["cases"].values()),
        "evaluation_states": 240, "evaluation_prefix_groups": 40, "validation": validation,
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("prepare", "fit", "report"), required=True)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--fold-index", type=int)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    old = json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text())
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    if registered != manifest(old):
        raise ValueError("Changed the registered future-transfer protocol")
    if args.operation == "report":
        cells = {(f["split"], f["index"]): json.loads((raw_directory(result_path(args.run_name, f))/"result.json").read_text()) for f in registered["folds"]}
        result = summarize(cells, registered)
        write_json(directory/"report/result.json", result)
        print(json.dumps({p: {s: {d: v["steps"][str(max(stopping.CHECKPOINTS))]["contrasts"]["mass_weighted_excess_nats"]
            for d, v in splits.items()} for s, splits in validation.items()} for p, validation in result["validation"].items()}, indent=2), flush=True)
        print("NATIVE_FUTURE_TRANSFER_REPORT_COMPLETE", flush=True)
        return
    bank = json.loads(source.source.coverage.bank_path(source.source.mean.SOURCE_RUN).read_text())
    if args.operation == "prepare":
        original = json.loads(source.mass_path(previous.previous.SOURCE_RUN).read_text())
        if original["contract"] != source.contract():
            raise ValueError("Changed the archived mean8 wait masses")
        masses = {state_key(r): r for r in original["targets"]}
        panels = source.diagnostic.load_panels()
        if len(bank["training_data"]) != 600 or set(panels) != {state_key(r) for r in registered["mean_states"]}:
            raise ValueError("Changed the600/240 half-future population")
        write_json(mass_path(args.run_name), {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
            "targets": half_masses(bank["training_data"], masses, panels)})
        write_json(directory/"prepare/receipt.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
            "input_states": 600, "mean_states": 240, "draws_per_half": 4, "unchanged_single_states": 360})
        print("NATIVE_FUTURE_TRANSFER_PREPARE_COMPLETE", flush=True)
        return
    if args.split is None or args.fold_index is None:
        parser.error("Fit requires split and fold index")
    fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (args.split, args.fold_index))
    prepared = json.loads(mass_path(args.run_name).read_text())
    if prepared["contract"] != contract():
        raise ValueError("Changed the prepared future halves")
    tables = {p: {h: {state_key(r): r for r in rs} for h, rs in halves.items()} for p, halves in prepared["targets"].items()}
    cell = fit_fold(bank, tables, fold, registered)
    qualify(cell, fold, registered)
    output = result_path(args.run_name, fold)
    write_json(raw_directory(output)/"result.json", cell)
    write_json(output, {k: v for k, v in cell.items() if k != "cases"})
    print(f"FUTURE_TRANSFER_FIT split={args.split} fold={args.fold_index} adamw={cell['adamw_steps']}", flush=True)
    print("NATIVE_FUTURE_TRANSFER_FIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
