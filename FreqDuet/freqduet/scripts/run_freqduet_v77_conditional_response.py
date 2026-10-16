#!/usr/bin/env python3
"""Bounded whole-day-excluded test of nonlinear and continuation-policy effects."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import freqduet_conditional_response as conditional
import run_freqduet_v75_resource_value as source


old, model, previous = source.old, source.model, source.previous
CHANNELS = ("zero", "action_only", "pooled_linear", "policy_linear", "pooled_nonlinear", "policy_nonlinear")
PRIMARY = "policy_nonlinear"
FACTOR_PAIRS = (("policy_nonlinear", "pooled_nonlinear"), ("policy_nonlinear", "policy_linear"),
                ("pooled_nonlinear", "pooled_linear"), ("policy_linear", "pooled_linear"))


def read_data(root):
    with np.load(root / "prepared_train.npz") as data:
        return {k: data[k] for k in ("states", "resources", "native", "outcomes", "days", "cases", "windows")}


def partition(data, held_day):
    training = np.flatnonzero(data["days"] != held_day)
    testing = np.flatnonzero(data["days"] == held_day)
    if not len(testing) or not len(training):
        raise ValueError("whole-day exclusion requires fitting and held-out contexts")
    return training, testing


def selected_rows(data, testing, predictions):
    rows = old.score_predictions(data, testing, predictions)
    for local, (index, row) in enumerate(zip(testing, rows)):
        native = int(data["native"][index])
        for channel, prediction in predictions.items():
            selected = model.choose_action(prediction[local], native)
            record = row["channels"][channel]
            record["predicted"] = {m: float(prediction[local, selected, j]) for j, m in enumerate(model.METRICS)}
            record["delta_action_s"] = model.ACTIONS[selected] - model.ACTIONS[native]
            contrasts = data["outcomes"][index, :, selected] - data["outcomes"][index, :, native]
            record["half_effects"] = [{m: float(part[:, j].mean()) for j, m in enumerate(old.OUTCOMES)}
                                      for part in np.array_split(contrasts, 2)]
    return rows


def training_choices(predictions, targets, native):
    selected = np.asarray([model.choose_action(p, int(n)) for p, n in zip(predictions, native)])
    effects = targets[np.arange(len(selected)), selected]
    return dict(contexts=len(selected), changed=int((selected != native).sum()),
        longer=int((np.asarray(model.ACTIONS)[selected] > np.asarray(model.ACTIONS)[native]).sum()),
        passenger_harms=int((effects[:, 0] > 0).sum()),
        mean_actual_passenger=float(effects[:, 0].mean()),
        mean_predicted_passenger=float(predictions[np.arange(len(selected)), selected, 0].mean()))


def fit_fold(v75_root, root, held_day):
    torch.set_num_threads(1)
    path = root / "folds" / str(held_day)
    if (path / "done.json").exists():
        return int(held_day)
    data = read_data(v75_root)
    training, testing = partition(data, held_day)
    target = model.mean_differences(data["outcomes"], data["native"])[:, :, old.TARGET_COLUMNS]
    fitting_days = sorted(set(data["days"][training].tolist()))
    predictions, fit_info = {}, {}
    path.mkdir(parents=True, exist_ok=True)
    for channel in CHANNELS[1:]:
        conditioned = channel.startswith("policy_")
        inputs = conditional.features(data, conditioned)
        args = (inputs[training], data["native"][training], target[training], data["days"][training])
        if channel.endswith("nonlinear"):
            fitted, info = conditional.fit_model(*args)
            torch.save(dict(state_dict=fitted.state_dict(), inputs=inputs.shape[1],
                            conditioned_on_known_actor=conditioned), path / (channel + ".pt"))
        else:
            kind = "action_only" if channel == "action_only" else "state_quadratic"
            fitted = model.fit_model(*args, kind)
            info = dict(parameters=int(fitted.coefficients.size),
                        effective_parameters_per_output=fitted.effective_parameters_per_output)
            source.save_model(path / (channel + ".npz"), fitted, channel, "v77_day_excluded")
        predictions[channel] = fitted.predict(inputs[testing], data["native"][testing])
        in_sample = fitted.predict(inputs[training], data["native"][training])
        info["training_choices"] = training_choices(in_sample, target[training], data["native"][training])
        fit_info[channel] = info
        if not np.isfinite(predictions[channel]).all():
            raise ValueError("non-finite held-day action effects")
    predictions["zero"] = np.zeros_like(predictions[PRIMARY])
    predictions = {c: predictions[c] for c in CHANNELS}
    rows = selected_rows(data, testing, predictions)
    previous.write_json(path / "done.json", dict(held_day=int(held_day), fitting_days=fitting_days,
        rows=rows, fits=fit_info, diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")))
    print("V77_FOLD_DONE " + json.dumps(dict(held_day=int(held_day), contexts=len(testing))), flush=True)
    return int(held_day)


def summary(rows, channel):
    records = [r["channels"][channel] for r in rows]
    changed = [r for r in records if r["changed"]]
    return dict(contexts=len(rows), changed=len(changed),
        longer=sum(r["delta_action_s"] > 0 for r in records), shorter=sum(r["delta_action_s"] < 0 for r in records),
        passenger_gains=sum(r["effects"]["passenger_person_s"] < 0 for r in changed),
        passenger_harms=sum(r["effects"]["passenger_person_s"] > 0 for r in changed),
        action_counts={str(a): sum(r["action_s"] == a for r in records) for a in model.ACTIONS},
        predicted={m: old.interval(rows, lambda r, m=m: r["channels"][channel]["predicted"][m]) for m in model.METRICS},
        effects={m: old.interval(rows, lambda r, m=m: r["channels"][channel]["effects"][m]) for m in old.OUTCOMES},
        both_halves_passenger_gain=sum(all(h["passenger_person_s"] < 0 for h in r["half_effects"]) for r in changed),
        both_halves_passenger_harm=sum(all(h["passenger_person_s"] > 0 for h in r["half_effects"]) for r in changed),
        changed_constraint_violations={m: sum(r["effects"][m] > 0 for r in changed) for m in model.METRICS[1:]})


def aggregate(v75_root, root):
    data = read_data(v75_root)
    days = sorted(set(data["days"].tolist()))
    folds = [json.loads((root / "folds" / str(day) / "done.json").read_text()) for day in days]
    for fold in folds:
        if fold["fitting_days"] != [d for d in days if d != fold["held_day"]]:
            raise ValueError("held-out demand day leaked into a fitting fold")
    rows = sorted([r for fold in folds for r in fold["rows"]], key=lambda r: r["context_index"])
    if [r["context_index"] for r in rows] != list(range(len(data["days"]))):
        raise ValueError("day-excluded predictions do not cover the saved contexts exactly once")
    channels = {c: summary(rows, c) for c in CHANNELS}
    errors = {}
    for c in CHANNELS:
        errors[c] = {}
        for m in old.PRIMARY:
            errors[c][m] = {loss: old.interval(rows, lambda r, c=c, m=m, loss=loss:
                r["channels"][c]["errors"][m][loss]) for loss in ("mae", "mse")}
            for kind in ("sign", "order"):
                stats = [r["channels"][c]["errors"][m] for r in rows]
                denominator = sum(s[kind + "_pairs"] for s in stats)
                errors[c][m][kind + "_agreement"] = sum(s[kind + "_correct"] for s in stats) / denominator if denominator else None
    contrasts = {a + "_minus_" + b: {
        "selected_effects": {m: old.interval(rows, lambda r, a=a, b=b, m=m:
            r["channels"][a]["effects"][m] - r["channels"][b]["effects"][m]) for m in model.METRICS},
        "prediction_mse": {m: old.interval(rows, lambda r, a=a, b=b, m=m:
            r["channels"][a]["errors"][m]["mse"] - r["channels"][b]["errors"][m]["mse"]) for m in old.PRIMARY}}
        for a, b in FACTOR_PAIRS}
    effects = channels[PRIMARY]["effects"]
    joint_pass = effects["passenger_person_s"]["ci95"][1] < 0 and all(effects[m]["ci95"][1] <= 0 for m in model.METRICS[1:])
    against_linear = old.interval(rows, lambda r: r["channels"][PRIMARY]["effects"]["passenger_person_s"] -
                                  r["channels"]["pooled_linear"]["effects"]["passenger_person_s"])
    development_pass = joint_pass and against_linear["ci95"][1] < 0
    passenger_by_day = [float(np.mean([r["channels"][PRIMARY]["effects"]["passenger_person_s"]
                        for r in rows if r["scenario_seed"] == day])) for day in days]
    gross = np.maximum(-np.asarray(passenger_by_day), 0.)
    previous.write_json(root / "oof_contexts.json", rows)
    previous.write_json(root / "summary" / "status.json", dict(status="conditional_response_day_exclusion_complete",
        primary_candidate=PRIMARY, demand_days=len(days), contexts=len(rows), channels=channels, errors=errors,
        factor_contrasts=contrasts, primary_minus_pooled_linear_passenger=against_linear,
        joint_effect_gate_passed=bool(joint_pass), development_stage_passed=bool(development_pass),
        next_step="new_day_single_intervention_confirmation" if development_pass else "stop_selector_expansion_return_system_constraint_learning",
        primary_by_actor={str(c): summary([r for r in rows if r["case_index"] == c], PRIMARY) for c in range(conditional.ACTORS)},
        primary_by_window={str(t): summary([r for r in rows if r["target_time_s"] == t], PRIMARY) for t in source.source.TARGETS},
        primary_by_direction={name: summary(subset, PRIMARY) for name in ("longer", "shorter")
            if (subset := [r for r in rows if (r["channels"][PRIMARY]["delta_action_s"] > 0 if name == "longer"
                                             else r["channels"][PRIMARY]["delta_action_s"] < 0)])},
        primary_half_effects=[{m: old.interval(rows, lambda r, h=h, m=m: r["channels"][PRIMARY]["half_effects"][h][m])
                              for m in model.METRICS} for h in range(2)],
        primary_largest_day_gross_gain_share=float(gross.max() / gross.sum()) if gross.sum() else None,
        by_day=[dict(scenario_seed=d, primary_passenger_effect=p) for d, p in zip(days, passenger_by_day)],
        fit_settings=dict(hidden=conditional.HIDDEN, steps=conditional.STEPS, lr=conditional.LR,
            weight_decay=conditional.WEIGHT_DECAY, initialization_seed=conditional.INIT_SEED),
        fitting_diagnostics=[dict(held_day=f["held_day"], fits=f["fits"]) for f in folds],
        actor_updates=0, dual_updates=0, rollouts_added=0, main_policy_changed=False, controller_admitted=False,
        original_training_seeds=2, full_day_exclusion=True, old_validation_labels_read=False,
        development_only=True, old_v75_gate_unchanged=True,
        diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")))
    print("V77_AGGREGATE " + json.dumps(dict(joint_effect_gate_passed=bool(joint_pass),
        development_stage_passed=bool(development_pass))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "folds", "aggregate"))
    parser.add_argument("--v75-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "aggregate":
        aggregate(args.v75_root, args.out_root)
        return
    days = sorted(set(read_data(args.v75_root)["days"].tolist()))
    if args.command == "smoke":
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
            str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v77_*.py", "-v"], check=True)
        fit_fold(args.v75_root, args.out_root, days[0])
        previous.write_json(args.out_root / "smoke" / "done.json", dict(status="real_fold_and_tests_passed", held_day=days[0]))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            pending = [pool.submit(fit_fold, args.v75_root, args.out_root, day) for day in days[args.start:args.end]]
            for task in pending:
                task.result()
        previous.write_json(args.out_root / "shards" / f"{args.start}-{args.end}.json", dict(status="fold_shard_complete"))


if __name__ == "__main__":
    main()
