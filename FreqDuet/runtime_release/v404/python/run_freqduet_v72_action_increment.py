#!/usr/bin/env python3
"""Whole-day-excluded development probe of full-day action-increment scores."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import freqduet_action_increment as model
import run_freqduet_v70_independent_labels as source


CHANNELS = ("zero", *model.KINDS)
PRIMARY = model.METRICS[:3]
OUTCOMES = source.labels.OUTCOMES
TARGET_COLUMNS = [OUTCOMES.index(m) for m in model.METRICS]


def prepare(root, v70_root):
    states, outcomes, natives, days, cases, windows = [], [], [], [], [], []
    for reference in range(len(source.REFERENCES)):
        bundle = torch.load(source.reference_path(v70_root, reference), map_location="cpu", weights_only=False)
        for window, target in enumerate(source.TARGETS):
            context = source.CONTEXTS.index((reference, window))
            spec = source.context_spec(context)
            capture = bundle["captures"][target]
            native = model.ACTIONS.index(capture["actor_action_s"])
            groups = [json.loads(source.group_path(v70_root, context * len(source.FUTURES) + f).read_text()) for f in source.FUTURES]
            for future, group in enumerate(groups):
                if (group["scenario_seed"], group["context_index"], group["future_index"], group["native_action_s"]) != (
                        spec["scenario_seed"], context, future, model.ACTIONS[native]):
                    raise ValueError("cached labels do not match their original V70 context")
                if [r["action_s"] for r in group["rows"]] != list(model.ACTIONS):
                    raise ValueError("cached context lacks the registered seven-action grid")
            states.append(model.observable_features(capture))
            outcomes.append([[[r[m] for m in OUTCOMES] for r in g["rows"]] for g in groups])
            natives.append(native)
            days.append(spec["scenario_seed"])
            cases.append(spec["case_index"])
            windows.append(target)
    payload = dict(states=np.asarray(states), outcomes=np.asarray(outcomes), native=np.asarray(natives),
                   days=np.asarray(days), cases=np.asarray(cases), windows=np.asarray(windows))
    if not np.isfinite(payload["states"]).all() or not np.isfinite(payload["outcomes"]).all():
        raise ValueError("non-finite causal inputs or action labels")
    np.savez_compressed(root / "prepared.npz", **payload)
    return payload


def read_data(root):
    with np.load(root / "prepared.npz") as payload:
        return dict(payload)


def save_model(path, fitted):
    np.savez_compressed(path, format="freqduet-v72-full-day-action-increment", kind=fitted.kind,
                        center=fitted.center, scale=fitted.scale, coefficients=fitted.coefficients,
                        metrics=np.asarray(model.METRICS), actions=np.asarray(model.ACTIONS), ridge=model.RIDGE)


def score_predictions(data, testing, predictions):
    truth = model.mean_differences(data["outcomes"], data["native"])
    targets = truth[:, :, TARGET_COLUMNS]
    rows = []
    for local, index in enumerate(testing):
        native = int(data["native"][index])
        row = dict(context_index=int(index), scenario_seed=int(data["days"][index]), case_index=int(data["cases"][index]),
                   target_time_s=int(data["windows"][index]), native_action_s=model.ACTIONS[native], channels={})
        for channel, values in predictions.items():
            prediction = values[local]
            np.testing.assert_array_equal(prediction[native], np.zeros(len(model.METRICS)))
            selected = model.choose_action(prediction, native)
            row["channels"][channel] = dict(action_s=model.ACTIONS[selected], changed=selected != native,
                effects={m: float(truth[index, selected, i]) for i, m in enumerate(OUTCOMES)},
                errors={m: model.error_statistics(prediction[:, i], targets[index, :, i], native) for i, m in enumerate(PRIMARY)})
        rows.append(row)
    return rows


def fit_fold(root, held_day):
    destination = root / "folds" / str(held_day)
    if (destination / "done.json").exists():
        return int(held_day)
    data = read_data(root)
    truth = model.mean_differences(data["outcomes"], data["native"])
    targets = truth[:, :, TARGET_COLUMNS]
    testing = np.flatnonzero(data["days"] == held_day)
    destination.mkdir(parents=True, exist_ok=True)
    fitted = {kind: model.fit_excluding_day(data["states"], data["native"], targets, data["days"], held_day, kind)
              for kind in model.KINDS}
    predictions = {kind: fitted[kind].predict(data["states"][testing], data["native"][testing]) for kind in model.KINDS}
    predictions["zero"] = np.zeros_like(predictions[model.KINDS[0]])
    rows = score_predictions(data, testing, {channel: predictions[channel] for channel in CHANNELS})
    for kind, fitted_model in fitted.items():
        save_model(destination / f"{kind}.npz", fitted_model)
    result = dict(held_day=int(held_day), fitting_days=sorted(set(data["days"].tolist()) - {int(held_day)}), rows=rows,
                  parameters={k: int(m.coefficients.size) for k, m in fitted.items()},
                  effective_parameters_per_output={k: m.effective_parameters_per_output for k, m in fitted.items()},
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    (destination / "done.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V72_FOLD_DONE " + json.dumps(dict(held_day=int(held_day), contexts=len(rows))), flush=True)
    return int(held_day)


def interval(rows, getter):
    values = [float(np.mean([getter(r) for r in rows if r["scenario_seed"] == day]))
              for day in sorted({r["scenario_seed"] for r in rows})]
    result = source.labels.expanded.paired_interval(values, 72000)
    return {k: result[k] for k in ("n_demand_days", "mean", "ci95")}


def gates(contrasts, effects):
    return dict(prediction_gate_passed=all(contrasts[control][metric][loss]["ci95"][1] < 0
                    for control in ("zero", "action_only") for metric in PRIMARY for loss in ("mae", "mse")),
                joint_effect_gate_passed=effects["passenger_person_s"]["ci95"][1] < 0
                    and all(effects[m]["ci95"][1] <= 0 for m in model.METRICS[1:]))


def aggregate(root):
    folds = [json.loads((root / "folds" / str(day) / "done.json").read_text()) for day in source.DEMAND_SEEDS]
    rows = [r for fold in folds for r in fold["rows"]]
    if sorted(r["context_index"] for r in rows) != list(range(len(source.CONTEXTS))):
        raise ValueError("whole-day-excluded predictions do not cover the full balanced V70 panel")
    for fold in folds:
        if fold["fitting_days"] != sorted(set(source.DEMAND_SEEDS) - {fold["held_day"]}):
            raise ValueError("held demand day entered the fitting set")
    errors, effects, counts = {}, {}, {}
    for channel in CHANNELS:
        errors[channel] = {}
        for metric in PRIMARY:
            stats = [r["channels"][channel]["errors"][metric] for r in rows]
            errors[channel][metric] = {loss: interval(rows, lambda r, loss=loss: r["channels"][channel]["errors"][metric][loss])
                                       for loss in ("mae", "mse")}
            errors[channel][metric]["rmse"] = float(np.sqrt(errors[channel][metric]["mse"]["mean"]))
            for prefix in ("sign", "order"):
                denominator = sum(s[prefix + "_pairs"] for s in stats)
                errors[channel][metric][prefix + "_agreement"] = sum(s[prefix + "_correct"] for s in stats) / denominator if denominator else None
        effects[channel] = {m: interval(rows, lambda r, m=m: r["channels"][channel]["effects"][m]) for m in OUTCOMES}
        counts[channel] = dict(changed=sum(r["channels"][channel]["changed"] for r in rows),
            passenger_improved=sum(r["channels"][channel]["effects"]["passenger_person_s"] < 0 for r in rows),
            passenger_worsened=sum(r["channels"][channel]["effects"]["passenger_person_s"] > 0 for r in rows),
            joint_nonworse=sum(r["channels"][channel]["effects"]["passenger_person_s"] < 0 and
                              all(r["channels"][channel]["effects"][m] <= 0 for m in model.METRICS[1:]) for r in rows))
    contrasts = {control: {m: {loss: interval(rows, lambda r, m=m, loss=loss, control=control:
        r["channels"]["state_quadratic"]["errors"][m][loss] - r["channels"][control]["errors"][m][loss])
        for loss in ("mae", "mse")} for m in PRIMARY} for control in ("zero", "action_only")}
    daily = [dict(scenario_seed=day, **{channel: {m: float(np.mean([r["channels"][channel]["effects"][m]
              for r in rows if r["scenario_seed"] == day])) for m in model.METRICS} for channel in CHANNELS})
              for day in source.DEMAND_SEEDS]
    passenger = np.asarray([d["state_quadratic"]["passenger_person_s"] for d in daily])
    improvement = np.maximum(-passenger, 0)
    known_day = source.DEMAND_SEEDS.index(700006)
    result = dict(status="full_day_action_increment_day_exclusion_complete", contexts=len(rows), demand_day_clusters=len(folds),
        futures_per_context=len(source.FUTURES), prediction_target="mean of 32 paired full-day native-action differences",
        primary_candidate="state_quadratic", ridge=model.RIDGE, errors=errors, prediction_contrasts=contrasts,
        effects=effects, choices=counts, by_day=daily, **gates(contrasts, effects["state_quadratic"]),
        effects_without_evaluation_day700006={m: interval([r for r in rows if r["scenario_seed"] != 700006],
                                     lambda r, m=m: r["channels"]["state_quadratic"]["effects"][m]) for m in model.METRICS},
        day700006_gross_improvement_share=float(improvement[known_day] / improvement.sum()) if improvement.sum() else None,
        day700006_net_improvement_share=float(passenger[known_day] / passenger.sum()) if passenger.sum() else None,
        parameters=folds[0]["parameters"], effective_parameters_per_output=[f["effective_parameters_per_output"] for f in folds],
        rollouts_added=0, actor_updates=0, dual_updates=0, production_value_updates=0, main_policy_changed=False,
        development_day_crossfit_only=True, single_intervention_labels=True, controller_admitted=False)
    data = read_data(root)
    targets = model.mean_differences(data["outcomes"], data["native"])[:, :, TARGET_COLUMNS]
    for kind in model.KINDS:
        fitted = model.fit_model(data["states"], data["native"], targets, data["days"], kind)
        save_model(root / f"development_{kind}.npz", fitted)
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V72_AGGREGATE " + json.dumps(dict(contexts=len(rows), **gates(contrasts, effects["state_quadratic"]))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "folds", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v70-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
        return
    if args.command == "smoke":
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v72_*.py", "-v"], check=True)
        args.out_root.mkdir(parents=True, exist_ok=True)
        prepare(args.out_root, args.v70_root)
        fit_fold(args.out_root, source.DEMAND_SEEDS[0])
        export = args.out_root / "smoke_export"
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            list(pool.map(fit_fold, itertools.repeat(args.out_root), source.DEMAND_SEEDS))
        export = args.out_root / "fold_exports"
    export.mkdir(exist_ok=True)
    (export / "done.json").write_text(json.dumps(dict(status=args.command + "_complete")) + "\n")


if __name__ == "__main__":
    main()
