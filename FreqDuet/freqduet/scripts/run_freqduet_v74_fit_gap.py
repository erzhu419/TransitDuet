#!/usr/bin/env python3
"""Read-only fitting/generalization diagnosis of cached V72/V73 ridge models."""

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

import run_freqduet_v73_fleet_state as previous


old, model, source = previous.old, previous.model, previous.source
CHANNELS = (*previous.CONTROLS, *previous.FAMILIES)
LOSSES = ("mae", "mse")
BLOCKS = ("states", "fleet_states", "plan_states")


def load_model(path):
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["actions"], model.ACTIONS)
        np.testing.assert_array_equal(saved["metrics"], model.METRICS)
        return model.IncrementModel(str(saved["kind"].item()), saved["center"], saved["scale"], saved["coefficients"], 0.)


def support(data, training, queries, center, scale):
    normalized, offset = {}, 0
    for block in BLOCKS:
        values = data[block]
        width = values.shape[1]
        normalized[block] = (values - center[offset:offset + width]) / scale[offset:offset + width]
        offset += width
    result = []
    for index in queries:
        matched = training[(data["cases"][training] == data["cases"][index]) &
                           (data["windows"][training] == data["windows"][index]) &
                           (data["days"][training] != data["days"][index])]
        if not len(matched):
            raise ValueError("matched actor/window support requires other training days")
        row = {}
        for block in BLOCKS:
            z = normalized[block]
            reference, query = z[matched], z[index]
            low, high = reference.min(axis=0), reference.max(axis=0)
            variable = high > low
            excess = np.maximum(np.maximum(low - query, query - high), 0.)
            distance = np.square(reference[:, variable] - query[variable]).sum(axis=1) / max(1, variable.sum())
            row[block] = dict(matched_days=len(matched), variable_features=int(variable.sum()),
                outside_fraction=float(((excess > 0) & variable).sum() / max(1, variable.sum())),
                novel_constant_features=int(((excess > 0) & ~variable).sum()),
                nearest_rms=float(np.sqrt(distance.min())))
        result.append(row)
    return result


def audit_fold(root, v72_root, v73_root, day):
    path = root / "folds" / f"{day}.json"
    if path.exists():
        return day
    data = old.read_data(v73_root)
    targets = model.mean_differences(data["outcomes"], data["native"])[:, :, old.TARGET_COLUMNS]
    training, testing = np.flatnonzero(data["days"] != day), np.flatnonzero(data["days"] == day)
    caches = [json.loads((r / "folds" / str(day) / "done.json").read_text()) for r in (v72_root, v73_root)]
    cached = {}
    for cache in caches:
        if cache["fitting_days"] != sorted(set(source.DEMAND_SEEDS) - {day}):
            raise ValueError("saved model does not exclude the complete declared demand day")
        for row in cache["rows"]:
            cached.setdefault(row["context_index"], {}).update(row["channels"])
    fits = {channel: load_model((v72_root if channel in model.KINDS else v73_root) /
                "folds" / str(day) / f"{channel}.npz") for channel in CHANNELS if channel != "zero"}
    rows = {}
    for split, indices in (("train", training), ("held", testing)):
        rows[split] = [dict(context_index=int(i), scenario_seed=int(data["days"][i]), case_index=int(data["cases"][i]),
                            target_time_s=int(data["windows"][i]), channels={}) for i in indices]
        for channel in CHANNELS:
            states = previous.inputs(data, channel) if channel in previous.FAMILIES else data["states"]
            predicted = np.zeros_like(targets[indices]) if channel == "zero" else fits[channel].predict(states[indices], data["native"][indices])
            for local, index in enumerate(indices):
                stats = {m: model.error_statistics(predicted[local, :, j], targets[index, :, j], int(data["native"][index]))
                         for j, m in enumerate(old.PRIMARY)}
                if split == "held":
                    expected = cached[int(index)][channel]
                    for metric in old.PRIMARY:
                        source.joint.assert_same_state(stats[metric], expected["errors"][metric], "cached_held_errors")
                    if model.ACTIONS[model.choose_action(predicted[local], int(data["native"][index]))] != expected["action_s"]:
                        raise AssertionError("read-only loaded model changed the original held-day choice")
                rows[split][local]["channels"][channel] = {m: {loss: stats[m][loss] for loss in LOSSES} for m in old.PRIMARY}
        fitted = fits[previous.PRIMARY]
        for row, coverage in zip(rows[split], support(data, training, indices, fitted.center, fitted.scale)):
            row["support"] = coverage
    previous.write_json(path, dict(held_day=day, **rows, cached_held_errors_and_choices_verified=True))
    print("V74_FOLD_DONE " + json.dumps(dict(held_day=day, held_contexts=len(testing), fitting_contexts=len(training))), flush=True)
    return day


def pair_rows(folds):
    held, training = {}, {}
    for fold in folds:
        for row in fold["held"]:
            index = row["context_index"]
            if index in held or row["scenario_seed"] != fold["held_day"]:
                raise ValueError("held-day diagnostic contains repeated or in-sample contexts")
            held[index] = row
        for row in fold["train"]:
            if row["scenario_seed"] == fold["held_day"]:
                raise ValueError("excluded day entered the fitting-error panel")
            training.setdefault(row["context_index"], []).append(row)
    rows = []
    for index in sorted(held):
        fit_rows = training[index]
        if len(fit_rows) != len(folds) - 1:
            raise ValueError("each prefix needs exactly eleven in-sample cached predictions")
        rows.append(dict(context_index=index, scenario_seed=held[index]["scenario_seed"],
            case_index=held[index]["case_index"], target_time_s=held[index]["target_time_s"],
            held=held[index]["channels"], train={c: {m: {loss: float(np.mean([r["channels"][c][m][loss] for r in fit_rows]))
                for loss in LOSSES} for m in old.PRIMARY} for c in CHANNELS},
            support_held=held[index]["support"], support_train={b: {k: float(np.mean([r["support"][b][k] for r in fit_rows]))
                for k in held[index]["support"][b]} for b in BLOCKS}))
    return rows


def describe(rows):
    return dict(errors={c: {m: {f"{split}_{loss}": float(np.mean([r[split][c][m][loss] for r in rows]))
        for split in ("train", "held") for loss in LOSSES} for m in old.PRIMARY} for c in CHANNELS},
        support={b: {f"{split}_{k}": float(np.mean([r["support_" + split][b][k] for r in rows]))
        for split in ("train", "held") for k in ("matched_days", "variable_features", "outside_fraction", "novel_constant_features", "nearest_rms")}
        for b in BLOCKS})


def aggregate(root):
    folds = [json.loads((root / "folds" / f"{day}.json").read_text()) for day in source.DEMAND_SEEDS]
    rows = pair_rows(folds)
    if [r["context_index"] for r in rows] != list(range(len(source.CONTEXTS))):
        raise ValueError("frozen-fit diagnosis lacks the balanced 144-prefix panel")
    contrasts = {split: {m: {loss: old.interval(rows, lambda r, m=m, loss=loss, split=split:
        r[split][previous.PRIMARY][m][loss] - r[split]["state_quadratic"][m][loss]) for loss in LOSSES}
        for m in old.PRIMARY} for split in ("train", "held")}
    gap = {c: {m: {loss: old.interval(rows, lambda r, c=c, m=m, loss=loss:
        r["held"][c][m][loss] - r["train"][c][m][loss]) for loss in LOSSES} for m in old.PRIMARY} for c in CHANNELS}
    previous.write_json(root / "context_audit.json", rows)
    previous.write_json(root / "summary" / "status.json", dict(status="frozen_fit_generalization_diagnosis_complete",
        contexts=len(rows), demand_day_clusters=len(folds), in_sample_fits_per_context=len(folds) - 1,
        **describe(rows), primary_minus_original=contrasts, held_minus_in_sample=gap,
        by_day=[dict(scenario_seed=day, **describe([r for r in rows if r["scenario_seed"] == day])) for day in source.DEMAND_SEEDS],
        by_actor=[dict(case_index=case, train_seed=seed, actor_scenario=scenario,
            **describe([r for r in rows if r["case_index"] == case])) for case, (seed, scenario) in enumerate(source.labels.deployment.CASES)],
        by_window=[dict(target_time_s=t, **describe([r for r in rows if r["target_time_s"] == t])) for t in source.TARGETS],
        cached_held_errors_and_choices_verified=all(f["cached_held_errors_and_choices_verified"] for f in folds),
        model_refits=0, rollouts=0, actor_updates=0, dual_updates=0, production_value_updates=0,
        main_policy_changed=False, gates_changed=False, development_diagnosis_only=True,
        diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")))
    print("V74_AGGREGATE " + json.dumps(dict(contexts=len(rows), primary_minus_original=contrasts)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v72-root", type=Path, required=True)
    parser.add_argument("--v73-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v74_*.py", "-v"], check=True)
    audit_fold(args.out_root, args.v72_root, args.v73_root, source.DEMAND_SEEDS[0])
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(audit_fold, itertools.repeat(args.out_root), itertools.repeat(args.v72_root),
                      itertools.repeat(args.v73_root), source.DEMAND_SEEDS))
    aggregate(args.out_root)


if __name__ == "__main__":
    main()
