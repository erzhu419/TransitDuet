#!/usr/bin/env python3
"""Read-only selected-action diagnosis after the fixed V75 validation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_freqduet_v74_fit_gap import load_model
import run_freqduet_v75_resource_value as value


model, old, previous = value.model, value.old, value.previous


def audit_rows(data, predictions):
    means = model.mean_differences(data["outcomes"], data["native"])
    targets = means[:, :, old.TARGET_COLUMNS]
    rows = []
    for index, prediction in enumerate(predictions):
        native = int(data["native"][index])
        selected = model.choose_action(prediction, native)
        optimistic = model.choose_action(targets[index], native)
        stats = model.error_statistics(prediction[:, 0], targets[index, :, 0], native)
        contrasts = data["outcomes"][index, :, selected, old.OUTCOMES.index("passenger_person_s")] - \
                    data["outcomes"][index, :, native, old.OUTCOMES.index("passenger_person_s")]
        rows.append(dict(context_index=index, scenario_seed=int(data["days"][index]), case_index=int(data["cases"][index]),
            target_time_s=int(data["windows"][index]), native_action_s=model.ACTIONS[native], action_s=model.ACTIONS[selected],
            delta_action_s=model.ACTIONS[selected] - model.ACTIONS[native], changed=selected != native,
            predicted={m: float(prediction[selected, j]) for j, m in enumerate(model.METRICS)},
            actual={m: float(means[index, selected, j]) for j, m in enumerate(old.OUTCOMES)},
            optimistic={m: float(targets[index, optimistic, j]) for j, m in enumerate(model.METRICS)},
            passenger_sign_correct=stats["sign_correct"], passenger_sign_pairs=stats["sign_pairs"],
            passenger_order_correct=stats["order_correct"], passenger_order_pairs=stats["order_pairs"],
            selected_half_means=[float(v.mean()) for v in np.array_split(contrasts, 2)]))
    return rows


def summarize(rows):
    changed = [r for r in rows if r["changed"]]
    result = dict(contexts=len(rows), changed=len(changed),
        passenger_improved=sum(r["actual"]["passenger_person_s"] < 0 for r in changed),
        passenger_worsened=sum(r["actual"]["passenger_person_s"] > 0 for r in changed),
        longer=sum(r["delta_action_s"] > 0 for r in rows), shorter=sum(r["delta_action_s"] < 0 for r in rows),
        predicted={m: old.interval(rows, lambda r, m=m: r["predicted"][m]) for m in model.METRICS},
        actual={m: old.interval(rows, lambda r, m=m: r["actual"][m]) for m in old.OUTCOMES},
        optimistic_mean32_choice={m: old.interval(rows, lambda r, m=m: r["optimistic"][m]) for m in model.METRICS},
        changed_constraint_violations={m: sum(r["actual"][m] > 0 for r in changed) for m in model.METRICS[1:]},
        both_halves_passenger_improve=sum(all(v < 0 for v in r["selected_half_means"]) for r in changed),
        both_halves_passenger_worsen=sum(all(v > 0 for v in r["selected_half_means"]) for r in changed),
        action_counts={str(a): sum(r["action_s"] == a for r in rows) for a in model.ACTIONS})
    for kind in ("sign", "order"):
        denominator = sum(r[f"passenger_{kind}_pairs"] for r in rows)
        result["passenger_" + kind + "_agreement"] = sum(r[f"passenger_{kind}_correct"] for r in rows) / denominator if denominator else None
    return result


def direction_summary(rows):
    return {name: dict(contexts=len(selected),
        mean_action_change_s=float(np.mean([r["delta_action_s"] for r in selected])) if selected else None,
        predicted_passenger=float(np.mean([r["predicted"]["passenger_person_s"] for r in selected])) if selected else None,
        actual_passenger=float(np.mean([r["actual"]["passenger_person_s"] for r in selected])) if selected else None,
        passenger_improved=sum(r["actual"]["passenger_person_s"] < 0 for r in selected),
        passenger_worsened=sum(r["actual"]["passenger_person_s"] > 0 for r in selected))
        for name, selected in (("longer", [r for r in rows if r["delta_action_s"] > 0]),
                               ("shorter", [r for r in rows if r["delta_action_s"] < 0]))}


def run(root, destination):
    cached = json.loads((root / "validation_context_audit.json").read_text())
    cached_summary = json.loads((root / "summary" / "status.json").read_text())
    training = dict(np.load(root / "prepared_train.npz"))
    validation = dict(np.load(root / "prepared_validation.npz"))
    channels, primary_rows = {}, {}
    for budget in value.BUDGETS:
        fit_days = value.source.DEMAND_SEEDS if budget == "old12" else (*value.source.DEMAND_SEEDS, *value.panel.TRAIN_DAYS)
        mask = np.isin(training["days"], fit_days)
        fit_data = {k: v[mask] for k, v in training.items()}
        for family in value.FAMILIES:
            channel = budget + "/" + family
            fitted = load_model(root / "models" / budget / f"{family}.npz")
            channels[channel] = {}
            for split, data in (("in_sample", fit_data), ("validation", validation)):
                predictions = fitted.predict(value.inputs(data, family), data["native"])
                if split == "validation":
                    reproduced = old.score_predictions(data, np.arange(len(data["days"])), {channel: predictions})
                    for row, expected in zip(reproduced, cached):
                        for key in ("context_index", "scenario_seed", "case_index", "target_time_s", "native_action_s"):
                            if row[key] != expected[key]:
                                raise ValueError("cached validation contexts are not paired")
                        value.source.joint.assert_same_state(row["channels"][channel], expected["channels"][channel], "V75_fixed_prediction_identity")
                rows = audit_rows(data, predictions)
                channels[channel][split] = summarize(rows)
                if channel == value.PRIMARY:
                    primary_rows[split] = rows
            print("V75_CHOICE_AUDIT_CHANNEL " + channel, flush=True)
    validation_rows = primary_rows["validation"]
    value.source.joint.assert_same_state(channels[value.PRIMARY]["validation"]["actual"],
                                       cached_summary["effects"][value.PRIMARY], "V75_effect_summary_identity")
    previous.write_json(destination / "primary_contexts.json", primary_rows)
    previous.write_json(destination / "summary" / "status.json", dict(status="fixed_v75_choices_diagnosed",
        channels=channels, primary_direction={split: direction_summary(rows) for split, rows in primary_rows.items()},
        primary_by_window={str(t): summarize([r for r in validation_rows if r["target_time_s"] == t]) for t in value.source.TARGETS},
        primary_by_actor={str(case): summarize([r for r in validation_rows if r["case_index"] == case]) for case in range(4)},
        cached_predictions_and_effects_verified=True, fits=0, rollouts=0, policy_changed=False,
        optimistic_choice_uses_evaluation_labels=True, controller_admitted=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v75-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v75_choice_audit.py", "-v"], check=True)
    run(args.v75_root, args.out_root)


if __name__ == "__main__":
    main()
