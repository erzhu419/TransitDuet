#!/usr/bin/env python3
"""Cross-fit absolute fleet supervision and a learned wait-feasible selector."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_balance_stage180 as source
from freq_hrl.domains.transit import native_floor_value as value
from freq_hrl.experiments.pointmaze_root_response import write_json

paired, occupancy = source.paired, source.value
EXPERIMENT_PROTOCOL = "native_transit_floor_value_stage181_v1"
PRIMARY = "floor_absolute:wait_feasible"


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL, "primary": PRIMARY,
        "methods": list(value.METHODS), "policies": list(value.POLICIES),
        "data": "unchanged_120_states_720_labels_20_first_second_scene_groups_plus_registered_baseline_terminal_peak_labels",
        "representation": "unchanged_causal_96_features_96_16_3_tanh_1603_parameters",
        "supervision": "future_fleet_cost_equals_original_terminal_fleet_plus_delta_minus_irreversible_causal_floor",
        "floor": "raw_absolute_control_or_softplus_future_cost_plus_observed_peak_cost",
        "loss": "unit_absolute_fleet_MSE_plus_three_head_paired_MSE_plus_native_total_cost_expected_regret",
        "scales": "training_only_future_fleet_RMS_and_paired_nonfleet_wait_RMS_floor_1e-6",
        "decision": "restore_physical_units_cost_is_fleet_plus_nonfleet_wait_is_a_separate_nonpositive_delta_constraint",
        "steps": occupancy.STEPS, "initialization_seed": occupancy.SEED,
        "learning_rate": occupancy.LEARNING_RATE, "weight_decay": occupancy.WEIGHT_DECAY,
        "validation": ["20_leave_scene_out", "5_leave_regime_out", "2_leave_lower_root_out"],
        "grouping": "all_six_callbacks_and_six_candidates_per_scene_together_no_outcome_selection",
        "budget": {"fits_per_method": 28, "critic_updates_per_method": 28*occupancy.STEPS,
            "new_native_episodes": 0, "actor_updates": 0, "selectors_share_each_fitted_model": True},
        "screen": "fixed_primary_negative_mean_cost_and_nonpositive_wait_for_both_roots_in_all_three_splits",
        "next": "separately_register_third_scene_frozen_transfer_before_repeated_learned_actions",
        "boundary": "development_cost_wait_value_screen_not_independent_confirmation_or_native_learned_HRL"}


def source_baselines():
    return [{k: row[k] for k in ("scene_seed", "peak_fleet")} | {"root": root, "scenario": scenario}
        for root in source.source.ROOTS for scenario in source.source.SCENARIOS
        for row in (source.source.labels.source_record(root, scenario, scene_index=i) for i in (0, 1))]


def training_rows(first, labels, second, baselines):
    rows = source.training_rows(first, labels, second)
    peaks = {(b["root"], b["scene_seed"]): b for b in baselines}
    if len(baselines) != len(peaks) or set(peaks) != {paired.scene_group(r) for r in rows}:
        raise ValueError("Absolute fleet supervision requires exactly the twenty source scenes")
    augmented = []
    for row in rows:
        baseline = peaks[paired.scene_group(row)]
        peak = baseline["peak_fleet"]
        if (baseline["scenario"] != row["scenario"] or int(peak) != peak
                or peak < row["occupancy_context"]["historical_peak"]):
            raise ValueError("Terminal fleet label is below its causal peak or from another scene")
        original_cost = max(peak-12, 0)**2/12.
        new_labels = [{**l, "future_fleet_cost": original_cost+l["fleet_cost_delta"]-value.floor_cost(row)}
            for l in row["labels"]]
        if any(l["future_fleet_cost"] < -1e-12 or not np.isfinite(l["future_fleet_cost"]) for l in new_labels):
            raise ValueError("Counterfactual terminal fleet label violates its irreversible causal floor")
        augmented.append({**row, "labels": new_labels})
    return augmented


def evaluate(rows, model, policy):
    result = paired.evaluate_action_value(rows, model, predictor=value.predict_value,
        selector=lambda row, model: value.choose_action(row, model, policy))
    for row, observed in zip(rows, result):
        observed.update(value.predict(row, model))
    return result


def summary(rows):
    return {**paired.summarize_choices(rows),
        "mean_predicted_selected_fleet_delta": float(np.mean([r["predicted_heads"][r["selected"]][0] for r in rows])),
        "mean_predicted_selected_wait_delta_min": float(np.mean([r["predicted_wait_delta_min"][r["selected"]] for r in rows])),
        "absolute_fleet_floor_violations": sum(c < r["causal_fleet_floor"] for r in rows for c in r["absolute_fleet_cost"]),
        "selected_positive_predicted_wait": sum(r["predicted_wait_delta_min"][r["selected"]] > 0 for r in rows)}


def fit_and_validate(first, labels, second, baselines, method, *, fitter=None):
    return fit_rows_and_validate(training_rows(first, labels, second, baselines), method, fitter=fitter)


def fit_rows_and_validate(rows, method, *, fitter=None):
    ledger = []
    def fit(train):
        model = (value.fit_value if fitter is None else fitter)(train, method)
        ledger.append({k: model[k] for k in ("training_groups", "training_states", "steps", "parameter_count",
            "target_scale", "regret_cost_scale", "parameter_update_l2", "loss_ledger")})
        if model.get("representation") == "timetable_sequence":
            ledger[-1].update({k: model[k] for k in ("encoder_parameter_update_l2", "sequence_projection_update_l2", "encoder_gradient_l2_max")})
        print(f"FLOOR_VALUE_FIT method={method} fit={len(ledger)} states={len(train)} updates={model['steps']}", flush=True)
        return model
    validation = {p: {} for p in value.POLICIES}
    for split, key in (("scene", paired.scene_group), ("regime", lambda r: r["scenario"]),
            ("lower_root", lambda r: r["root"])):
        folds, predictions = ({p: [] for p in value.POLICIES} for _ in range(2))
        for group in sorted({key(r) for r in rows}):
            train, test = [r for r in rows if key(r) != group], [r for r in rows if key(r) == group]
            model = fit(train)
            for policy in value.POLICIES:
                evaluated = evaluate(test, model, policy)
                folds[policy].append({"held_out": group, "training_groups": model["training_groups"],
                    "training_states": len(train), **summary(evaluated)})
                predictions[policy].extend(evaluated)
        for policy in value.POLICIES:
            prediction = predictions[policy]
            validation[policy][split] = {"split": split, "folds": folds[policy], "predictions": prediction,
                "overall": summary(prediction), "by_root": {str(root): summary([r for r in prediction if r["root"] == root])
                    for root in source.source.ROOTS}}
    model = fit(rows)
    gates = {p: {"cost": all(v["mean_actual_cost_delta"] < 0 for s in validation[p].values() for v in s["by_root"].values()),
        "wait": all(v["mean_wait_delta_min"] <= 0 for s in validation[p].values() for v in s["by_root"].values())}
        for p in value.POLICIES}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "method": method,
        "model": model, "fit_ledger": ledger, "validation": validation, "gates": gates,
        "critic_updates": sum(f["steps"] for f in ledger), "actor_updates": 0, "new_native_steps": 0,
        "apparent_training": {p: summary(evaluate(rows, model, p)) for p in value.POLICIES},
        "third_scene_screen_eligible": method == "floor_absolute" and all(gates["wait_feasible"].values()),
        "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=value.METHODS, required=True)
    for name in ("first", "labels", "second", "baselines", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = fit_and_validate(*(json.loads(getattr(args, n).read_text())
        for n in ("first", "labels", "second", "baselines")), args.method)
    write_json(args.output, result)
    print(json.dumps({"method": args.method, "third_scene_screen_eligible": result["third_scene_screen_eligible"],
        "validation": {p: {s: v["overall"] for s, v in splits.items()} for p, splits in result["validation"].items()}}, indent=2), flush=True)
    print("NATIVE_FLOOR_VALUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
