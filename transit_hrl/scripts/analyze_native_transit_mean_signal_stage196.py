#!/usr/bin/env python3
"""Separate conditional-draw noise from the frozen value predictors' errors."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_mean_coverage_stage195 as source
from scripts.analyze_native_transit_continuation_noise_stage191 import choice
from scripts.analyze_native_transit_expected_value_stage192 import qualify
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_mean_signal_stage196_v1"
SOURCE_RUN = "native_transit_mean_coverage_stage195_development_20261012_r1"
FIELDS = ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "same240_states_40_prefix_groups_eight_draws_and47_completed_grouped_fits",
        "predictions": "frozen_Stage195_single_and_mean_target_models_both_selectors_all_three_splits",
        "pair_error": "mean_squared_prediction_error_minus_sample_pair_variance_over8_no_clipping",
        "zero_reference": "zero_action_difference_pair_MSE_minus_same_noise_estimate",
        "halves": "odd_even_four_draw_oracle_choice_cross_evaluated_on_other_half_wait_feasible_identity_ties",
        "feasibility": "selected_mean_wait_harm_split_into_both_halves_harm_and_half_disagreement",
        "budget": {"new_fits": 0, "critic_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "draw_labels_and_frozen_predictions_server_only_small_diagnostic_summary",
        "boundary": "development_noise_and_model_error_diagnostic_not_CI_or_new_policy_performance"}


def oracle_panel(values, original):
    halves = (values[::2].mean(axis=0), values[1::2].mean(axis=0))
    selected = [choice(h[:, [0, 3]], original) for h in halves]
    full = choice(values.mean(axis=0)[:, [0, 3]], original)
    return {"half_choice_agreement": int(selected[0] == selected[1]),
        "same_half_cost": float(np.mean([h[s, 0] for h, s in zip(halves, selected)])),
        "cross_half_cost": float(np.mean([halves[1][selected[0], 0], halves[0][selected[1], 0]])),
        "cross_half_wait_min": float(np.mean([halves[1][selected[0], 3], halves[0][selected[1], 3]])),
        "mean8_hindsight_cost": float(values.mean(axis=0)[full, 0]),
        "mean8_hindsight_wait_min": float(values.mean(axis=0)[full, 3])}


def prediction_panel(values, prediction):
    heads = np.asarray(prediction["predicted_heads"])
    predicted = np.c_[heads.sum(axis=1), heads, prediction["predicted_wait_delta_min"]]
    a, b = np.triu_indices(6, 1)
    gaps = values[:, a]-values[:, b]
    means, noise = gaps.mean(axis=0), gaps.var(axis=0, ddof=1)/len(values)
    errors = (predicted[a]-predicted[b]-means)**2
    selected, original = prediction["selected"], prediction["original_choice"]
    wait = float(values[:, selected, 3].mean())
    half_wait = (float(values[::2, selected, 3].mean()), float(values[1::2, selected, 3].mean()))
    false = selected != original and predicted[selected, 3] <= 0 and wait > 0
    return {"pair_metrics": {f: {"observed_pair_mse": float(errors[:, i].mean()),
                "mean_label_noise": float(noise[:, i].mean()),
                "noise_corrected_pair_mse": float((errors[:, i]-noise[:, i]).mean()),
                "zero_difference_pair_mse": float((means[:, i]**2-noise[:, i]).mean()),
                "error_change_vs_zero": float((errors[:, i]-means[:, i]**2).mean())}
            for i, f in enumerate(FIELDS)},
        "changed_choice": int(selected != original), "false_feasible": int(false),
        "false_feasible_both_halves_harm": int(false and all(w > 0 for w in half_wait)),
        "selected_wait_half_disagreement": int((half_wait[0] > 0) != (half_wait[1] > 0)),
        "selected_mean_cost": float(values[:, selected, 0].mean()), "selected_mean_wait_min": wait}


def summary(rows):
    counts = ("changed_choice", "false_feasible", "false_feasible_both_halves_harm",
        "selected_wait_half_disagreement", "half_choice_agreement")
    result = {"states": len(rows),
        "prefix_groups": len({(r["root"], r["scene_seed"]) for r in rows})}
    for key in rows[0]:
        if key in ("root", "scenario", "scene_seed", "probe_index", "pair_metrics"): continue
        result[key if key in counts else "mean_"+key] = (sum(r[key] for r in rows) if key in counts
            else float(np.mean([r[key] for r in rows])))
    if "pair_metrics" in rows[0]:
        result["pair_metrics"] = {f: {k: float(np.mean([r["pair_metrics"][f][k] for r in rows]))
            for k in rows[0]["pair_metrics"][f]} for f in FIELDS}
    return result


def scopes(rows):
    return {"overall": summary(rows),
        "by_root": {str(r): summary([x for x in rows if x["root"] == r]) for r in source.ROOTS},
        "by_regime": {s: summary([x for x in rows if x["scenario"] == s]) for s in source.SCENARIOS}}


def load_panels():
    result = {}
    for root, scenario, replica, probe in source.states():
        key = root, scenario, replica, probe
        paths = [source.noise.result_path(ROOT/"results"/source.NOISE_RUN, root, scenario, probe, d)
            if source.reused(key) else source.draw_path(SOURCE_RUN, *key, d) for d in source.DRAWS]
        cells = [json.loads((raw_directory(p)/"result.json").read_text()) for p in paths]
        first = cells[0]
        values = np.asarray([[[l[f] for f in FIELDS] for l in c["labels"]] for c in cells])
        if values.shape != (8, 6, 4) or not np.all(np.isfinite(values)):
            raise ValueError("Conditional diagnostic lost the registered draw/action/head axes")
        result[source.state_key(first)] = {**{k: first[k] for k in
            ("root", "scenario", "scene_seed", "probe_index", "original_choice")}, "values": values}
    return result


def analyze(panels, cells, registered):
    expected_keys = {source.state_key(r) for r in registered["mean_states"]}
    if set(panels) != expected_keys or set(cells) != {(f["split"], f["index"]) for f in registered["folds"]}:
        raise ValueError("Diagnostic requires the same full mean-target cohort and all47 frozen fits")
    for fold in registered["folds"]:
        qualify(cells[fold["split"], fold["index"]], "tail_summary", fold, registered, protocol_spec=source)
    oracle = [{**{k: p[k] for k in ("root", "scenario", "scene_seed", "probe_index")},
        **oracle_panel(p["values"], p["original_choice"])} for p in panels.values()]
    variants = {}
    for variant in source.VARIANTS:
        for policy in source.source.value.POLICIES:
            variants[f"{variant}:{policy}"] = {}
            for split in source.SPLITS:
                rows = []
                predictions = [p for fold in registered["folds"] if fold["split"] == split
                    for p in cells[split, fold["index"]]["predictions"][variant]["mean8"][policy]]
                for prediction in predictions:
                    panel = panels[source.state_key(prediction)]
                    if prediction["original_choice"] != panel["original_choice"]:
                        raise ValueError("Frozen predictor changed the original control action")
                    rows.append({**{k: panel[k] for k in ("root", "scenario", "scene_seed", "probe_index")},
                        **prediction_panel(panel["values"], prediction)})
                variants[f"{variant}:{policy}"][split] = scopes(rows)
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "states": len(panels), "prefix_groups": 40, "reused_fits": 47, "oracle": scopes(oracle),
        "variants": variants, "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/SOURCE_RUN
    prior = json.loads((directory/"report/result.json").read_text())
    if not (prior["protocol"] == source.EXPERIMENT_PROTOCOL and prior["contract"] == source.contract()
            and prior["software_qualified"] and prior["new_fits"] == 47):
        raise ValueError("Finish qualified Stage195 before diagnosing frozen predictions")
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    cells = {(f["split"], f["index"]): json.loads((raw_directory(source.result_path(
        SOURCE_RUN, "tail_summary", f))/"result.json").read_text()) for f in registered["folds"]}
    result = analyze(load_panels(), cells, registered)
    write_json(args.output, result)
    print(json.dumps(result["oracle"]["overall"], indent=2), flush=True)
    print("NATIVE_MEAN_SIGNAL_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
