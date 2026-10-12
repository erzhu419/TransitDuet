#!/usr/bin/env python3
"""Aggregate nested-calibrated folds without syncing models or predictions."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_calibrated_value_stage190 as spec
from scripts.submit_native_transit_coverage_value_stage188_scheduleurm import result_path as source_path
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json


def state_key(row):
    return row["root"], row["scene_seed"], row["probe_index"]


def qualify(cell, mode, fold, registered):
    if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["mode"] == mode and cell["fold"] == fold
            and len(cell["fits"]) == spec.calibrated.MEMBERS
            and cell["critic_updates"] == spec.calibrated.MEMBERS*spec.STEPS
            and cell["actor_updates"] == cell["new_native_steps"] == 0):
        raise ValueError("Changed nested fold or critic budget")
    for fit, groups in zip(cell["fits"], fold["partition"]["member_groups"], strict=True):
        if not (fit["training_groups"] == groups and fit["training_states"] == 6*len(groups)
                and fit["steps"] == spec.STEPS and fit["representation"] == mode
                and fit["objective"] == "proper_mse" and fit["initialization_seed"] == spec.source.floor.occupancy.SEED
                and fit["parameter_count"] == spec.source.contract()["parameters"][mode]):
            raise ValueError("Critic used different groups, units or optimization")
        if mode == "timetable_sequence" and not all(fit[k] > 0 for k in
                ("encoder_parameter_update_l2", "sequence_projection_update_l2", "encoder_gradient_l2_max")):
            raise ValueError("Sequence member received no learning gradient")
    cal = cell["calibration"]
    expected_cal = fold["partition"]["calibration_groups"]
    scores = np.asarray([s["score"] for s in cal["group_scores"]])
    if not (cal["groups"] == expected_cal and cal["states"] == 6*len(expected_cal)
            and [s["group"] for s in cal["group_scores"]] == expected_cal
            and cal["quantile"] == spec.calibrated.QUANTILE
            and cal["rank"] == min(len(expected_cal), int(np.ceil((len(expected_cal)+1)*cal["quantile"])))
            and np.array_equal(np.sort(scores, axis=0)[cal["rank"]-1], cal["multipliers"])
            and np.all(np.isfinite(scores)) and np.all(scores >= 0)):
        raise ValueError("Margins were not fitted from the registered training-only calibration scenes")
    key = {"scene": lambda g: [g["root"], g["scene_seed"]], "regime": lambda g: g["scenario"],
        "lower_root": lambda g: g["root"]}[fold["split"]]
    expected = {(g["root"], g["scene_seed"], p) for g in registered["groups"]
        if key(g) == fold["held_out"] for p in spec.source.collection.PROBES}
    if set(cell["predictions"]) != set(spec.calibrated.POLICIES):
        raise ValueError("Missing shared-fit selector comparison")
    for policy, rows in cell["predictions"].items():
        if len(rows) != len(expected) or {state_key(r) for r in rows} != expected:
            raise ValueError("Missing or repeated outer held-out states")
        for row in rows:
            multiplier = cal["multipliers"] if policy == "calibrated" else [0., 0.]
            mean, uncertainty, original = np.asarray(row["mean"]), np.asarray(row["uncertainty"]), row["original_choice"]
            if not (np.all(mean[original] == 0) and np.all(uncertainty[original] == 0)
                    and np.all(np.isfinite(mean)) and np.all(np.isfinite(uncertainty)) and np.all(uncertainty >= 0)
                    and np.array_equal(mean+uncertainty*multiplier, row["upper"])
                    and row["selected"] == spec.calibrated.choose(row, original, multiplier)
                    and np.isclose(row["actual_cost_delta"], row["fleet_cost_delta"]+row["nonfleet_cost_delta"], rtol=0, atol=1e-12)):
                raise ValueError("Changed calibrated selector, physical units or identity fallback")
    predictions = [{state_key(r): r for r in cell["predictions"][p]} for p in spec.calibrated.POLICIES]
    if any(predictions[0][k][field] != predictions[1][k][field] for k in expected for field in ("mean", "uncertainty")):
        raise ValueError("Selectors do not share the ensemble")


def summarize(cells, registered, references):
    required = {(m, f["split"], f["index"]) for m in spec.MODES for f in registered["folds"]}
    if set(cells) != required:
        raise ValueError("Missing or mixed calibrated folds")
    for (mode, split, index), cell in cells.items():
        fold = next(f for f in registered["folds"] if (f["split"], f["index"]) == (split, index))
        qualify(cell, mode, fold, registered)
    old = {(g["root"], g["scene_seed"], p) for g in registered["groups"] if g["origin"] == "existing"
        for p in spec.source.collection.PROBES}
    variants = {}
    for mode in spec.MODES:
        for policy in (*spec.calibrated.POLICIES, "stage188_reference"):
            validation = {}
            for split in spec.SPLITS:
                rows = references[mode, split] if policy == "stage188_reference" else [r for f in registered["folds"]
                    if f["split"] == split for r in cells[mode, split, f["index"]]["predictions"][policy]]
                if len(rows) != 600 or len({state_key(r) for r in rows}) != 600:
                    raise ValueError("Aggregate does not cover the same 600 test states")
                validation[split] = {"overall": spec.calibrated.summary(rows),
                    "by_root": {str(root): spec.calibrated.summary([r for r in rows if r["root"] == root])
                        for root in spec.source.collection.ROOTS},
                    "old_state_subset": spec.calibrated.summary([r for r in rows if state_key(r) in old])}
            gates = {"cost": all(v["mean_actual_cost_delta"] < 0 for s in validation.values() for v in s["by_root"].values()),
                "wait": all(v["mean_wait_delta_min"] <= 0 for s in validation.values() for v in s["by_root"].values())}
            variants[f"{mode}:{policy}"] = {"validation": validation, "gates": gates}
    primary = variants[spec.contract()["primary"]]
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "outer_fit_tasks": len(cells), "fits": len(cells)*spec.calibrated.MEMBERS,
        "critic_updates": sum(c["critic_updates"] for c in cells.values()), "actor_updates": 0, "new_native_steps": 0,
        "variants": variants, "runtime": {mode: sum(c["fit_wall_s"] for (m, _, _), c in cells.items() if m == mode)
            for mode in spec.MODES},
        "third_scene_screen_eligible": all(primary["gates"].values()), "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    bank = json.loads(spec.source.bank_path(spec.SOURCE_RUN).read_text())
    if registered != spec.manifest(bank["manifest"]):
        raise ValueError("Aggregate changed preregistered nested splits")
    cells = {(m, f["split"], f["index"]): json.loads((raw_directory(spec.result_path(args.run_name, m, f))/"result.json").read_text())
        for m in spec.MODES for f in registered["folds"]}
    rows = {state_key(r): r for r in bank["training_data"]}
    references = {}
    for mode in spec.MODES:
        for split in spec.SPLITS:
            reference = []
            for fold in registered["folds"]:
                if fold["split"] != split: continue
                cell = json.loads(source_path(ROOT/"results"/spec.SOURCE_RUN, mode, fold).read_text())
                for p in cell["predictions"]["wait_feasible"]:
                    prediction = {"mean": np.c_[np.asarray(p["predicted_heads"]).sum(axis=1), p["predicted_wait_delta_min"]].tolist(),
                        "uncertainty": np.zeros((6, 2)).tolist()}
                    evaluated = spec.calibrated.evaluate(rows[state_key(p)], prediction, [0., 0.])
                    if evaluated["selected"] != p["selected"]:
                        raise ValueError("Cached reference differs from its original selector")
                    reference.append(evaluated)
            references[mode, split] = reference
    result = summarize(cells, registered, references)
    write_json(args.output, result)
    print(json.dumps({"fits": result["fits"], "gates": {k: v["gates"] for k, v in result["variants"].items()}}, indent=2))
    print("NATIVE_CALIBRATED_VALUE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
