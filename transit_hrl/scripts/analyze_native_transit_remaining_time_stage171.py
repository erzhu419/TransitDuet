#!/usr/bin/env python3
"""Fit on source telemetry; evaluate remaining-time accuracy on fresh scenes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_remaining_time_stage171 as spec
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from freq_hrl.domains.transit.native_remaining_time import fit_remaining_model, corrected_remaining, remaining_features
from freq_hrl.experiments.pointmaze_root_response import write_json


def qualify(cells, sources):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required:
        raise ValueError("Incomplete remaining-time matrix")
    for (root, scenario), cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["seed"] == root and cell["scenario"] == scenario and cell["software_qualified"]
                and cell["training_updates"] == 0 and cell["native_steps"] == 3*61380
                and [r["condition"] for r in cell["evaluation"]] == list(spec.CONDITIONS)):
            raise ValueError("Changed remaining-time source contract/budget")
        fit = next(r for r in sources[root, scenario]["calibration"] if r["split"] == "fit")
        for row in cell["evaluation"]:
            ref = fit if row["condition"] == "fit_audit" else cell["evaluation"][1]
            scene = fit["scene_seed"] if row["condition"] == "fit_audit" else spec.check_scene(root, scenario)
            if not (row["scenario"] == scenario and row["scene_seed"] == scene and row["N_fleet"] == 12
                    and row["simulation_end_time_s"] == 61380
                    and row["source_control_reproduced"] == (row["condition"] != "check_control")
                    and row["schedule"]["upper_decisions"] == 44 and row["schedule"]["scheduled_trips"] == 262
                    and len(row["snapshots"]) == (0 if row["condition"] == "check_control" else 44)):
                raise ValueError("Changed remaining-time scenes or native controls")
            spec.incumbent.source.distributional.source.check_episode(row, ref, baseline=True)
            validate_schedule(row["schedule"], ref["schedule"])
            if row["schedule"] != ref["schedule"]:
                raise ValueError("Passive audit changed native control schedules")
            ledger = row["completed_trips"]
            if len(ledger) != 262 or len({r["trip_id"] for r in ledger}) != 262:
                raise ValueError("Incomplete remaining-time trip truth")
            if row["condition"] == "fit_audit":
                expected = sorted(({"trip_id": tid, "direction": d == "True", "launch_time_s": launch,
                    "completion_time_s": completion} for d, r in fit["directions"].items()
                    for tid, launch, completion in zip(r["trip_ids"], r["launch_s"], r["completion_s"])), key=lambda t: t["trip_id"])
                if ledger != expected:
                    raise ValueError("Passive fit trip truth differs from the exact source")
            elif ledger != ref["completed_trips"]:
                raise ValueError("Passive check trip truth differs from its paired control")
            for direction in (False, True):
                schedule = next(r for r in row["schedule"]["schedules"] if r["direction"] == direction)
                trips = sorted((r for r in ledger if r["direction"] == direction), key=lambda t: t["trip_id"])
                if [r["launch_time_s"] for r in trips] != schedule["actual_s"] or any(
                        not (r["launch_time_s"] < r["completion_time_s"] <= 61380) for r in trips):
                    raise ValueError("Passive trip truth differs from executed native launches")
            for snapshot in row["snapshots"]:
                ids = [s["trip_id"] for s in snapshot["active"]]
                # Callbacks precede driving, but may follow another launch in this tick.
                required = {t["trip_id"] for t in ledger if t["launch_time_s"] < snapshot["time_s"] <= t["completion_time_s"]}
                allowed = {t["trip_id"] for t in ledger if t["launch_time_s"] <= snapshot["time_s"] <= t["completion_time_s"]}
                if len(ids) != len(set(ids)) or not required <= set(ids) <= allowed:
                    raise ValueError("Passive snapshot lost an active native trip")
                for sample in snapshot["active"]:
                    trip = next(t for t in ledger if t["trip_id"] == sample["trip_id"])
                    if (sample["actual_remaining_s"] != trip["completion_time_s"]-snapshot["time_s"]
                            or not np.all(np.isfinite(remaining_features(sample)))
                            or not 0 <= sample["remaining_fraction"] <= 1):
                        raise ValueError("Changed remaining-time label or observed feature")
            spec.label_snapshots(row["snapshots"], ledger)


def errors(rows, model):
    samples = [s for r in rows for snapshot in r["snapshots"] for s in snapshot["active"]]
    if not samples:
        raise ValueError("No check observations for remaining-time accuracy")
    result = {"samples": len(samples)}
    for kind in ("progress", "corrected"):
        e = np.asarray([(s["predicted_remaining_s"] if kind == "progress" else corrected_remaining(s, model))
            - s["actual_remaining_s"] for s in samples])
        result[kind] = {"mae_s": float(np.mean(np.abs(e))), "bias_s": float(np.mean(e)),
            "q95_abs_error_s": float(np.quantile(np.abs(e), .95)), "underprediction_rate": float(np.mean(e < 0))}
    return result


def summarize(cells, sources):
    qualify(cells, sources)
    models, panels, roots = {}, [], []
    for root in spec.ROOTS:
        fits = [cells[root, s]["evaluation"][0] for s in spec.SCENARIOS]
        models[str(root)] = model = fit_remaining_model(fits)
        checks = [cells[root, s]["evaluation"][2] for s in spec.SCENARIOS]
        roots.append({"root": root, **errors(checks, model)})
        panels.extend({"root": root, "scenario": s, **errors([row], model)} for s, row in zip(spec.SCENARIOS, checks))
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 30*61380, "source_control_reproductions": 20,
        "fit_episodes": 10, "fresh_check_episodes": 10, "models": models,
        "root_summary": roots, "regime_panels": panels, "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    result = summarize(cells, {(r, s): spec.source_cell(r, s) for r, s in cells})
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "models", "regime_panels"}}, indent=2))


if __name__ == "__main__":
    main()
