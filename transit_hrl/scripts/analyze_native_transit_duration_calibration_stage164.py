#!/usr/bin/env python3
"""Fit only training scenes; compare duration priors on independent checks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_duration_calibration_stage164 as spec
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from freq_hrl.experiments.pointmaze_root_response import write_json


def durations(row, direction):
    trips = row["directions"][direction]
    launch, completion = np.asarray(trips["launch_s"]), np.asarray(trips["completion_s"])
    if not (len(launch) > 0 and len(trips["trip_ids"]) == len(launch)
            and launch.shape == completion.shape and np.all(np.isfinite(launch))
            and np.all(np.isfinite(completion)) and np.all(launch >= 0)
            and np.all(completion > launch) and np.all(completion <= 61380)):
        raise ValueError("Invalid actual completed-trip duration observations")
    scheduled = next(s for s in row["schedule"]["schedules"] if str(s["direction"]) == direction)
    if not all(t in scheduled["actual_s"] for t in launch.tolist()):
        raise ValueError("Observed trip launch did not execute in the native schedule")
    return completion-launch


def summarize(cells):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required:
        raise ValueError("Incomplete duration calibration matrix")
    rows = {r: [] for r in spec.ROOTS}
    for (root, scenario), cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
                and cell["training_updates"] == 0 and cell["native_steps"] == 2*61380
                and len(cell["calibration"]) == 2 and [r["split"] for r in cell["calibration"]] == list(spec.SPLITS)):
            raise ValueError("Invalid duration calibration contract/budget/splits")
        for row in cell["calibration"]:
            if not (row["scene_seed"] == spec.scene_seed(root, scenario, row["split"])
                    and row["scenario"] == scenario and row["N_fleet"] == 12
                    and row["simulation_end_time_s"] == 61380 and row["schedule"]["upper_decisions"] == 44
                    and row["schedule"]["scheduled_trips"] == 262
                    and all(np.isfinite(row[k]) for k in spec.native.source.source_spec.authority.routing.METRICS)
                    and set(row["initial_prior_s"]) == {"False", "True"}
                    and all(np.isfinite(v) and v > 0 for v in row["initial_prior_s"].values())):
                raise ValueError("Changed duration calibration scene or physical controls")
            validate_schedule(row["schedule"], row["schedule"])
            ids = [tid for d in ("False", "True") for tid in row["directions"][d]["trip_ids"]]
            if not (len(ids) == len(set(ids)) == row["trips_completed"]
                    and all(isinstance(t, int) and 0 <= t < 262 for t in ids)):
                raise ValueError("Duplicate or missing actual completed trips")
            for direction in ("False", "True"):
                durations(row, direction)
            rows[root].append(row)
    priors, panels, diagnostics = {}, [], []
    for root, root_rows in rows.items():
        priors[str(root)] = {}
        for direction in ("False", "True"):
            fitted = np.concatenate([durations(r, direction) for r in root_rows if r["split"] == "fit"])
            mean = float(fitted.mean())
            priors[str(root)][direction] = mean
            checks = []
            for row in root_rows:
                if row["split"] != "check":
                    continue
                observed = durations(row, direction)
                panel = {"root": root, "scenario": row["scenario"], "direction": direction,
                    "trips": len(observed), "actual_mean_s": float(observed.mean()),
                    "actual_q05_q50_q95_s": np.quantile(observed, [.05, .5, .95]).tolist(),
                    "initial_freeflow_s": row["initial_prior_s"][direction], "fitted_prior_s": mean,
                    "freeflow_episode_mean_abs_error_s": abs(row["initial_prior_s"][direction]-float(observed.mean())),
                    "fitted_episode_mean_abs_error_s": abs(mean-float(observed.mean()))}
                panels.append(panel)
                checks.append(panel)
            diagnostics.append({"root": root, "direction": direction, "fit_trips": len(fitted),
                "fit_mean_s": mean, "fit_std_s": float(fitted.std()), "check_scenes": len(checks),
                **{k: float(np.mean([c[k] for c in checks])) for k in
                   ("freeflow_episode_mean_abs_error_s", "fitted_episode_mean_abs_error_s")}})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 20*61380, "fit_episodes": 10, "check_episodes": 10,
        "duration_prior_s": priors, "duration_diagnostics": diagnostics, "check_panels": panels,
        "performance": "not_tested_prior_accuracy_is_not_closed_loop_allocation_value"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    result = summarize(cells)
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "check_panels"}}, indent=2))


if __name__ == "__main__":
    main()
