#!/usr/bin/env python3
"""Separate rolling authority, fleet jumps and physical service outcomes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_rolling_authority_stage161 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def validate_schedule(schedule, reference):
    if not (schedule["queries"] == schedule["scheduled_trips"] == reference["queries"]
            and schedule["upper_decisions"] == 44 and schedule["committed_unchanged"]
            and schedule["daily_endpoint_error_max_s"] == 0 and len(schedule["schedules"]) == 2):
        raise ValueError("Changed rolling decision/event budget or committed departures")
    count, end_shifts, shifts = 0, [], []
    if {s["direction"] for s in schedule["schedules"]} != {False, True}:
        raise ValueError("Incomplete rolling direction schedules")
    for s in schedule["schedules"]:
        ref = next(r for r in reference["schedules"] if r["direction"] == s["direction"])
        nominal, planned, targets = (np.asarray(s[k]) for k in ("nominal_s", "planned_s", "target_headways_s"))
        if not (np.array_equal(nominal, ref["nominal_s"]) and planned.shape == nominal.shape == targets.shape
                and len(planned) == len(s["actual_s"]) and np.all(np.isfinite(planned))
                and np.array_equal(planned[[0, -1]], nominal[[0, -1]])
                and np.all(np.diff(planned) >= 240) and np.all(np.diff(planned) <= 480)
                and np.array_equal(targets[1:], np.diff(planned))):
            raise ValueError("Changed daily endpoints, cross-window interval bounds or native lower goals")
        if any(t is not None and (not np.isfinite(t) or t < p) for t, p in zip(s["actual_s"], planned)):
            raise ValueError("Actual native launch precedes its committed schedule")
        end_indices = list(range(5, len(planned), 6))
        if not end_indices or end_indices[-1] != len(planned) - 1:
            end_indices.append(len(planned) - 1)
        end_shifts.extend(np.abs(planned[end_indices] - nominal[end_indices]))
        shifts.extend(np.abs(planned - nominal))
        count += len(planned)
    if not (count == schedule["scheduled_trips"]
            and np.isclose(np.mean(end_shifts), schedule["window_endpoint_shift_abs_mean_s"], rtol=0, atol=1e-8)
            and np.isclose(np.mean(shifts), schedule["planned_shift_abs_mean_s"], rtol=0, atol=1e-8)):
        raise ValueError("Inconsistent rolling schedule or authority diagnostics")


def summarize(cells, sources):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != set(spec.ROOTS):
        raise ValueError("Incomplete rolling-authority cells")
    metrics = spec.native.source.source_spec.authority.routing.METRICS
    panels, contrasts = [], []
    all_rows = {root: [] for root in spec.ROOTS}
    for (root, scenario), cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["seed"] == root and cell["scenario"] == scenario and cell["software_qualified"]
                and cell["training_updates"] == 0 and cell["native_steps"] == 20 * 61380):
            raise ValueError("Unqualified frozen rolling-authority cell")
        scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False)
        rows = {(r["condition"], r["scene_seed"]): r for r in cell["evaluation"]}
        if set(rows) != {(c, scene) for c in spec.CONDITIONS for scene in scenes} or len(rows) != len(cell["evaluation"]):
            raise ValueError("Incomplete or duplicate rolling-authority evaluation")
        for (condition, scene), row in rows.items():
            ref = rows["window_nominal", scene]
            if row["scenario"] != scenario or not np.all(np.isfinite([row[k] for k in metrics])):
                raise ValueError("Nonfinite or mismatched rolling-authority outcomes")
            validate_schedule(row["schedule"], ref["schedule"])
            if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380):
                raise ValueError("Changed rolling native fleet or clock")
            source_condition = "nominal" if condition in {"window_nominal", "rolling_nominal"} else "forecast"
            registered = next(r for r in sources[root]["evaluation"] if r["condition"] == source_condition
                              and r["scenario"] == scenario and r["scene_seed"] == scene)
            reproduced = condition in {"window_forecast", "window_nominal", "rolling_nominal"}
            if reproduced and not row["source_control_reproduced"]:
                raise ValueError("Missing native rolling control reproduction")
            spec.source.previous.check_episode(row, registered, baseline=reproduced)
            if reproduced and row["schedule"]["window_endpoint_shift_abs_mean_s"] != 0:
                raise ValueError("Locked-window or nominal control moved a window endpoint")
            all_rows[root].append(row)
        for condition in spec.CONDITIONS:
            panel = [r for (c, _), r in rows.items() if c == condition]
            panels.append({"root": root, "scenario": scenario, "condition": condition,
                **{k: float(np.mean([r[k] for r in panel])) for k in metrics},
                "window_endpoint_shift_abs_mean_s": float(np.mean([
                    r["schedule"]["window_endpoint_shift_abs_mean_s"] for r in panel]))})
    for root, rows in all_rows.items():
        for condition in ("rolling_forecast", "rolling_reversed"):
            differences, fleet, wins = [], [], 0
            for row in rows:
                if row["condition"] != condition:
                    continue
                ref = next(r for r in rows if r["condition"] == "window_forecast"
                           and r["scenario"] == row["scenario"] and r["scene_seed"] == row["scene_seed"])
                differences.append({k: row[k] - ref[k] for k in metrics})
                fleet.append((max(row["peak_fleet"] - 12, 0) ** 2 - max(ref["peak_fleet"] - 12, 0) ** 2) / 12)
                wins += row["service_cost_restricted"] < ref["service_cost_restricted"]
            delta = {k: float(np.mean([d[k] for d in differences])) for k in metrics}
            contrasts.append({"root": root, "condition": condition, "scenes": len(differences), "cost_wins": wins,
                **delta, "fleet_cost_delta": float(np.mean(fleet)),
                "nonfleet_cost_delta": delta["service_cost_restricted"] - float(np.mean(fleet))})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": sum(c["native_steps"] for c in cells.values()),
        "stage": spec.contract()["boundary"], "source_control_reproductions": 120,
        "rolling_minus_window_forecast": contrasts, "regime_root_means": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {(r, s): json.loads((directory / "cells" / s / f"seed_{r}/result.json").read_text())
             for r in spec.ROOTS for s in spec.SCENARIOS}
    result = summarize(cells, {r: spec.source_cell(r) for r in spec.ROOTS})
    write_json(directory / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_means"}}, indent=2))


if __name__ == "__main__":
    main()
