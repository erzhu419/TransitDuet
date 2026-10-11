#!/usr/bin/env python3
"""Report every preregistered allocation strength, including fleet tradeoffs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_allocation_frontier_stage162 as spec
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from freq_hrl.experiments.pointmaze_root_response import write_json


def contrast(rows, condition, metrics):
    differences, fleets, wins = [], [], 0
    for row in rows:
        if row["condition"] != condition:
            continue
        ref = next(r for r in rows if r["condition"] == "window_forecast"
            and r["scenario"] == row["scenario"] and r["scene_seed"] == row["scene_seed"])
        differences.append({k: row[k] - ref[k] for k in metrics})
        fleets.append((max(row["peak_fleet"] - 12, 0) ** 2 - max(ref["peak_fleet"] - 12, 0) ** 2) / 12)
        wins += row["service_cost_restricted"] < ref["service_cost_restricted"]
    delta = {k: float(np.mean([d[k] for d in differences])) for k in metrics}
    return {"condition": condition, "strength": spec.STRENGTHS[condition], "scenes": len(differences),
        "cost_wins": wins, **delta, "fleet_cost_delta": float(np.mean(fleets)),
        "nonfleet_cost_delta": delta["service_cost_restricted"] - float(np.mean(fleets))}


def summarize(cells, sources):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required:
        raise ValueError("Incomplete allocation-frontier cells or sources")
    metrics = spec.native.source.source_spec.authority.routing.METRICS
    all_rows, panels = {root: [] for root in spec.ROOTS}, []
    for (root, scenario), cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["seed"] == root and cell["scenario"] == scenario and cell["software_qualified"]
                and cell["training_updates"] == 0 and cell["native_steps"] == 24 * 61380):
            raise ValueError("Unqualified frozen allocation-frontier cell")
        scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False)
        rows = {(r["condition"], r["scene_seed"]): r for r in cell["evaluation"]}
        if len(rows) != len(cell["evaluation"]) or set(rows) != {(c, s) for c in spec.CONDITIONS for s in scenes}:
            raise ValueError("Incomplete or duplicate allocation-frontier evaluation")
        source = sources[root, scenario]
        if not (source["protocol"] == spec.source.EXPERIMENT_PROTOCOL
                and source["contract"] == spec.source.contract() and source["software_qualified"]
                and source["seed"] == root and source["scenario"] == scenario
                and source["training_updates"] == 0 and source["native_steps"] == 20 * 61380):
            raise ValueError("Invalid allocation-frontier source")
        for (condition, scene), row in rows.items():
            ref = next(r for r in source["evaluation"]
                if r["condition"] == spec.SOURCE_CONTROLS.get(condition, "window_forecast") and r["scene_seed"] == scene)
            nominal = next(r for r in source["evaluation"] if r["condition"] == "window_nominal" and r["scene_seed"] == scene)
            if not (row["scenario"] == scenario and row["strength"] == spec.STRENGTHS.get(condition)
                    and row["source_control_reproduced"] == (condition in spec.SOURCE_CONTROLS)
                    and row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and np.all(np.isfinite([row[k] for k in metrics]))):
                raise ValueError("Changed allocation strength, physical outcomes or reproduction status")
            validate_schedule(row["schedule"], nominal["schedule"])
            spec.source.source.previous.check_episode(row, ref, baseline=condition in spec.SOURCE_CONTROLS)
            if condition in spec.SOURCE_CONTROLS and row["schedule"] != ref["schedule"]:
                raise ValueError("An endpoint control changed its registered native schedule")
            all_rows[root].append(row)
        panels.extend({"root": root, "scenario": scenario, **contrast(list(rows.values()), c, metrics)}
                      for c in spec.STRENGTHS)
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": sum(c["native_steps"] for c in cells.values()),
        "source_control_reproductions": 120, "stage": spec.contract()["boundary"],
        "strength_minus_window_forecast": [{"root": r, **contrast(rows, c, metrics)}
            for r, rows in all_rows.items() for c in spec.STRENGTHS], "regime_root_contrasts": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {(r, s): json.loads((directory / "cells" / s / f"seed_{r}/result.json").read_text())
             for r in spec.ROOTS for s in spec.SCENARIOS}
    result = summarize(cells, {(r, s): spec.source_cell(r, s) for r, s in cells})
    write_json(directory / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_contrasts"}}, indent=2))


if __name__ == "__main__":
    main()
