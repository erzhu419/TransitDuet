#!/usr/bin/env python3
"""Analyze actual supply-coupling outcomes, not just predicted fleet costs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_supply_allocation_stage163 as spec
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from freq_hrl.experiments.pointmaze_root_response import write_json


def validate_supply(row):
    supply = row["supply"]
    if row["condition"] in spec.CONTROLS:
        if supply is not None:
            raise ValueError("Source controls must not use a supply selector")
        return
    if supply["condition"] != row["condition"] or len(supply["decisions"]) != 44:
        raise ValueError("Supply selector changed its registered decision budget")
    counts = [0]*5
    times, completions = [], []
    for d in supply["decisions"]:
        candidates = d["candidates"]
        if [c["strength"] for c in candidates] != list(spec.STRENGTHS):
            raise ValueError("Changed supply allocation candidates")
        for c in candidates:
            fleet = max(c["predicted_peak"]-12, 0)**2/12
            score = c["wait_cost"] + (fleet if row["condition"] == "supply_aware" else 0)
            if not (np.all(np.isfinite(list(c.values()))) and c["predicted_peak"] >= 0
                    and np.isclose(c["fleet_cost"], fleet, rtol=0, atol=1e-10)
                    and np.isclose(c["score"], score, rtol=0, atol=1e-10)):
                raise ValueError("Invalid predicted supply score")
        selected = min(range(5), key=lambda i: candidates[i]["score"])
        if d["selected"] != selected:
            raise ValueError("Selector did not execute its registered causal score minimum")
        counts[selected] += 1
        times.append(d["time_s"])
        completions.append(d["observed_completions"])
    if not (counts == supply["choice_counts"] and np.all(np.diff(times) >= 0)
            and np.all(np.diff(completions) >= 0) and completions[-1] == supply["completed_trip_count"]
            and np.all(np.isfinite(list(supply["trip_duration_mean_s"].values())))):
        raise ValueError("Inconsistent causal supply observations or choice counts")


def contrast(rows, condition, reference, metrics):
    differences, fleets, wins = [], [], 0
    for row in rows:
        if row["condition"] != condition:
            continue
        ref = next(r for r in rows if r["condition"] == reference
                   and r["scenario"] == row["scenario"] and r["scene_seed"] == row["scene_seed"])
        differences.append({k: row[k] - ref[k] for k in metrics})
        fleets.append((max(row["peak_fleet"]-12, 0)**2 - max(ref["peak_fleet"]-12, 0)**2)/12)
        wins += row["service_cost_restricted"] < ref["service_cost_restricted"]
    delta = {k: float(np.mean([d[k] for d in differences])) for k in metrics}
    return {"condition": condition, "reference": reference, "scenes": len(differences), "cost_wins": wins,
        **delta, "fleet_cost_delta": float(np.mean(fleets)),
        "nonfleet_cost_delta": delta["service_cost_restricted"]-float(np.mean(fleets))}


def summarize(cells, sources):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required:
        raise ValueError("Incomplete native supply-allocation cells")
    metrics = spec.native.source.source_spec.authority.routing.METRICS
    panels, all_rows = [], {r: [] for r in spec.ROOTS}
    for (root, scenario), cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
                and cell["training_updates"] == 0 and cell["native_steps"] == 20*61380):
            raise ValueError("Invalid supply-allocation contract or native budget")
        scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False)
        rows = {(r["condition"], r["scene_seed"]): r for r in cell["evaluation"]}
        if len(rows) != len(cell["evaluation"]) or set(rows) != {(c, s) for c in spec.CONDITIONS for s in scenes}:
            raise ValueError("Incomplete or duplicate supply-allocation episodes")
        source = sources[root, scenario]
        if not (source["protocol"] == spec.source.EXPERIMENT_PROTOCOL and source["contract"] == spec.source.contract()
                and source["software_qualified"] and source["seed"] == root and source["scenario"] == scenario
                and source["training_updates"] == 0 and source["native_steps"] == 24*61380):
            raise ValueError("Invalid supply-allocation source")
        for (condition, scene), row in rows.items():
            registered = next(r for r in source["evaluation"]
                if r["condition"] == (condition if condition in spec.CONTROLS else "window_forecast") and r["scene_seed"] == scene)
            ref = next(r for r in source["evaluation"] if r["condition"] == "rolling_0" and r["scene_seed"] == scene)
            if not (row["scenario"] == scenario and row["source_control_reproduced"] == (condition in spec.CONTROLS)
                    and row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and np.all(np.isfinite([row[k] for k in metrics]))):
                raise ValueError("Changed supply-allocation physical controls")
            validate_schedule(row["schedule"], ref["schedule"])
            spec.source.source.source.previous.check_episode(row, registered, baseline=condition in spec.CONTROLS)
            if condition in spec.CONTROLS and row["schedule"] != registered["schedule"]:
                raise ValueError("Supply observation changed an existing native source schedule")
            validate_supply(row)
            all_rows[root].append(row)
        panels.extend({"root": root, "scenario": scenario, **contrast(list(rows.values()), c, ref, metrics)}
            for c, ref in (("demand_only", "window_forecast"), ("supply_aware", "window_forecast"), ("supply_aware", "demand_only")))
    diagnostics = []
    for root, rows in all_rows.items():
        for condition in ("demand_only", "supply_aware"):
            chosen = [r for r in rows if r["condition"] == condition]
            counts = np.sum([r["supply"]["choice_counts"] for r in chosen], axis=0)
            errors = [r["peak_fleet"] - max(d["candidates"][d["selected"]]["predicted_peak"]
                for d in r["supply"]["decisions"]) for r in chosen]
            decisions = [d for r in chosen for d in r["supply"]["decisions"]]
            diagnostics.append({"root": root, "condition": condition, "choice_counts": counts.tolist(),
                "selected_mean_strength": float(counts@np.asarray(spec.STRENGTHS)/counts.sum()),
                "decisions": len(decisions),
                "candidate_peak_distinguishes": int(sum(np.ptp([c["predicted_peak"] for c in d["candidates"]]) > 0
                    for d in decisions)),
                "candidate_fleet_cost_distinguishes": int(sum(np.ptp([c["fleet_cost"] for c in d["candidates"]]) > 0
                    for d in decisions)),
                "fleet_term_changes_wait_only_argmin": sum(d["selected"] != min(range(5),
                    key=lambda i: d["candidates"][i]["wait_cost"]) for d in decisions),
                "first_decision_candidate_peak_range": [min(c["predicted_peak"]
                    for r in chosen for c in r["supply"]["decisions"][0]["candidates"]),
                    max(c["predicted_peak"] for r in chosen for c in r["supply"]["decisions"][0]["candidates"])],
                "actual_minus_max_predicted_peak_mean": float(np.mean(errors)),
                "actual_minus_max_predicted_peak_abs_mean": float(np.mean(np.abs(errors)))})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "native_steps": sum(c["native_steps"] for c in cells.values()), "training_updates": 0,
        "source_control_reproductions": 120, "stage": spec.contract()["boundary"],
        "paired_contrasts": [{"root": root, **contrast(rows, c, ref, metrics)} for root, rows in all_rows.items()
            for c, ref in (("demand_only", "window_forecast"), ("supply_aware", "window_forecast"), ("supply_aware", "demand_only"))],
        "regime_root_contrasts": panels, "supply_diagnostics": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    result = summarize(cells, {(r, s): spec.source_cell(r, s) for r, s in cells})
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_contrasts"}}, indent=2))


if __name__ == "__main__":
    main()
