#!/usr/bin/env python3
"""Qualify ensemble scores and matched native physical performance."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_distributional_supply_stage166 as spec
from scripts.analyze_native_transit_calibrated_supply_stage165 import validate_prior
from scripts.analyze_native_transit_supply_allocation_stage163 import contrast
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from freq_hrl.domains.transit.native_distributional_supply_plan import ensemble_peaks, residual_draws, validate_model
from freq_hrl.domains.transit.native_supply_plan import occupancy_peak
from freq_hrl.experiments.pointmaze_root_response import write_json


def conditional_calibration(models):
    panels = []
    for root, model in models.items():
        for scenario in spec.SCENARIOS:
            cell = json.loads((ROOT/"results"/spec.source.CALIBRATION_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
            row = next(r for r in cell["calibration"] if r["split"] == "check")
            starts, ends, means, ids, directions = [], [], [], [], []
            for direction in ("False", "True"):
                trips = row["directions"][direction]
                starts.extend(trips["launch_s"])
                ends.extend(trips["completion_s"])
                ids.extend(trips["trip_ids"])
                directions.extend([direction == "True"]*len(trips["trip_ids"]))
                means.extend([model["duration_prior_s"][direction]]*len(trips["trip_ids"]))
            actual = occupancy_peak(starts, ends)
            if actual != row["peak_fleet"]:
                raise ValueError("Duration ledger does not reproduce native concurrent peak")
            draws = residual_draws(model, list(zip(ids, directions)))
            peaks = ensemble_peaks(starts, np.maximum(np.asarray(means)+np.stack([draws[t] for t in ids], axis=1), 1.))
            point = occupancy_peak(starts, np.asarray(starts)+means)
            actual_cost = max(actual-12, 0)**2/12
            mean_cost = max(point-12, 0)**2/12
            expected = float(np.mean(np.maximum(peaks-12, 0)**2/12))
            panels.append({"root": root, "scenario": scenario, "scene_seed": row["scene_seed"],
                "actual_peak": actual, "mean_duration_peak": point, "ensemble_mean_peak": float(peaks.mean()),
                "actual_fleet_cost": actual_cost, "mean_duration_fleet_cost": mean_cost,
                "ensemble_fleet_cost": expected, "mean_abs_cost_error": abs(actual_cost-mean_cost),
                "ensemble_abs_cost_error": abs(actual_cost-expected)})
    return {"panels": panels, "boundary": "held_out_duration_check_conditioned_on_realized_launch_schedule_not_prospective_control",
        "mean_abs_cost_error": float(np.mean([p["mean_abs_cost_error"] for p in panels])),
        "ensemble_abs_cost_error": float(np.mean([p["ensemble_abs_cost_error"] for p in panels]))}


def validate_distributional(supply, model):
    validate_prior(supply, model["duration_prior_s"])
    if not (supply["condition"] == "supply_aware" and supply["fleet_model"] == spec.FLEET_MODEL
            and supply["ensemble_members"] == spec.MEMBERS and supply["ensemble_seed"] == model["ensemble_seed"]
            and supply["fit_scene_seeds"] == model["fit_scene_seeds"] and len(supply["decisions"]) == 44):
        raise ValueError("Changed distributional supply model or decision budget")
    choices, times, completions = [0]*5, [], []
    for d in supply["decisions"]:
        candidates = d["candidates"]
        if [c["strength"] for c in candidates] != list(spec.STRENGTHS):
            raise ValueError("Changed distributional supply candidates")
        for c in candidates:
            histogram = c["ensemble_peak_counts"]
            if not (sum(histogram.values()) == spec.MEMBERS
                    and all(int(p) >= d["historical_peak"] and isinstance(n, int) and n > 0 for p, n in histogram.items())):
                raise ValueError("Invalid member peak histogram or removed historical floor")
            fleet = sum(n*max(int(p)-12, 0)**2/12 for p, n in histogram.items())/spec.MEMBERS
            if not (np.all(np.isfinite([c[k] for k in ("wait_cost", "headway_cost", "predicted_peak", "fleet_cost", "score")]))
                    and np.isclose(c["fleet_cost"], fleet, rtol=0, atol=1e-10)
                    and np.isclose(c["score"], c["wait_cost"]+fleet, rtol=0, atol=1e-10)):
                raise ValueError("Changed expected native fleet cost or candidate score")
        if d["selected"] != min(range(5), key=lambda i: candidates[i]["score"]):
            raise ValueError("Distributional selector did not execute minimum expected score")
        choices[d["selected"]] += 1
        times.append(d["time_s"])
        completions.append(d["observed_completions"])
    if not (choices == supply["choice_counts"] and np.all(np.diff(times) >= 0)
            and np.all(np.diff(completions) >= 0) and completions[-1] == supply["completed_trip_count"]):
        raise ValueError("Inconsistent distributional choices or causal observations")


def summarize(cells, sources, models):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required or set(models) != set(spec.ROOTS):
        raise ValueError("Incomplete distributional supply matrix")
    metrics = spec.native.source.source_spec.authority.routing.METRICS
    all_rows, panels, diagnostics = {r: [] for r in spec.ROOTS}, [], []
    for root, model in models.items():
        validate_model(model)
        if model["root"] != root or model["fit_scene_seeds"] != [spec.calibration.scene_seed(root, s, "fit") for s in spec.SCENARIOS]:
            raise ValueError("Changed training-only duration model roots/scenes")
    for (root, scenario), cell in cells.items():
        source = sources[root, scenario]
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
                and cell["training_updates"] == 0 and cell["native_steps"] == 12*61380
                and cell["fleet_model"] == spec.FLEET_MODEL and source["duration_prior_s"] == models[root]["duration_prior_s"]
                and source["protocol"] == spec.source.EXPERIMENT_PROTOCOL and source["contract"] == spec.source.contract()
                and source["software_qualified"] and source["seed"] == root and source["scenario"] == scenario
                and source["native_steps"] == 16*61380 and source["training_updates"] == 0):
            raise ValueError("Changed distributional or source contract/model/budget")
        scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False)
        rows = {(r["condition"], r["scene_seed"]): r for r in cell["evaluation"]}
        if len(rows) != len(cell["evaluation"]) or set(rows) != {(c, s) for c in spec.CONDITIONS for s in scenes}:
            raise ValueError("Incomplete or duplicate distributional episodes")
        for (condition, scene), row in rows.items():
            reproduced = condition in spec.CONTROLS
            ref = next(r for r in source["evaluation"] if r["scene_seed"] == scene and r["condition"] == (
                condition if reproduced else "calibrated_supply"))
            if not (row["scenario"] == scenario and row["source_control_reproduced"] == reproduced
                    and row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and np.all(np.isfinite([row[k] for k in metrics]))):
                raise ValueError("Changed distributional physical controls")
            spec.source.check_episode(row, ref, baseline=reproduced)
            validate_schedule(row["schedule"], ref["schedule"])
            if reproduced and (row["schedule"] != ref["schedule"] or row["supply"] != ref["supply"]):
                raise ValueError("Changed distributional source-control schedule or selector")
            if not reproduced:
                validate_distributional(row["supply"], models[root])
            all_rows[root].append(row)
        panels.extend({"root": root, "scenario": scenario, **contrast(list(rows.values()), "distributional_supply", ref, metrics)}
            for ref in spec.CONTROLS)
    for root, rows in all_rows.items():
        selected = [r for r in rows if r["condition"] == "distributional_supply"]
        decisions = [d for r in selected for d in r["supply"]["decisions"]]
        diagnostics.append({"root": root, "decisions": len(decisions),
            "choice_counts": np.sum([r["supply"]["choice_counts"] for r in selected], axis=0).tolist(),
            "candidate_fleet_cost_distinguishes": int(sum(np.ptp([c["fleet_cost"] for c in d["candidates"]]) > 0 for d in decisions)),
            "fleet_term_changes_wait_only_argmin": sum(d["selected"] != min(range(5), key=lambda i: d["candidates"][i]["wait_cost"]) for d in decisions),
            "all_member_peaks_at_historical_floor": sum(all(set(c["ensemble_peak_counts"]) == {str(d["historical_peak"])}
                for c in d["candidates"]) for d in decisions)})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 120*61380, "source_control_reproductions": 80,
        "paired_contrasts": [{"root": root, **contrast(rows, "distributional_supply", ref, metrics)}
            for root, rows in all_rows.items() for ref in spec.CONTROLS],
        "regime_root_contrasts": panels, "supply_diagnostics": diagnostics, "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    models = {int(r): m for r, m in json.loads((directory/"inputs/duration_models.json").read_text()).items()}
    if any(m != spec.fit_model(r) for r, m in models.items()):
        raise ValueError("Run input differs from the registered training-only fit")
    result = summarize(cells, {(r, s): spec.source_cell(r, s) for r, s in cells}, models)
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_contrasts"}}, indent=2))


if __name__ == "__main__":
    main()
