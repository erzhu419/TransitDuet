#!/usr/bin/env python3
"""Check fit-only joint duration dependence before changing native control."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_distributional_supply_stage166 as source
from scripts.analyze_native_transit_duration_calibration_stage164 import summarize as qualify_calibration
from freq_hrl.domains.transit.native_distributional_supply_plan import ensemble_peaks, residual_draws
from freq_hrl.domains.transit.native_joint_duration import joint_residual_draws, marginal_reproduction
from freq_hrl.domains.transit.native_supply_plan import occupancy_peak
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_joint_duration_stage170_v1"
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
CALIBRATION_RUN = source.source.CALIBRATION_RUN


def contract():
    return {"calibration_run": CALIBRATION_RUN, "roots": list(ROOTS), "scenarios": list(SCENARIOS),
        "fit": "five_registered_Stage164_fit_episodes_only_joint_paths_aligned_by_native_trip_id",
        "dependence": "empirical_path_rank_reordering_with_common_template_and_tie_order",
        "unchanged": "every_trip_same_64_Stage166_residual_values_prior_and_known_native_launches",
        "coupling_seed": "170000+root_local_generator_not_native_RNG",
        "check": "all_ten_Stage164_check_episodes_no_model_selection_or_residual_fitting",
        "primary": "full_episode_known_launch_expected_peak_cost_absolute_error_joint_vs_independent",
        "boundary": "conditional_known_launch_calibration_not_prospective_fleet_value_or_policy_improvement",
        "additional_native_steps": 0, "training_updates": 0}


def trip_arrays(row):
    trips = sorted((tid, d == "True", launch, completion-launch)
        for d, r in row["directions"].items()
        for tid, launch, completion in zip(r["trip_ids"], r["launch_s"], r["completion_s"]))
    return ([t[0] for t in trips], [t[1] for t in trips],
        np.asarray([t[2] for t in trips]), np.asarray([t[3] for t in trips]))


def fit_joint_model(root, cells):
    model = source.fit_model(root)
    paths = []
    for scenario in SCENARIOS:
        row = next(r for r in cells[root, scenario]["calibration"] if r["split"] == "fit")
        ids, directions, _, durations = trip_arrays(row)
        if row["scene_seed"] != source.calibration.scene_seed(root, scenario, "fit"):
            raise ValueError("Changed joint-duration fitting scene")
        if paths and (ids != model["joint_trip_ids"] or directions != model["joint_directions"]):
            raise ValueError("Fitting paths changed native trip identities")
        model.update(joint_trip_ids=ids, joint_directions=directions)
        paths.append((durations - [model["duration_prior_s"][str(d)] for d in directions]).tolist())
    model["joint_fit_residual_paths_s"] = paths
    return model


def serial_errors(row, model):
    records = []
    for direction, trips in row["directions"].items():
        launches = np.asarray(trips["launch_s"])
        completions = np.asarray(trips["completion_s"])
        durations = completions-launches
        order = np.argsort(completions)
        position, mean, errors = 0, model["duration_prior_s"][direction], []
        for launch, duration in zip(launches, durations):
            while position < len(order) and completions[order[position]] < launch:
                observed = durations[order[position]]
                mean = observed if position == 0 else .8*mean+.2*observed
                position += 1
            errors.append(duration-mean)
        def lag1(values):
            return float(np.corrcoef(values[:-1], values[1:])[0, 1]) if np.ptp(values) > 0 else None
        records.append({"split": row["split"], "direction": direction,
            "raw_lag1": lag1(durations), "causal_launch_error_lag1": lag1(np.asarray(errors))})
    return records


def check_models(models, cells):
    panels = []
    for root in ROOTS:
        model = models[root]
        trips = list(zip(model["joint_trip_ids"], model["joint_directions"]))
        if not marginal_reproduction(model, trips):
            raise ValueError("Joint predictor changed marginal duration samples")
        mean = np.asarray([model["duration_prior_s"][str(d)] for _, d in trips])
        draws = {"independent": residual_draws(model, trips), "joint": joint_residual_draws(model, trips)}
        for scenario in SCENARIOS:
            row = next(r for r in cells[root, scenario]["calibration"] if r["split"] == "check")
            ids, directions, starts, durations = trip_arrays(row)
            if list(zip(ids, directions)) != trips or row["scene_seed"] != source.calibration.scene_seed(root, scenario, "check"):
                raise ValueError("Changed independent check trip identities/scene")
            actual_peak = occupancy_peak(starts, starts+durations)
            if actual_peak != row["peak_fleet"]:
                raise ValueError("Known-launch trip ledger differs from native peak")
            actual_cost = max(actual_peak-12, 0)**2/12
            panel = {"root": root, "scenario": scenario, "scene_seed": row["scene_seed"],
                "actual_peak": actual_peak, "actual_fleet_cost": actual_cost}
            for kind, samples in draws.items():
                matrix = mean + np.stack([samples[tid] for tid in ids], axis=1)
                peaks = ensemble_peaks(starts, matrix)
                cost = float(np.mean(np.maximum(peaks-12, 0)**2/12))
                panel[kind] = {"expected_fleet_cost": cost, "abs_cost_error": abs(cost-actual_cost),
                    "mean_peak": float(peaks.mean())}
            panels.append(panel)
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "marginal_samples_reproduced": 524, "check_episodes": 10, "additional_native_steps": 0,
        "serial_diagnostics": [{"root": root, "scenario": scenario, **record}
            for (root, scenario), cell in cells.items() for row in cell["calibration"]
            for record in serial_errors(row, models[root])],
        "panels": panels, "root_summary": [{"root": root,
            "joint_error_wins": sum(p["joint"]["abs_cost_error"] < p["independent"]["abs_cost_error"] for p in panels if p["root"] == root),
            **{f"{kind}_cost_mae": float(np.mean([p[kind]["abs_cost_error"] for p in panels if p["root"] == root]))
                for kind in ("independent", "joint")}} for root in ROOTS]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    write_json(directory/"preregistration.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    cells = {(r, s): json.loads((ROOT/"results"/CALIBRATION_RUN/"cells"/s/f"seed_{r}/result.json").read_text())
        for r in ROOTS for s in SCENARIOS}
    qualify_calibration(cells)
    models = {r: fit_joint_model(r, cells) for r in ROOTS}
    write_json(directory/"inputs/joint_models.json", {str(r): m for r, m in models.items()})
    result = check_models(models, cells)
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "serial_diagnostics"}}, indent=2))


if __name__ == "__main__":
    main()
