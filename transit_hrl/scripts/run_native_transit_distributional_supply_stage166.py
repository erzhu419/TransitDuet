#!/usr/bin/env python3
"""Isolate expected peak costs with the existing native mean-prior controls."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_distributional_supply_plan import (
    NativeDistributionalSupplyPlan, MEMBERS, FLEET_MODEL, validate_model)
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_calibrated_supply_stage165 as source
from scripts.analyze_native_transit_duration_calibration_stage164 import durations

native, calibration = source.native, source.calibration
EXPERIMENT_PROTOCOL = "native_transit_distributional_supply_stage166_v1"
SOURCE_RUN = "native_transit_calibrated_supply_stage165_frozen_20261011_r2"
ROOTS, SCENARIOS, STRENGTHS = source.ROOTS, source.SCENARIOS, source.STRENGTHS
CONDITIONS = ("window_forecast", "calibrated_supply", "distributional_supply")
CONTROLS = CONDITIONS[:2]


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "calibration_run": source.CALIBRATION_RUN, "roots": list(ROOTS), "scenarios": list(SCENARIOS),
        "conditions": list(CONDITIONS), "episodes_per_cell": 12, "clock_s": 61380, "fleet": 12,
        "training_updates": 0, "strengths": list(STRENGTHS), "ensemble_members": MEMBERS,
        "ensemble_seed": "166000+root_fixed_across_scenes_candidates_and_decisions_independent_of_native_RNG",
        "fit": "same_all_five_Stage164_fit_scenes_only_direction_centered_individual_trip_residuals",
        "change": "expected_native_peak_fleet_penalty_not_penalty_at_mean_duration",
        "active": "same_progress_fraction_times_sampled_total_duration_at_least_1s",
        "online": "same_prior_first_completion_replacement_EWMA0.2_fixed_fit_residuals",
        "score": "same_wait_proxy+mean_64(max(member_peak-12,0)^2/12)",
        "unchanged": "historical_peak_floor_candidates_lower_weights_physical_cost_endpoints_44_commitments",
        "qualification": "80_exact_source_physics_schedules_and_calibrated_selector_reproductions",
        "scenes": "all_four_registered_stage165_scenes_per_regime_no_selection",
        "primary": "distributional_minus_mean_supply_and_window_actual_cost_wait_fleet_nonfleet",
        "boundary": "frozen_distributional_mechanism_development_not_learned_authority_or_confirmation"}


def fit_model(root):
    rows = []
    for scenario in SCENARIOS:
        cell = json.loads((ROOT/"results"/source.CALIBRATION_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
        if not (cell["software_qualified"] and cell["protocol"] == calibration.EXPERIMENT_PROTOCOL
                and cell["contract"] == calibration.contract() and cell["seed"] == root
                and cell["scenario"] == scenario and cell["training_updates"] == 0):
            raise ValueError("Distributional supply needs qualified duration fitting cells")
        row = next(r for r in cell["calibration"] if r["split"] == "fit")
        if row["scene_seed"] != calibration.scene_seed(root, scenario, "fit"):
            raise ValueError("Changed duration fitting scene")
        rows.append(row)
    pools = {d: np.concatenate([durations(r, d) for r in rows]) for d in ("False", "True")}
    model = {"root": root, "members": MEMBERS, "ensemble_seed": 166000+root,
        "fit_scene_seeds": [r["scene_seed"] for r in rows],
        "duration_prior_s": {d: float(p.mean()) for d, p in pools.items()},
        "centered_residuals_s": {d: (p-p.mean()).tolist() for d, p in pools.items()}}
    validate_model(model)
    return model


def source_cell(root, scenario):
    cell = json.loads((ROOT/"results"/SOURCE_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
    if not (cell["software_qualified"] and cell["protocol"] == source.EXPERIMENT_PROTOCOL
            and cell["contract"] == source.contract() and cell["seed"] == root and cell["scenario"] == scenario
            and cell["training_updates"] == 0 and cell["native_steps"] == 16*61380):
        raise ValueError("Distributional supply requires qualified Stage165 controls")
    return cell


def episode(root, scenario, scene, raw, checkpoint, condition, model):
    prior = {d: model["duration_prior_s"][str(d)] for d in (False, True)}
    if condition in CONTROLS:
        return source.episode(root, scenario, scene, raw, checkpoint, condition, prior)
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = NativeDistributionalSupplyPlan(runner.env, model)
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Distributional supply changed frozen native weights")
    return row, plan.summarize(), plan.supply_summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--duration-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = native.NativeRoutingTracker
    torch.set_num_threads(1)
    model = json.loads(args.duration_model.read_text())[str(args.seed)]
    validate_model(model)
    if model["root"] != args.seed:
        raise ValueError("Changed fitted duration root")
    registered = source_cell(args.seed, args.scenario)
    if model["duration_prior_s"] != registered["duration_prior_s"]:
        raise ValueError("Distributional supply changed the Stage165 mean prior")
    _, checkpoint = native.load_source(args.seed)
    raw, rows = raw_directory(args.output), []
    scenes = native.source.source_spec.scene_seeds(native.LOWER_ROOT[args.seed], args.scenario, preflight=False)
    for scene in scenes:
        for condition in CONDITIONS:
            row, schedule, supply = episode(args.seed, args.scenario, scene,
                raw/condition/str(scene), checkpoint, condition, model)
            ref = next(r for r in registered["evaluation"] if r["condition"] == (
                condition if condition in CONTROLS else "calibrated_supply") and r["scene_seed"] == scene)
            reproduced = condition in CONTROLS
            source.check_episode(row, ref, baseline=reproduced)
            if reproduced and (schedule != ref["schedule"] or supply != ref["supply"]):
                raise RuntimeError("Distributional supply changed an exact source control")
            if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and schedule["upper_decisions"] == 44 and schedule["scheduled_trips"] == 262
                    and np.all(np.isfinite([row[k] for k in native.source.source_spec.authority.routing.METRICS]))):
                raise RuntimeError("Distributional supply changed the physical/decision budget")
            rows.append({"condition": condition, "scenario": args.scenario, "scene_seed": scene,
                **native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule,
                "supply": supply, "source_control_reproduced": reproduced})
            print(f"distributional_supply root={args.seed} {args.scenario}/{condition}/{scene} "
                f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                f"peak={row['peak_fleet']} choices={supply['choice_counts'] if supply else None}", flush=True)
            if scene == scenes[0] and condition == "calibrated_supply":
                print("DISTRIBUTIONAL_SUPPLY_FIRST_SCENE_SOURCE_QUALIFIED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "fleet_model": FLEET_MODEL, "native_steps": 12*61380, "evaluation": rows})
    print("NATIVE_DISTRIBUTIONAL_SUPPLY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
