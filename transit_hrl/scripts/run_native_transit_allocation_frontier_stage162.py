#!/usr/bin/env python3
"""Measure the frozen rolling allocation service/resource frontier."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_rolling_plan import NativeTemperedRollingPlan
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_rolling_authority_stage161 as source

native = source.native
EXPERIMENT_PROTOCOL = "native_transit_allocation_frontier_stage162_v1"
SOURCE_RUN = "native_transit_rolling_authority_stage161_frozen_20261011_r1"
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
STRENGTHS = {"rolling_0": 0., "rolling_025": .25, "rolling_05": .5, "rolling_075": .75, "rolling_1": 1.}
# Qualify both endpoints and the window control before intermediate allocations.
CONDITIONS = ("window_forecast", "rolling_0", "rolling_1", "rolling_025", "rolling_05", "rolling_075")
SOURCE_CONTROLS = {"window_forecast": "window_forecast", "rolling_0": "rolling_nominal",
                   "rolling_1": "rolling_forecast"}


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "conditions": list(CONDITIONS),
        "strengths": STRENGTHS, "episodes_per_cell": 24, "clock_s": 61380, "fleet": 12,
        "training_updates": 0, "scenes": "all_four_registered_scenes_per_regime_no_outcome_selection",
        "allocation": "(1-alpha)*current_tentative_intervals+alpha*causal_forecast_intervals_before_projection",
        "execution": "same_remaining_horizon_projection_six_trip_commitment_44_decisions_daily_endpoints",
        "bounds": "all_adjacent_intervals_240_to_480_including_cross_window",
        "qualification": "window_forecast_and_alpha0_alpha1_exact_stage161_physical_and_schedule_reproduction",
        "primary": "paired_cost_wait_fleet_nonfleet_vs_window_forecast_all_strengths_no_winner_selection",
        "selection": "none_no_strength_fitted_or_selected_as_deployed_policy",
        "boundary": "frozen_development_frontier_not_learned_adaptation_or_confirmation"}


def source_cell(root, scenario):
    path = ROOT / "results" / SOURCE_RUN / "cells" / scenario / f"seed_{root}/result.json"
    cell = json.loads(path.read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["seed"] == root and cell["scenario"] == scenario and cell["software_qualified"]
            and cell["training_updates"] == 0 and cell["native_steps"] == 20 * 61380):
        raise ValueError("Allocation frontier requires qualified completed rolling-authority cells")
    return cell


def episode(root, scenario, scene, raw, checkpoint, condition):
    if condition == "window_forecast":
        return source.episode(root, scenario, scene, raw, checkpoint, condition)
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = NativeTemperedRollingPlan(runner.env, STRENGTHS[condition])
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Frozen native weights changed during allocation frontier")
    return row, plan.summarize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = native.NativeRoutingTracker
    torch.set_num_threads(1)
    registered = source_cell(args.seed, args.scenario)
    _, checkpoint = native.load_source(args.seed)
    raw, rows = raw_directory(args.output), []
    scenes = native.source.source_spec.scene_seeds(native.LOWER_ROOT[args.seed], args.scenario, preflight=False)
    for scene in scenes:
        for condition in CONDITIONS:
            row, schedule = episode(args.seed, args.scenario, scene, raw / condition / str(scene), checkpoint, condition)
            reproduced = condition in SOURCE_CONTROLS
            ref = next(r for r in registered["evaluation"]
                if r["condition"] == SOURCE_CONTROLS.get(condition, "window_forecast") and r["scene_seed"] == scene)
            source.source.previous.check_episode(row, ref, baseline=reproduced)
            if reproduced and schedule != ref["schedule"]:
                raise RuntimeError("Allocation endpoint changed its registered native schedule")
            if not (row["simulation_end_time_s"] == 61380 and row["N_fleet"] == 12
                    and np.all(np.isfinite([row[k] for k in native.source.source_spec.authority.routing.METRICS]))
                    and schedule["upper_decisions"] == 44
                    and schedule["scheduled_trips"] == ref["schedule"]["scheduled_trips"]):
                raise RuntimeError("Allocation frontier changed native clock/fleet/decision/event budget")
            rows.append({"condition": condition, "strength": STRENGTHS.get(condition),
                "scenario": args.scenario, "scene_seed": scene,
                **native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule,
                "source_control_reproduced": reproduced})
            print(f"allocation_frontier root={args.seed} {args.scenario}/{condition}/{scene} "
                f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                f"peak={row['peak_fleet']}", flush=True)
            if scene == scenes[0] and condition == "rolling_1":
                print("ALLOCATION_FRONTIER_FIRST_SCENE_SOURCE_QUALIFIED", flush=True)
    print("ALLOCATION_FRONTIER_ALL_SOURCE_CONTROLS_REPRODUCED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "native_steps": 24 * 61380, "evaluation": rows})
    print("NATIVE_ALLOCATION_FRONTIER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
