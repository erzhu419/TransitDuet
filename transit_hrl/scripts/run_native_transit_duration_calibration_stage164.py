#!/usr/bin/env python3
"""Measure independent fitting/check trip times before changing supply priors."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_supply_allocation_stage163 as source
from freq_hrl.domains.transit.native_residual_plan import NativeResidualPlan
from freq_hrl.domains.transit.native_supply_plan import NativeSupplyRollingPlan
from freq_hrl.domains.transit.native_rolling_plan import schedule_summary
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

native = source.native
EXPERIMENT_PROTOCOL = "native_transit_duration_calibration_stage164_v1"
SOURCE_RUN = "native_transit_supply_allocation_stage163_frozen_20261011_r1"
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
SPLITS = ("fit", "check")


def scene_seed(root, scenario, split):
    return 810000000 + root*1000 + SCENARIOS.index(scenario)*10 + SPLITS.index(split)


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "splits": list(SPLITS),
        "scene_seeds": "810000000+root*1000+regime_index*10+split_index_disjoint_from_previous_evaluation",
        "episodes_per_cell": 2, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "policy": "unchanged_window_forecast_zero_residual_frozen_native_lower",
        "observations": "actual_completed_trip_launch_and_completion_times_passive_native_ledger",
        "initial_prior": "same_current_route_freeflow_plus_4s_per_segment_at_first_callback",
        "fit": "one_trip_weighted_direction_mean_per_root_all_five_fit_regimes_only",
        "check": "five_independent_check_scenes_per_root_never_included_in_fit",
        "primary": "held_out_episode_mean_duration_error_freeflow_vs_fitted_prior_all_regimes",
        "boundary": "duration_calibration_not_allocation_performance_or_learned_upper_authority"}


class DurationCalibrationPlan(NativeResidualPlan):
    def __init__(self, env):
        super().__init__(env, lambda s: np.zeros(2, dtype=np.float32))
        self.initial_prior = None

    def __call__(self, state, trip):
        if self.initial_prior is None:
            predictor = NativeSupplyRollingPlan(self.env, "demand_only")
            self.initial_prior = {str(d): predictor.trip_duration(d) for d in (False, True)}
        return super().__call__(state, trip)


def episode(root, scenario, split, raw, checkpoint):
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = DurationCalibrationPlan(runner.env)
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12,
        scenario_seed=scene_seed(root, scenario, split), record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Duration calibration changed frozen native weights")
    observations = runner.env.completed_trip_observations
    if not (row["simulation_end_time_s"] == 61380 and row["N_fleet"] == 12
            and len(observations) == row["trips_completed"]):
        raise RuntimeError("Duration calibration changed native clock/fleet or lost completed trips")
    directions = {}
    for direction in (False, True):
        trips = sorted((int(tid), t) for tid, t in observations.items() if bool(t["direction"]) == direction)
        directions[str(direction)] = {"trip_ids": [tid for tid, t in trips],
            "launch_s": [t["launch_time_s"] for tid, t in trips],
            "completion_s": [t["completion_time_s"] for tid, t in trips]}
    return {"split": split, "scenario": scenario, "scene_seed": scene_seed(root, scenario, split),
        **native.source.source_spec.authority.routing.compact_row(row),
        "trips_completed": row["trips_completed"], "initial_prior_s": plan.initial_prior,
        "schedule": schedule_summary(plan), "directions": directions}


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
    source.source_cell(args.seed, args.scenario)
    _, checkpoint = native.load_source(args.seed)
    rows, raw = [], raw_directory(args.output)
    for split in SPLITS:
        row = episode(args.seed, args.scenario, split, raw/split, checkpoint)
        rows.append(row)
        means = {d: float(np.mean(np.asarray(t["completion_s"])-t["launch_s"])) for d, t in row["directions"].items()}
        print(f"duration_calibration root={args.seed} {args.scenario}/{split} "
            f"prior={row['initial_prior_s']} actual_mean={means} trips={row['trips_completed']}", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "seed": args.seed, "scenario": args.scenario, "software_qualified": True,
        "training_updates": 0, "native_steps": 2*61380, "calibration": rows})
    print("NATIVE_DURATION_CALIBRATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
