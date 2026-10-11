#!/usr/bin/env python3
"""Test cross-window planning authority before another learned-upper trainer."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_rolling_plan import NativeRollingServicePlan, schedule_summary
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_mc_upper_stage160 as source

native = source.source
EXPERIMENT_PROTOCOL = "native_transit_rolling_authority_stage161_v1"
SOURCE_RUN = "native_transit_mc_upper_stage160_development_20261011_r1"
ROOTS = source.ROOTS
SCENARIOS = tuple(native.contract()["scenarios"])
CONDITIONS = ("window_forecast", "window_nominal", "rolling_nominal", "rolling_forecast", "rolling_reversed")
METHOD = "rolling_authority"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "conditions": list(CONDITIONS),
        "scenes": "all_four_registered_scenes_per_regime_no_outcome_selection", "episodes_per_cell": 20,
        "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "commitment": "six_trips_once_same_44_upper_decisions_and_daily_trip_count",
        "rolling": "reallocate_uncommitted_remaining_direction_horizon_current_causal_forecast_only",
        "constraints": "daily_first_last_fixed_all_adjacent_intervals_240_to_480_including_cross_window",
        "tentative": "future_uncommitted_launch_times_revisable_committed_times_and_goals_never_rewritten",
        "qualification": "all_window_forecast_nominal_and_rolling_nominal_exact_source_reproduction",
        "boundary": "frozen_planning_authority_not_learned_policy_or_confirmation"}


def source_cell(root):
    p = ROOT / "results" / SOURCE_RUN / "cells" / source.METHOD / f"seed_{root}/result.json"
    cell = json.loads(p.read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["seed"] == root and cell["software_qualified"] and cell["training"]["updates"] == 768):
        raise ValueError("Rolling authority requires completed MC upper source")
    return cell


def episode(root, scenario, scene, raw, checkpoint, condition):
    if condition in {"window_forecast", "window_nominal"}:
        row, plan, _, _ = native.episode(root, scenario, scene, raw, checkpoint,
            lambda s: np.zeros(2, dtype=np.float32), nominal=condition == "window_nominal")
        return row, schedule_summary(plan)
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    mode = {"rolling_nominal": "nominal_plan", "rolling_forecast": "causal_forecast",
            "rolling_reversed": "forecast_reversed"}[condition]
    plan = NativeRollingServicePlan(runner.env, mode)
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Frozen native weights changed during rolling-authority evaluation")
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
    registered = source_cell(args.seed)
    _, checkpoint = native.load_source(args.seed)
    raw, rows = raw_directory(args.output), []
    scenes = native.source.source_spec.scene_seeds(native.LOWER_ROOT[args.seed], args.scenario, preflight=False)
    for scene in scenes:
        for condition in CONDITIONS:
            row, schedule = episode(args.seed, args.scenario, scene, raw / condition / str(scene), checkpoint, condition)
            ref_condition = "nominal" if condition in {"window_nominal", "rolling_nominal"} else "forecast"
            ref = next(r for r in registered["evaluation"] if r["condition"] == ref_condition
                       and r["scenario"] == args.scenario and r["scene_seed"] == scene)
            reproduced = condition in {"window_forecast", "window_nominal", "rolling_nominal"}
            source.previous.check_episode(row, ref, baseline=reproduced)
            if not (row["simulation_end_time_s"] == 61380 and row["N_fleet"] == 12
                    and np.all(np.isfinite([row[k] for k in native.source.source_spec.authority.routing.METRICS]))
                    and schedule["upper_decisions"] == 44
                    and schedule["scheduled_trips"] == ref["execution"]["queries"]):
                raise RuntimeError("Rolling authority changed native clock/fleet/decision/event budget")
            rows.append({"condition": condition, "scenario": args.scenario, "scene_seed": scene,
                **native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule,
                "source_control_reproduced": reproduced})
            print(f"rolling_authority root={args.seed} {args.scenario}/{condition}/{scene} "
                  f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                  f"window_shift={schedule['window_endpoint_shift_abs_mean_s']}", flush=True)
            if scene == scenes[0] and condition == "rolling_nominal":
                print("ROLLING_AUTHORITY_FIRST_SCENE_SOURCE_QUALIFIED", flush=True)
    print("ROLLING_AUTHORITY_ALL_SOURCE_CONTROLS_REPRODUCED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "native_steps": 20 * 61380, "evaluation": rows})
    print("NATIVE_ROLLING_AUTHORITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
