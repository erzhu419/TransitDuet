#!/usr/bin/env python3
"""Isolate causal supply cost in the frozen rolling allocation selector."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_supply_plan import NativeSupplyRollingPlan, STRENGTHS
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_allocation_frontier_stage162 as source

native = source.native
EXPERIMENT_PROTOCOL = "native_transit_supply_allocation_stage163_v1"
SOURCE_RUN = "native_transit_allocation_frontier_stage162_frozen_20261011_r1"
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
CONDITIONS = ("window_forecast", "rolling_0", "rolling_1", "demand_only", "supply_aware")
CONTROLS = CONDITIONS[:3]


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "conditions": list(CONDITIONS),
        "strengths": list(STRENGTHS), "episodes_per_cell": 20, "clock_s": 61380, "fleet": 12,
        "training_updates": 0, "scenes": "all_four_registered_scenes_per_regime_no_outcome_selection",
        "change": "same_candidate_selector_with_or_without_predicted_peak_fleet_cost",
        "candidates": "original_nominal_interval_preferences_to_causal_forecast_projected_on_current_feasible_endpoints",
        "duration": "per_direction_EWMA_alpha0.2_completed_actual_trip_times_current_route_freeflow_initial_prior",
        "active": "current_progress_times_observed_mean_duration_no_future_realized_arrivals",
        "occupancy": "both_directions_all_known_future_launches_committed_and_tentative_plus_active_buses",
        "score": "forecast_weighted_interval_wait/600+optional(max(predicted_peak-12,0)^2/12)",
        "proxy_boundary": "interval_CV_diagnostic_only_no_headway_unserved_capacity_proxy_in_selector_actual_cost_unchanged",
        "physics": "unchanged_dispatch_cap_buffer_lower_weights_daily_endpoints_and_44_commitments",
        "observation": "passive_completed_trip_ledger_recorded_after_actual_native_arrival",
        "qualification": "all_three_stage162_controls_exact_physical_and_schedule_reproduction",
        "primary": "supply_aware_minus_demand_only_and_window_forecast_actual_cost_wait_fleet_nonfleet",
        "selection": "minimum_causal_proxy_at_each_decision_lower_strength_breaks_exact_ties",
        "boundary": "frozen_supply_coupling_mechanism_not_learned_upper_or_confirmation"}


def source_cell(root, scenario):
    p = ROOT / "results" / SOURCE_RUN / "cells" / scenario / f"seed_{root}/result.json"
    cell = json.loads(p.read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
            and cell["training_updates"] == 0 and cell["native_steps"] == 24*61380):
        raise ValueError("Supply allocation requires completed qualified Stage162 cells")
    return cell


def episode(root, scenario, scene, raw, checkpoint, condition):
    if condition in CONTROLS:
        row, schedule = source.episode(root, scenario, scene, raw, checkpoint, condition)
        return row, schedule, None
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = NativeSupplyRollingPlan(runner.env, condition)
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Supply allocation changed frozen native weights")
    return row, plan.summarize(), plan.supply_summary()


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
            row, schedule, supply = episode(args.seed, args.scenario, scene,
                raw / condition / str(scene), checkpoint, condition)
            reproduced = condition in CONTROLS
            ref = next(r for r in registered["evaluation"] if r["condition"] == (
                condition if reproduced else "window_forecast") and r["scene_seed"] == scene)
            source.source.source.previous.check_episode(row, ref, baseline=reproduced)
            if reproduced and schedule != ref["schedule"]:
                raise RuntimeError("Passive completion observations changed native control schedules")
            if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and schedule["upper_decisions"] == 44 and schedule["scheduled_trips"] == ref["schedule"]["scheduled_trips"]
                    and np.all(np.isfinite([row[k] for k in native.source.source_spec.authority.routing.METRICS]))
                    and (reproduced or len(supply["decisions"]) == 44)):
                raise RuntimeError("Supply allocation changed physical/decision budget")
            rows.append({"condition": condition, "scenario": args.scenario, "scene_seed": scene,
                **native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule,
                "supply": supply, "source_control_reproduced": reproduced})
            print(f"supply_allocation root={args.seed} {args.scenario}/{condition}/{scene} "
                f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                f"peak={row['peak_fleet']} choices={supply['choice_counts'] if supply else None}", flush=True)
            if scene == scenes[0] and condition == "rolling_1":
                print("SUPPLY_ALLOCATION_FIRST_SCENE_SOURCE_QUALIFIED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "native_steps": 20*61380, "evaluation": rows})
    print("NATIVE_SUPPLY_ALLOCATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
