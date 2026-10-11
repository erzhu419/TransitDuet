#!/usr/bin/env python3
"""Isolate fitted cold-start duration priors in the frozen native control loop."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_calibrated_supply_plan import NativeCalibratedSupplyPlan
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_supply_allocation_stage163 as source
from scripts import run_native_transit_duration_calibration_stage164 as calibration
from scripts.run_native_transit_diagnostics_stage147 import check_episode

native = source.native
EXPERIMENT_PROTOCOL = "native_transit_calibrated_supply_stage165_v1"
SOURCE_RUN = "native_transit_supply_allocation_stage163_frozen_20261011_r1"
CALIBRATION_RUN = "native_transit_duration_calibration_stage164_frozen_20261011_r1"
ROOTS, SCENARIOS, STRENGTHS = source.ROOTS, source.SCENARIOS, source.STRENGTHS
CONDITIONS = ("window_forecast", "legacy_supply", "calibrated_demand", "calibrated_supply")
CONTROLS = CONDITIONS[:3]
SOURCE_CONDITIONS = {"window_forecast": "window_forecast", "legacy_supply": "supply_aware",
    "calibrated_demand": "demand_only", "calibrated_supply": "supply_aware"}


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "calibration_run": CALIBRATION_RUN, "calibration_protocol": calibration.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "conditions": list(CONDITIONS),
        "source_controls": {c: SOURCE_CONDITIONS[c] for c in CONTROLS},
        "strengths": list(STRENGTHS), "episodes_per_cell": 16, "clock_s": 61380, "fleet": 12,
        "training_updates": 0, "scenes": "all_four_registered_stage163_scenes_per_regime_no_selection",
        "change": "only_initial_direction_duration_prior_before_first_real_completion",
        "prior": "registered_stage164_trip_weighted_direction_means_from_five_fit_scenes_only",
        "online": "first_completion_replaces_prior_then_original_EWMA_alpha0.2_no_future_realized_trips",
        "unchanged": "candidate_strengths_wait_score_fleet_score_occupancy_physics_frozen_weights_projection_44_commitments",
        "qualification": "window_and_legacy_supply_exact_source_physics_schedule_and_selector_reproduction_calibrated_demand_exact_source_physics_and_schedule",
        "primary": "calibrated_supply_minus_legacy_supply_calibrated_demand_and_window_actual_cost_wait_fleet_nonfleet",
        "boundary": "frozen_cold_start_mechanism_development_not_learned_upper_or_confirmation"}


def duration_prior(root, summary_path):
    result = json.loads(summary_path.read_text())
    if not (result["protocol"] == calibration.EXPERIMENT_PROTOCOL and result["contract"] == calibration.contract()
            and result["software_qualified"] and result["fit_episodes"] == result["check_episodes"] == 10
            and result["training_updates"] == 0 and result["native_steps"] == 20*61380):
        raise ValueError("Calibrated supply requires the completed fit-only Stage164 prior")
    prior = result["duration_prior_s"][str(root)]
    if set(prior) != {"False", "True"} or not all(np.isfinite(v) and v > 0 for v in prior.values()):
        raise ValueError("Invalid fitted direction times")
    return {d: prior[str(d)] for d in (False, True)}


def source_cell(root, scenario):
    cell = json.loads((ROOT/"results"/SOURCE_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
            and cell["training_updates"] == 0 and cell["native_steps"] == 20*61380):
        raise ValueError("Calibrated supply requires qualified Stage163 controls")
    return cell


def episode(root, scenario, scene, raw, checkpoint, condition, prior):
    if condition in {"window_forecast", "legacy_supply"}:
        return source.episode(root, scenario, scene, raw, checkpoint, SOURCE_CONDITIONS[condition])
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = NativeCalibratedSupplyPlan(runner.env, SOURCE_CONDITIONS[condition], prior)
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Calibrated supply changed frozen native weights")
    return row, plan.summarize(), plan.supply_summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = native.NativeRoutingTracker
    torch.set_num_threads(1)
    registered, prior = source_cell(args.seed, args.scenario), duration_prior(args.seed, args.calibration_summary)
    _, checkpoint = native.load_source(args.seed)
    raw, rows = raw_directory(args.output), []
    scenes = native.source.source_spec.scene_seeds(native.LOWER_ROOT[args.seed], args.scenario, preflight=False)
    for scene in scenes:
        for condition in CONDITIONS:
            row, schedule, supply = episode(args.seed, args.scenario, scene,
                raw/condition/str(scene), checkpoint, condition, prior)
            ref = next(r for r in registered["evaluation"] if r["condition"] == SOURCE_CONDITIONS[condition] and r["scene_seed"] == scene)
            reproduced = condition in CONTROLS
            check_episode(row, ref, baseline=reproduced)
            if reproduced and schedule != ref["schedule"]:
                raise RuntimeError("Calibrated supply changed a source-control schedule")
            if condition == "legacy_supply" and supply != ref["supply"]:
                raise RuntimeError("Calibrated supply changed the legacy selector")
            if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and schedule["upper_decisions"] == 44 and schedule["scheduled_trips"] == 262
                    and np.all(np.isfinite([row[k] for k in native.source.source_spec.authority.routing.METRICS]))
                    and (condition == "window_forecast" or len(supply["decisions"]) == 44)):
                raise RuntimeError("Calibrated supply changed physical/decision budget")
            rows.append({"condition": condition, "scenario": args.scenario, "scene_seed": scene,
                **native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule,
                "supply": supply, "source_control_reproduced": reproduced})
            print(f"calibrated_supply root={args.seed} {args.scenario}/{condition}/{scene} "
                f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                f"peak={row['peak_fleet']} choices={supply['choice_counts'] if supply else None}", flush=True)
            if scene == scenes[0] and condition == "calibrated_demand":
                print("CALIBRATED_SUPPLY_FIRST_SCENE_SOURCE_QUALIFIED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "duration_prior_s": {str(d): v for d, v in prior.items()},
        "native_steps": 16*61380, "evaluation": rows})
    print("NATIVE_CALIBRATED_SUPPLY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
