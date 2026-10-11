#!/usr/bin/env python3
"""Collect passive in-flight telemetry without changing window-forecast control."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_duration_calibration_stage164 as source
from scripts import run_native_transit_incumbent_window_stage169 as incumbent
from freq_hrl.domains.transit.native_remaining_time import NativeRemainingTimeAudit
from freq_hrl.domains.transit.native_rolling_plan import schedule_summary
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

native = source.native
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
EXPERIMENT_PROTOCOL = "native_transit_remaining_time_stage171_v1"
SOURCE_RUN = incumbent.source.distributional.source.CALIBRATION_RUN
CONDITIONS = ("fit_audit", "check_control", "check_audit")


def check_scene(root, scenario):
    return 811000000+root*1000+SCENARIOS.index(scenario)*10


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "conditions": list(CONDITIONS),
        "fit_scenes": "exact_Stage164_fit_replays_only_no_check_labels_in_fitting",
        "check_scenes": "811000000+root*1000+regime_index*10_fresh_disjoint_from_Stage164_and_rolling_evaluation",
        "episodes_per_cell": 3, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "policy": "unchanged_window_forecast_frozen_native_lower_no_supply_candidate_selection",
        "observations": "44_callbacks_active_bus_progress_age_load_speed_holding_dwelling_and_causal_EWMA",
        "labels": "join_actual_completion_ledger_only_after_native_episode_finishes",
        "model": "one_fixed_ridge0.001_additive_correction_per_root_all_five_fit_regimes_no_tuning",
        "qualification": "10_exact_source_fit_physics_schedules_plus_10_fresh_check_control_audit_pairs",
        "primary": "fresh_check_remaining_time_MAE_bias_underprediction_and_q95_error_by_root_and_regime",
        "boundary": "passive_remaining_time_accuracy_not_native_allocation_value_or_learned_upper"}


def source_cell(root, scenario):
    cell = json.loads((ROOT/"results"/SOURCE_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["seed"] == root and cell["scenario"] == scenario and cell["software_qualified"]
            and cell["training_updates"] == 0 and cell["native_steps"] == 2*61380):
        raise ValueError("Remaining-time audit needs qualified fitting sources")
    return cell


def label_snapshots(snapshots, trips):
    ledger = {t["trip_id"]: t for t in trips}
    for snapshot in snapshots:
        now = snapshot["time_s"]
        for sample in snapshot["active"]:
            trip = ledger[sample["trip_id"]]
            if not (sample["direction"] == trip["direction"] and sample["launch_time_s"] == trip["launch_time_s"]
                    and trip["launch_time_s"] <= now <= trip["completion_time_s"]
                    and sample["elapsed_s"] == now-trip["launch_time_s"]
                    and np.isclose(sample["predicted_remaining_s"], max(sample["mean_total_s"]*sample["remaining_fraction"], 1.), rtol=0, atol=1e-10)):
                raise ValueError("Passive active snapshot differs from the native trip ledger")
            sample["actual_remaining_s"] = trip["completion_time_s"]-now


def episode(root, scenario, scene, raw, checkpoint, audit, model):
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = NativeRemainingTimeAudit(runner.env, model) if audit else source.DurationCalibrationPlan(runner.env)
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Passive remaining-time audit changed native weights")
    trips = [{"trip_id": int(t), **r} for t, r in sorted(runner.env.completed_trip_observations.items())]
    if len(trips) != row["trips_completed"] or len(trips) != 262:
        raise RuntimeError("Passive remaining-time audit lost completed trips")
    snapshots = plan.snapshots if audit else []
    label_snapshots(snapshots, trips)
    return {**native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule_summary(plan),
        "snapshots": snapshots, "completed_trips": trips}


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
    incumbent.source.distributional.validate_model(model)
    if model["root"] != args.seed:
        raise ValueError("Changed remaining-time duration model root")
    fitting = next(r for r in source_cell(args.seed, args.scenario)["calibration"] if r["split"] == "fit")
    _, checkpoint = native.load_source(args.seed)
    rows, raw = [], raw_directory(args.output)
    for condition in CONDITIONS:
        scene = fitting["scene_seed"] if condition == "fit_audit" else check_scene(args.seed, args.scenario)
        row = episode(args.seed, args.scenario, scene, raw/condition, checkpoint, condition != "check_control", model)
        ref = fitting if condition == "fit_audit" else rows[-1] if condition == "check_audit" else None
        if ref is not None:
            incumbent.source.distributional.source.check_episode(row, ref, baseline=True)
            if row["schedule"] != ref["schedule"]:
                raise RuntimeError("Passive remaining-time telemetry changed native departures/goals")
        if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                and row["schedule"]["upper_decisions"] == 44 and row["schedule"]["scheduled_trips"] == 262
                and len(row["snapshots"]) == (0 if condition == "check_control" else 44)):
            raise RuntimeError("Remaining-time audit changed native callback/physical budget")
        rows.append({"condition": condition, "scenario": args.scenario, "scene_seed": scene,
            "source_control_reproduced": ref is not None, **row})
        print(f"remaining_time root={args.seed} {args.scenario}/{condition}/{scene} "
            f"cost={row['service_cost_restricted']} active_samples="
            f"{sum(len(s['active']) for s in row['snapshots'])} exact_control={ref is not None}", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "native_steps": 3*61380, "evaluation": rows})
    print("NATIVE_REMAINING_TIME_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
