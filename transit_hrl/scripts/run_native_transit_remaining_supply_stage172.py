#!/usr/bin/env python3
"""Test fit-only remaining-time correction in the frozen native selector."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_remaining_time_stage171 as source
from freq_hrl.domains.transit.native_remaining_time import NativeRemainingSupplyPlan, remaining_features
from freq_hrl.domains.transit.native_incumbent_window_plan import NativeIncumbentWindowPlan
from scripts.analyze_native_transit_commitment_audit_stage167 import validate_audits
from freq_hrl.domains.transit.native_distributional_supply_plan import residual_draws
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

native, incumbent = source.native, source.incumbent
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
EXPERIMENT_PROTOCOL = "native_transit_remaining_supply_stage172_v1"
SOURCE_RUN = "native_transit_remaining_time_stage171_frozen_20261011_r1"
CONDITIONS = ("window_forecast", "rolling_plus_incumbent", "zero_remaining_correction", "corrected_remaining")
CONTROLS = ("window_forecast", "zero_remaining_correction")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "conditions": list(CONDITIONS),
        "episodes_per_cell": 4, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "scenes": "all_ten_Stage171_check_scenes_reused_for_bounded_control_development_not_confirmation",
        "change": "same_fit_only_remaining_correction_added_to_active_member_seconds_then_floor1",
        "unchanged": "64_duration_draws_future_trip_columns_EWMA_historical_floor_six_candidates_prefix_wait_full_fleet_score_and_weights_native_cost_lower_cap_and_44_commitments",
        "fit": "exact_Stage171_coefficients_fit_scenes_only_no_refitting_tuning_or_joint_copula",
        "qualification": "10_exact_window_source_reproductions_and_10_zero_correction_physics_schedules_and_selector_reproductions",
        "window_accounting": "30_selected_window_native_trip_ledger_qualifications",
        "primary": "corrected_minus_window_and_original_actual_cost_wait_reward_fleet_nonfleet_all_root_regimes",
        "boundary": "frozen_remaining_prediction_control_development_not_learned_upper_or_independent_confirmation"}


def source_cell(root, scenario):
    cell = json.loads((ROOT/"results"/SOURCE_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["seed"] == root and cell["scenario"] == scenario and cell["software_qualified"]
            and cell["training_updates"] == 0 and cell["native_steps"] == 3*61380):
        raise ValueError("Remaining-supply test needs completed passive telemetry cells")
    return cell


def correction_model(condition, model):
    return {**model, "coefficients": [0.]*11} if condition == "zero_remaining_correction" else model


def validate_selector(row, duration_model, remaining_model, trips):
    incumbent.validate_candidates({**row, "condition": "rolling_plus_incumbent"}, duration_model)
    snapshots = row["remaining_corrections"]
    expected = 44 if row["condition"] in CONDITIONS[2:] else 0
    if len(snapshots) != expected:
        raise ValueError("Changed active correction observation budget")
    if not snapshots:
        return
    model = correction_model(row["condition"], remaining_model)
    draws = residual_draws(duration_model, [(t["trip_id"], t["direction"]) for t in trips])
    for snapshot, decision in zip(snapshots, row["supply"]["decisions"]):
        if any(snapshot[k] != decision[k] for k in ("time_s", "direction")):
            raise ValueError("Active correction differs from executed selector clock")
        ids = [s["trip_id"] for s in snapshot["active"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate active trip correction")
        for s in snapshot["active"]:
            shift = 2000.*float(remaining_features(s)@model["coefficients"])
            original = np.maximum((s["mean_total_s"]+draws[s["trip_id"]])*s["remaining_fraction"], 1.)
            if not (s["trip_id"] in draws and s["mean_total_s"] == decision["duration_estimates_s"][str(s["direction"])]
                    and s["elapsed_s"] == snapshot["time_s"]-s["launch_time_s"]
                    and s["elapsed_s"] >= 0 and 0 <= s["remaining_fraction"] <= 1
                    and np.all(np.isfinite(remaining_features(s))) and s["actual_remaining_s"] >= 0
                    and np.isclose(s["predicted_remaining_s"], max(s["mean_total_s"]*s["remaining_fraction"], 1.), rtol=0, atol=1e-10)
                    and np.allclose([s["correction_s"], s["original_sample_mean_s"], s["corrected_sample_mean_s"]],
                        [shift, original.mean(), np.maximum(original+shift, 1.).mean()], rtol=0, atol=1e-10)):
                raise ValueError("Executed active ensemble shift changed its fitted model or samples")


def episode(root, scenario, scene, raw, checkpoint, condition, duration_model, remaining_model):
    if condition == "window_forecast":
        result = source.episode(root, scenario, scene, raw, checkpoint, False, duration_model)
        return result, None, None, [], result["completed_trips"], []
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = (NativeIncumbentWindowPlan(runner.env, duration_model, "rolling_plus_incumbent")
        if condition == "rolling_plus_incumbent" else NativeRemainingSupplyPlan(runner.env,
            duration_model, correction_model(condition, remaining_model)))
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Remaining-supply selector changed native learned weights")
    trips = [{"trip_id": int(t), **r} for t, r in sorted(runner.env.completed_trip_observations.items())]
    if len(trips) != row["trips_completed"] or len(trips) != 262:
        raise RuntimeError("Remaining-supply selector lost native completions")
    snapshots = [] if condition == "rolling_plus_incumbent" else plan.snapshots
    source.label_snapshots(snapshots, trips)
    return row, plan.summarize(), plan.supply_summary(), plan.audits, trips, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = native.NativeRoutingTracker
    torch.set_num_threads(1)
    pack = json.loads(args.models.read_text())
    duration, remaining = pack["duration_models"][str(args.seed)], pack["remaining_models"][str(args.seed)]
    incumbent.source.distributional.validate_model(duration)
    if duration["root"] != args.seed or len(remaining["coefficients"]) != 11:
        raise ValueError("Changed correction model root or feature width")
    registered = source_cell(args.seed, args.scenario)
    window = next(r for r in registered["evaluation"] if r["condition"] == "check_control")
    _, checkpoint = native.load_source(args.seed)
    scene, rows, raw = source.check_scene(args.seed, args.scenario), [], raw_directory(args.output)
    for condition in CONDITIONS:
        row, schedule, supply, audits, trips, snapshots = episode(args.seed, args.scenario, scene,
            raw/condition, checkpoint, condition, duration, remaining)
        schedule = row["schedule"] if condition == "window_forecast" else schedule
        ref = rows[1] if condition == "zero_remaining_correction" else window
        reproduced = condition in CONTROLS
        incumbent.source.distributional.source.check_episode(row, ref, baseline=reproduced)
        if reproduced and schedule != ref["schedule"]:
            raise RuntimeError("Remaining correction changed an exact control schedule")
        if condition == "zero_remaining_correction" and supply != ref["supply"]:
            raise RuntimeError("Zero correction changed native candidate scores or selector")
        if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                and schedule["upper_decisions"] == 44 and schedule["scheduled_trips"] == 262):
            raise RuntimeError("Remaining correction changed native decision/physical budget")
        compact = {"condition": condition, "scenario": args.scenario, "scene_seed": scene,
            **native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule,
            "supply": supply, "remaining_corrections": snapshots, "source_control_reproduced": reproduced}
        if condition != "window_forecast":
            compact["window_panels"] = validate_audits({**compact, "audits": audits, "completed_trips": trips})
            compact["window_ledger_qualified"] = True
            validate_selector(compact, duration, remaining, window["completed_trips"])
            if condition == "zero_remaining_correction" and compact["window_panels"] != ref["window_panels"]:
                raise RuntimeError("Zero correction changed native window accounting")
        rows.append(compact)
        print(f"remaining_supply root={args.seed} {args.scenario}/{condition}/{scene} "
            f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
            f"peak={row['peak_fleet']} exact_control={reproduced}", flush=True)
        if condition == "zero_remaining_correction":
            print("REMAINING_SUPPLY_ZERO_CONTROL_QUALIFIED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "native_steps": 4*61380, "evaluation": rows})
    print("NATIVE_REMAINING_SUPPLY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
