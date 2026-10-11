#!/usr/bin/env python3
"""Test missing fixed-window authority without changing the hybrid score."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_incumbent_window_plan import NativeIncumbentWindowPlan
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_commitment_scoring_stage168 as source
from scripts.analyze_native_transit_commitment_scoring_stage168 import validate_scoring
from scripts.analyze_native_transit_commitment_audit_stage167 import validate_audits

native = source.native
EXPERIMENT_PROTOCOL = "native_transit_incumbent_window_stage169_v1"
SOURCE_RUN = "native_transit_commitment_scoring_stage168_frozen_20261011_r1"
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
CONDITIONS = ("window_forecast", "prefix_wait_full_fleet", "incumbent_only", "rolling_plus_incumbent")
CONTROLS = CONDITIONS[:2]
CANDIDATES = {"prefix_wait_full_fleet": source.distributional.STRENGTHS,
    "incumbent_only": (None,), "rolling_plus_incumbent": (*source.distributional.STRENGTHS, None)}


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "conditions": list(CONDITIONS),
        "episodes_per_cell": 16, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "candidates": {k: list(v) for k, v in CANDIDATES.items()},
        "change": "append_fixed_current_six_trip_window_forecast_to_same_five_rolling_candidates",
        "incumbent": "budgeted_window_internal_times_only_first_last_next_anchor_and_tentative_tail_unchanged",
        "score": "same_prefix_wait_plus_full_tail_expected_native_fleet_penalty_no_new_weights",
        "tie": "same_first_minimum_existing_candidates_before_appended_incumbent",
        "unchanged": "fit_model_64_draws_historical_floor_EWMA_native_RNG_lower_cost_cap_clock_and_44_commitments",
        "qualification": "80_exact_source_trajectories_and_selectors_plus_40_forced_incumbent_window_physics_and_schedules",
        "window_accounting": "120_selected_window_trip_ledger_qualifications",
        "scenes": "all_four_registered_stage168_scenes_per_regime_no_selection",
        "primary": "augmented_minus_five_and_window_actual_cost_wait_reward_fleet_nonfleet",
        "boundary": "frozen_action_pool_development_not_learned_upper_or_independent_confirmation"}


def source_cell(root, scenario):
    cell = json.loads((ROOT/"results"/SOURCE_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
            and cell["training_updates"] == 0 and cell["native_steps"] == 20*61380):
        raise ValueError("Incumbent experiment requires qualified Stage168 cells")
    return cell


def validate_candidates(row, model):
    validate_scoring({**row, "condition": "prefix_wait_full_fleet"}, model, strengths=CANDIDATES[row["condition"]])


def episode(root, scenario, scene, raw, checkpoint, condition, model):
    if condition in CONTROLS:
        return source.episode(root, scenario, scene, raw, checkpoint, condition, model)
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = NativeIncumbentWindowPlan(runner.env, model, condition)
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Incumbent experiment changed frozen native weights")
    trips = [{"trip_id": int(t), **r} for t, r in sorted(runner.env.completed_trip_observations.items())]
    if len(trips) != row["trips_completed"] or len(trips) != 262:
        raise RuntimeError("Incumbent experiment lost native trip completions")
    return row, plan.summarize(), plan.supply_summary(), plan.audits, trips


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
    source.distributional.validate_model(model)
    if model["root"] != args.seed:
        raise ValueError("Changed frozen duration model root")
    registered = source_cell(args.seed, args.scenario)
    _, checkpoint = native.load_source(args.seed)
    rows, raw = [], raw_directory(args.output)
    scenes = native.source.source_spec.scene_seeds(native.LOWER_ROOT[args.seed], args.scenario, preflight=False)
    for scene in scenes:
        for condition in CONDITIONS:
            row, schedule, supply, audits, trips = episode(args.seed, args.scenario, scene,
                raw/condition/str(scene), checkpoint, condition, model)
            ref = next(r for r in registered["evaluation"] if r["scene_seed"] == scene and r["condition"] == (
                "window_forecast" if condition in {"window_forecast", "incumbent_only"} else "prefix_wait_full_fleet"))
            reproduced, incumbent = condition in CONTROLS, condition == "incumbent_only"
            source.distributional.source.check_episode(row, ref, baseline=reproduced or incumbent)
            if (reproduced or incumbent) and schedule != ref["schedule"]:
                raise RuntimeError("Incumbent experiment changed an exact control schedule")
            if reproduced and supply != ref["supply"]:
                raise RuntimeError("Incumbent experiment changed a source selector")
            if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and schedule["upper_decisions"] == 44 and schedule["scheduled_trips"] == 262
                    and np.all(np.isfinite([row[k] for k in native.source.source_spec.authority.routing.METRICS]))):
                raise RuntimeError("Incumbent experiment changed physical/decision budget")
            compact = {"condition": condition, "scenario": args.scenario, "scene_seed": scene,
                **native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule,
                "supply": supply, "source_control_reproduced": reproduced, "incumbent_control_reproduced": incumbent}
            if condition != "window_forecast":
                compact["window_panels"] = validate_audits({**compact, "audits": audits, "completed_trips": trips})
                compact["window_ledger_qualified"] = True
                validate_candidates(compact, model)
                if reproduced and compact["window_panels"] != ref["window_panels"]:
                    raise RuntimeError("Incumbent experiment changed source window accounting")
            rows.append(compact)
            print(f"incumbent_window root={args.seed} {args.scenario}/{condition}/{scene} "
                f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                f"peak={row['peak_fleet']} exact_source={reproduced} exact_incumbent={incumbent}", flush=True)
            if scene == scenes[0] and incumbent:
                print("INCUMBENT_WINDOW_FIRST_SCENE_SOURCE_QUALIFIED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "native_steps": 16*61380, "evaluation": rows})
    print("NATIVE_INCUMBENT_WINDOW_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
