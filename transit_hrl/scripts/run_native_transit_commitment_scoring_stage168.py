#!/usr/bin/env python3
"""Test a two-by-two score-horizon change under frozen native physics."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_commitment_audit import NativeCommitmentAudit
from freq_hrl.domains.transit.native_commitment_plan import NativeCommitmentScoringPlan, HORIZONS
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_commitment_audit_stage167 as source

native, distributional = source.native, source.source
EXPERIMENT_PROTOCOL = "native_transit_commitment_scoring_stage168_v1"
SOURCE_RUN = "native_transit_commitment_audit_stage167_frozen_20261011_r1"
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
CONDITIONS = ("window_forecast", "full_wait_full_fleet", *HORIZONS)
CONTROLS = CONDITIONS[:2]


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "window_source_run": source.SOURCE_RUN, "roots": list(ROOTS), "scenarios": list(SCENARIOS),
        "conditions": list(CONDITIONS), "episodes_per_cell": 20, "clock_s": 61380, "fleet": 12,
        "training_updates": 0, "strengths": list(distributional.STRENGTHS),
        "change": "two_by_two_full_tail_or_next_same_direction_commitment_window_wait_and_fleet_scores",
        "window": source.contract()["window"], "wait": source.contract()["wait"],
        "fleet_score": "same_64_member_expected_native_quadratic_penalty_with_historical_floor",
        "unchanged": "fit_model_native_RNG_mean_EWMA_candidates_lower_weights_cost_cap_endpoints_and_44_commitments",
        "qualification": "40_exact_window_and_40_exact_full_physics_schedule_selector_replays_plus_all_window_ledger_checks",
        "scenes": "all_four_registered_scenes_per_regime_no_selection",
        "primary": "actual_restricted_service_cost_vs_full_and_window_with_wait_reward_fleet_nonfleet_components",
        "factorial": "prefix_wait_main_effect_prefix_fleet_main_effect_and_interaction_in_actual_cost",
        "boundary": "frozen_score_horizon_mechanism_development_not_learned_authority_or_independent_confirmation"}


def source_cell(root, scenario):
    cell = json.loads((ROOT/"results"/SOURCE_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
            and cell["training_updates"] == 0 and cell["native_steps"] == 4*61380):
        raise ValueError("Commitment scoring requires qualified Stage167 controls")
    return cell


def episode(root, scenario, scene, raw, checkpoint, condition, model):
    if condition == "window_forecast":
        row, schedule, supply = distributional.episode(root, scenario, scene, raw, checkpoint, condition, model)
        return row, schedule, supply, [], []
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = (NativeCommitmentAudit(runner.env, model) if condition == "full_wait_full_fleet"
        else NativeCommitmentScoringPlan(runner.env, model, condition))
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Commitment scoring changed frozen native weights")
    trips = [{"trip_id": int(t), **r} for t, r in sorted(runner.env.completed_trip_observations.items())]
    if len(trips) != row["trips_completed"] or len(trips) != 262:
        raise RuntimeError("Commitment scoring lost actual trip completions")
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
    distributional.validate_model(model)
    registered = source_cell(args.seed, args.scenario)
    window_source = source.source_cell(args.seed, args.scenario)
    if model["root"] != args.seed:
        raise ValueError("Changed fitted duration root")
    _, checkpoint = native.load_source(args.seed)
    from scripts.analyze_native_transit_commitment_audit_stage167 import validate_audits
    from scripts.analyze_native_transit_commitment_scoring_stage168 import validate_scoring
    raw, rows = raw_directory(args.output), []
    scenes = native.source.source_spec.scene_seeds(native.LOWER_ROOT[args.seed], args.scenario, preflight=False)
    for scene in scenes:
        for condition in CONDITIONS:
            row, schedule, supply, audits, trips = episode(args.seed, args.scenario, scene,
                raw/condition/str(scene), checkpoint, condition, model)
            ref = next(r for r in (window_source if condition == "window_forecast" else registered)["evaluation"]
                if r["scene_seed"] == scene and r["condition"] == (
                    "window_forecast" if condition == "window_forecast" else "distributional_supply"))
            reproduced = condition in CONTROLS
            distributional.source.check_episode(row, ref, baseline=reproduced)
            if reproduced and (schedule != ref["schedule"] or supply != ref["supply"]):
                raise RuntimeError("Commitment scoring changed an exact source control")
            compact = {"condition": condition, "scenario": args.scenario, "scene_seed": scene,
                **native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule,
                "supply": supply, "source_control_reproduced": reproduced}
            if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and schedule["upper_decisions"] == 44 and schedule["scheduled_trips"] == 262
                    and np.all(np.isfinite([row[k] for k in native.source.source_spec.authority.routing.METRICS]))):
                raise RuntimeError("Commitment scoring changed the physical/decision budget")
            if condition != "window_forecast":
                if reproduced and (audits != ref["audits"] or trips != ref["completed_trips"]):
                    raise RuntimeError("Full score control changed passive audit accounting")
                compact["window_panels"] = validate_audits({**compact, "audits": audits, "completed_trips": trips})
                compact["window_ledger_qualified"] = True
                validate_scoring(compact, model)
            rows.append(compact)
            print(f"commitment_scoring root={args.seed} {args.scenario}/{condition}/{scene} "
                f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                f"peak={row['peak_fleet']} exact_source={reproduced}", flush=True)
            if scene == scenes[0] and condition == "full_wait_full_fleet":
                print("COMMITMENT_SCORING_FIRST_SCENE_SOURCE_QUALIFIED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "native_steps": 20*61380, "evaluation": rows})
    print("NATIVE_COMMITMENT_SCORING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
