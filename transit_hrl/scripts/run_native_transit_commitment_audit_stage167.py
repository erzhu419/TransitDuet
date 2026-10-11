#!/usr/bin/env python3
"""Replay all frozen distributional trajectories with passive replan audits."""

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
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_distributional_supply_stage166 as source

native = source.native
EXPERIMENT_PROTOCOL = "native_transit_commitment_audit_stage167_v1"
SOURCE_RUN = "native_transit_distributional_supply_stage166_frozen_20261011_r1"
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "episodes_per_cell": 4,
        "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "policy": "exact_Stage166_distributional_supply_all_four_registered_scenes_no_selection",
        "change": "passive_scores_and_completed_trip_ledger_only_no_policy_or_RNG_change",
        "window": "current_decision_to_selected_seventh_departure_minus_native_lookahead_or_simulation_end",
        "forecast": "same_64_member_draws_full_tail_vs_clipped_next_same_direction_replan",
        "wait": "diagnostic_first_six_intervals_including_next_anchor_not_cumulative_native_wait",
        "actual": "joint_direction_half_open_completed_trip_occupancy_in_selected_window_with_historical_floor",
        "qualification": "40_exact_physics_schedule_and_original_selector_reproductions_plus_window_ledger_accounting",
        "primary": "tail_only_discrimination_prefix_rank_disagreement_and_selected_prefix_peak_cost_error",
        "boundary": "passive_credit_horizon_diagnosis_not_counterfactual_policy_value_or_new_performance"}


def source_cell(root, scenario):
    cell = json.loads((ROOT/"results"/SOURCE_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
            and cell["training_updates"] == 0 and cell["native_steps"] == 12*61380):
        raise ValueError("Commitment audit requires qualified Stage166 trajectories")
    return cell


def episode(root, scenario, scene, raw, checkpoint, model):
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = NativeCommitmentAudit(runner.env, model)
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Passive audit changed frozen native weights")
    observations = [{"trip_id": int(tid), **r} for tid, r in sorted(runner.env.completed_trip_observations.items())]
    if len(observations) != row["trips_completed"] or len(observations) != 262:
        raise RuntimeError("Commitment audit lost actual native trip completions")
    return row, plan.summarize(), plan.supply_summary(), plan.audits, observations


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
    source.validate_model(model)
    if model["root"] != args.seed:
        raise ValueError("Changed duration model root")
    registered = source_cell(args.seed, args.scenario)
    _, checkpoint = native.load_source(args.seed)
    from scripts.analyze_native_transit_commitment_audit_stage167 import validate_audits
    rows, raw = [], raw_directory(args.output)
    scenes = native.source.source_spec.scene_seeds(native.LOWER_ROOT[args.seed], args.scenario, preflight=False)
    for scene in scenes:
        row, schedule, supply, audits, observations = episode(args.seed, args.scenario, scene,
            raw/str(scene), checkpoint, model)
        ref = next(r for r in registered["evaluation"] if r["condition"] == "distributional_supply" and r["scene_seed"] == scene)
        source.source.check_episode(row, ref, baseline=True)
        if schedule != ref["schedule"] or supply != ref["supply"]:
            raise RuntimeError("Passive audit changed native schedules or original selector")
        compact = {"condition": "distributional_supply", "scenario": args.scenario, "scene_seed": scene,
            **native.source.source_spec.authority.routing.compact_row(row), "schedule": schedule,
            "supply": supply, "audits": audits, "completed_trips": observations, "source_control_reproduced": True}
        validate_audits(compact)
        rows.append(compact)
        print(f"commitment_audit root={args.seed} {args.scenario}/{scene} cost={row['service_cost_restricted']} "
            f"wait={row['restricted_wait_horizon_min']} peak={row['peak_fleet']} exact_source=True", flush=True)
        if scene == scenes[0]:
            print("COMMITMENT_AUDIT_FIRST_SCENE_SOURCE_QUALIFIED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0,
        "native_steps": 4*61380, "evaluation": rows})
    print("NATIVE_COMMITMENT_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
