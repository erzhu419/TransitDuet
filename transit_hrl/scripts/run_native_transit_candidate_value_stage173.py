#!/usr/bin/env python3
"""Measure six one-action native continuations at matched pre-action states."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_remaining_supply_stage172 as source
from scripts.analyze_native_transit_commitment_scoring_stage168 import validate_scoring
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from freq_hrl.domains.transit.native_candidate_value import NativeCandidateValuePlan
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

native, ROOTS, SCENARIOS = source.native, source.ROOTS, source.SCENARIOS
EXPERIMENT_PROTOCOL = "native_transit_candidate_value_stage173_v1"
SOURCE_RUN = "native_transit_remaining_supply_stage172_frozen_20261011_r1"
CANDIDATES = tuple(range(6))
METRICS = native.source.source_spec.authority.routing.METRICS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "candidates": list(CANDIDATES),
        "probe": "first_original_corrected_choice_divergence_else_fixed_index22_zero_based_no_reward_selection",
        "episodes_per_cell": 7, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "intervention": "one_original_selector_candidate_then_resume_original_greedy_selector",
        "pairing": "exact_prior_scores_actions_native_states_prefix_cost_and_python_numpy_torch_scenario_RNG",
        "qualification": "10_observer_source_reproductions_plus_10_original_selected_candidate_reproductions_70_window_ledgers",
        "models": "registered_Stage172_duration_draws_and_corrected_probe_scores_no_new_fit_or_uncertainty_change",
        "primary": "six_actual_terminal_costs_waits_peaks_and_ranking_regret_at_each_common_state",
        "boundary": "retrospective_reused_scene_single_action_value_diagnostic_not_deployable_oracle_or_learned_HRL"}


def source_cell(root, scenario):
    cell = json.loads((ROOT/"results"/SOURCE_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
            and cell["training_updates"] == 0 and cell["native_steps"] == 4*61380):
        raise ValueError("Candidate values require completed Stage172 controls")
    return cell


def probe_index(cell):
    original, corrected = cell["evaluation"][1], cell["evaluation"][3]
    return next((i for i, (a, b) in enumerate(zip(original["supply"]["decisions"], corrected["supply"]["decisions"]))
        if a["selected"] != b["selected"]), 22)


def policy_trace(supply):
    return [{"time_s": d["time_s"], "direction": d["direction"], "selected": d["selected"],
        "greedy": min(CANDIDATES, key=lambda i: d["candidates"][i]["score"])} for d in supply["decisions"]]


def episode(root, scenario, scene, raw, checkpoint, model, index, *, candidate=None, reference=None):
    runner = native.make_runner(root, scenario, raw, checkpoint)
    before = native.model_arrays(runner)
    plan = NativeCandidateValuePlan(runner.env, model, index, forced_candidate=candidate, reference=reference)
    runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = native.model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Native candidate values changed frozen learned weights")
    trips = [{"trip_id": int(t), **r} for t, r in sorted(runner.env.completed_trip_observations.items())]
    if len(trips) != row["trips_completed"] or len(trips) != 262 or plan.probe is None:
        raise RuntimeError("Native candidate values lost completions or the registered probe")
    compact = {"condition": "observer" if candidate is None else f"candidate_{candidate}",
        "scenario": scenario, "scene_seed": scene, **native.source.source_spec.authority.routing.compact_row(row),
        "schedule": plan.summarize(), "supply": plan.supply_summary(), "probe": plan.probe,
        "window_ledger_qualified": True}
    compact["window_panels"] = source.validate_audits({**compact, "audits": plan.audits, "completed_trips": trips})
    validate_scoring({**compact, "condition": "prefix_wait_full_fleet"}, model,
        strengths=source.incumbent.CANDIDATES["rolling_plus_incumbent"],
        choice_overrides={} if candidate is None else {index: candidate})
    return compact, plan


def qualify(row, baseline, *, candidate=None):
    reproduced = candidate is None or candidate == baseline["probe"]["greedy"]
    source.incumbent.source.distributional.source.check_episode(row, baseline, baseline=reproduced)
    validate_schedule(row["schedule"], baseline["schedule"])
    if reproduced and any(row[k] != baseline[k] for k in ("schedule", "supply", "window_panels")):
        raise RuntimeError("Candidate value observer or identity changed the native source")
    if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
            and row["schedule"]["upper_decisions"] == 44 and row["schedule"]["scheduled_trips"] == 262
            and np.all(np.isfinite([row[k] for k in METRICS]))):
        raise RuntimeError("Changed candidate value physical or decision budget")
    row["source_control_reproduced"] = reproduced


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
    model = pack["duration_models"][str(args.seed)]
    source.incumbent.source.distributional.validate_model(model)
    registered = source_cell(args.seed, args.scenario)
    original, corrected = registered["evaluation"][1], registered["evaluation"][3]
    index, scene = probe_index(registered), original["scene_seed"]
    _, checkpoint = native.load_source(args.seed)
    raw = raw_directory(args.output)
    baseline, plan = episode(args.seed, args.scenario, scene, raw/"observer", checkpoint, model, index)
    qualify(baseline, {**original, "probe": baseline["probe"]})
    print("CANDIDATE_VALUE_OBSERVER_SOURCE_QUALIFIED", flush=True)
    rows = []
    # Run the selected candidate first so exact identity qualifies before alternatives.
    order = [baseline["probe"]["greedy"], *[i for i in CANDIDATES if i != baseline["probe"]["greedy"]]]
    for candidate in order:
        row, _ = episode(args.seed, args.scenario, scene, raw/f"candidate_{candidate}", checkpoint,
            model, index, candidate=candidate, reference=plan)
        qualify(row, baseline, candidate=candidate)
        row["policy_trace"] = policy_trace(row.pop("supply"))
        rows.append(row)
        print(f"candidate_value root={args.seed} {args.scenario}/{index}/{candidate} "
            f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
            f"peak={row['peak_fleet']} identity={row['source_control_reproduced']}", flush=True)
        if candidate == baseline["probe"]["greedy"]:
            print("CANDIDATE_VALUE_IDENTITY_SOURCE_QUALIFIED", flush=True)
    baseline["policy_trace"] = policy_trace(baseline["supply"])
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "software_qualified": True, "training_updates": 0, "native_steps": 7*61380,
        "probe_index": index, "observer": baseline, "branches": sorted(rows, key=lambda r: r["probe"]["selected"]),
        "corrected_probe": corrected["supply"]["decisions"][index]})
    print("NATIVE_CANDIDATE_VALUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
