#!/usr/bin/env python3
"""Collect fit-only native candidate cost labels at fixed decision indices."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_candidate_value_stage173 as shared
from scripts import run_native_transit_incumbent_window_stage169 as source
from scripts.analyze_native_transit_candidate_value_stage173 import validate_continuations
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

native, ROOTS, SCENARIOS = shared.native, shared.ROOTS, shared.SCENARIOS
EXPERIMENT_PROTOCOL = "native_transit_value_data_stage174_v1"
SOURCE_RUN = "native_transit_incumbent_window_stage169_frozen_20261011_r1"
PROBES = (8, 22, 36)


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "probes": list(PROBES),
        "scene": "first_registered_Stage169_scene_per_regime_training_only_disjoint_from_811_Stage173_checks",
        "episodes_per_cell": 7, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "intervention": shared.contract()["intervention"], "pairing": shared.contract()["pairing"],
        "qualification": "30_exact_observers_30_exact_identity_controls_180_paired_interventions_210_window_ledgers",
        "state": "causal_native_v2_frequency_state16_plus_preaction_queue_onboard_active_speed_and_prefix_cost",
        "labels": "postepisode_actual_terminal_cost_minus_original_same_state_continuation_fleet_nonfleet_wait_and_reward",
        "model": "unchanged_Stage169_fit_only_duration_prior_64_draws_and_original_greedy_continuation",
        "boundary": "30_correlated_training_states_180_counterfactual_labels_not_policy_training_or_performance_confirmation"}


def source_record(root, scenario):
    cell = json.loads((ROOT/"results"/SOURCE_RUN/"cells"/scenario/f"seed_{root}/result.json").read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
            and cell["training_updates"] == 0 and cell["native_steps"] == 16*61380):
        raise ValueError("Value training labels require completed Stage169 controls")
    rows = [r for r in cell["evaluation"] if r["condition"] == "rolling_plus_incumbent"]
    row = min(rows, key=lambda r: r["scene_seed"])
    if row["scene_seed"] == shared.source.source.check_scene(root, scenario):
        raise ValueError("Value training scene overlaps diagnosed Stage173 checks")
    return row


def training_context(snapshot):
    active = [b for b in snapshot["buses"] if b["on_route"]]
    return {"queues": {str(d): sum(len(q[2]) for q in snapshot["queues"] if q[1] == d) for d in (False, True)},
        "active": {str(d): sum(b["direction"] == d for b in active) for d in (False, True)},
        "onboard": {str(d): sum(len(b["passengers"]) for b in active if b["direction"] == d) for d in (False, True)},
        "route_speed_mean_mps": float(np.mean([r[1] for r in snapshot["routes"]])),
        "cost_before": snapshot["cost_before"]}


def records(cell):
    baseline, probe = cell["observer"], cell["observer"]["probe"]
    base_fleet = max(baseline["peak_fleet"]-12, 0)**2/12
    return {"root": cell["seed"], "scenario": cell["scenario"], "scene_seed": baseline["scene_seed"],
        "probe_index": cell["probe_index"], "time_s": probe["time_s"], "direction": probe["direction"],
        "native_state_v2": probe["native_state_v2"], "context": cell["training_context"],
        "nominal_s": probe["nominal_s"], "original_choice": probe["greedy"], "candidates": probe["candidates"],
        "labels": [{"candidate": r["probe"]["selected"],
            "actual_cost_delta": (delta := r["service_cost_restricted"]-baseline["service_cost_restricted"]),
            "physical_return_delta": -100.*delta,
            "fleet_cost_delta": (fleet := max(r["peak_fleet"]-12, 0)**2/12-base_fleet),
            "nonfleet_cost_delta": delta-fleet,
            "wait_delta_min": r["restricted_wait_horizon_min"]-baseline["restricted_wait_horizon_min"],
            "episode_reward_delta": r["ep_reward"]-baseline["ep_reward"]} for r in cell["branches"]]}


def collect(root, scenario, index, model, output):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = native.NativeRoutingTracker
    torch.set_num_threads(1)
    shared.source.incumbent.source.distributional.validate_model(model)
    registered = source_record(root, scenario)
    scene = registered["scene_seed"]
    _, checkpoint = native.load_source(root)
    raw = raw_directory(output)
    baseline, plan = shared.episode(root, scenario, scene, raw/"observer", checkpoint, model, index)
    shared.qualify(baseline, {**registered, "probe": baseline["probe"]})
    context = training_context(plan.snapshot)
    print("VALUE_DATA_OBSERVER_SOURCE_QUALIFIED", flush=True)
    order = [baseline["probe"]["greedy"], *[i for i in shared.CANDIDATES if i != baseline["probe"]["greedy"]]]
    branches = []
    for candidate in order:
        row, _ = shared.episode(root, scenario, scene, raw/f"candidate_{candidate}", checkpoint,
            model, index, candidate=candidate, reference=plan)
        shared.qualify(row, baseline, candidate=candidate)
        row["policy_trace"] = shared.policy_trace(row.pop("supply"))
        branches.append(row)
        print(f"value_data root={root} {scenario}/{index}/{candidate} "
            f"cost={row['service_cost_restricted']} peak={row['peak_fleet']} identity={row['source_control_reproduced']}", flush=True)
        if candidate == baseline["probe"]["greedy"]:
            print("VALUE_DATA_IDENTITY_SOURCE_QUALIFIED", flush=True)
    baseline["policy_trace"] = shared.policy_trace(baseline["supply"])
    branches.sort(key=lambda r: r["probe"]["selected"])
    validate_continuations(baseline, branches)
    return {"seed": root, "scenario": scenario, "probe_index": index, "software_qualified": True,
        "training_updates": 0, "native_steps": 7*61380, "observer": baseline,
        "branches": branches, "training_context": context}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--probe", type=int, choices=PROBES, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.models.read_text())[str(args.seed)]
    result = collect(args.seed, args.scenario, args.probe, model, args.output)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), **result})
    print("NATIVE_VALUE_TRAINING_DATA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
