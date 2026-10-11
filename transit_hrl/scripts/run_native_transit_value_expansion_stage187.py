#!/usr/bin/env python3
"""Collect fixed fresh development scenes, with full preaction timetable inputs."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_timetable_context_stage185 as context_source
from scripts import run_native_transit_sequence_value_stage186 as prior
from scripts.analyze_native_transit_candidate_value_stage173 import validate_continuations
from freq_hrl.domains.transit.native_timetable_context import NativeTimetableContextAudit, qualify_timetable
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

labels, shared = context_source.labels, context_source.shared
native, ROOTS, SCENARIOS, PROBES = labels.native, labels.ROOTS, labels.SCENARIOS, context_source.PROBES
EXPERIMENT_PROTOCOL = "native_transit_value_expansion_stage187_v1"
SOURCE_RUN = "native_transit_sequence_value_stage186_fit_20261012_r1"
REPLICAS = tuple(range(8))
OUTCOME_KEYS = (*shared.METRICS, "scenario", "scene_seed", "N_fleet", "simulation_end_time_s",
    "passengers_generated", "passengers_unserved", "trips_completed", "ep_steps", "window_ledger_qualified")


def scene_seed(root, scenario, replica):
    return 910000000 + root*10000 + SCENARIOS.index(scenario)*100 + replica


def cells(*, preflight=False):
    return [(r, s, i, p) for r in ROOTS for s in SCENARIOS for i in REPLICAS for p in PROBES
        if not preflight or (i == 0 and p == PROBES[0])]


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": prior.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "replicas": list(REPLICAS), "probes": list(PROBES),
        "scene_seeds": "910000000+root*10000+regime_index*100+replica",
        "role": "fresh_development_only_no_outcome_selection_no_Stage169_third_or_fourth_action_labels",
        "independent_scene_groups": 80, "correlated_preaction_states": 480, "paired_candidate_labels": 2880,
        "episodes_per_cell": 7, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "intervention": shared.contract()["intervention"], "pairing": shared.contract()["pairing"],
        "policy": "frozen_original_native_lower_and_unchanged_original_greedy_continuation",
        "duration_model": context_source.MODEL_SOURCE_RUN + "/inputs/models.json_unchanged_64_draws",
        "context": "preaction_native_state_occupancy_and_all_six_full_timetable_curves_in_both_directions",
        "targets": "physical_terminal_fleet_nonfleet_wait_reward_differences_after_one_forced_action",
        "qualification": "frozen_weights_valid_scoring_window_ledgers_exact_identity_and_matched_prefix_state_RNG",
        "preflight": "ten_replica0_probe8_cells_cover_both_roots_all_regimes_before_remaining470",
        "storage": "full_observer_branches_and_raw_server_only_compact_training_JSON_for_sync",
        "boundary": "development_value_data_not_confirmation_or_repeated_learned_upper_actions"}


def validate_scene_roles(reserved):
    new = {scene_seed(r, s, i) for r in ROOTS for s in SCENARIOS for i in REPLICAS}
    if len(new) != 80 or new.intersection(reserved):
        raise ValueError("Fresh development scenes overlap reserved scenes or each other")


def qualify_observer(row):
    # A fresh observer has no earlier outcome to reproduce; only identity is an exact replay.
    shared.validate_schedule(row["schedule"], row["schedule"])
    if not (row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
            and row["trips_completed"] == 262
            and row["schedule"]["upper_decisions"] == 44 and row["schedule"]["scheduled_trips"] == 262
            and row["window_ledger_qualified"] and np.all(np.isfinite([row[k] for k in shared.METRICS]))):
        raise ValueError("Fresh observer changed the native physical or decision budget")


def training_record(root, scenario, index, baseline, branches, snapshot, context):
    cell = {"seed": root, "scenario": scenario, "probe_index": index,
        "observer": baseline, "branches": branches, "training_context": labels.training_context(snapshot)}
    row = labels.records(cell)
    row["occupancy_context"] = {k: copy.deepcopy(context[k]) for k in context_source.OCCUPANCY_KEYS}
    row["timetable_context"] = copy.deepcopy(context["timetable_context"])
    context_source.qualify_contexts([context], [row], baseline)
    floor = max(context["historical_peak"]-12, 0)**2/12
    for target, branch in zip(row["labels"], branches):
        target["future_fleet_cost"] = max(branch["peak_fleet"]-12, 0)**2/12 - floor
        if target["future_fleet_cost"] < 0:
            raise ValueError("Terminal fleet cost is below the preaction historical floor")
    return row


def collect(root, scenario, replica, index, model, output):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = native.NativeRoutingTracker
    torch.set_num_threads(1)
    shared.source.incumbent.source.distributional.validate_model(model)
    scene = scene_seed(root, scenario, replica)
    _, checkpoint = native.load_source(root)
    raw = raw_directory(output)
    baseline, plan = shared.episode(root, scenario, scene, raw/"observer", checkpoint, model, index,
        plan_class=NativeTimetableContextAudit)
    qualify_observer(baseline)
    context = next(c for c in plan.contexts if c["index"] == index)
    print(f"VALUE_EXPANSION_OBSERVER root={root} regime={scenario} scene={scene} probe={index}", flush=True)
    branches = []
    order = [baseline["probe"]["greedy"], *[i for i in shared.CANDIDATES if i != baseline["probe"]["greedy"]]]
    for candidate in order:
        row, _ = shared.episode(root, scenario, scene, raw/f"candidate_{candidate}", checkpoint, model,
            index, candidate=candidate, reference=plan)
        shared.qualify(row, baseline, candidate=candidate)
        row["policy_trace"] = shared.policy_trace(row.pop("supply"))
        branches.append(row)
        print(f"value_expansion {root}/{scenario}/{replica}/{index}/{candidate} "
            f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
            f"identity={row['source_control_reproduced']}", flush=True)
    branches.sort(key=lambda r: r["probe"]["selected"])
    baseline["policy_trace"] = shared.policy_trace(baseline["supply"])
    validate_continuations(baseline, branches)
    record = training_record(root, scenario, index, baseline, branches, plan.snapshot, context)
    write_json(raw/"paired_rollouts.json", {"observer": baseline, "branches": branches, "context": context})
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": root, "scenario": scenario,
        "replica": replica, "scene_seed": scene, "probe_index": index, "software_qualified": True,
        "training_updates": 0, "native_steps": 7*61380, "frozen_weights_qualified_episodes": 7,
        "matched_prefix_state_rng_branches": 6, "window_ledger_qualified_episodes": 7,
        "exact_identity_reproductions": 1, "continuation_qualified": True,
        "observer_origin": "fresh_development_forward_control_not_prior_source_reproduction",
        "observer": {k: baseline[k] for k in OUTCOME_KEYS}, "policy_trace": baseline["policy_trace"],
        "branches": [{"candidate": r["probe"]["selected"],
            "identity_control_reproduced": r["source_control_reproduced"],
            **{k: r[k] for k in OUTCOME_KEYS}} for r in branches],
        "training_record": record, "full_rollouts_path": str(raw/"paired_rollouts.json")}
    qualify_cell(result)
    return result


def qualify_cell(cell):
    r, s, i, p = cell["seed"], cell["scenario"], cell["replica"], cell["probe_index"]
    if not ((r, s, i, p) in cells() and cell["scene_seed"] == scene_seed(r, s, i)
            and cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract()
            and cell["software_qualified"] and cell["training_updates"] == 0 and cell["native_steps"] == 7*61380
            and cell["frozen_weights_qualified_episodes"] == cell["window_ledger_qualified_episodes"] == 7
            and cell["matched_prefix_state_rng_branches"] == 6 and cell["exact_identity_reproductions"] == 1
            and cell["continuation_qualified"]):
        raise ValueError("Changed fresh development scene, qualification or episode budget")
    row, baseline, branches = cell["training_record"], cell["observer"], cell["branches"]
    if (row["root"], row["scenario"], row["scene_seed"], row["probe_index"]) != (r, s, cell["scene_seed"], p):
        raise ValueError("Training record differs from its registered development scene")
    if len(cell["policy_trace"]) != 44 or cell["policy_trace"][p]["selected"] != row["original_choice"]:
        raise ValueError("Training record changed its original continuation control")
    qualify_timetable({"timetable_context": row["timetable_context"]}, row)
    if [b["candidate"] for b in branches] != list(shared.CANDIDATES) or [t["candidate"] for t in row["labels"]] != list(shared.CANDIDATES):
        raise ValueError("Missing or reordered paired candidate outcomes")
    floor = max(row["occupancy_context"]["historical_peak"]-12, 0)**2/12
    base_fleet = max(baseline["peak_fleet"]-12, 0)**2/12
    for branch, target in zip(branches, row["labels"]):
        cost = branch["service_cost_restricted"]-baseline["service_cost_restricted"]
        fleet = max(branch["peak_fleet"]-12, 0)**2/12 - base_fleet
        expected = {"actual_cost_delta": cost, "physical_return_delta": -100.*cost, "fleet_cost_delta": fleet,
            "nonfleet_cost_delta": cost-fleet, "wait_delta_min": branch["restricted_wait_horizon_min"]-baseline["restricted_wait_horizon_min"],
            "episode_reward_delta": branch["ep_reward"]-baseline["ep_reward"],
            "future_fleet_cost": max(branch["peak_fleet"]-12, 0)**2/12-floor}
        identity = branch["candidate"] == row["original_choice"]
        if not (branch["identity_control_reproduced"] == identity and target["future_fleet_cost"] >= 0
                and all(np.isfinite(target[k]) and np.isclose(target[k], v, rtol=0, atol=1e-12) for k, v in expected.items())):
            raise ValueError("Changed physical label units, signs, fleet floor or identity")
        if identity and any(branch[k] != baseline[k] for k in OUTCOME_KEYS):
            raise ValueError("Original-choice continuation failed exact physical reproduction")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--replica", type=int, choices=REPLICAS, required=True)
    parser.add_argument("--probe", type=int, choices=PROBES, required=True)
    for name in ("models", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.models.read_text())[str(args.seed)]
    write_json(args.output, collect(args.seed, args.scenario, args.replica, args.probe, model, args.output))
    print("NATIVE_VALUE_EXPANSION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
