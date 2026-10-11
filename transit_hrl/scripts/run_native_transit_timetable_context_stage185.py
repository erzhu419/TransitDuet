#!/usr/bin/env python3
"""Recover full timetable inputs on the twenty already-labelled native scenes."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_tail_value_stage184 as prior
from scripts import run_native_transit_value_context_stage177 as context_source
from scripts import run_native_transit_value_data_stage174 as labels
from freq_hrl.domains.transit.native_timetable_context import NativeTimetableContextAudit, qualify_timetable, EVENT_NAMES
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

shared, native, ROOTS, SCENARIOS = labels.shared, labels.native, labels.ROOTS, labels.SCENARIOS
PROBES = context_source.PROBES
OCCUPANCY_KEYS = ("historical_peak", "active_buses", "queue_by_station", "route_speed_mps")
EXPERIMENT_PROTOCOL = "native_transit_timetable_context_stage185_v1"
SOURCE_RUN = "native_transit_tail_value_stage184_fit_20261012_r1"
MODEL_SOURCE_RUN = "native_transit_value_context_stage177_frozen_20261011_r1"
OBSERVER_KEYS = (*shared.METRICS, "scenario", "scene_seed", "N_fleet", "simulation_end_time_s",
    "source_control_reproduced", "window_ledger_qualified")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": prior.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "scene_indices": [0, 1], "probes": list(PROBES),
        "episodes_per_cell": 1, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "policy": "unchanged_original_greedy_selector_and_frozen_native_lower",
        "context": "all_six_full_candidate_time_curves_and_ordered_pending_trips_in_both_directions_with_commitment_and_goals",
        "event_features": list(EVENT_NAMES),
        "qualification": "exact_source_physics_schedule_supply_windows_six_labelled_states_and_existing_occupancy_context",
        "labels": "unchanged_720_counterfactual_labels_on_120_states_in_20_development_scenes",
        "storage": "full_observer_and_raw_server_only_compact_context_JSON_only_for_sync",
        "boundary": "passive_causal_representation_data_not_new_policy_performance_or_independent_confirmation"}


def qualify_contexts(contexts, rows, registered):
    context_source.qualify_contexts(contexts, rows, registered)
    by_index = {r["probe_index"]: r for r in rows}
    for c in contexts:
        row = by_index[c["index"]]
        if {k: c[k] for k in OCCUPANCY_KEYS} != row["occupancy_context"]:
            raise ValueError("Full timetable replay changed the previously labelled occupancy state")
        qualify_timetable(c, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--scene-index", type=int, choices=(0, 1), required=True)
    for name in ("models", "training-data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = native.NativeRoutingTracker
    torch.set_num_threads(1)
    registered = labels.source_record(args.seed, args.scenario, scene_index=args.scene_index)
    rows = [r for r in json.loads(args.training_data.read_text())
        if (r["root"], r["scenario"], r["scene_seed"]) == (args.seed, args.scenario, registered["scene_seed"])]
    if len(rows) != 6 or {r["probe_index"] for r in rows} != set(PROBES):
        raise ValueError("Full timetable audit must reuse all six callbacks of its labelled scene")
    model = json.loads(args.models.read_text())[str(args.seed)]
    shared.source.incumbent.source.distributional.validate_model(model)
    _, checkpoint = native.load_source(args.seed)
    raw = raw_directory(args.output)
    row, plan = shared.episode(args.seed, args.scenario, registered["scene_seed"], raw/"audit",
        checkpoint, model, PROBES[-1], plan_class=NativeTimetableContextAudit)
    shared.qualify(row, {**registered, "probe": row["probe"]})
    qualify_contexts(plan.contexts, rows, registered)
    write_json(raw/"observer.json", row)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "scene_index": args.scene_index, "scene_seed": registered["scene_seed"],
        "software_qualified": True, "training_updates": 0, "native_steps": 61380,
        "observer": {k: row[k] for k in OBSERVER_KEYS}, "full_observer_path": str(raw/"observer.json"),
        "policy_trace": shared.policy_trace(row["supply"]), "contexts": plan.contexts})
    print("NATIVE_TIMETABLE_CONTEXT_SOURCE_QUALIFIED", flush=True)


if __name__ == "__main__":
    main()
