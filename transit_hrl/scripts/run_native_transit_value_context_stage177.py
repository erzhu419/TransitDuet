#!/usr/bin/env python3
"""Recover preaction occupancy context without rerunning labelled branches."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_action_value_stage176 as prior
from freq_hrl.domains.transit.native_value_context import NativeValueContextAudit, PROBES
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

shared = prior.source.shared
ROOTS, SCENARIOS = prior.source.ROOTS, prior.source.SCENARIOS
EXPERIMENT_PROTOCOL = "native_transit_value_context_stage177_v1"
SOURCE_RUN = prior.SOURCE_RUN


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": prior.source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "probes": list(PROBES),
        "episodes_per_cell": 1, "clock_s": 61380, "fleet": 12, "training_updates": 0,
        "policy": "unchanged_original_greedy_selector_and_frozen_native_lower",
        "context": "preaction_historical_peak_active_bus_progress_age_load_speed_hold_dwell_local_queues_route_speed",
        "qualification": "exact_source_physics_schedule_supply_windows_and_all_six_causal_label_states",
        "labels": "reuse_360_qualified_Stage174_175_labels_no_new_counterfactuals",
        "boundary": "passive_value_representation_data_not_a_policy_improvement"}


def qualify_contexts(contexts, rows, registered):
    expected = {r["probe_index"]: r for r in rows}
    if not (len(contexts) == len(expected) == len(rows)
            and sorted(c["index"] for c in contexts) == sorted(expected)):
        raise ValueError("Missing or duplicate passive value contexts")
    for c in contexts:
        r = expected[c["index"]]
        d = registered["supply"]["decisions"][c["index"]]
        if not (all(c[k] == r[k] for k in ("time_s", "direction", "native_state_v2", "nominal_s", "candidates"))
                and c["greedy"] == r["original_choice"] == d["selected"]
                and c["historical_peak"] == d["historical_peak"]):
            raise ValueError("Passive value context changed a labelled preaction state or source fleet floor")
        context = r["context"]
        for direction in (False, True):
            buses = [b for b in c["active_buses"] if b["direction"] == direction]
            if not (len(buses) == context["active"][str(direction)]
                    and sum(b["onboard_count"] for b in buses) == context["onboard"][str(direction)]
                    and sum(q[2] for q in c["queue_by_station"] if q[1] == direction) == context["queues"][str(direction)]):
                raise ValueError("Passive queues/fleet counts differ from original causal context")
        if not np.isclose(np.mean(c["route_speed_mps"]), context["route_speed_mean_mps"], rtol=0, atol=1e-12):
            raise ValueError("Passive route speeds differ from original context")
        for bus in c["active_buses"]:
            if not (bus["launch_time_s"] <= c["time_s"] and bus["elapsed_s"] == c["time_s"]-bus["launch_time_s"]
                    and 0 <= bus["remaining_fraction"] <= 1
                    and bus["load_fraction"] == bus["onboard_count"]/bus["capacity"]
                    and np.isclose(bus["predicted_remaining_s"], max(bus["mean_total_s"]*bus["remaining_fraction"], 1.), rtol=0, atol=1e-10)
                    and np.all(np.isfinite([v for v in bus.values()]))):
                raise ValueError("Invalid preaction active-trip context")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(shared.native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = shared.native.NativeRoutingTracker
    torch.set_num_threads(1)
    rows = [r for r in prior.training_rows(json.loads(args.training_data.read_text()))
        if (r["root"], r["scenario"]) == (args.seed, args.scenario)]
    registered = prior.source.source_record(args.seed, args.scenario)
    if len(rows) != 6 or {r["scene_seed"] for r in rows} != {registered["scene_seed"]}:
        raise ValueError("Passive audit must reuse the exact registered value-fitting scene")
    model = json.loads(args.models.read_text())[str(args.seed)]
    shared.source.incumbent.source.distributional.validate_model(model)
    _, checkpoint = shared.native.load_source(args.seed)
    row, plan = shared.episode(args.seed, args.scenario, registered["scene_seed"], raw_directory(args.output)/"audit",
        checkpoint, model, PROBES[-1], plan_class=NativeValueContextAudit)
    shared.qualify(row, {**registered, "probe": row["probe"]})
    qualify_contexts(plan.contexts, rows, registered)
    row["policy_trace"] = shared.policy_trace(row["supply"])
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "scene_seed": registered["scene_seed"], "software_qualified": True,
        "training_updates": 0, "native_steps": 61380, "observer": row, "contexts": plan.contexts})
    print("NATIVE_VALUE_CONTEXT_SOURCE_QUALIFIED", flush=True)


if __name__ == "__main__":
    main()
