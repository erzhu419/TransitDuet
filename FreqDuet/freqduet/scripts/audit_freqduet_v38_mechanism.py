#!/usr/bin/env python3
"""Four-arm event accounting for V37's dominant development context."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_freqduet_v36_joint_space import PRIMARY


ARMS = ("reference", "first_only", "second_only", "joint")


def cumulative_at(timeline, time):
    index = np.searchsorted(timeline[:, 0], time, side="right") - 1
    return timeline[index, 1:] if index >= 0 else np.zeros(3)


def contrast_events(reference, candidate):
    launches = reference["launches"].merge(candidate["launches"], on="trip_id", suffixes=("_ref", "_new"),
                                           validate="one_to_one")
    for field in ("time_s", "scheduled_s", "readiness_s"):
        launches[f"delta_{field}"] = launches[f"{field}_new"] - launches[f"{field}_ref"]
    changed = launches.loc[(launches.bus_id_new != launches.bus_id_ref)
                           | (launches.delta_time_s != 0) | (launches.delta_scheduled_s != 0)]
    launch_fields = ["trip_id", "bus_id_ref", "bus_id_new", "time_s_ref", "time_s_new",
                     "delta_time_s", "delta_scheduled_s", "delta_readiness_s"]
    holds = reference["holds"].merge(candidate["holds"], on=["trip_id", "station_id"], how="outer",
                                     suffixes=("_ref", "_new"), validate="one_to_one")
    holds["delta_action_s"] = holds.action_s_new.fillna(0) - holds.action_s_ref.fillna(0)
    changed_holds = holds.loc[holds.delta_action_s != 0]
    totals = changed_holds.groupby("trip_id").delta_action_s.sum()
    return dict(launches_changed=len(changed), vehicles_reassigned=int((launches.bus_id_new != launches.bus_id_ref).sum()),
                first_changed_launches=changed.sort_values("time_s_ref")[launch_fields].head(8).to_dict("records"),
                largest_readiness_reductions=launches.nsmallest(6, "delta_readiness_s")[launch_fields].to_dict("records"),
                holding_commands_changed=len(changed_holds),
                unmatched_holding_decisions=int(holds.action_s_new.isna().sum() + holds.action_s_ref.isna().sum()),
                first_changed_holds=changed_holds.sort_values("time_s_ref")[
                    ["trip_id", "station_id", "time_s_ref", "time_s_new", "action_s_ref", "action_s_new"]].head(10).to_dict("records"),
                largest_trip_hold_reductions=[dict(trip_id=int(trip), delta_s=float(delta))
                                              for trip, delta in totals.nsmallest(6).items()])


def aggregate(root):
    traces = {}
    for arm in ARMS:
        cell = root / "observed" / arm
        snapshot = json.loads((cell / "snapshot.json").read_text())
        if not snapshot["matches_v37"]:
            raise ValueError("read-only tracing did not reproduce V37")
        traces[arm] = dict(snapshot=snapshot, timeline=np.load(cell / "exposure.npz")["timeline"],
                           **{name: pd.read_csv(cell / f"{name}.csv")
                              for name in ("launches", "terminals", "holds", "denied")})
    reference = traces["reference"]
    decisions = reference["snapshot"]["decisions"]
    target_trips = [int(decisions[key]["identity"]["trip_id"]) for key in ("first", "second")]
    target_buses = [int(decisions[key]["identity"]["bus_id"]) for key in ("first", "second")]
    returns = reference["terminals"].set_index("trip_id").loc[target_trips].time_s
    first_time = decisions["first"]["identity"]["time_s"]
    second_time = decisions["second"]["identity"]["time_s"]
    cuts = [first_time, second_time, float(returns.max()), np.inf]
    stage_names = ("before_first", "between_decisions", "after_second_before_reference_returns", "after_reference_returns")
    metrics = [*PRIMARY, "holding_vehicle_seconds", "commanded_holding_vehicle_seconds",
               "holding_passenger_seconds", "commanded_holding_passenger_seconds"]
    arms = {}
    for arm, data in traces.items():
        episode = data["snapshot"]["episode"]
        command_sum = float(data["holds"].action_s.sum())
        if command_sum != episode["commanded_holding_vehicle_seconds"]:
            raise ValueError(f"{arm}: holding event sum differs from commanded metric")
        cumulative = np.stack([cumulative_at(data["timeline"], cut) for cut in cuts])
        stages = np.diff(np.vstack([np.zeros(3), cumulative]), axis=0)
        ref_cumulative = np.stack([cumulative_at(reference["timeline"], cut) for cut in cuts])
        ref_stages = np.diff(np.vstack([np.zeros(3), ref_cumulative]), axis=0)
        launch_sequence = data["launches"].loc[(data["launches"].bus_id.isin(target_buses))
                                               & (data["launches"].time_s >= first_time)]
        arms[arm] = dict(metrics={metric: episode[metric] for metric in metrics},
                         holding_event_sum_s=command_sum,
                         target_terminal_returns=data["terminals"].loc[
                             data["terminals"].trip_id.isin(target_trips)].to_dict("records"),
                         target_vehicle_next_launches=launch_sequence.groupby("bus_id").head(3).to_dict("records"),
                         exposure_delta_vs_reference={name: dict(zip(
                             ("waiting_person_seconds", "onboard_person_seconds", "dispatch_backlog_trip_seconds"),
                             row.tolist())) for name, row in zip(stage_names, stages - ref_stages)},
                         event_contrast=contrast_events(reference, data),
                         second_observed_state=data["snapshot"]["decisions"]["second"]["features"])
    interaction = {metric: arms["joint"]["metrics"][metric] - arms["first_only"]["metrics"][metric]
                   - arms["second_only"]["metrics"][metric] + arms["reference"]["metrics"][metric]
                   for metric in metrics}
    result = dict(scope="posthoc four-arm mechanism audit; not causal mediation identification",
                  target_trips=target_trips, target_buses=target_buses,
                  common_stage_cuts_s=cuts[:-1],
                  first_observed_state=decisions["first"]["features"],
                  interaction_joint_minus_singles_plus_reference=interaction, arms=arms,
                  exposure_units="tick-integrated counts; diagnostic, not the restricted journey estimator")
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "mechanism.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="mechanism_complete", interaction=interaction,
                          joint=arms["joint"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    aggregate(parser.parse_args().out_root)
