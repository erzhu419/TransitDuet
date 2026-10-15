#!/usr/bin/env python3
"""Read-only fleet-pool forecast validation on the frozen V37 cohort."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_freqduet_v38_observed_replay as observed
from freqduet_fleet_pool_forecast import PastFleetTelemetry, freeze_pool, forecast_pool, joint_resource_grid


joint = observed.joint


class PoolObserver(observed.Observer):
    def __init__(self, target, detailed):
        super().__init__(target, detailed)
        self.telemetry = PastFleetTelemetry()

    def attach(self, runner):
        super().attach(runner)
        env, telemetry = runner.env, self.telemetry
        original_step = env.step

        def step(action, *args, **kwargs):
            now = float(env.current_time)
            before = {bus.bus_id: (bus.on_route, bus.trip_id, bool(bus.direction),
                                   float(bus.absolute_distance), bus.last_board_time,
                                   int(bus.last_station.station_id)) for bus in env.bus_all}
            result = original_step(action, *args, **kwargs)
            for bus in env.bus_all:
                previous = before.get(bus.bus_id)
                if previous and previous[0] and previous[1] == bus.trip_id:
                    telemetry.record_movement(previous[2], previous[5],
                                              abs(float(bus.absolute_distance) - previous[3]), env.time_step)
                    if not bus.on_route:
                        telemetry.cycles[previous[2]].append(now - float(bus.launch_time))
                if bus.last_board_time == now and (not previous or previous[4] != bus.last_board_time):
                    if bus.on_route:
                        telemetry.record_stop("dwell", bool(bus.direction), int(bus.last_station.station_id),
                                              bus.last_service_dwell_s)
                if bus.last_action_time == now:
                    telemetry.record_stop("hold", bool(bus.direction), int(bus.last_action_station_id), bus.last_action_s)
            return result

        env.step = step

    def observe(self, runner, obs, key, previous, action):
        existing = set(self.decisions)
        super().observe(runner, obs, key, previous, action)
        for decision in set(self.decisions) - existing:
            capture = self.decisions[decision]
            snapshot = freeze_pool(runner, self.telemetry, capture["identity"], float(np.asarray(action).item()))
            capture["fleet_pool"] = snapshot
            capture["pool_forecasts"] = {str(horizon): forecast_pool(snapshot, horizon_s=horizon)
                                         for horizon in (900, 1800, 3600)}
            if decision == "first":
                capture["joint_resource_grid"] = joint_resource_grid(snapshot, joint.ACTIONS)


def items():
    references = [(*context, "reference") for context in itertools.product(joint.SEEDS, joint.SCENARIOS, joint.TARGET_TIMES)]
    return references + [(933013, 45007, 7200, arm) for arm in ("first_only", "second_only", "joint")]


def run_item(arguments):
    root, v37, seed, scenario, target, arm = arguments
    original = observed.Observer
    observed.Observer = PoolObserver
    try:
        if arm == "reference":
            bank = pd.read_csv(v37 / "records" / f"{joint.shard_name(seed, scenario, target, 0)}.csv")
            first, second = int(bank.first_actor_s.iloc[0]), None
        else:
            first, second = {"first_only": (15, 0), "second_only": (0, 45), "joint": (15, 45)}[arm]
        label = f"seed{seed}_eval{scenario}_t{target}_{arm}"
        observed.replay(root, v37, seed, scenario, target, label, first, second, True)
    finally:
        observed.Observer = original
    return label


def batch(root, v37, start, end, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        complete = list(pool.map(run_item, [(root, v37, *item) for item in items()[start:end]]))
    export = root / "exports" / f"batch_{start}_{end}"
    export.mkdir(parents=True, exist_ok=True)
    (export / "shadow_done.json").write_text(json.dumps(dict(completed=complete,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")


def holding_components(snapshot, vehicle, commands, actual_return):
    route = snapshot["routes"][str(vehicle["direction"])]
    station, now = vehicle["station_index"], snapshot["time_s"]
    current = 0.0
    if vehicle["phase"] == "current_decision":
        current = snapshot["decision"]["actor_action_s"]
    elif vehicle["phase"] == "commanded_hold":
        current = max(vehicle["remaining_stop_s"] - snapshot["dt_s"], 0.0)
    elif vehicle["phase"] == "service":
        current = route["hold_s"][station]
    predicted = current + sum(route["hold_s"][station + 1:-1])
    actual = sum(max(0.0, min(actual_return, row["time_s"] + row["action_s"])
                     - max(now - snapshot["dt_s"], row["time_s"])) for row in commands)
    return float(predicted), float(actual)


def aggregate(root):
    return_rows, launch_rows, identity_rows, resource_rows, target_rows = [], [], [], [], []
    dominant_first, source_commits = None, set()
    for seed, scenario, target, arm in items():
        label = f"seed{seed}_eval{scenario}_t{target}_{arm}"
        cell = root / "observed" / label
        record = json.loads((cell / "snapshot.json").read_text())
        source_commits.add(record["source_commit"])
        if (seed, scenario, target) == (933013, 45007, 7200):
            first_pool = record["decisions"]["first"]["fleet_pool"]
            if dominant_first is not None:
                joint.assert_same_state(dominant_first, first_pool, "fleet prefix before any intervention")
            dominant_first = first_pool
        actual_returns = pd.read_csv(cell / "terminals.csv").set_index("trip_id")
        actual_launches = pd.read_csv(cell / "launches.csv").set_index("trip_id")
        holding = pd.read_csv(cell / "holds.csv")
        commands = {trip: group.to_dict("records") for trip, group in holding.groupby("trip_id")}
        for stage, capture in record["decisions"].items():
            if not record["matches_v37"]:
                raise ValueError("shadow instrumentation differs from frozen V37")
            snapshot = capture["fleet_pool"]
            current = {row["trip_id"]: row for row in snapshot["vehicles"] if row["on_route"]}
            meta = dict(train_seed=seed, eval_seed=scenario, target_time_s=target, arm=arm, stage=stage)
            target_trip = capture["identity"]["trip_id"]
            target_actual = actual_returns.loc[target_trip].time_s
            target_predicted = next(row["time_s"] for row in capture["pool_forecasts"]["3600"]["returns"]
                                    if row["trip_id"] == target_trip)
            target_rows.append(dict(meta,
                pooled_error_s=float(target_predicted - target_actual),
                v38_proxy_error_s=float(snapshot["time_s"] + capture["features"]["return_eta_proxy_s"] - target_actual)))
            for horizon, forecast in capture["pool_forecasts"].items():
                for row in forecast["returns"]:
                    if row["trip_id"] in current:
                        actual = actual_returns.loc[row["trip_id"]]
                        vehicle = current[row["trip_id"]]
                        predicted_hold, actual_hold = holding_components(
                            snapshot, vehicle, commands[row["trip_id"]], float(actual.time_s))
                        return_rows.append(dict(meta, horizon_s=int(horizon), trip_id=row["trip_id"],
                            direction=vehicle["direction"], phase=vehicle["phase"],
                            predicted_hold_s=predicted_hold, actual_hold_s=actual_hold,
                            holding_error_s=predicted_hold - actual_hold,
                            nonholding_error_s=float(row["time_s"] - actual.time_s) - predicted_hold + actual_hold,
                            remaining_actual_s=float(actual.time_s - snapshot["time_s"]),
                            error_s=float(row["time_s"] - actual.time_s)))
                for row in snapshot["commitments"]:
                    trip = row["trip_id"]
                    if row["scheduled_s"] > snapshot["time_s"] + int(horizon):
                        continue
                    predicted = next((launch for launch in forecast["launches"] if launch["trip_id"] == trip), None)
                    actual = actual_launches.loc[trip]
                    launched_in_window = bool(actual.time_s <= snapshot["time_s"] + int(horizon))
                    launch_rows.append(dict(meta, horizon_s=int(horizon), trip_id=trip,
                        predicted_launched=predicted is not None, actual_launched=launched_in_window,
                        error_s=float(predicted["time_s"] - actual.time_s) if predicted else np.nan,
                        vehicle_matches=predicted["bus_id"] == actual.bus_id if predicted else False,
                        readiness_error_s=float(predicted["readiness_s"] - actual.readiness_s) if predicted else np.nan))
            if stage == "first":
                for row in capture["joint_resource_grid"]:
                    predicted_second = row.pop("predicted_second")
                    resource_rows.append(dict(meta, **row))
                    if row["first_action_s"] == snapshot["decision"]["actor_action_s"] and row["second_action_s"] == 0:
                        actual_second = record["decisions"]["second"]["identity"]
                        identity_rows.append(dict(meta, predicted=predicted_second, actual=actual_second,
                            matches=predicted_second is not None and all(predicted_second[field] == actual_second[field]
                                                                         for field in ("trip_id", "bus_id")),
                            time_error_s=predicted_second["time_s"] - actual_second["time_s"] if predicted_second else None))
    returns, launches = pd.DataFrame(return_rows), pd.DataFrame(launch_rows)
    # Reference contexts are the calibration cohort; interventions stay separate.
    ref_returns = returns.loc[(returns.arm == "reference") & (returns.horizon_s == 3600)]
    ref_launches = launches.loc[launches.arm == "reference"]
    short_returns = ref_returns.loc[ref_returns.remaining_actual_s <= 900]
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    return_groups = []
    for name, group in (("all_current", ref_returns), ("actual_remaining_le_900", short_returns)):
        error = group.error_s.to_numpy()
        return_groups.append(dict(group=name, count=len(group), bias_s=float(error.mean()),
                                 mae_s=float(np.abs(error).mean()), p90_abs_s=float(np.quantile(np.abs(error), .9)),
                                 within_45s=float((np.abs(error) <= 45).mean())))
    pd.DataFrame(return_groups).to_csv(summary / "return_calibration.csv", index=False)
    target_frame = pd.DataFrame(target_rows)
    target_comparison = target_frame.loc[target_frame.arm == "reference"].groupby("train_seed").agg(
        pooled_bias_s=("pooled_error_s", "mean"), pooled_mae_s=("pooled_error_s", lambda values: values.abs().mean()),
        v38_bias_s=("v38_proxy_error_s", "mean"), v38_mae_s=("v38_proxy_error_s", lambda values: values.abs().mean()))
    target_comparison.to_csv(summary / "target_return_comparison.csv")
    component_rows = []
    for name, group in (("all_current", ref_returns), ("actual_remaining_le_900", short_returns)):
        for component in ("holding_error_s", "nonholding_error_s"):
            component_rows.append(dict(group=name, component=component, bias_s=float(group[component].mean()),
                                       mae_s=float(group[component].abs().mean())))
    pd.DataFrame(component_rows).to_csv(summary / "error_components.csv", index=False)
    worst = ref_returns.assign(abs_error=ref_returns.error_s.abs()).nlargest(8, "abs_error")
    (summary / "largest_errors.json").write_text(worst.drop(columns="abs_error").to_json(orient="records", indent=2) + "\n")
    launch_summary = ref_launches.groupby("horizon_s").apply(lambda frame: pd.Series(dict(
        count=len(frame), launch_window_accuracy=(frame.predicted_launched == frame.actual_launched).mean(),
        predicted_missing=(~frame.predicted_launched).sum(), mae_s=frame.error_s.abs().mean(),
        p90_abs_s=frame.error_s.abs().quantile(.9), readiness_mae_s=frame.readiness_error_s.abs().mean(),
        vehicle_accuracy=frame.loc[frame.predicted_launched, "vehicle_matches"].mean())))
    launch_summary.to_csv(summary / "launch_calibration.csv")
    identities = [row for row in identity_rows if row["arm"] == "reference"]
    status = dict(status="complete_observable_fleet_shadow", reference_contexts=12, matched_replays=len(items()),
                  independent_confirmation=False, policy_changed=False, passenger_value_available=False,
                  short_return_within45_rate=return_groups[1]["within_45s"],
                  short_return_resolution_pass=return_groups[1]["within_45s"] >= .9,
                  second_identity_accuracy=float(np.mean([row["matches"] for row in identities])),
                  replay_source_commits=sorted(source_commits),
                  model_boundary="current published plan; past speed/dwell/hold and cycle continuation; no future policy cascade")
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    (summary / "second_identity.json").write_text(json.dumps(identity_rows, indent=2) + "\n")
    # Full prediction/label rows remain on server; only the summary is exported.
    audit = root / "calibration_records"
    audit.mkdir(parents=True, exist_ok=True)
    returns.to_csv(audit / "returns.csv", index=False)
    launches.to_csv(audit / "launches.csv", index=False)
    pd.DataFrame(resource_rows).to_csv(audit / "joint_resource_grid.csv", index=False)
    target_frame.to_csv(audit / "target_returns.csv", index=False)
    print(json.dumps(status), flush=True)
    print(pd.DataFrame(return_groups).to_string(index=False), flush=True)
    print(launch_summary.to_string(), flush=True)
    print(pd.DataFrame(component_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v37-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "batch":
        batch(args.out_root, args.v37_root, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root)
