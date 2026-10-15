#!/usr/bin/env python3
"""Read-only online-prefix holding and APC passenger-flow forecast validation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
import pandas as pd

from freqduet_closed_loop_forecast import FlowHistory, HoldingHistory, fluid_rollout
from freqduet_fleet_pool_forecast import freeze_pool
import run_freqduet_v40_uncertainty as uncertainty


class ClosedLoopObserver(uncertainty.IntervalObserver):
    def __init__(self, target, detailed):
        super().__init__(target, detailed)
        self.flows, self.holding_history = FlowHistory(), HoldingHistory()
        self.closed_snapshots = {}

    def capture_flow(self, runner, label, pool):
        env, now = runner.env, pool["time_s"]
        pool["flows"] = self.flows.freeze(pool)
        pool["holding_model"] = self.holding_history.freeze(now)
        for vehicle, bus in zip(pool["vehicles"], env.bus_all):
            vehicle.update(target_headway_s=float(getattr(bus, "_target_headway", 360.0)),
                           previous_hold_s=float(bus.last_action_s),
                           predecessor_trip=getattr(bus, "forward_predecessor_trip_id", None))
        targets = {int(trip.launch_turn): float(trip.target_headway) for trip in env.timetables}
        for trip in pool["commitments"]:
            trip["target_headway_s"] = targets[trip["trip_id"]]
        recorder = env.headway_events
        pool["last_arrivals"] = {f"{direction}:{station}": dict(time_s=time,
            trip_id=int(recorder._last_arrival_trip[(station, direction)]))
            for (station, direction), time in recorder._last_arrival.items()}
        pool["last_departures"] = {f"{direction}:{station}": dict(entry)
                                   for (station, direction), entry in recorder._last_departure.items()}
        intervals = {}
        for vehicle in pool["vehicles"]:
            if vehicle["on_route"]:
                predicted = now + uncertainty.remaining_trip_s(pool, vehicle)
                intervals[str(vehicle["trip_id"])] = self.intervals.interval(now, predicted)
        self.closed_snapshots[label] = dict(pool=pool, return_intervals=intervals,
            forecasts={f"{mode}_{horizon}": fluid_rollout(pool, horizon, mode)
                       for mode in ("historical", "response") for horizon in (900, 1800)})

    def attach(self, runner):
        super().attach(runner)
        env, original = runner.env, runner.env.step

        def step(action, *args, **kwargs):
            before = {bus.bus_id: (bus.trip_id, bool(bus.direction), bus.last_board_time,
                                   len(bus.passengers), bus.next_station.station_id) for bus in env.bus_all}
            result = original(action, *args, **kwargs)
            now = float(env.current_time)
            for bus in env.bus_all:
                previous = before.get(bus.bus_id)
                if bus.last_board_time == now - env.time_step and (previous is None or previous[2] != bus.last_board_time):
                    same_trip = previous is not None and previous[0] == bus.trip_id
                    direction = previous[1] if same_trip else bool(bus.direction)
                    preload = previous[3] if same_trip else 0
                    boarded = int(bus.last_board_count)
                    alighted = preload + boarded - len(bus.passengers)
                    self.flows.arrival(direction, int(bus.last_board_station_id), float(bus.last_board_time),
                                       boarded, alighted, preload, int(bus.last_left_behind_count))
            if now in (7200, 21600, 36000) and not env.done:
                self.capture_flow(runner, f"clock_{int(now)}", freeze_pool(runner, self.telemetry, None, None))
            return result

        env.step = step

    def observe(self, runner, obs, key, previous, action):
        existing = set(self.decisions)
        super().observe(runner, obs, key, previous, action)
        for stage in set(self.decisions) - existing:
            self.capture_flow(runner, stage, self.decisions[stage]["fleet_pool"])
        bus = runner._bus_for_agent(key)
        context = runner._pending_lower_action_context[int(key)]
        if (not runner._lower_terminal_action_masked(bus) and context.evidence_valid
                and not runner._fixed_expert_active):
            values = dict(forward_gap_s=float(context.forward_headway_s or 0.0),
                          target_headway_s=float(context.target_headway_s),
                          follower_gap_s=float(context.follower_departure_gap_s or 0.0),
                          follower_valid=bool(context.follower_evidence_valid),
                          load_ratio=len(bus.passengers) / float(bus.capacity), previous_hold_s=float(previous),
                          station_phase=bus.effective_station.index(bus.last_station) / (len(bus.effective_station) - 1),
                          direction=bool(bus.direction), time_s=float(runner.env.current_time))
            self.holding_history.record(values["time_s"], values, float(np.asarray(action).item()))


def items():
    return [(seed, scenario, "reference") for seed, scenario in uncertainty.items()] + [
        (933013, 45007, arm) for arm in ("first_only", "second_only", "joint")]


def run_item(arguments):
    root, v37, seed, scenario, arm = arguments
    observed = uncertainty.shadow.observed
    original, instance = observed.Observer, []

    class CaptureObserver(ClosedLoopObserver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instance.append(self)

    observed.Observer = CaptureObserver
    try:
        if arm == "reference":
            bank = pd.read_csv(v37 / "records" / f"{uncertainty.shadow.joint.shard_name(seed, scenario, 7200, 0)}.csv")
            first, second = int(bank.first_actor_s.iloc[0]), None
        else:
            first, second = {"first_only": (15, 0), "second_only": (0, 45), "joint": (15, 45)}[arm]
        label = f"seed{seed}_eval{scenario}_{arm}"
        observed.replay(root, v37, seed, scenario, 7200, label, first, second, True)
    finally:
        observed.Observer = original
    cell = root / "observed" / label
    (cell / "closed_snapshots.json").write_text(json.dumps(instance[0].closed_snapshots) + "\n")
    pd.DataFrame(instance[0].intervals.rows).to_csv(cell / "interval_predictions.csv", index=False)
    return label


def batch(root, v37, start, end, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        completed = list(pool.map(run_item, [(root, v37, *item) for item in items()[start:end]]))
    export = root / "exports" / f"batch_{start}_{end}"
    export.mkdir(parents=True, exist_ok=True)
    (export / "closed_done.json").write_text(json.dumps(dict(completed=completed,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")


def at(timeline, clock):
    index = np.searchsorted(timeline[:, 0], clock, side="right") - 1
    return timeline[index, 1:] if index >= 0 else np.zeros(3)


def aggregate(root):
    rows = []
    for seed, scenario, arm in items():
        cell = root / "observed" / f"seed{seed}_eval{scenario}_{arm}"
        snapshots = json.loads((cell / "closed_snapshots.json").read_text())
        timeline = np.load(cell / "exposure.npz")["timeline"]
        holds = pd.read_csv(cell / "holds.csv")
        for label, capture in snapshots.items():
            now = capture["pool"]["time_s"]
            model = capture["pool"]["holding_model"]
            if model["latest_time_s"] is not None and model["latest_time_s"] >= now:
                raise ValueError("holding-response fit used simultaneous or future decisions")
            for name, forecast in capture["forecasts"].items():
                mode, horizon = name.rsplit("_", 1)
                horizon = int(horizon)
                actual = at(timeline, now + horizon) - at(timeline, now)
                actual_hold = np.maximum(0, np.minimum(now + horizon - 1, holds.time_s + holds.action_s)
                                          - np.maximum(now - 1, holds.time_s)).sum()
                outcomes = dict(waiting_person_seconds=actual[0], onboard_person_seconds=actual[1],
                                backlog_trip_seconds=actual[2], holding_vehicle_seconds=float(actual_hold))
                row = dict(train_seed=seed, eval_seed=scenario, arm=arm, origin=label,
                           origin_s=now, horizon_s=horizon, mode=mode, holding_samples=model["samples"])
                for metric, truth in outcomes.items():
                    row[f"actual_{metric}"] = float(truth)
                    row[f"predicted_{metric}"] = forecast[metric]
                    row[f"error_{metric}"] = forecast[metric] - truth
                rows.append(row)
    frame = pd.DataFrame(rows)
    reference = frame.loc[(frame.arm == "reference") & frame.origin.str.startswith("clock_")]
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    result = []
    metrics = ("waiting_person_seconds", "onboard_person_seconds", "backlog_trip_seconds", "holding_vehicle_seconds")
    for (mode, horizon), group in reference.groupby(["mode", "horizon_s"]):
        for metric in metrics:
            result.append(dict(mode=mode, horizon_s=horizon, metric=metric, contexts=len(group),
                               mae=float(group[f"error_{metric}"].abs().mean()),
                               bias=float(group[f"error_{metric}"].mean()),
                               mean_actual=float(group[f"actual_{metric}"].mean())))
    pd.DataFrame(result).to_csv(summary / "prediction_errors.csv", index=False)
    record = root / "calibration_records"
    record.mkdir(parents=True, exist_ok=True)
    frame.to_csv(record / "closed_predictions.csv", index=False)
    # Same-clock four-arm responses are diagnostic; the two forced actions are
    # not supplied to a forecast made before they have actually occurred.
    dominant = frame.loc[(frame.train_seed == 933013) & (frame.eval_seed == 45007) & (frame.origin == "clock_7200")]
    dominant.to_csv(summary / "dominant_preaction.csv", index=False)
    status = dict(status="complete_causal_closed_loop_shadow", days=4, matched_replays=7,
                  reference_origins=12, policy_changed=False, independent_confirmation=False,
                  actor_model="past-decision ridge response surrogate, not the original neural actor",
                  passenger_model="APC plus observed left-behind flow conservation and stop alighting fractions",
                  scope="15/30 min holding, waiting/onboard exposure, and backlog; no learned tail value")
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(pd.DataFrame(result).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v37-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "batch":
        batch(args.out_root, args.v37_root, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root)
