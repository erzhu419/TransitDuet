#!/usr/bin/env python3
"""Read-only fleet-event tracing and causal snapshots for the V37 action bank."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_freqduet_v37_joint_lower as joint
from audit_freqduet_v38_dominant_case import select_branches


FEATURE_NAMES = (
    "clock_hours", "station_fraction", "direction", "target_headway_s",
    "forward_gap_s", "follower_gap_s", "follower_valid", "load_ratio",
    "left_behind_ratio", "last_service_dwell_s", "previous_hold_s", "actor_hold_s",
    "ready_origin", "ready_return", "overdue_return_count", "overdue_return_age_s",
    "published_return_deadline_s", "has_return_departure", "remaining_distance_m",
    "observed_route_speed_mps", "return_eta_proxy_s", "return_margin_proxy_s",
    "past_mean_hold_s",
)


def observable_snapshot(runner, bus, raw, previous, actor, mean_hold):
    """Only clock, APC/AVL, route geometry and the currently published plan."""
    env = runner.env
    now, direction = float(env.current_time), bool(bus.direction)
    context = runner._pending_lower_action_context[int(bus.bus_id)]
    routes = bus.effective_route
    distances = np.asarray([route.distance for route in routes], dtype=float)
    speed = max(float(np.mean([route.speed_limit for route in routes])), 1.0)
    remaining = max(float(distances.sum()) - float(bus.travel_distance), 0.0)
    stops = int((np.cumsum(distances) > float(bus.travel_distance)).sum())
    eta = remaining / speed + stops * (float(bus.last_service_dwell_s) + mean_hold)
    pending = [float(getattr(trip, "_freqduet_scheduled_launch", trip.launch_time))
               for trip in env.timetables if not trip.launched and bool(trip.direction) != direction]
    deadline = min(pending) - now if pending else 0.0
    overdue = [now - time for time in pending if time <= now]
    return dict(zip(FEATURE_NAMES, (
        now / 3600.0, float(raw[1]) / max(len(env.stations) - 1, 1), float(direction),
        float(context.target_headway_s), float(context.forward_headway_s or 0.0),
        float(context.follower_departure_gap_s or 0.0), float(context.follower_evidence_valid),
        len(bus.passengers) / max(float(bus.capacity), 1.0),
        float(bus.last_left_behind_count) / max(float(bus.capacity), 1.0),
        float(bus.last_service_dwell_s), float(previous), float(actor),
        float(env._ready_vehicle_count(direction)), float(env._ready_vehicle_count(not direction)),
        float(len(overdue)), max(overdue, default=0.0), deadline, float(bool(pending)),
        remaining, speed, eta, deadline - eta if pending else 0.0, mean_hold)))


class Observer:
    def __init__(self, target, detailed):
        self.target, self.detailed = target, detailed
        self.decisions, self.launches, self.terminals, self.holds = {}, [], [], []
        self.denied, self.timeline = {}, []
        self.hold_sum, self.hold_count = 0.0, 0
        self.exposure = np.zeros(3, dtype=float)

    def attach(self, runner):
        env = runner.env
        original_launch, original_step = env.launch_bus, env.step
        tracker = env._upper_interval_outcome_tracker
        original_record = tracker.record_step

        def record_step(**kwargs):
            original_record(**kwargs)
            if self.detailed:
                self.exposure += float(kwargs["dt_s"]) * np.array([
                    sum(kwargs["waiting_by_direction"].values()),
                    sum(kwargs["onboard_by_direction"].values()),
                    sum(kwargs["dispatch_backlog_by_direction"].values())], dtype=float)
                self.timeline.append([float(env.current_time) + float(kwargs["dt_s"]), *self.exposure])

        def launch(trip, actual_launch=None):
            result = original_launch(trip, actual_launch=actual_launch)
            if self.detailed:
                now, tid = float(env.current_time), int(trip.launch_turn)
                if result is None:
                    entry = self.denied.setdefault(tid, dict(trip_id=tid, direction=bool(trip.direction),
                                                            first_time_s=now, last_time_s=now, attempts=0))
                    entry["last_time_s"], entry["attempts"] = now, entry["attempts"] + 1
                else:
                    self.launches.append(dict(time_s=now, trip_id=tid, bus_id=int(result.bus_id),
                                              direction=bool(trip.direction),
                                              scheduled_s=float(getattr(trip, "_freqduet_scheduled_launch", trip.launch_time)),
                                              readiness_s=max(0.0, now - float(trip._freqduet_first_eligible_time))))
            return result

        def step(action, *args, **kwargs):
            now = float(env.current_time)
            before = {bus.bus_id: (bool(bus.on_route), int(bus.trip_id), bool(bus.direction)) for bus in env.bus_all}
            result = original_step(action, *args, **kwargs)
            for bus in env.bus_all:
                previous = before.get(bus.bus_id)
                if self.detailed and previous and previous[0] and not bus.on_route:
                    self.terminals.append(dict(time_s=float(bus.back_to_terminal_time), bus_id=int(bus.bus_id),
                                               trip_id=previous[1], direction=previous[2]))
                if bus.last_action_time == now:
                    hold = float(bus.last_action_s)
                    self.hold_sum += hold
                    self.hold_count += 1
                    if self.detailed:
                        self.holds.append(dict(time_s=now, bus_id=int(bus.bus_id), trip_id=int(bus.trip_id),
                                               station_id=int(bus.last_action_station_id), action_s=hold))
            return result

        tracker.record_step, env.launch_bus, env.step = record_step, launch, step

    def observe(self, runner, obs, key, previous, action):
        if len(self.decisions) == 2:
            return
        bus = runner._bus_for_agent(key)
        context = runner._pending_lower_action_context[int(key)]
        eligible = (not runner._lower_terminal_action_masked(bus)
                    and not runner._fixed_expert_active and context.evidence_valid)
        identity = dict(time_s=float(runner.env.current_time), bus_id=int(key), trip_id=int(bus.trip_id),
                        station_id=int(obs[1]), direction=bool(bus.direction))
        decision = None
        if eligible and not self.decisions and identity["time_s"] >= self.target:
            decision = "first"
        elif eligible and "first" in self.decisions and joint.second_matches(self.decisions["first"]["identity"], identity):
            decision = "second"
        if decision:
            self.decisions[decision] = dict(identity=identity, features=observable_snapshot(
                runner, bus, obs, previous, float(np.asarray(action).item()),
                self.hold_sum / max(self.hold_count, 1)))


def replay(root, v37_root, seed, scenario, target, label, first, second, detailed):
    observer = Observer(target, detailed)
    original_runtime = joint.optimized_runtime

    def observed_runtime(out_root):
        matrix, Runner, config, torch = original_runtime(out_root)

        class ObservedRunner(Runner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                observer.attach(self)

            def _lower_action_for_agent(self, obs, key, last_action=0.0, deterministic=False):
                action = super()._lower_action_for_agent(obs, key, last_action=last_action, deterministic=deterministic)
                observer.observe(self, obs, key, last_action, action)
                return action

        return matrix, ObservedRunner, config, torch

    joint.optimized_runtime = observed_runtime
    try:
        branch = joint.run_branch(root, seed, scenario, target, label, first, second)
    finally:
        joint.optimized_runtime = original_runtime
    for decision in ("first", "second"):
        joint.assert_same_state(observer.decisions[decision]["identity"], branch["captures"][decision]["identity"])
    source = pd.read_csv(v37_root / "records" / f"{joint.shard_name(seed, scenario, target, int(first))}.csv")
    second_actor = branch["captures"]["second"]["actor_action_s"]
    match = source.loc[source.second_action_s == (second_actor if second is None else second)]
    if len(match) != 1:
        raise ValueError("instrumented branch lacks a V37 control")
    joint.assert_same_state({metric: branch["episode"][metric] for metric in joint.OUTCOMES},
                            match.iloc[0][list(joint.OUTCOMES)].to_dict(), "V38_matches_V37")
    cell = root / "observed" / label
    cell.mkdir(parents=True, exist_ok=True)
    record = dict(seed=seed, scenario=scenario, target_time_s=target, first_action_s=first,
                  second_action_s=second, decisions=observer.decisions, episode=branch["episode"],
                  matches_v37=True, source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    (cell / "snapshot.json").write_text(json.dumps(record, default=lambda value: value.item()) + "\n")
    if detailed:
        for name in ("launches", "terminals", "holds"):
            pd.DataFrame(getattr(observer, name)).to_csv(cell / f"{name}.csv", index=False)
        pd.DataFrame(list(observer.denied.values())).to_csv(cell / "denied.csv", index=False)
        np.savez_compressed(cell / "exposure.npz", timeline=np.asarray(observer.timeline))
    print(f"OBSERVED_REPLAY_DONE {label} matches_v37=True", flush=True)
    return label


def feature_item(item):
    root, v37, seed, scenario, target, first = item
    label = joint.shard_name(seed, scenario, target, first)
    return replay(root, v37, seed, scenario, target, label, first, None, False)


def features(root, v37, start, end, workers):
    items = joint.pilot_items()[start:end]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        labels = list(pool.map(feature_item, [(root, v37, *item) for item in items]))
    export = root / "exports" / f"batch_{start}_{end}"
    export.mkdir(parents=True, exist_ok=True)
    (export / "features_done.json").write_text(json.dumps(dict(completed=labels)) + "\n")


def trace(root, v37, label):
    group = pd.concat([pd.read_csv(v37 / "records" / f"seed933013_eval45007_t7200_a{a}.csv")
                       for a in joint.ACTIONS], ignore_index=True)
    selected = select_branches(group)[label]
    replay(root, v37, 933013, 45007, 7200, label,
           int(selected.first_action_s), int(selected.second_action_s), True)
    export = root / "exports" / label
    export.mkdir(parents=True, exist_ok=True)
    (export / "trace_done.json").write_text(json.dumps(dict(label=label, matches_v37=True)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["trace", "features"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v37-root", type=Path, required=True)
    parser.add_argument("--label", choices=["reference", "first_only", "second_only", "joint"], default="reference")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=84)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "trace":
        trace(args.out_root, args.v37_root, args.label)
    else:
        features(args.out_root, args.v37_root, args.start, args.end, args.workers)
