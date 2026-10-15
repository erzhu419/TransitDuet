"""Causal APC flow and holding-response surrogate for short-window validation."""

from collections import defaultdict, deque
import copy
import heapq
import math

import numpy as np


ACTIONS = np.array([0, 5, 10, 15, 20, 30, 45], dtype=float)


def holding_features(values):
    gap = values["forward_gap_s"] / max(values["target_headway_s"], 1.0) - 1
    follower = values["follower_gap_s"] / max(values["target_headway_s"], 1.0) - 1
    valid, load = float(values["follower_valid"]), values["load_ratio"]
    phase = values["station_phase"]
    return np.array([1, gap, gap * gap, valid * follower, valid, load, gap * load,
                     values["previous_hold_s"] / 45, phase, phase * phase,
                     float(values["direction"]), values["time_s"] / 50400], dtype=float)


class HoldingHistory:
    def __init__(self):
        self.rows = deque(maxlen=4096)

    def record(self, time, features, action):
        self.rows.append((float(time), holding_features(features), float(action) / 45))

    def freeze(self, now):
        past = [row for row in self.rows if row[0] < now]
        if len(past) < 32:
            return dict(coef=None, samples=len(past), latest_time_s=max((row[0] for row in past), default=None))
        design = np.stack([row[1] for row in past])
        labels = np.array([row[2] for row in past])
        scale = np.maximum(np.sqrt(np.mean(design * design, axis=0)), 1e-8)
        design /= scale
        penalty = np.eye(design.shape[1]) * 10
        penalty[0, 0] = 0
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ labels) / scale
        return dict(coef=coef.tolist(), samples=len(past), latest_time_s=past[-1][0])


class FlowHistory:
    def __init__(self):
        self.last = {}
        self.gaps = defaultdict(lambda: deque(maxlen=32))
        self.alighting = defaultdict(lambda: deque(maxlen=32))

    def arrival(self, direction, station, time, boarded, alighted, preload, left_behind):
        key = (bool(direction), int(station))
        previous = self.last.get(key)
        if previous is not None and time > previous[0]:
            inferred_arrivals = max(0.0, boarded + left_behind - previous[1])
            self.gaps[key].append((inferred_arrivals, time - previous[0]))
        self.last[key] = (float(time), float(left_behind))
        self.alighting[key].append((float(alighted), float(preload)))

    def freeze(self, pool):
        now, output = pool["time_s"], {}
        for direction in (True, False):
            rows = []
            for station in pool["routes"][str(direction)]["station_ids"]:
                key = (direction, station)
                gaps, alighting = self.gaps[key], self.alighting[key]
                rate = sum(row[0] for row in gaps) / max(sum(row[1] for row in gaps), 1)
                last = self.last.get(key)
                queue = last[1] + rate * (now - last[0]) if last is not None else 0.0
                fraction = sum(row[0] for row in alighting) / max(sum(row[1] for row in alighting), 1)
                rows.append(dict(rate_per_s=rate, queue=queue, alight_fraction=min(fraction, 1.0),
                                 observed_gaps=len(gaps), last_observed_s=last[0] if last else None))
            output[str(direction)] = rows
        return output


def response_action(model, values, historical):
    if model["coef"] is None:
        return historical
    predicted = float(holding_features(values) @ np.asarray(model["coef"])) * 45
    return float(ACTIONS[np.argmin(np.abs(ACTIONS - predicted))])


def fluid_rollout(observed, horizon_s, holding_mode):
    """Propagate service, queues, onboard loads, holding and physical reuse."""
    now, dt = observed["time_s"], observed["dt_s"]
    end, clock = now + horizon_s, now
    vehicles = {row["bus_id"]: dict(row) for row in observed["vehicles"]}
    flows = copy.deepcopy(observed["flows"])
    pending = {row["trip_id"]: dict(row) for row in observed["commitments"]}
    pools, events, wakes = {True: [], False: []}, [], set()
    last_dispatch = {direction: observed["last_dispatch"][str(direction)] for direction in pools}
    arrivals, departures = copy.deepcopy(observed["last_arrivals"]), copy.deepcopy(observed["last_departures"])
    launches, returns, commands = [], [], []
    waiting, onboard, holding = 0.0, 0.0, 0.0
    alighted_total = 0.0
    max_load = max((vehicle["load"] for vehicle in vehicles.values()), default=0.0)
    serial = 0

    def tick(time):
        return math.ceil(time / dt) * dt

    def schedule(time, kind, bus_id=None):
        nonlocal serial
        if kind == "dispatch":
            if time in wakes:
                return
            wakes.add(time)
        priority = 0 if kind == "available" else 1 if kind == "dispatch" else 2
        heapq.heappush(events, (tick(time), priority, serial, kind, bus_id))
        serial += 1

    def key(direction, station):
        return f"{direction}:{station}"

    def route_of(vehicle):
        return observed["routes"][str(vehicle["direction"])]

    def integrate(time):
        nonlocal clock, waiting, onboard
        duration = time - clock
        for route_flows in flows.values():
            for stop in route_flows:
                waiting += stop["queue"] * duration + .5 * stop["rate_per_s"] * duration * duration
                stop["queue"] += stop["rate_per_s"] * duration
        onboard += sum(vehicle["load"] for vehicle in vehicles.values() if vehicle["on_route"]) * duration
        clock = time

    def begin_travel(vehicle, time, partial=False):
        route = route_of(vehicle)
        station = vehicle["station_index"]
        fraction = 1.0
        if partial:
            end_position = sum(route["lengths_m"][:station + 1])
            fraction = np.clip((end_position - vehicle["progress_m"]) / route["lengths_m"][station], 0, 1)
        travel = max(float(fraction) * route["travel_s"][station], dt)
        vehicle.update(phase="travel", travel_start_s=time, travel_end_s=time + travel,
                       start_progress_m=vehicle["progress_m"], end_progress_m=sum(route["lengths_m"][:station + 1]))
        schedule(time + travel, "arrival", vehicle["bus_id"])

    def execute_hold(vehicle, time, action):
        nonlocal holding
        action = float(max(action, 0))
        commands.append(dict(time_s=time, bus_id=vehicle["bus_id"], trip_id=vehicle["trip_id"],
                             direction=vehicle["direction"], station_id=route_of(vehicle)["station_ids"][vehicle["station_index"]],
                             action_s=action))
        holding += min(action, max(end - time, 0))
        vehicle.update(previous_hold_s=action, phase="commanded_hold")
        schedule(time + max(action, dt), "departure", vehicle["bus_id"])

    def dispatch(time):
        for trip_id in list(pending):
            trip = pending[trip_id]
            direction = trip["direction"]
            if trip["scheduled_s"] > time:
                continue
            eligible = last_dispatch[direction] + trip["min_gap_s"]
            if eligible > time:
                schedule(eligible, "dispatch")
                continue
            if trip["first_eligible_s"] is None:
                trip["first_eligible_s"] = time
            if not pools[direction]:
                continue
            _, bus_id = heapq.heappop(pools[direction])
            vehicle = vehicles[bus_id]
            vehicle.update(direction=direction, trip_id=trip_id, on_route=True, load=0.0,
                           station_index=0, progress_m=0.0, previous_hold_s=0.0,
                           target_headway_s=trip["target_headway_s"], phase="service", predecessor_trip=None)
            launches.append(dict(time_s=time, trip_id=trip_id, bus_id=bus_id, direction=direction,
                                 scheduled_s=trip["scheduled_s"], readiness_s=time - trip["first_eligible_s"]))
            last_dispatch[direction] = time
            schedule(time + dt, "hold", bus_id)
            del pending[trip_id]

    for vehicle in vehicles.values():
        if not vehicle["on_route"]:
            heapq.heappush(pools[vehicle["direction"]], (vehicle["ready_since_s"], vehicle["bus_id"]))
        elif vehicle["phase"] == "travel":
            begin_travel(vehicle, now, partial=True)
        elif vehicle["phase"] == "commanded_hold":
            remaining = max(vehicle["remaining_stop_s"] - dt, 0.0)
            holding += min(remaining, horizon_s)
            schedule(now + remaining, "departure", vehicle["bus_id"])
        elif vehicle["phase"] == "current_decision":
            execute_hold(vehicle, now, observed["decision"]["actor_action_s"])
        else:
            schedule(now + vehicle["remaining_service_s"], "hold", vehicle["bus_id"])
    for trip in pending.values():
        schedule(max(now, trip["scheduled_s"]), "dispatch")
    while events:
        time, _, _, kind, bus_id = heapq.heappop(events)
        if time > end:
            break
        integrate(time)
        if kind == "dispatch":
            wakes.remove(time)
            dispatch(time)
            continue
        vehicle = vehicles[bus_id]
        route = route_of(vehicle)
        station = vehicle["station_index"]
        if kind == "available":
            heapq.heappush(pools[vehicle["direction"]], (vehicle["ready_since_s"], bus_id))
            schedule(time, "dispatch")
        elif kind == "departure":
            departures[key(vehicle["direction"], route["station_ids"][station])] = dict(time_s=time, trip_id=vehicle["trip_id"])
            begin_travel(vehicle, time)
        elif kind == "arrival":
            station += 1
            vehicle.update(station_index=station, progress_m=sum(route["lengths_m"][:station]))
            station_key = key(vehicle["direction"], route["station_ids"][station])
            vehicle["predecessor_trip"] = arrivals.get(station_key, {}).get("trip_id")
            arrivals[station_key] = dict(time_s=time, trip_id=vehicle["trip_id"])
            stop = flows[str(vehicle["direction"])][station]
            alight = vehicle["load"] if station == len(route["station_ids"]) - 1 else vehicle["load"] * stop["alight_fraction"]
            board = 0.0 if station == len(route["station_ids"]) - 1 else min(
                stop["queue"], vehicle["capacity"] - vehicle["load"] + alight)
            vehicle["load"] += board - alight
            alighted_total += alight
            max_load = max(max_load, vehicle["load"])
            stop["queue"] -= board
            if station == len(route["station_ids"]) - 1:
                returns.append(dict(time_s=time, trip_id=vehicle["trip_id"], bus_id=bus_id))
                vehicle.update(on_route=False, load=0.0, direction=not vehicle["direction"], ready_since_s=time)
                schedule(time + dt, "available", bus_id)
            else:
                vehicle["phase"] = "service"
                dwell = max(alight, 2 * board) + 4
                schedule(time + dwell + dt, "hold", bus_id)
        elif kind == "hold":
            station_key = key(vehicle["direction"], route["station_ids"][station])
            predecessor = vehicle["predecessor_trip"]
            departure = departures.get(station_key)
            valid = (station > 0 and not route["holding_mask"][station]
                     and departure is not None and departure["trip_id"] == predecessor)
            action = 0.0
            if valid:
                follower_gaps = []
                for follower in vehicles.values():
                    if not follower["on_route"] or follower["direction"] != vehicle["direction"] or follower["bus_id"] == bus_id:
                        continue
                    progress = follower["progress_m"]
                    if follower["phase"] == "travel":
                        fraction = np.clip((time - follower["travel_start_s"]) /
                                           max(follower["travel_end_s"] - follower["travel_start_s"], dt), 0, 1)
                        progress = follower["start_progress_m"] + fraction * (follower["end_progress_m"] - follower["start_progress_m"])
                    gap = vehicle["progress_m"] - progress
                    if gap > 0:
                        mean_speed = sum(route["lengths_m"]) / max(sum(route["travel_s"]), 1)
                        follower_gaps.append(gap / mean_speed + route["dwell_s"][station])
                values = dict(forward_gap_s=time - departure["time_s"], target_headway_s=vehicle["target_headway_s"],
                              follower_gap_s=min(follower_gaps, default=0.0), follower_valid=bool(follower_gaps),
                              load_ratio=vehicle["load"] / vehicle["capacity"], previous_hold_s=vehicle["previous_hold_s"],
                              station_phase=station / (len(route["station_ids"]) - 1), direction=vehicle["direction"], time_s=time)
                historical = route["hold_s"][station]
                action = historical if holding_mode == "historical" else response_action(observed["holding_model"], values, historical)
            execute_hold(vehicle, time, action)
    integrate(end)
    backlog = sum(max(0, min(row["time_s"], end) - max(now, row["scheduled_s"])) for row in launches)
    backlog += sum(max(0, end - max(now, row["scheduled_s"])) for row in pending.values())
    return dict(waiting_person_seconds=waiting, onboard_person_seconds=onboard,
                holding_vehicle_seconds=holding, backlog_trip_seconds=backlog,
                launches=launches, returns=returns, commands=commands,
                vehicles_created=0, max_load=max_load, alighted_total=alighted_total,
                final_queue=sum(stop["queue"] for rows in flows.values() for stop in rows),
                final_onboard=sum(vehicle["load"] for vehicle in vehicles.values() if vehicle["on_route"]))
