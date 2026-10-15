"""Observable fleet-pool service forecast, separate from simulator state."""

from collections import defaultdict, deque
import heapq
import math

import numpy as np


class PastFleetTelemetry:
    def __init__(self):
        self.movement = defaultdict(lambda: deque(maxlen=512))
        self.dwell = defaultdict(lambda: deque(maxlen=32))
        self.hold = defaultdict(lambda: deque(maxlen=32))
        self.cycles = defaultdict(lambda: deque(maxlen=12))

    def record_movement(self, direction, station, distance, duration):
        if distance > 0:
            for key in ((bool(direction), int(station)), (bool(direction), None)):
                self.movement[key].append((float(distance), float(duration)))

    def record_stop(self, kind, direction, station, seconds):
        table = self.dwell if kind == "dwell" else self.hold
        for key in ((bool(direction), int(station)), (bool(direction), None)):
            table[key].append(float(seconds))

    def speed(self, direction, station):
        samples = self.movement[(direction, station)] or self.movement[(direction, None)]
        if not samples:
            raise ValueError("no past AVL movement available")
        return max(sum(row[0] for row in samples) / sum(row[1] for row in samples), 1.0)

    def stop_mean(self, kind, direction, station):
        table = self.dwell if kind == "dwell" else self.hold
        samples = table[(direction, station)] or table[(direction, None)]
        return float(np.mean(samples)) if samples else 0.0


def freeze_pool(runner, telemetry, identity, actor):
    """Whitelist current AVL/APC, past commands, geometry and published service."""
    env, now = runner.env, float(runner.env.current_time)
    split = (len(env.stations) + 1) // 2
    routes = {}
    index = {(route.start_stop, route.end_stop): route for route in env.routes}
    for direction, stations in ((True, env.stations[:split]), (False, env.stations[split - 1:])):
        lengths = [float(index[(a.station_name, b.station_name)].distance)
                   for a, b in zip(stations[:-1], stations[1:])]
        ids = [int(station.station_id) for station in stations]
        travel = [length / telemetry.speed(direction, station) + env.time_step
                  for length, station in zip(lengths, ids[:-1])]
        dwell = [telemetry.stop_mean("dwell", direction, station) for station in ids]
        hold = [telemetry.stop_mean("hold", direction, station) for station in ids]
        cycle = telemetry.cycles[direction]
        routes[str(direction)] = dict(station_ids=ids, lengths_m=lengths, travel_s=travel,
                                      dwell_s=dwell, hold_s=hold,
                                      cycle_s=float(np.mean(cycle)) if cycle else
                                      float(sum(travel) + sum(dwell[:-1]) + sum(hold[:-1]) + 2 * len(lengths)))
    vehicles = []
    for bus in env.bus_all:
        direction = bool(bus.direction)
        route = routes[str(direction)]
        station = route["station_ids"].index(int(bus.last_station.station_id))
        phase = "travel"
        remaining_stop, remaining_service = 0.0, 0.0
        if bus.in_station:
            phase = "service"
            if bus.last_action_time is not None and (bus.last_board_time is None
                                                     or bus.last_action_time >= bus.last_board_time):
                phase = "commanded_hold"
                remaining_stop = max(0.0, float(bus.last_action_s) + 2 * env.time_step
                                     - (now - float(bus.last_action_time)))
            else:
                elapsed = now - float(bus.last_board_time if bus.last_board_time is not None else bus.launch_time)
                remaining_service = max(0.0, float(bus.last_service_dwell_s) - elapsed)
                remaining_stop = remaining_service
                remaining_stop += route["hold_s"][station] + 2 * env.time_step
        if int(bus.bus_id) == identity["bus_id"]:
            phase, remaining_stop = "current_decision", float(actor) + 2 * env.time_step
        ready_since = bus.back_to_terminal_time
        vehicles.append(dict(bus_id=int(bus.bus_id), trip_id=int(bus.trip_id), direction=direction,
                             on_route=bool(bus.on_route), station_index=station,
                             progress_m=float(bus.travel_distance), phase=phase,
                             remaining_stop_s=remaining_stop, remaining_service_s=remaining_service,
                             load=float(len(bus.passengers)),
                             capacity=float(bus.capacity),
                             ready_since_s=float(ready_since) if ready_since is not None and np.isfinite(ready_since) else -1.0))
    commitments = []
    for trip in env.timetables:
        if trip.launched:
            continue
        terminal = bool(getattr(trip, "_freqduet_terminal_dispatch", False))
        min_gap = (getattr(trip, "_freqduet_min_dispatch_headway", 0.0) or 0.0) if terminal else (
            0.0 if hasattr(trip, "_delta_t") else float(trip.target_headway))
        scheduled = float(getattr(trip, "_freqduet_scheduled_launch", trip.launch_time))
        if not terminal and hasattr(trip, "_delta_t"):
            scheduled = max(scheduled, float(trip._original_launch + trip._delta_t))
        commitments.append(dict(trip_id=int(trip.launch_turn), direction=bool(trip.direction),
                                scheduled_s=scheduled, min_gap_s=float(min_gap),
                                first_eligible_s=getattr(trip, "_freqduet_first_eligible_time", None),
                                upper_queried=bool(getattr(trip, "_upper_queried", False))))
    return dict(time_s=now, dt_s=float(env.time_step), routes=routes, vehicles=vehicles,
                commitments=commitments, decision=dict(identity, actor_action_s=float(actor)),
                last_dispatch={str(direction): float(env._last_dispatch_time[direction]) for direction in (True, False)})


def remaining_trip_s(observed, vehicle, stop=None):
    route = observed["routes"][str(vehicle["direction"])]
    lengths = route["lengths_m"]
    start = vehicle["station_index"]
    finish = len(lengths) if stop is None else stop
    if finish < start or (finish == start and vehicle["phase"] == "travel"):
        return None
    elapsed = vehicle["remaining_stop_s"] if finish > start else vehicle["remaining_service_s"]
    position = sum(lengths[:start])
    for leg in range(start, finish):
        fraction = 1.0
        if leg == start:
            fraction = np.clip((position + lengths[leg] - vehicle["progress_m"]) / lengths[leg], 0, 1)
        elapsed += float(fraction) * route["travel_s"][leg]
        if leg + 1 < len(lengths):
            elapsed += route["dwell_s"][leg + 1]
            if leg + 1 < finish:
                elapsed += route["hold_s"][leg + 1] + 2 * observed["dt_s"]
        position += lengths[leg]
    return elapsed


def forecast_pool(observed, horizon_s=3600.0, return_offsets=None):
    """Event rollout with fixed current published plan and physical FIFO reuse."""
    now, dt = observed["time_s"], observed["dt_s"]
    end = now + horizon_s
    offsets = return_offsets or {}
    pools = {True: [], False: []}
    events, serial, wakes = [], 0, set()
    returns, launches, inventory = [], [], []
    last_dispatch = {direction: observed["last_dispatch"][str(direction)] for direction in pools}
    pending = {row["trip_id"]: dict(row) for row in observed["commitments"]}
    initial = {vehicle["trip_id"]: vehicle for vehicle in observed["vehicles"] if vehicle["on_route"]}

    def tick(time):
        return math.ceil(time / dt) * dt

    def event(time, kind, payload):
        nonlocal serial
        if kind == "due":
            if time in wakes:
                return
            wakes.add(time)
        heapq.heappush(events, (time, 0 if kind == "available" else 1, serial, kind, payload))
        serial += 1

    def add_return(bus_id, trip_id, direction, arrival):
        returns.append(dict(bus_id=bus_id, trip_id=trip_id, direction=direction, time_s=arrival))
        event(tick(arrival) + dt, "available", (not direction, bus_id, arrival))

    for vehicle in observed["vehicles"]:
        if vehicle["on_route"]:
            remaining = remaining_trip_s(observed, vehicle)
            add_return(vehicle["bus_id"], vehicle["trip_id"], vehicle["direction"],
                       now + max(remaining + offsets.get(vehicle["trip_id"], 0.0), 0.0))
        else:
            heapq.heappush(pools[vehicle["direction"]], (vehicle["ready_since_s"], vehicle["bus_id"]))
    for row in pending.values():
        event(max(now, tick(row["scheduled_s"])), "due", None)
    inventory.append([now, len(pools[True]), len(pools[False])])
    while events:
        time, _, _, kind, payload = heapq.heappop(events)
        if time > end:
            break
        if kind == "due":
            wakes.remove(time)
        if kind == "available":
            direction, bus_id, arrival = payload
            heapq.heappush(pools[direction], (arrival, bus_id))
        for trip_id in list(pending):
            trip = pending[trip_id]
            direction = trip["direction"]
            if trip["scheduled_s"] > time:
                continue
            eligible = tick(last_dispatch[direction] + trip["min_gap_s"])
            if eligible > time:
                event(eligible, "due", None)
                continue
            if trip["first_eligible_s"] is None:
                trip["first_eligible_s"] = time
            if not pools[direction]:
                continue
            _, bus_id = heapq.heappop(pools[direction])
            launches.append(dict(trip_id=trip_id, bus_id=bus_id, direction=direction, time_s=time,
                                 scheduled_s=trip["scheduled_s"],
                                 readiness_s=max(time - trip["first_eligible_s"], 0.0)))
            last_dispatch[direction] = time
            cycle = observed["routes"][str(direction)]["cycle_s"]
            add_return(bus_id, trip_id, direction, time + max(cycle + offsets.get(trip_id, 0.0), dt))
            del pending[trip_id]
        inventory.append([time, len(pools[True]), len(pools[False])])
    backlog = sum(max(0.0, min(row["time_s"], end) - max(now, row["scheduled_s"])) for row in launches)
    backlog += sum(max(0.0, end - max(now, row["scheduled_s"])) for row in pending.values())
    return dict(returns=returns, launches=launches, inventory=inventory,
                backlog_trip_seconds=backlog, unfulfilled_within_window=sum(row["scheduled_s"] <= end for row in pending.values()),
                horizon_s=horizon_s, active_trips=list(initial))


def forecast_second_identity(observed, forecast):
    first = observed["decision"]
    route = observed["routes"][str(first["direction"]) ]
    stop = route["station_ids"].index(first["station_id"])
    candidates = []
    for vehicle in observed["vehicles"]:
        if (vehicle["on_route"] and vehicle["direction"] == first["direction"]
                and vehicle["bus_id"] != first["bus_id"]):
            eta = remaining_trip_s(observed, vehicle, stop=stop)
            if eta is not None:
                candidates.append(dict(bus_id=vehicle["bus_id"], trip_id=vehicle["trip_id"],
                                       time_s=observed["time_s"] + eta))
    for launch in forecast["launches"]:
        if launch["direction"] == first["direction"] and launch["bus_id"] != first["bus_id"]:
            eta = sum(route["travel_s"][:stop]) + sum(route["dwell_s"][:stop + 1])
            eta += sum(route["hold_s"][:stop]) + 2 * observed["dt_s"] * stop
            candidates.append(dict(bus_id=launch["bus_id"], trip_id=launch["trip_id"], time_s=launch["time_s"] + eta))
    return min(candidates, key=lambda row: row["time_s"]) if candidates else None


def joint_resource_grid(observed, actions):
    baseline = forecast_pool(observed)
    first = observed["decision"]
    stop = observed["routes"][str(first["direction"])]["station_ids"].index(first["station_id"])
    continuation_hold = observed["routes"][str(first["direction"])]["hold_s"][stop]
    rows = []
    for action1 in actions:
        offsets = {first["trip_id"]: action1 - first["actor_action_s"]}
        first_forecast = forecast_pool(observed, return_offsets=offsets)
        second = forecast_second_identity(observed, first_forecast)
        for action2 in actions:
            pair_offsets = dict(offsets)
            if second is not None:
                pair_offsets[second["trip_id"]] = action2 - continuation_hold
            pair = forecast_pool(observed, return_offsets=pair_offsets)
            rows.append(dict(first_action_s=action1, second_action_s=action2,
                             predicted_second=second, second_reference_hold_s=continuation_hold,
                             delta_backlog_trip_seconds=pair["backlog_trip_seconds"] - baseline["backlog_trip_seconds"],
                             unfulfilled_within_window=pair["unfulfilled_within_window"]))
    return rows
