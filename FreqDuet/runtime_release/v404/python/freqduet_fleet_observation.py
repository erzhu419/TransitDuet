"""Current fleet telemetry and published dispatch commitments, without forecasts."""

import numpy as np

MOTION = ("HOLDING", "WAITING_ACTION", "DWELLING", "TRAVEL")
FLEET_FIELDS = ("present", "on_route", "direction", "active_trip", "progress", "speed", "load",
                "actual_launch", "terminal_return", "terminal_return_seen", "remaining_command_hold",
                "remaining_service", *MOTION)
PLAN_FIELDS = ("direction", "launched", "upper_queried", "published_eligibility", "published_release",
               "target_headway", "terminal_mode", "offset_mode", "min_gap", "min_gap_valid",
               "first_eligible_seen", "first_eligible", "actual_launch")


def capture_fleet(env):
    fleet = []
    for vehicle in env.bus_all:
        active = bool(vehicle.on_route)
        returned = not active and vehicle.trip_id >= 0
        state = vehicle.state.name if active else "TERMINAL"
        fleet.append(dict(bus_id=int(vehicle.bus_id), on_route=active, direction=bool(vehicle.direction),
                          active_trip_id=int(vehicle.trip_id) if active else None,
                          progress_m=float(vehicle.travel_distance) if active else 0., speed_mps=float(vehicle.current_speed) if active else 0.,
                          load=len(vehicle.passengers) if active else 0, capacity=float(vehicle.capacity),
                          actual_launch_s=float(vehicle.launch_time) if active else None,
                          terminal_return_s=float(vehicle.back_to_terminal_time) if returned else None, state=state,
                          remaining_command_hold_s=float(vehicle.dwelling_time) if state == "DWELLING" else 0.,
                          remaining_service_s=float(vehicle.holding_time) if state == "HOLDING" else 0.))
    plans = []
    for trip in env.timetables:
        terminal = bool(getattr(trip, "_freqduet_terminal_dispatch", False))
        offset = not terminal and hasattr(trip, "_delta_t")
        published = float(getattr(trip, "_freqduet_scheduled_launch", trip.launch_time))
        release = float(trip._original_launch + trip._delta_t) if offset else published
        plans.append(dict(trip_id=int(trip.launch_turn), direction=bool(trip.direction), launched=bool(trip.launched),
                          upper_queried=bool(getattr(trip, "_upper_queried", False)),
                          published_eligibility_s=published, published_release_s=max(published, release),
                          target_headway_s=float(trip.target_headway), terminal_mode=terminal, offset_mode=offset,
                          min_gap_s=getattr(trip, "_freqduet_min_dispatch_headway", None) if terminal else None,
                          first_eligible_s=getattr(trip, "_freqduet_first_eligible_time", None),
                          actual_launch_s=float(trip._actual_launch_time) if trip.launched else None))
    routes = env.routes
    middle = len(routes) // 2
    return dict(time_s=float(env.current_time), buses=int(env.max_agent_num), trips=len(plans),
                horizon_s=float(env.protocol.evaluation_end_time_s),
                route_lengths_m=[float(sum(r.distance for r in routes[middle:])), float(sum(r.distance for r in routes[:middle]))],
                fleet=fleet, plans=plans)


def encode_fleet(snapshot):
    horizon, trips = snapshot["horizon_s"], snapshot["trips"]
    fleet = np.zeros((snapshot["buses"], len(FLEET_FIELDS)))
    seen = set()
    for row in snapshot["fleet"]:
        bus = row["bus_id"]
        if bus in seen or not 0 <= bus < len(fleet):
            raise ValueError("fleet telemetry does not align with physical bus slots")
        seen.add(bus)
        active, returned = row["on_route"], row["terminal_return_s"] is not None
        fleet[bus] = [1., float(active), float(row["direction"]), row["active_trip_id"] / trips if active else 0.,
                      row["progress_m"] / snapshot["route_lengths_m"][int(row["direction"])], row["speed_mps"] / 15.,
                      row["load"] / row["capacity"], row["actual_launch_s"] / horizon if active else 0.,
                      row["terminal_return_s"] / horizon if returned else 0., float(returned),
                      row["remaining_command_hold_s"] / 45., row["remaining_service_s"] / 600.,
                      *[float(row["state"] == state) for state in MOTION]]
    plans = np.zeros((trips, len(PLAN_FIELDS)))
    seen = set()
    for row in snapshot["plans"]:
        trip = row["trip_id"]
        if trip in seen or not 0 <= trip < trips:
            raise ValueError("published plan does not align with logical trip slots")
        seen.add(trip)
        first, actual, gap = row["first_eligible_s"], row["actual_launch_s"], row["min_gap_s"]
        plans[trip] = [float(row["direction"]), float(row["launched"]), float(row["upper_queried"]),
                       row["published_eligibility_s"] / horizon, row["published_release_s"] / horizon,
                       row["target_headway_s"] / 600., float(row["terminal_mode"]), float(row["offset_mode"]),
                       gap / 600. if gap is not None else 0., float(gap is not None),
                       float(first is not None), first / horizon if first is not None else 0.,
                       actual / horizon if row["launched"] else 0.]
    if len(seen) != trips:
        raise ValueError("published dispatch roster is incomplete")
    if not np.isfinite(fleet).all() or not np.isfinite(plans).all():
        raise ValueError("non-finite current fleet observation")
    return fleet.ravel(), plans.ravel()
