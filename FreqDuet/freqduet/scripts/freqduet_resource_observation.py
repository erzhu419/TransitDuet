"""Controlled-vehicle resource state from current telemetry and published plans."""

import numpy as np


OWN = ("remaining_route", "speed", "load", "trip_elapsed", "remaining_service", "target_headway")
TERMINAL = ("ready", "other_inbound", "inbound_load", "inbound_remaining_mean", "inbound_nearest_remaining",
            "inbound_hold", "inbound_pending_hold", "inbound_service", "due", "oldest_due", "next_release_wait",
            "has_plan", "plans_15m", "next_headway", "recent_return_age", "return_seen")
FEATURES = (*OWN, *[role + "_" + name for role in ("origin", "return") for name in TERMINAL])


def encode_resources(capture, snapshot):
    now, bus_id, trip_id, _, direction = capture["identity"]
    if snapshot["time_s"] != now:
        raise ValueError("resource observation is not from the current decision tick")
    fleet = snapshot["fleet"]
    own = next(v for v in fleet if v["bus_id"] == bus_id)
    if not own["on_route"] or own["active_trip_id"] != trip_id or own["direction"] != direction:
        raise ValueError("controlled vehicle does not match its causal prefix")
    n = float(snapshot["buses"])
    remaining = lambda v: 1. - v["progress_m"] / snapshot["route_lengths_m"][int(v["direction"])]
    values = dict(remaining_route=remaining(own), speed=own["speed_mps"] / 15., load=own["load"] / own["capacity"],
                  trip_elapsed=(now - own["actual_launch_s"]) / 3600., remaining_service=own["remaining_service_s"] / 600.,
                  target_headway=capture["observation"]["target_headway_s"] / 600.)
    pending = capture["observation"]["pending_commands"]
    for role, terminal in (("origin", bool(direction)), ("return", not bool(direction))):
        ready = [v for v in fleet if not v["on_route"] and v["direction"] == terminal]
        # Operating direction flips on return: these vehicles supply this terminal.
        inbound = [v for v in fleet if v["on_route"] and v["direction"] != terminal and v["bus_id"] != bus_id]
        inbound_ids = {v["bus_id"] for v in inbound}
        distances = [remaining(v) for v in inbound]
        plans = sorted((p for p in snapshot["plans"] if not p["launched"] and p["direction"] == terminal),
                       key=lambda p: (p["published_release_s"], p["trip_id"]))
        due = [now - p["published_release_s"] for p in plans if p["published_release_s"] <= now]
        returns = [now - v["terminal_return_s"] for v in ready if v["terminal_return_s"] is not None]
        block = dict(ready=len(ready) / n, other_inbound=len(inbound) / n,
            inbound_load=sum(v["load"] / v["capacity"] for v in inbound) / n,
            inbound_remaining_mean=float(np.mean(distances)) if distances else 0.,
            inbound_nearest_remaining=min(distances, default=0.),
            inbound_hold=sum(v["remaining_command_hold_s"] for v in inbound) / (45. * n),
            inbound_pending_hold=sum(c["action_s"] for c in pending if c["bus_id"] in inbound_ids) / (45. * n),
            inbound_service=sum(v["remaining_service_s"] for v in inbound) / (600. * n),
            due=len(due) / n, oldest_due=max(due, default=0.) / 3600.,
            next_release_wait=max(0., plans[0]["published_release_s"] - now) / 3600. if plans else 0.,
            has_plan=float(bool(plans)), plans_15m=sum(now < p["published_release_s"] <= now + 900. for p in plans) / n,
            next_headway=plans[0]["target_headway_s"] / 600. if plans else 0.,
            recent_return_age=min(returns, default=0.) / 3600., return_seen=len(returns) / n)
        values.update({role + "_" + key: value for key, value in block.items()})
    result = np.asarray([values[name] for name in FEATURES], dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("non-finite causal resource state")
    return result
