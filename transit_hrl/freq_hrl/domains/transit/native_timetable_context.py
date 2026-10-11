"""Preaction full candidate timetables and both directions' pending departures."""

import copy

import numpy as np

from freq_hrl.domains.transit.native_value_context import NativeValueContextAudit, PROBES

EVENT_NAMES = ("nominal_offset", "scheduled_offset", "candidate_offset", "same_direction",
    "committed", "current_headway_goal")


def pending_timetable(plan):
    return [{"trip_id": int(t.launch_turn), "direction": bool(t.direction),
        "nominal_s": float(t.launch_time),
        "scheduled_s": float(getattr(t, "_freqduet_scheduled_launch", t.launch_time)),
        "committed": int(t.launch_turn) in plan._targets, "target_headway_s": float(t.target_headway)}
        for t in sorted((t for t in plan.env.timetables if not t.launched),
            key=lambda t: (t.launch_time, bool(t.direction)))]


class NativeTimetableContextAudit(NativeValueContextAudit):
    def preferred_intervals(self, nominal, trip, now):
        self._pending_timetable = pending_timetable(self)
        self._full_candidate_plans = []
        return super().preferred_intervals(nominal, trip, now)

    def candidate_fleet_cost(self, peak, planned_by_id, now):
        result = super().candidate_fleet_cost(peak, planned_by_id, now)
        self._full_candidate_plans.append([float(planned_by_id[int(t.launch_turn)]) for t in self._audit_remaining])
        return result

    def select_candidate(self, candidates):
        index = len(self.decisions)
        selected = super().select_candidate(candidates)
        if index in PROBES:
            self.contexts[-1]["timetable_context"] = {
                "pending_trips": copy.deepcopy(self._pending_timetable),
                "candidate_trip_ids": [int(t.launch_turn) for t in self._audit_remaining],
                "candidate_planned_s": copy.deepcopy(self._full_candidate_plans)}
        return selected


def qualify_timetable(context, row):
    table = context["timetable_context"]
    trips = table["pending_trips"]
    expected = [t for t in trips if t["direction"] == row["direction"] and not t["committed"]]
    ids = [t["trip_id"] for t in trips]
    if (len(ids) != len(set(ids)) or trips != sorted(trips, key=lambda t: (t["nominal_s"], t["direction"]))
            or table["candidate_trip_ids"] != [t["trip_id"] for t in expected]
            or [t["scheduled_s"] for t in expected] != row["nominal_s"]
            or len(table["candidate_planned_s"]) != len(row["candidates"])):
        raise ValueError("Full timetable does not match labelled pending trips and causal anchors")
    for proposed, candidate in zip(table["candidate_planned_s"], row["candidates"]):
        prefix = candidate["commitment"]["prefix_planned_s"]
        if (len(proposed) != len(expected) or proposed[:len(prefix)] != prefix
                or not np.all(np.isfinite(proposed)) or proposed[0] < row["time_s"]
                or proposed[0] != row["nominal_s"][0] or proposed[-1] != row["nominal_s"][-1]
                or np.any(np.diff(proposed) < 240) or np.any(np.diff(proposed) > 480)):
            raise ValueError("Full candidate curve changed its labelled prefix, endpoints or interval bounds")
    if not np.all(np.isfinite([[t[k] for k in ("nominal_s", "scheduled_s", "target_headway_s")] for t in trips])):
        raise ValueError("Nonfinite causal pending timetable")


def event_features(row):
    """Keep every pending event; trip IDs join actions but are not network inputs."""
    table, now = row["timetable_context"], row["time_s"]
    trips = sorted(table["pending_trips"], key=lambda t: (t["nominal_s"], t["direction"]))
    result = []
    for proposed in table["candidate_planned_s"]:
        by_id = dict(zip(table["candidate_trip_ids"], proposed))
        result.append([[(t["nominal_s"]-now)/61380., (t["scheduled_s"]-now)/61380.,
            (by_id.get(t["trip_id"], t["scheduled_s"])-now)/61380.,
            float(t["direction"] == row["direction"]), float(t["committed"]), t["target_headway_s"]/360.]
            for t in trips])
    return np.asarray(result)
