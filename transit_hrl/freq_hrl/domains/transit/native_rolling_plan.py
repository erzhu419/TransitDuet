"""Causal remaining-horizon allocation with irrevocable six-trip commitments."""

import numpy as np

from freq_hrl.core.time_allocation import projected_time_points
from freq_hrl.domains.transit.native_service_plan import NativeServicePlan


def schedule_summary(plan):
    env, trips = plan.env, plan.env.timetables
    if len(plan.queries) != len(trips) or len(plan._targets) != len(trips):
        raise RuntimeError("Incomplete native rolling-plan queries or commitments")
    for block in plan.blocks:
        for tid, committed in zip(block["trip_ids"], block["planned_s"]):
            tt = next(t for t in trips if t.launch_turn == tid)
            if tt._freqduet_scheduled_launch != committed:
                raise RuntimeError("A later rolling proposal rewrote a committed departure")
    schedules, lateness = [], []
    for direction in (False, True):
        ordered = sorted((t for t in trips if bool(t.direction) == direction), key=lambda t: t.launch_time)
        nominal = np.asarray([t.launch_time for t in ordered], dtype=float)
        planned = np.asarray([t._freqduet_scheduled_launch for t in ordered], dtype=float)
        targets = np.asarray([t.target_headway for t in ordered], dtype=float)
        if (not np.array_equal(planned[[0, -1]], nominal[[0, -1]])
                or np.any(np.diff(planned) < 240) or np.any(np.diff(planned) > 480)
                or not np.array_equal(targets[1:], np.diff(planned))):
            raise RuntimeError("Rolling plan changed daily endpoints or violated cross-window interval bounds/goals")
        actual = []
        for tt, launch in zip(ordered, planned):
            if plan.queries[tt.launch_turn] > launch or tt.target_headway != plan._targets[tt.launch_turn]:
                raise RuntimeError("Late native planning or changed committed lower goal")
            time = float(tt._actual_launch_time) if tt.launched else None
            if time is not None:
                if time < launch:
                    raise RuntimeError("Native launch preceded its causal committed plan")
                lateness.append(time - launch)
            actual.append(time)
        schedules.append({"direction": direction, "nominal_s": nominal.tolist(),
            "planned_s": planned.tolist(), "actual_s": actual, "target_headways_s": targets.tolist()})
    return {"queries": len(plan.queries), "upper_decisions": len(plan.blocks), "scheduled_trips": len(trips),
        "committed_unchanged": True, "daily_endpoint_error_max_s": 0,
        "window_endpoint_shift_abs_mean_s": float(np.mean([
            abs(b["planned_s"][-1] - b["nominal_s"][-1]) for b in plan.blocks])),
        "planned_shift_abs_mean_s": float(np.mean([abs(p - n) for s in schedules
            for p, n in zip(s["planned_s"], s["nominal_s"])])),
        "release_lateness_mean_s": float(np.mean(lateness)) if lateness else 0., "schedules": schedules}


class NativeRollingServicePlan(NativeServicePlan):
    """Replan only uncommitted trips; commit the next six at each native callback."""

    def __init__(self, env, condition):
        if condition not in {"nominal_plan", "causal_forecast", "forecast_reversed"}:
            raise ValueError(condition)
        super().__init__(env, condition)

    def __call__(self, state, trip):
        now, tid = float(self.env.current_time), int(trip.launch_turn)
        self.queries[tid] = now
        if tid in self._targets:
            return self._targets[tid]
        remaining = sorted((t for t in self.env.timetables if bool(t.direction) == bool(trip.direction)
            and not t.launched and t.launch_turn not in self._targets), key=lambda t: t.launch_time)
        if not remaining or remaining[0] is not trip:
            raise RuntimeError("Rolling horizon must begin with the next uncommitted native trip")
        anchor = np.asarray([getattr(t, "_freqduet_scheduled_launch", t.launch_time) for t in remaining], dtype=float)
        if len(anchor) >= 3:
            preferred, _ = self.preferred_intervals(anchor, trip, now)
            planned = projected_time_points(anchor, preferred, minimum=240, maximum=480)
        else:
            planned = anchor.astype(np.int64)
        if planned[0] < now:
            raise RuntimeError("Rolling forecast attempted to commit a departure in the past")
        count = min(self.block_size, len(remaining))
        previous = next((b for b in reversed(self.blocks) if b["direction"] == bool(trip.direction)), None)
        first_goal = float(trip.target_headway) if previous is None else float(planned[0] - previous["planned_s"][-1])
        targets = np.r_[first_goal, np.diff(planned[:count])]
        # Tentative times make the next callback eligible before an advanced trip.
        # Only the next six become irrevocable; later forecasts may rewrite the rest.
        for t, launch in zip(remaining, planned):
            t._freqduet_scheduled_launch = int(launch)
        for t, target in zip(remaining[:count], targets):
            t._original_launch, t._delta_t = t.launch_time, 0
            t._freqduet_terminal_dispatch = True
            t.target_headway = float(target)
            self._targets[int(t.launch_turn)] = float(target)
        self.blocks.append({"decision_time_s": now, "direction": bool(trip.direction),
            "trip_ids": [int(t.launch_turn) for t in remaining[:count]],
            "nominal_s": [float(t.launch_time) for t in remaining[:count]],
            "planned_s": planned[:count].tolist(), "remaining_trip_count": len(remaining)})
        return self._targets[tid]

    def summarize(self):
        return schedule_summary(self)


class NativeTemperedRollingPlan(NativeRollingServicePlan):
    """Scale forecast interval preferences before the unchanged budget projection."""

    def __init__(self, env, strength):
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Allocation strength must be in [0, 1]")
        super().__init__(env, "causal_forecast")
        self.strength = float(strength)

    def preferred_intervals(self, nominal, trip, now):
        preferred, rates = super().preferred_intervals(nominal, trip, now)
        return (1 - self.strength) * np.diff(nominal) + self.strength * preferred, rates
