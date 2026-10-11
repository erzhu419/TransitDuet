"""Causal joint-direction occupancy costs for rolling service allocation."""

import numpy as np

from freq_hrl.core.time_allocation import projected_time_points
from freq_hrl.domains.transit.native_rolling_plan import NativeRollingServicePlan

STRENGTHS = (0., .25, .5, .75, 1.)


def occupancy_peak(starts, ends):
    """Count overlapping half-open trip intervals; returns precede tied launches."""
    starts, ends = np.asarray(starts), np.asarray(ends)
    if starts.shape != ends.shape or np.any(ends <= starts):
        raise ValueError("Occupancy needs one positive duration per departure")
    if not len(starts):
        return 0
    times = np.r_[starts, ends]
    changes = np.r_[np.ones(len(starts), dtype=int), -np.ones(len(ends), dtype=int)]
    return int(np.cumsum(changes[np.lexsort((changes, times))]).max())


class NativeSupplyRollingPlan(NativeRollingServicePlan):
    """Select the same strength candidates with or without a fleet-cost term."""

    def __init__(self, env, condition):
        if condition not in {"demand_only", "supply_aware"}:
            raise ValueError(condition)
        super().__init__(env, "causal_forecast")
        self.selection = condition
        self.seen_completions, self.duration_mean, self.decisions = set(), {}, []

    def observe_completions(self, now):
        for tid, row in self.env.completed_trip_observations.items():
            if tid in self.seen_completions:
                continue
            duration = row["completion_time_s"] - row["launch_time_s"]
            if duration <= 0 or row["completion_time_s"] > now:
                raise RuntimeError("Supply estimate needs completed causal trip observations")
            direction = bool(row["direction"])
            self.duration_mean[direction] = (duration if direction not in self.duration_mean
                else .8 * self.duration_mean[direction] + .2 * duration)
            self.seen_completions.add(tid)

    def trip_duration(self, direction):
        if direction in self.duration_mean:
            return self.duration_mean[direction]
        routes = self.env.routes
        half = round(len(routes) / 2)
        routes = routes[:half] if direction else routes[half:]
        return sum(float(r.distance) / max(float(r.speed_limit), 1.) + 4. for r in routes)

    def predicted_peak(self, planned_by_id, now):
        starts, ends = [], []
        for bus in self.env.bus_all:
            if not bus.on_route:
                continue
            total = sum(float(r.distance) for r in bus.effective_route)
            remaining_fraction = np.clip(1 - float(bus.travel_distance) / max(total, 1.), 0., 1.)
            remaining = self.trip_duration(bool(bus.direction)) * remaining_fraction
            starts.append(now)
            ends.append(now + max(remaining, 1.))
        for tt in self.env.timetables:
            if tt.launched:
                continue
            start = max(now, planned_by_id.get(int(tt.launch_turn),
                float(getattr(tt, "_freqduet_scheduled_launch", tt.launch_time))))
            starts.append(start)
            ends.append(start + self.trip_duration(bool(tt.direction)))
        return max(self.env._peak_concurrent, occupancy_peak(starts, ends))

    def candidate_fleet_cost(self, peak, planned_by_id, now):
        return max(peak - self.env._n_fleet_target, 0) ** 2 / self.env._n_fleet_target, {}

    def preferred_intervals(self, nominal, trip, now):
        self.observe_completions(now)
        forecast, rates = super().preferred_intervals(nominal, trip, now)
        remaining = sorted((t for t in self.env.timetables if bool(t.direction) == bool(trip.direction)
            and not t.launched and t.launch_turn not in self._targets), key=lambda t: t.launch_time)
        tracker = self.env.frequency_tracker
        base = np.diff([t.launch_time for t in remaining])
        candidates, preferences = [], []
        for strength in STRENGTHS:
            # A low strength must be able to undo earlier tentative compression.
            preference = (1 - strength) * base + strength * forecast
            planned = projected_time_points(nominal, preference, minimum=240, maximum=480)
            intervals = np.diff(planned)
            offsets = ((planned[:-1] + planned[1:]) / 2 - now) / tracker.bin_interval_s
            demand = np.asarray([max(float(tracker.global_state.forecast(float(t))), 1.) for t in offsets])
            # Uniform arrivals within each predicted interval: mean wait h/2.
            wait_cost = float(np.sum(demand * intervals**2) / (2 * np.sum(demand * intervals)) / 600)
            regularity = float(intervals.std() / intervals.mean())
            planned_by_id = {int(t.launch_turn): float(p) for t, p in zip(remaining, planned)}
            peak = self.predicted_peak(planned_by_id, now)
            fleet_cost, diagnostics = self.candidate_fleet_cost(peak, planned_by_id, now)
            score = wait_cost + (fleet_cost if self.selection == "supply_aware" else 0.)
            candidates.append({"strength": strength, "wait_cost": wait_cost, "headway_cost": regularity,
                "predicted_peak": peak, "fleet_cost": float(fleet_cost), "score": float(score), **diagnostics})
            preferences.append(preference)
        selected = min(range(len(candidates)), key=lambda i: candidates[i]["score"])
        self.decisions.append({"time_s": now, "direction": bool(trip.direction),
            "selected": selected, "observed_completions": len(self.seen_completions), "candidates": candidates})
        return preferences[selected], rates

    def supply_summary(self):
        choices = [STRENGTHS[d["selected"]] for d in self.decisions]
        return {"condition": self.selection, "decisions": self.decisions,
            "choice_counts": [choices.count(s) for s in STRENGTHS],
            "completed_trip_count": len(self.seen_completions),
            "trip_duration_mean_s": {str(k): v for k, v in self.duration_mean.items()},
            "boundary": "point_estimate_occupancy_and_wait_proxy_not_a_fleet_guarantee"}
