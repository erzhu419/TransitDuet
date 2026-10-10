"""Committed service-interval plans for the copied native terminal dispatcher."""

import numpy as np

from freq_hrl.core.time_allocation import budgeted_time_points


CONDITIONS = ("nominal_plan", "frontload", "backload", "causal_forecast", "forecast_reversed")


class NativeServicePlan:
    """Allocate six departures at a time, leaving each block's endpoints fixed."""

    def __init__(self, env, condition, *, block_size=6):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.env, self.condition, self.block_size = env, condition, block_size
        self.blocks, self.queries = [], {}
        self._targets = {}

    def __call__(self, state, trip):
        now, tid = float(self.env.current_time), int(trip.launch_turn)
        self.queries[tid] = now
        if tid not in self._targets:
            trips = sorted((tt for tt in self.env.timetables
                if bool(tt.direction) == bool(trip.direction) and not tt.launched
                and tt.launch_turn not in self._targets), key=lambda tt: tt.launch_time)[:self.block_size]
            if trips[0] is not trip:
                raise RuntimeError("Service window must start at the next uncommitted departure")
            nominal = np.asarray([tt.launch_time for tt in trips], dtype=float)
            rates = []
            if len(trips) < 3:
                planned = nominal.astype(np.int64)
            else:
                base = np.diff(nominal)
                preference = base.copy()
                if self.condition in {"frontload", "backload"}:
                    sign = 1 if self.condition == "frontload" else -1
                    preference += sign * 120 * np.linspace(-1, 1, len(base))
                elif self.condition in {"causal_forecast", "forecast_reversed"}:
                    tracker = self.env.frequency_tracker
                    offsets = ((nominal[:-1] + nominal[1:]) / 2 - now) / tracker.bin_interval_s
                    rates = [max(float(tracker.global_state.forecast(float(t))), 1.0) for t in offsets]
                    inverse_sqrt = 1 / np.sqrt(rates)
                    forecast_intervals = inverse_sqrt / inverse_sqrt.sum() * base.sum()
                    preference = (forecast_intervals if self.condition == "causal_forecast"
                                  else 2 * base - forecast_intervals)
                planned = budgeted_time_points(nominal, preference, minimum=240, maximum=480)
            targets = np.r_[float(trips[0].target_headway), np.diff(planned)]
            if planned[0] < now:
                raise RuntimeError("Cannot commit a departure in the past")
            for tt, launch, target in zip(trips, planned, targets):
                tt._original_launch = tt.launch_time
                tt._delta_t = 0
                tt._freqduet_terminal_dispatch = True
                tt._freqduet_scheduled_launch = int(launch)
                tt.target_headway = float(target)
                self._targets[int(tt.launch_turn)] = float(target)
            self.blocks.append({"decision_time_s": now, "direction": bool(trip.direction),
                "trip_ids": [int(tt.launch_turn) for tt in trips], "nominal_s": nominal.tolist(),
                "planned_s": planned.tolist(), "target_headways_s": targets.tolist(),
                "forecast_rates": rates})
        return self._targets[tid]

    def summarize(self):
        trips = self.env.timetables
        if len(self.queries) != len(trips) or len(self._targets) != len(trips):
            raise RuntimeError("Incomplete service-plan execution")
        changes, shifts, late, actual_gaps, target_errors = [], [], [], [], []
        previous = {}
        for tt in sorted(trips, key=lambda tt: tt.launch_time):
            tid = int(tt.launch_turn)
            planned = int(tt._freqduet_scheduled_launch)
            if self.queries[tid] > planned:
                raise RuntimeError("Service decision arrived after its committed departure")
            if float(tt.target_headway) != self._targets[tid]:
                raise RuntimeError("Lower target no longer matches the committed interval plan")
            shifts.append(planned - tt.launch_time)
            changes.append(self._targets[tid] - 360)
            if tt.launched:
                actual = float(tt._actual_launch_time)
                if actual < planned:
                    raise RuntimeError("Native departure preceded the service plan")
                late.append(actual - planned)
                if tt.direction in previous:
                    gap = actual - previous[tt.direction]
                    actual_gaps.append(gap)
                    target_errors.append(abs(gap - self._targets[tid]))
                previous[tt.direction] = actual
        if not late:
            raise RuntimeError("No native departures executed")
        return {"blocks": len(self.blocks), "queries": len(self.queries), "launched": len(late),
            "changed_headway_count": sum(abs(v) >= 1 for v in changes),
            "headway_change_abs_mean_s": float(np.mean(np.abs(changes))),
            "planned_shift_abs_mean_s": float(np.mean(np.abs(shifts))),
            "planned_shift_max_abs_s": float(np.max(np.abs(shifts))),
            "release_lateness_mean_s": float(np.mean(late)),
            "actual_headway_std_s": float(np.std(actual_gaps)),
            "actual_target_error_abs_mean_s": float(np.mean(target_errors)),
            "endpoint_error_max_s": max(max(abs(b["planned_s"][i] - b["nominal_s"][i])
                for i in (0, -1)) for b in self.blocks)}
