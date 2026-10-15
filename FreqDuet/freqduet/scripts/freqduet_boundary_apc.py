"""Observable terminal clocks and APC mass-balance diagnostics."""

import pandas as pd

from freqduet_closed_loop_forecast import FlowHistory


class APCTrace(FlowHistory):
    def __init__(self):
        super().__init__()
        self.rows = []

    def arrival(self, direction, station, time, boarded, alighted, preload, left_behind):
        self.rows.append(dict(direction=bool(direction), station_id=int(station), time_s=float(time),
            boarded=float(boarded), alighted=float(alighted), preload=float(preload), left_behind=float(left_behind)))
        super().arrival(direction, station, time, boarded, alighted, preload, left_behind)


def boundary_events(events, launches, terminals, routes):
    added = []
    for frame, stage, index in ((launches, "departure", 0), (terminals, "arrival", -1)):
        for row in frame.itertuples():
            added.append(dict(stage=stage, direction=bool(row.direction), trip_id=int(row.trip_id),
                bus_id=int(row.bus_id), station_id=routes[str(bool(row.direction))]["station_ids"][index],
                time_s=float(row.time_s)))
    result = pd.concat([events, pd.DataFrame(added)], ignore_index=True)
    if result.duplicated(["direction", "trip_id", "station_id", "stage"]).any():
        raise ValueError("terminal event duplicated an existing stop clock")
    return result


def arrival_intervals(apc):
    records = []
    for (direction, station), group in apc.groupby(["direction", "station_id"]):
        group = group.sort_values("time_s")
        times, left = group.time_s.to_numpy(), group.left_behind.to_numpy()
        for i in range(1, len(group)):
            if times[i] <= times[i - 1]:
                raise ValueError("two APC services share a station/direction clock")
            records.append(dict(direction=bool(direction), station_id=int(station), start_s=times[i - 1],
                end_s=times[i], duration_s=times[i] - times[i - 1],
                arrivals=group.boarded.iloc[i] + left[i] - left[i - 1]))
    return pd.DataFrame(records)


def audit_flows(apc, intervals, observed, horizon=1800):
    """Future APC clocks/loads are scoring labels, not rollout inputs."""
    now, rows = observed["time_s"], []
    for direction, route in observed["routes"].items():
        for station, flow in zip(route["station_ids"], observed["flows"][direction]):
            services = apc.loc[(apc.direction == (direction == "True")) & (apc.station_id == station)]
            history = services.loc[services.time_s < now].sort_values("time_s")
            gaps = intervals.loc[(intervals.direction == (direction == "True")) &
                                 (intervals.station_id == station)]
            past = gaps.loc[gaps.end_s < now].sort_values("end_s").tail(4)
            if history.empty or past.empty:
                continue
            recent_rate = float(past.arrivals.sum() / past.duration_s.sum())
            if recent_rate < 0:
                raise ValueError("negative APC arrival mass; repair the telemetry contract before fitting")
            future = services.loc[(services.time_s > now) & (services.time_s <= now + horizon)].sort_values("time_s")
            last = history.iloc[-1]
            complete = gaps.loc[(gaps.start_s >= now) & (gaps.end_s <= now + horizon)]
            for mode, rate in (("legacy_32", flow["rate_per_s"]), ("recent_4", recent_rate)):
                if len(future):
                    first = future.iloc[0]
                    predicted = last.left_behind + rate * (first.time_s - last.time_s)
                    actual = first.boarded + first.left_behind
                    rows.append(dict(mode=mode, metric="next_service_preboard_queue", error=predicted - actual,
                                     actual=actual, station_id=station, direction=direction))
                for row in complete.itertuples():
                    rows.append(dict(mode=mode, metric="completed_gap_arrivals", error=rate * row.duration_s - row.arrivals,
                                     actual=row.arrivals, station_id=station, direction=direction))
            for row in future.itertuples():
                rows.append(dict(mode="legacy_32", metric="teacher_forced_alighted", actual=row.alighted,
                    error=flow["alight_fraction"] * row.preload - row.alighted,
                    station_id=station, direction=direction))
    return rows
