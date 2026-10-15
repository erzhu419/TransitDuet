"""Completed AVL legs and shared, past-only 300-second travel-time blocks."""

import numpy as np
import pandas as pd

from freqduet_closed_loop_forecast import fluid_rollout
from freqduet_holding_distribution import DRAWS, METRICS, probabilities, sample_action

BIN_S = 300
HISTORY_S = 10800
MODES = ("completed_mean", "independent", "shared_block")


def completed_legs(events, routes):
    rows = []
    for direction, route in routes.items():
        ids = route["station_ids"]
        arrival = events.loc[(events.stage == "arrival") & (events.direction == (direction == "True")),
                             ["trip_id", "station_id", "time_s"]].rename(columns={"time_s": "end_s"})
        departure = events.loc[(events.stage == "departure") & (events.direction == (direction == "True")),
                               ["trip_id", "station_id", "time_s"]].rename(columns={"time_s": "start_s"})
        arrival["station_id"] = arrival.station_id.map(dict(zip(ids[1:], ids[:-1])))
        paired = departure.merge(arrival, on=["trip_id", "station_id"], validate="one_to_one")
        for row in paired.itertuples():
            if row.end_s <= row.start_s:
                raise ValueError("nonpositive observed AVL leg")
            rows.append(dict(direction=direction, station_id=int(row.station_id),
                             start_s=float(row.start_s), end_s=float(row.end_s),
                             duration_s=float(row.end_s - row.start_s)))
    return pd.DataFrame(rows)


def fit_travel(legs, observed, horizon=1800):
    now, dt = observed["time_s"], observed["dt_s"]
    # Require a completed leg; a departure alone is not a travel-time label.
    past = legs.loc[(legs.end_s < now) & (legs.start_s >= max(0, now - HISTORY_S))]
    first_bin = int(max(0, now - HISTORY_S) // BIN_S)
    last_bin = int(now // BIN_S) - 1
    bins = np.arange(first_bin, last_bin + 1)
    steps = int(np.ceil(horizon / BIN_S))
    if len(bins) < steps:
        raise ValueError("travel blocks require at least one completed horizon of AVL history")
    profiles, means, counts = {}, {}, {}
    for direction, route in observed["routes"].items():
        for station, fallback in zip(route["station_ids"][:-1], route["travel_s"]):
            key = f"{direction}:{station}"
            records = past.loc[(past.direction == direction) & (past.station_id == station)]
            mean = float(records.duration_s.mean()) if len(records) else float(fallback)
            means[key], counts[key] = mean, len(records)
            values = np.full(len(bins), mean)
            if len(records):
                grouped = records.assign(bin=(records.start_s // BIN_S).astype(int)).groupby("bin").duration_s.mean()
                latest = mean
                for i, bin_id in enumerate(bins):
                    if bin_id in grouped.index:
                        latest = float(grouped.loc[bin_id])
                    values[i] = latest
            # Recenter the shared path, so independent and block sampling have
            # the same per-leg marginal and the same mean as completed_mean.
            profiles[key] = (values * mean / values.mean()).tolist()
    return dict(now_s=now, dt_s=dt, bin_s=BIN_S, horizon_s=horizon,
                means=means, profiles=profiles, counts=counts, samples=len(past),
                latest_completion_s=float(past.end_s.max()) if len(past) else None,
                block_starts=list(range(len(bins))))


class TravelPath:
    def __init__(self, model, mode, rng):
        self.model, self.mode, self.rng = model, mode, rng
        self.start = int(rng.choice(model["block_starts"]))

    def __call__(self, vehicle, time, fraction):
        model = self.model
        key = f"{vehicle['direction']}:{vehicle['travel_station_id']}"
        if self.mode == "completed_mean":
            duration = model["means"][key]
        else:
            lag = min(int((time - model["now_s"]) // model["bin_s"]),
                      int(np.ceil(model["horizon_s"] / model["bin_s"])) - 1)
            start = self.start if self.mode == "shared_block" else int(self.rng.choice(model["block_starts"]))
            profile = model["profiles"][key]
            duration = profile[(start + lag) % len(profile)]
        return max(fraction * duration, model["dt_s"])


def travel_ensemble(observed, horizon, holding_model, travel_model, mode, draws=DRAWS, event_sink=None):
    metrics = {metric: [] for metric in METRICS}
    routes = observed["routes"]
    for draw in range(draws):
        holding_rng = np.random.default_rng(20261015 + draw)
        travel = TravelPath(travel_model, mode, np.random.default_rng(20261016 + draw))

        def predict(vehicle, time, fraction):
            values = dict(vehicle, travel_station_id=routes[str(vehicle["direction"])]["station_ids"][vehicle["station_index"]])
            return travel(values, time, fraction)

        forecast = fluid_rollout(observed, horizon, "response",
            action_predictor=lambda values: sample_action(probabilities(holding_model, values), holding_rng.random()),
            event_sink=None if event_sink is None else lambda row: event_sink(draw, row),
            travel_predictor=predict)
        for metric in metrics:
            metrics[metric].append(forecast[metric])
    return dict(mean={metric: float(np.mean(values)) for metric, values in metrics.items()}, draws=metrics)


def score_draws(values, actual):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    lo, hi = np.quantile(values, [.05, .95])
    crps = np.mean(abs(values - actual)) - np.sum((2 * np.arange(n) - n + 1) * values) / n ** 2
    return dict(error=float(values.mean() - actual), crps=float(crps),
                covered=float(lo <= actual <= hi), width=float(hi - lo), spread=float(values.std()))
