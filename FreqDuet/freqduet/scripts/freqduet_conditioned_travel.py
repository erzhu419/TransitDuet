"""Completed-prefix empirical-Bayes travel means and short Gaussian-copula blocks."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from freqduet_travel_distribution import HISTORY_S, BIN_S

MODES = ("conditioned_mean", "conditioned_independent", "conditioned_shared")


def road_key(direction, station):
    return f"{direction}:{int(station)}"


def fit_mean(legs, observed):
    now = observed["time_s"]
    past = legs.loc[(legs.end_s < now) & (legs.start_s >= max(0, now - HISTORY_S))].copy()
    past["road"] = [road_key(d, s) for d, s in zip(past.direction, past.station_id)]
    hour = now // 3600 * 3600
    recent_start = max(hour, now - 900)
    keys, fallback = [], {}
    for direction, route in observed["routes"].items():
        for station, value in zip(route["station_ids"][:-1], route["travel_s"]):
            key = road_key(direction, station)
            keys.append(key)
            fallback[key] = float(value)
    groups = dict(tuple(past.groupby("road")))
    means, uncertainty, noise, counts = {}, {}, {}, {}
    for key in keys:
        group = groups.get(key, past.iloc[:0])
        previous = group.loc[(group.start_s >= hour - 3600) & (group.start_s < hour)]
        prior = previous if len(previous) else group.loc[group.start_s < hour]
        means[key] = float(prior.duration_s.mean()) if len(prior) else fallback[key]
        counts[key] = len(group)
        # Within-hour variation separates regime changes from sampling noise.
        centered = group.duration_s - group.groupby(group.start_s // 3600).duration_s.transform("mean")
        dof = len(group) - group.groupby(group.start_s // 3600).ngroups
        noise[key] = float(centered @ centered / dof) if dof else 0.0
        uncertainty[key] = noise[key] / max(len(prior), 1)
    relative = [noise[k] / means[k] ** 2 for k in keys if counts[k] > 1]
    pooled = float(np.mean(relative)) if relative else 0.0
    for key in keys:
        if counts[key] <= 1:
            noise[key] = pooled * means[key] ** 2
            uncertainty[key] = noise[key]

    def update(selected):
        current = {key: group for key, group in selected.groupby("road")}
        changes = [float((g.duration_s.mean() - means[k]) ** 2 - noise[k] / len(g) - uncertainty[k])
                   for k, g in current.items()]
        tau2 = max(float(np.mean(changes)), 0.0) if changes else 0.0
        for key, group in current.items():
            prior_var = tau2 + uncertainty[key]
            observation_var = noise[key] / len(group)
            weight = prior_var / (prior_var + observation_var) if prior_var + observation_var else 0.0
            means[key] += weight * (float(group.duration_s.mean()) - means[key])
            uncertainty[key] = (1 - weight) * prior_var
        return tau2

    period_tau2 = update(past.loc[(past.start_s >= hour) & (past.start_s < recent_start)])
    recent_tau2 = update(past.loc[past.start_s >= recent_start])
    return dict(now_s=now, dt_s=observed["dt_s"], bin_s=BIN_S, horizon_s=1800, keys=keys,
        means=means, mean_variances=uncertainty, counts=counts, samples=len(past),
        latest_completion_s=float(past.end_s.max()) if len(past) else None,
        period_tau2=period_tau2, recent_tau2=recent_tau2)


def fit_noise(model, innovations, legs):
    now, keys = model["now_s"], model["keys"]
    residuals = innovations.loc[(innovations.end_s < now) & (innovations.time_s >= max(0, now - HISTORY_S))].copy()
    residuals["road"] = [road_key(d, s) for d, s in zip(residuals.direction, residuals.station_id)]
    rows, prequential_legs = [], 0
    for key in keys:
        selected = residuals.loc[residuals.road == key]
        if len(selected):
            prequential_legs += 1
            rows.extend(dict(road=key, start_s=r.time_s, ratio=r.actual / r.expected) for r in selected.itertuples())
        else:
            # At the first issued origin no new-model innovations have matured.
            # Use only completed historical within-hour deviations as a prior.
            direction, station = key.split(":")
            past = legs.loc[(legs.end_s < now) & (legs.start_s >= max(0, now - HISTORY_S)) &
                (legs.direction == direction) & (legs.station_id == int(station))].copy()
            if len(past):
                reference = past.groupby(past.start_s // 3600).duration_s.transform("mean")
                rows.extend(dict(road=key, start_s=t, ratio=v) for t, v in zip(past.start_s, past.duration_s / reference))
    frame = pd.DataFrame(rows, columns=["road", "start_s", "ratio"])
    pools = {key: [1.0] for key in keys}
    if len(frame):
        grouped = frame.groupby("road")
        frame["z"] = [NormalDist().inv_cdf(p) for p in
            ((grouped.ratio.rank(method="average") - .5) / grouped.ratio.transform("size"))]
        scale = frame.groupby("road").z.transform(lambda v: float(np.sqrt(np.mean(v ** 2))))
        frame["z"] = frame.z / scale.where(scale > 0, 1)
        for key, group in grouped:
            values = group.ratio.to_numpy()
            pools[key] = np.sort(values / values.mean()).tolist()
        frame["bin"] = (frame.start_s // BIN_S).astype(int)
        clusters = frame.loc[scale > 0].groupby(["road", "bin"]).z.agg(["size", "var", "mean"])
        dof = int((clusters["size"] - 1).sum())
        within = float(((clusters["size"] - 1) * clusters["var"].fillna(0)).sum()) / dof if dof else 1.0
        shared_fraction = float(np.clip(1 - within, 0, 1))
        matrix = clusters["mean"].unstack("road").reindex(columns=keys).fillna(0).to_numpy()
        if not len(matrix):
            matrix = np.zeros((1, len(keys)))
        matrix -= matrix.mean(axis=0)
        norms = np.sqrt(np.sum(matrix ** 2, axis=0))
        matrix /= np.where(norms > 0, norms, 1)
        correlation = matrix.T @ matrix
        correlation[np.diag_indices_from(correlation)] += (norms == 0)
    else:
        dof, shared_fraction, correlation = 0, 0.0, np.eye(len(keys))
    values, vectors = np.linalg.eigh(correlation)
    # A Gram correlation can be rank deficient; preserve its supported subspace.
    factor = vectors * np.sqrt(np.maximum(values, 0))
    return dict(model, ratios=pools, shared_factor=factor.tolist(), shared_fraction=shared_fraction,
        within_bin_dof=dof, prequential_noise_legs=prequential_legs,
        noise_latest_completion_s=float(residuals.end_s.max()) if len(residuals) else None)


class ConditionedTravelPath:
    def __init__(self, model, mode, draw):
        self.model, self.mode = model, mode
        self.rng = np.random.default_rng(20261020 + draw)
        self.indices = {key: i for i, key in enumerate(model["keys"])}
        steps = int(np.ceil(model["horizon_s"] / model["bin_s"]))
        self.shared = self.rng.standard_normal((steps, len(self.indices))) @ np.asarray(model["shared_factor"]).T

    def mean(self, key, time):
        return self.model["means"][key]

    def __call__(self, vehicle, time, fraction):
        key = road_key(vehicle["direction"], vehicle["travel_station_id"])
        duration = self.mean(key, time)
        if self.mode != "conditioned_mean":
            z = float(self.rng.standard_normal())
            if self.mode == "conditioned_shared":
                lag = min(int((time - self.model["now_s"]) // self.model["bin_s"]), len(self.shared) - 1)
                weight = self.model["shared_fraction"]
                z = math.sqrt(weight) * self.shared[lag, self.indices[key]] + math.sqrt(1 - weight) * z
            quantile = .5 * (1 + math.erf(z / math.sqrt(2)))
            pool = self.model["ratios"][key]
            duration *= pool[min(int(quantile * len(pool)), len(pool) - 1)]
        return max(fraction * duration, self.model["dt_s"])
