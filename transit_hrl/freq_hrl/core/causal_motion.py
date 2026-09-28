"""Action-independent motion forecasts from observed causal velocity lags."""

from __future__ import annotations

import numpy as np


class CausalMotionForecaster:
    def __init__(self, *, observed_dim, velocity_channels, horizon_steps,
                 dt_seconds, velocity_lags=(1, 10, 25, 50), ridge_alpha=1.):
        self.observed_dim = observed_dim
        self.velocity_channels = tuple(velocity_channels)
        self.horizon_steps = tuple(horizon_steps)
        self.velocity_lags = tuple(velocity_lags)
        self.dt_seconds, self.ridge_alpha = float(dt_seconds), float(ridge_alpha)
        if (self.dt_seconds <= 0 or self.ridge_alpha <= 0
                or not self.horizon_steps or min(self.horizon_steps) <= 0
                or not self.velocity_lags or min(self.velocity_lags) <= 0):
            raise ValueError("motion time scales and ridge penalty must be positive")
        self.fitted = None

    def features(self, observed_history):
        history = np.asarray(observed_history, dtype=np.float64)
        if (history.ndim != 3 or history.shape[2] != self.observed_dim
                or history.shape[1] <= max(self.velocity_lags)):
            raise ValueError("motion forecast requires complete observed velocity lags")
        current = history[:, -1]
        slopes = [(current[:, self.velocity_channels] - history[:, -1-lag, self.velocity_channels])
                  / (lag*self.dt_seconds) for lag in self.velocity_lags]
        return np.column_stack((current, *slopes))

    def fit(self, observed_history, future_rates):
        design = self.features(observed_history)
        rates = np.asarray(future_rates, dtype=np.float64)
        if rates.shape != (len(design), len(self.horizon_steps), self.observed_dim):
            raise ValueError("motion labels differ from the registered forecast horizons")
        mean, scale = design.mean(axis=0), design.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.)
        z = np.column_stack((np.ones(len(design)), (design-mean)/scale))
        penalty = np.eye(z.shape[1])*self.ridge_alpha
        penalty[0, 0] = 0.
        weights = np.linalg.solve(z.T@z+penalty, z.T@rates.reshape(len(design), -1))
        self.fitted = {"feature_mean":mean, "feature_scale":scale, "weights":weights,
                       "training_rows":len(design), "parameter_count":weights.size,
                       "ridge_alpha":self.ridge_alpha}
        return self

    def predict_rates(self, observed_history):
        if self.fitted is None:
            raise RuntimeError("motion forecaster has not been fitted")
        design = self.features(observed_history)
        model = self.fitted
        z = np.column_stack((np.ones(len(design)), (design-model["feature_mean"])/model["feature_scale"]))
        return (z@model["weights"]).reshape(len(design), len(self.horizon_steps), self.observed_dim)

    def predict_levels(self, observed_history):
        rates = self.predict_rates(observed_history)
        durations = np.asarray(self.horizon_steps)*self.dt_seconds
        return np.asarray(observed_history)[:, -1:, :] + rates*durations[None, :, None]
