"""Plan-conditioned response credit from frozen causal target forecasts."""

from __future__ import annotations

import numpy as np


def plan_response_features(*, current_state, position, velocity, retained_plan,
                           candidate_plan, observed_target, forecast_targets):
    current, position, velocity, old, new, target, future = (
        np.asarray(value, dtype=np.float64) for value in
        (current_state, position, velocity, retained_plan, candidate_plan, observed_target, forecast_targets))
    displacement = future-target[:, None]
    old_now = np.sum((target-old)**2, axis=1)
    new_now = np.sum((target-new)**2, axis=1)
    old_change = np.sum((future-old[:, None])**2, axis=2)-old_now[:, None]
    new_change = np.sum((future-new[:, None])**2, axis=2)-new_now[:, None]
    velocity_alignment = np.sum(displacement*velocity[:, None], axis=2)
    return np.column_stack((current, new-position, new-old, old_now, new_now,
                            displacement.reshape(len(current), -1), old_change, new_change, velocity_alignment))


class PlanResponseCritic:
    def __init__(self, *, durations_seconds, ridge_alpha=1.):
        self.durations = np.asarray(durations_seconds, dtype=np.float64)
        self.ridge_alpha = float(ridge_alpha)
        if np.any(self.durations <= 0) or self.ridge_alpha <= 0:
            raise ValueError("plan-response durations and ridge penalty must be positive")
        self.fitted = None

    def fit(self, features, keep_minus_renew_curves):
        x, curves = np.asarray(features, dtype=np.float64), np.asarray(keep_minus_renew_curves, dtype=np.float64)
        if x.ndim != 2 or curves.shape != (len(x), len(self.durations)):
            raise ValueError("plan-response labels differ from the response horizons")
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.)
        design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
        penalty = np.eye(design.shape[1])*self.ridge_alpha
        penalty[0, 0] = 0.
        weights = np.linalg.solve(design.T@design+penalty, design.T@(curves/self.durations))
        self.fitted = {"feature_mean":mean, "feature_scale":scale, "weights":weights,
                       "training_rows":len(x), "parameter_count":weights.size, "ridge_alpha":self.ridge_alpha}
        return self

    def predict_rates(self, features):
        if self.fitted is None:
            raise RuntimeError("plan-response critic has not been fitted")
        x, model = np.asarray(features, dtype=np.float64), self.fitted
        design = np.column_stack((np.ones(len(x)), (x-model["feature_mean"])/model["feature_scale"]))
        return design@model["weights"]
