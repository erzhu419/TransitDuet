"""Causal demand-intensity estimators for FreqDuet.

The harmonic estimator is intentionally lightweight: recursive least squares on
``log1p(arrival_rate)`` with Fourier time-of-day bases. It gives a causal
low-frequency intensity forecast and an innovation residual without introducing
the complexity of a full particle-filter negative-binomial state-space model.
"""

import math
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8192)
def _cached_harmonic_features(step, update_interval_s, period_s, fourier_k):
    step = int(step)
    update_interval_s = float(update_interval_s)
    period_s = float(period_s)
    fourier_k = max(0, int(fourier_k))
    elapsed_s = float(step) * max(update_interval_s, 1e-6)
    period_s = max(period_s, max(update_interval_s, 1e-6))
    phase = 2.0 * math.pi * (elapsed_s % period_s) / period_s
    trend = 2.0 * ((elapsed_s % period_s) / period_s) - 1.0
    feats = [1.0, trend]
    for k in range(1, fourier_k + 1):
        feats.append(math.sin(k * phase))
        feats.append(math.cos(k * phase))
    arr = np.asarray(feats, dtype=np.float64)
    arr.setflags(write=False)
    return arr


def harmonic_features(step, update_interval_s, period_s, fourier_k):
    return _cached_harmonic_features(
        int(step), float(update_interval_s), float(period_s), int(fourier_k))


def fit_harmonic_prior(rates, update_interval_s, period_s=50400.0,
                       fourier_k=4, ridge=1e-2):
    """Fit Fourier coefficients for ``log1p(rate)`` from historical rates."""
    y = np.log1p(np.clip(np.asarray(rates, dtype=np.float64), 0.0, None))
    if y.size == 0:
        return np.zeros(2 + 2 * max(0, int(fourier_k)), dtype=np.float64)
    x = np.stack([
        harmonic_features(i, update_interval_s, period_s, fourier_k)
        for i in range(y.size)
    ])
    reg = max(float(ridge), 0.0) * np.eye(x.shape[1], dtype=np.float64)
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


class CausalHarmonicBandState:
    """Online harmonic smoother with innovation residuals.

    Interface matches the simple EMA/Haar band states used by
    ``DemandFrequencyTracker``: after each update it exposes ``low``, ``high``,
    slopes, and high-frequency energy.
    """

    def __init__(
        self,
        update_interval_s,
        period_s=50400.0,
        fourier_k=4,
        forgetting_factor=0.995,
        prior_var=100.0,
        energy_alpha=0.01,
        residual_alpha=0.2,
        prior_theta=None,
    ):
        self.update_interval_s = max(float(update_interval_s), 1e-6)
        self.period_s = max(float(period_s), self.update_interval_s)
        self.fourier_k = max(0, int(fourier_k))
        self.forgetting_factor = float(np.clip(forgetting_factor, 0.90, 0.9999))
        self.prior_var = max(float(prior_var), 1e-6)
        self.energy_alpha = float(np.clip(energy_alpha, 1e-6, 1.0))
        self.residual_alpha = float(np.clip(residual_alpha, 1e-6, 1.0))
        self.dim = 2 + 2 * self.fourier_k
        if prior_theta is None:
            self.initial_theta = np.zeros(self.dim, dtype=np.float64)
        else:
            theta = np.asarray(prior_theta, dtype=np.float64).reshape(-1)
            if theta.size != self.dim:
                raise ValueError(
                    f"harmonic prior has dim {theta.size}, expected {self.dim}")
            self.initial_theta = theta.copy()
        self.reset()

    def reset(self):
        self.theta = self.initial_theta.copy()
        self.cov = np.eye(self.dim, dtype=np.float64) * self.prior_var
        self.low = self._rate_from_log(float(self._features(0) @ self.theta))
        self.fast = 0.0
        self.prev_low = self.low
        self.high = 0.0
        self.prev_high = 0.0
        self.high_energy = 0.0
        self._raw_high = 0.0
        self.uncertainty = 1.0
        self.n = 0

    def _features(self, step):
        return harmonic_features(
            step, self.update_interval_s, self.period_s, self.fourier_k)

    def _rate_from_log(self, log_rate):
        x = float(log_rate)
        if x < -20.0:
            x = -20.0
        elif x > 20.0:
            x = 20.0
        return max(0.0, float(math.expm1(x)))

    def update(self, value):
        value = max(float(value), 0.0)
        phi = self._features(self.n)
        pred_log_before = float(phi @ self.theta)
        pred_rate_before = self._rate_from_log(pred_log_before)

        y = math.log1p(value)
        cov_phi = self.cov @ phi
        denom = self.forgetting_factor + float(phi @ cov_phi)
        gain = cov_phi / max(denom, 1e-9)
        innovation = y - pred_log_before
        self.theta = self.theta + gain * innovation
        phi_cov = phi @ self.cov
        self.cov = (
            self.cov - np.outer(gain, phi_cov)
        ) / self.forgetting_factor

        pred_log_after = float(phi @ self.theta)
        pred_rate_after = self._rate_from_log(pred_log_after)

        self.prev_low = self.low
        self.prev_high = self.high
        self.low = pred_rate_after
        self.fast = value
        self._raw_high = value - pred_rate_before
        if self.n == 0:
            self.high = self._raw_high
        else:
            self.high = (
                (1.0 - self.residual_alpha) * self.high
                + self.residual_alpha * self._raw_high
            )
        if self.n == 0:
            self.high_energy = 0.0
        else:
            self.high_energy = (
                (1.0 - self.energy_alpha) * self.high_energy
                + self.energy_alpha * (self.high ** 2)
            )
        self.uncertainty = float(math.sqrt(max(phi @ self.cov @ phi, 0.0)))
        self.n += 1

    @property
    def low_slope(self):
        return self.low - self.prev_low if self.n > 1 else 0.0

    @property
    def high_slope(self):
        return self.high - self.prev_high if self.n > 1 else 0.0
