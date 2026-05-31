"""Causal dynamic Fourier encoder.

The encoder uses recursive least squares on harmonic time bases.  Unlike the
transit-specific FreqDuet harmonic demand estimator, this version operates on
generic signed vectors, so it can be reused for market returns or any other
exogenous time-series features.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping
import math

import numpy as np

from .base import CausalSpectralEncoder
from .causal_ema import alpha_from_period
from ..core.types import ExogenousBin, FrequencyFeatures


@lru_cache(maxsize=16384)
def harmonic_features(step: int, update_interval_s: float, period_s: float, fourier_k: int) -> np.ndarray:
    step = int(step)
    update_interval_s = float(update_interval_s)
    period_s = max(float(period_s), update_interval_s)
    fourier_k = max(0, int(fourier_k))
    elapsed_s = float(step) * max(update_interval_s, 1e-9)
    phase = 2.0 * math.pi * (elapsed_s % period_s) / period_s
    trend = 2.0 * ((elapsed_s % period_s) / period_s) - 1.0
    values = [1.0, trend]
    for k in range(1, fourier_k + 1):
        values.append(math.sin(k * phase))
        values.append(math.cos(k * phase))
    arr = np.asarray(values, dtype=np.float64)
    arr.setflags(write=False)
    return arr


class _FourierState:
    def __init__(
        self,
        dim: int,
        update_interval_s: float,
        period_s: float,
        fourier_k: int,
        forgetting_factor: float,
        prior_var: float,
        residual_alpha: float,
        mid_alpha: float,
        energy_alpha: float,
        persistence_alpha: float,
        persistence_threshold: float,
        forecast_steps: int,
    ) -> None:
        self.dim = int(dim)
        self.update_interval_s = float(update_interval_s)
        self.period_s = float(period_s)
        self.fourier_k = int(fourier_k)
        self.forgetting_factor = float(np.clip(forgetting_factor, 0.90, 0.9999))
        self.prior_var = max(float(prior_var), 1e-9)
        self.residual_alpha = float(np.clip(residual_alpha, 1e-9, 1.0))
        self.mid_alpha = float(np.clip(mid_alpha, 1e-9, 1.0))
        self.energy_alpha = float(np.clip(energy_alpha, 1e-9, 1.0))
        self.persistence_alpha = float(np.clip(persistence_alpha, 1e-9, 1.0))
        self.persistence_threshold = max(float(persistence_threshold), 0.0)
        self.forecast_steps = max(1, int(forecast_steps))
        self.basis_dim = 2 + 2 * self.fourier_k
        self.reset()

    def reset(self) -> None:
        self.theta = np.zeros((self.basis_dim, self.dim), dtype=np.float64)
        self.cov = np.eye(self.basis_dim, dtype=np.float64) * self.prior_var
        self.raw = np.zeros(self.dim, dtype=np.float64)
        self.low = np.zeros(self.dim, dtype=np.float64)
        self.prev_low = np.zeros(self.dim, dtype=np.float64)
        self.high = np.zeros(self.dim, dtype=np.float64)
        self.prev_high = np.zeros(self.dim, dtype=np.float64)
        self.mid = np.zeros(self.dim, dtype=np.float64)
        self.high_energy = np.zeros(self.dim, dtype=np.float64)
        self.high_persistence = np.zeros(self.dim, dtype=np.float64)
        self.shock_age = np.zeros(self.dim, dtype=np.float64)
        self.uncertainty = np.ones(self.dim, dtype=np.float64)
        self.n = 0

    def _phi(self, step: int) -> np.ndarray:
        return harmonic_features(step, self.update_interval_s, self.period_s, self.fourier_k)

    def update(self, value: np.ndarray) -> None:
        y = np.asarray(value, dtype=np.float64).reshape(-1)
        if y.size != self.dim:
            raise ValueError(f"expected x dim {self.dim}, got {y.size}")
        phi = self._phi(self.n)
        pred_before = phi @ self.theta
        cov_phi = self.cov @ phi
        denom = max(self.forgetting_factor + float(phi @ cov_phi), 1e-12)
        innovation = y - pred_before
        self.theta += np.outer(cov_phi / denom, innovation)
        self.cov = (self.cov - np.outer(cov_phi, cov_phi) / denom) / self.forgetting_factor
        pred_after = phi @ self.theta

        self.raw = y.copy()
        self.prev_low = self.low.copy()
        self.prev_high = self.high.copy()
        self.low = pred_after
        if self.n == 0:
            self.high = innovation.copy()
            self.mid = innovation.copy()
        else:
            self.high = (1.0 - self.residual_alpha) * self.high + self.residual_alpha * innovation
            self.mid = (1.0 - self.mid_alpha) * self.mid + self.mid_alpha * innovation
        self.high_energy = (
            (1.0 - self.energy_alpha) * self.high_energy
            + self.energy_alpha * (self.high ** 2)
        )
        active = (np.abs(self.high) >= self.persistence_threshold).astype(np.float64)
        self.high_persistence = (
            (1.0 - self.persistence_alpha) * self.high_persistence
            + self.persistence_alpha * active
        )
        self.shock_age = np.where(active > 0.0, self.shock_age + 1.0, 0.0)
        self.uncertainty = np.full(
            self.dim, math.sqrt(max(float(phi @ (self.cov @ phi)), 0.0)),
            dtype=np.float64,
        )
        self.n += 1

    def feature_object(self, timestamp: float, entity_id: Any) -> FrequencyFeatures:
        low_slope = self.low - self.prev_low if self.n > 1 else np.zeros_like(self.low)
        high_delta = self.high - self.prev_high if self.n > 1 else np.zeros_like(self.high)
        forecasts = []
        for i in range(1, self.forecast_steps + 1):
            forecasts.append(self._phi(self.n + i) @ self.theta)
        shock_age_norm = np.clip(self.shock_age / max(self.forecast_steps, 1), 0.0, 1.0)
        return FrequencyFeatures(
            timestamp=float(timestamp),
            entity_id=entity_id,
            x_raw=self.raw,
            x_low=self.low,
            x_low_slope=low_slope,
            x_low_forecast=np.asarray(forecasts, dtype=np.float64),
            x_low_uncertainty=self.uncertainty,
            x_mid=self.mid,
            x_high=self.high,
            x_high_delta=high_delta,
            x_high_energy=self.high_energy,
            x_high_persistence=self.high_persistence,
            shock_age=shock_age_norm,
            metadata={"encoder": "causal_fourier", "updates": self.n},
        )


class CausalFourierEncoder(CausalSpectralEncoder):
    """Recursive harmonic trend encoder with causal innovation residuals."""

    def __init__(
        self,
        update_interval_s: float = 60.0,
        period_s: float = 24 * 3600.0,
        fourier_k: int = 4,
        forgetting_factor: float = 0.995,
        prior_var: float = 100.0,
        residual_period_s: float = 300.0,
        mid_period_s: float = 1800.0,
        energy_period_s: float = 600.0,
        persistence_period_s: float = 900.0,
        persistence_threshold: float = 1.0,
        forecast_horizon_s: float = 3600.0,
    ) -> None:
        self.update_interval_s = max(float(update_interval_s), 1e-9)
        self.period_s = max(float(period_s), self.update_interval_s)
        self.fourier_k = max(0, int(fourier_k))
        self.forgetting_factor = float(forgetting_factor)
        self.prior_var = float(prior_var)
        self.residual_alpha = alpha_from_period(self.update_interval_s, residual_period_s)
        self.mid_alpha = alpha_from_period(self.update_interval_s, mid_period_s)
        self.energy_alpha = alpha_from_period(self.update_interval_s, energy_period_s)
        self.persistence_alpha = alpha_from_period(self.update_interval_s, persistence_period_s)
        self.persistence_threshold = max(float(persistence_threshold), 0.0)
        self.forecast_steps = max(
            1, int(round(float(forecast_horizon_s) / self.update_interval_s))
        )
        self._states: dict[Any, _FourierState] = {}
        self._latest_entity: Any = "global"
        self._latest_t = 0.0

    def reset(self, episode_id: int | None = None) -> None:
        self._states.clear()
        self._latest_entity = "global"
        self._latest_t = 0.0

    def _new_state(self, dim: int) -> _FourierState:
        return _FourierState(
            dim=dim,
            update_interval_s=self.update_interval_s,
            period_s=self.period_s,
            fourier_k=self.fourier_k,
            forgetting_factor=self.forgetting_factor,
            prior_var=self.prior_var,
            residual_alpha=self.residual_alpha,
            mid_alpha=self.mid_alpha,
            energy_alpha=self.energy_alpha,
            persistence_alpha=self.persistence_alpha,
            persistence_threshold=self.persistence_threshold,
            forecast_steps=self.forecast_steps,
        )

    def update(self, x_bin: Mapping[str, Any], t: float | None = None) -> None:
        bin_obj = ExogenousBin.from_mapping(x_bin, default_t=t)
        state = self._states.get(bin_obj.entity_id)
        if state is None:
            state = self._new_state(bin_obj.x_raw.size)
            self._states[bin_obj.entity_id] = state
        state.update(bin_obj.x_raw)
        self._latest_entity = bin_obj.entity_id
        self._latest_t = bin_obj.timestamp

    def features(self, t: float | None = None, entity_id: Any = "global") -> dict[str, Any]:
        if entity_id == "global" and entity_id not in self._states:
            entity_id = self._latest_entity
        state = self._states.get(entity_id)
        if state is None:
            raise KeyError(f"no features for entity_id={entity_id!r}")
        timestamp = self._latest_t if t is None else float(t)
        return state.feature_object(timestamp, entity_id).to_mapping()
