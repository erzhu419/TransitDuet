"""Causal dynamic harmonic encoder for count/intensity streams.

This is the Transit-oriented harmonic state-space path from the development
manual: low frequency is a causal Fourier intensity state, while high
frequency is the one-step count residual.  The update uses a diagonal online
Newton step for Poisson or negative-binomial observations, so uncertainty and
forecast curvature reflect count noise rather than squared-error residuals.
"""

from __future__ import annotations

from typing import Any, Mapping
import math

import numpy as np

from .base import CausalSpectralEncoder
from .causal_ema import alpha_from_period
from .causal_fourier import harmonic_features
from ..core.types import ExogenousBin, FrequencyFeatures


class _PoissonHarmonicState:
    def __init__(
        self,
        dim: int,
        update_interval_s: float,
        period_s: float,
        fourier_k: int,
        learning_rate: float,
        ridge: float,
        observation_model: str,
        nb_dispersion: float,
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
        self.learning_rate = float(np.clip(learning_rate, 1e-6, 1.0))
        self.ridge = max(float(ridge), 1e-9)
        self.observation_model = str(observation_model or "poisson").lower()
        self.nb_dispersion = max(float(nb_dispersion), 1e-6)
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
        self.cov = np.eye(self.basis_dim, dtype=np.float64) * (1.0 / self.ridge)
        self.precision = np.ones((self.basis_dim, self.dim), dtype=np.float64) * self.ridge
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

    def _rate(self, eta: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(eta, dtype=np.float64), 0.0)

    def _variance(self, mu: np.ndarray) -> np.ndarray:
        if self.observation_model in {"nb", "negative_binomial", "neg_binomial"}:
            return mu + (mu * mu) / self.nb_dispersion
        return np.maximum(mu, 1e-6)

    def update(self, value: np.ndarray) -> None:
        y = np.maximum(np.asarray(value, dtype=np.float64).reshape(-1), 0.0)
        if y.size != self.dim:
            raise ValueError(f"expected x dim {self.dim}, got {y.size}")
        phi = self._phi(self.n)
        if self.n == 0:
            self.theta[0, :] = np.maximum(y, 0.0)
            self.raw = y.copy()
            self.low = y.copy()
            self.prev_low = y.copy()
            self.high = np.zeros(self.dim, dtype=np.float64)
            self.prev_high = np.zeros(self.dim, dtype=np.float64)
            self.mid = np.zeros(self.dim, dtype=np.float64)
            self.high_energy = np.zeros(self.dim, dtype=np.float64)
            self.uncertainty = np.sqrt(1.0 / np.maximum(np.mean(self.precision, axis=0), 1e-9))
            self.n = 1
            return
        pred_before = self._rate(phi @ self.theta)
        innovation = y - pred_before

        variance = self._variance(pred_before)
        weight = float(np.mean(1.0 / np.maximum(variance, 1e-6)))
        cov_phi = self.cov @ phi
        denom = max(0.999 + weight * float(phi @ cov_phi), 1e-9)
        gain = cov_phi * weight / denom
        self.theta += np.clip(self.learning_rate * np.outer(gain, innovation), -3.0, 3.0)
        self.cov = (self.cov - weight * np.outer(cov_phi, cov_phi) / denom) / 0.999
        diag_precision = np.maximum(np.diag(np.linalg.pinv(self.cov)), self.ridge)
        self.precision = np.repeat(diag_precision[:, None], self.dim, axis=1)
        pred_after = self._rate(phi @ self.theta)

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
        self.uncertainty = np.sqrt(1.0 / np.maximum(np.mean(self.precision, axis=0), 1e-9))
        self.n += 1

    def feature_object(self, timestamp: float, entity_id: Any) -> FrequencyFeatures:
        low_slope = self.low - self.prev_low if self.n > 1 else np.zeros_like(self.low)
        high_delta = self.high - self.prev_high if self.n > 1 else np.zeros_like(self.high)
        forecasts = []
        for i in range(1, self.forecast_steps + 1):
            forecasts.append(self._rate(self._phi(self.n + i) @ self.theta))
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
            metadata={
                "encoder": "causal_poisson_harmonic",
                "updates": self.n,
                "observation_model": self.observation_model,
            },
        )

    def promote_residual(self, strength: float = 1.0, gain: float = 0.10) -> float:
        if self.n <= 0:
            return 0.0
        strength = float(np.clip(strength, 0.0, 1.0))
        gain = max(float(gain), 0.0)
        if strength <= 0.0 or gain <= 0.0:
            return 0.0
        phi = self._phi(max(0, self.n - 1))
        low_before = self.low.copy()
        target = np.maximum(low_before + gain * strength * self.high, 0.0)
        delta_eta = target - np.maximum(low_before, 0.0)
        denom = max(float(phi @ phi), 1e-9)
        self.theta += phi[:, None] * (delta_eta / denom)[None, :]
        low_after = self._rate(phi @ self.theta)
        absorbed = low_after - low_before
        self.prev_low = self.low.copy()
        self.low = low_after
        self.prev_high = self.high.copy()
        self.high = self.high - absorbed
        self.mid = self.mid - absorbed
        self.high_energy = (
            (1.0 - self.energy_alpha) * self.high_energy
            + self.energy_alpha * (self.high ** 2)
        )
        return float(np.sqrt(np.mean(absorbed * absorbed)))


class CausalPoissonHarmonicEncoder(CausalSpectralEncoder):
    """Dynamic harmonic state-space encoder for nonnegative demand counts."""

    def __init__(
        self,
        update_interval_s: float = 60.0,
        period_s: float = 24 * 3600.0,
        fourier_k: int = 4,
        learning_rate: float = 0.4,
        ridge: float = 1.0,
        observation_model: str = "poisson",
        nb_dispersion: float = 20.0,
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
        self.learning_rate = float(learning_rate)
        self.ridge = float(ridge)
        self.observation_model = str(observation_model or "poisson").lower()
        self.nb_dispersion = float(nb_dispersion)
        self.residual_alpha = alpha_from_period(self.update_interval_s, residual_period_s)
        self.mid_alpha = alpha_from_period(self.update_interval_s, mid_period_s)
        self.energy_alpha = alpha_from_period(self.update_interval_s, energy_period_s)
        self.persistence_alpha = alpha_from_period(self.update_interval_s, persistence_period_s)
        self.persistence_threshold = max(float(persistence_threshold), 0.0)
        self.forecast_steps = max(
            1, int(round(float(forecast_horizon_s) / self.update_interval_s))
        )
        self._states: dict[Any, _PoissonHarmonicState] = {}
        self._latest_entity: Any = "global"
        self._latest_t = 0.0

    def reset(self, episode_id: int | None = None) -> None:
        self._states.clear()
        self._latest_entity = "global"
        self._latest_t = 0.0

    def _new_state(self, dim: int) -> _PoissonHarmonicState:
        return _PoissonHarmonicState(
            dim=dim,
            update_interval_s=self.update_interval_s,
            period_s=self.period_s,
            fourier_k=self.fourier_k,
            learning_rate=self.learning_rate,
            ridge=self.ridge,
            observation_model=self.observation_model,
            nb_dispersion=self.nb_dispersion,
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

    def promote_residual(
        self,
        entity_id: Any = "global",
        strength: float = 1.0,
        gain: float = 0.10,
    ) -> float:
        if entity_id == "global" and entity_id not in self._states:
            entity_id = self._latest_entity
        state = self._states.get(entity_id)
        if state is None:
            return 0.0
        return state.promote_residual(strength=strength, gain=gain)
