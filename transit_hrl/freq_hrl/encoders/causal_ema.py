"""Causal EMA frequency splitter.

This is the Level-0 encoder in the development manual.  It is intentionally
simple, fully online, and domain-agnostic; transit demand, market returns, or
any other exogenous stream can pass through the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

import numpy as np

from .base import CausalSpectralEncoder
from ..core.types import ExogenousBin, FrequencyFeatures


def alpha_from_period(update_interval_s: float, period_s: float) -> float:
    period_s = max(float(period_s), 1e-9)
    update_interval_s = max(float(update_interval_s), 1e-9)
    return float(1.0 - math.exp(-update_interval_s / period_s))


@dataclass
class _EMAState:
    dim: int
    low_alpha: float
    fast_alpha: float
    mid_alpha: float
    energy_alpha: float
    persistence_alpha: float
    persistence_threshold: float
    shock_age_norm: float
    forecast_steps: int

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.raw = np.zeros(self.dim, dtype=np.float64)
        self.low = np.zeros(self.dim, dtype=np.float64)
        self.fast = np.zeros(self.dim, dtype=np.float64)
        self.mid = np.zeros(self.dim, dtype=np.float64)
        self.prev_low = np.zeros(self.dim, dtype=np.float64)
        self.high = np.zeros(self.dim, dtype=np.float64)
        self.prev_high = np.zeros(self.dim, dtype=np.float64)
        self.high_energy = np.zeros(self.dim, dtype=np.float64)
        self.high_persistence = np.zeros(self.dim, dtype=np.float64)
        self.shock_age = np.zeros(self.dim, dtype=np.float64)
        self.n = 0

    def update(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if value.size != self.dim:
            raise ValueError(f"expected x dim {self.dim}, got {value.size}")
        self.raw = value.copy()
        if self.n == 0:
            self.low = value.copy()
            self.fast = value.copy()
            self.mid.fill(0.0)
            self.high.fill(0.0)
            self.prev_low = self.low.copy()
            self.prev_high = self.high.copy()
            self.n = 1
            return

        self.prev_low = self.low.copy()
        self.prev_high = self.high.copy()
        self.low = (1.0 - self.low_alpha) * self.low + self.low_alpha * value
        self.fast = (1.0 - self.fast_alpha) * self.fast + self.fast_alpha * value
        self.high = self.fast - self.low
        self.mid = (1.0 - self.mid_alpha) * self.mid + self.mid_alpha * (value - self.fast)
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
        self.n += 1

    def feature_object(self, timestamp: float, entity_id: Any) -> FrequencyFeatures:
        low_slope = self.low - self.prev_low if self.n > 1 else np.zeros_like(self.low)
        high_delta = self.high - self.prev_high if self.n > 1 else np.zeros_like(self.high)
        horizon = max(1, int(self.forecast_steps))
        multipliers = np.arange(1, horizon + 1, dtype=np.float64)[:, None]
        forecast = self.low[None, :] + multipliers * low_slope[None, :]
        forecast = np.maximum(forecast, 0.0) if np.all(self.raw >= 0.0) else forecast
        shock_age = np.clip(self.shock_age / max(self.shock_age_norm, 1.0), 0.0, 1.0)
        return FrequencyFeatures(
            timestamp=float(timestamp),
            entity_id=entity_id,
            x_raw=self.raw,
            x_low=self.low,
            x_low_slope=low_slope,
            x_low_forecast=forecast,
            x_low_uncertainty=np.sqrt(np.maximum(self.high_energy, 0.0)),
            x_mid=self.mid,
            x_high=self.high,
            x_high_delta=high_delta,
            x_high_energy=self.high_energy,
            x_high_persistence=self.high_persistence,
            shock_age=shock_age,
            metadata={"encoder": "causal_ema", "updates": self.n},
        )

    def promote_residual(self, strength: float = 1.0, gain: float = 0.10) -> float:
        """Causally absorb part of persistent HF residual into the LF state."""
        if self.n <= 0:
            return 0.0
        strength = float(np.clip(strength, 0.0, 1.0))
        gain = max(float(gain), 0.0)
        if strength <= 0.0 or gain <= 0.0:
            return 0.0
        absorbed = gain * strength * self.high
        self.prev_low = self.low.copy()
        self.prev_high = self.high.copy()
        self.low = self.low + absorbed
        self.high = self.fast - self.low
        self.mid = self.mid - absorbed
        self.high_energy = (
            (1.0 - self.energy_alpha) * self.high_energy
            + self.energy_alpha * (self.high ** 2)
        )
        return float(np.sqrt(np.mean(absorbed * absorbed)))


class CausalEMAEncoder(CausalSpectralEncoder):
    """Generic causal low/mid/high splitter using trailing EMA filters."""

    def __init__(
        self,
        update_interval_s: float = 60.0,
        low_period_s: float = 7200.0,
        fast_period_s: float = 300.0,
        mid_period_s: float = 1800.0,
        energy_period_s: float = 600.0,
        persistence_period_s: float = 900.0,
        persistence_threshold: float = 1.0,
        forecast_horizon_s: float = 3600.0,
    ) -> None:
        self.update_interval_s = max(float(update_interval_s), 1e-9)
        self.low_alpha = alpha_from_period(self.update_interval_s, low_period_s)
        self.fast_alpha = alpha_from_period(self.update_interval_s, fast_period_s)
        self.mid_alpha = alpha_from_period(self.update_interval_s, mid_period_s)
        self.energy_alpha = alpha_from_period(self.update_interval_s, energy_period_s)
        self.persistence_alpha = alpha_from_period(
            self.update_interval_s, persistence_period_s
        )
        self.persistence_threshold = max(float(persistence_threshold), 0.0)
        self.forecast_steps = max(
            1, int(round(float(forecast_horizon_s) / self.update_interval_s))
        )
        self._states: dict[Any, _EMAState] = {}
        self._latest_entity: Any = "global"
        self._latest_t = 0.0

    def reset(self, episode_id: int | None = None) -> None:
        self._states.clear()
        self._latest_entity = "global"
        self._latest_t = 0.0

    def _new_state(self, dim: int) -> _EMAState:
        return _EMAState(
            dim=dim,
            low_alpha=self.low_alpha,
            fast_alpha=self.fast_alpha,
            mid_alpha=self.mid_alpha,
            energy_alpha=self.energy_alpha,
            persistence_alpha=self.persistence_alpha,
            persistence_threshold=self.persistence_threshold,
            shock_age_norm=max(1.0, self.forecast_steps),
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
