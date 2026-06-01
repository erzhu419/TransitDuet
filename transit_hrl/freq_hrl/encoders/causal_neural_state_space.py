"""Causal neural state-space encoder with a lightweight PINN constraint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import CausalSpectralEncoder
from .causal_ema import alpha_from_period
from ..core.types import ExogenousBin, FrequencyFeatures


def _stable_matrix(rows: int, cols: int, scale: float, offset: int = 0) -> np.ndarray:
    idx = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) + float(offset)
    return scale * np.sin(0.37 * idx + 0.11 * (idx % 7))


@dataclass
class _NeuralStateSpaceState:
    dim: int
    hidden_dim: int
    low_alpha: float
    residual_alpha: float
    mid_alpha: float
    slope_alpha: float
    energy_alpha: float
    persistence_alpha: float
    persistence_threshold: float
    shock_age_norm: float
    forecast_steps: int
    learn_rate: float
    ridge: float
    pinn_gain: float
    max_correction: float

    def __post_init__(self) -> None:
        self.w_in = _stable_matrix(self.hidden_dim, self.dim, 0.20, offset=1)
        self.w_h = _stable_matrix(self.hidden_dim, self.hidden_dim, 0.08, offset=101)
        self.w_slope = _stable_matrix(self.hidden_dim, self.dim, 0.10, offset=211)
        self.w_out = np.zeros((self.dim, self.hidden_dim), dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.raw = np.zeros(self.dim, dtype=np.float64)
        self.low = np.zeros(self.dim, dtype=np.float64)
        self.prev_low = np.zeros(self.dim, dtype=np.float64)
        self.prev2_low = np.zeros(self.dim, dtype=np.float64)
        self.slope = np.zeros(self.dim, dtype=np.float64)
        self.hidden = np.zeros(self.hidden_dim, dtype=np.float64)
        self.high = np.zeros(self.dim, dtype=np.float64)
        self.prev_high = np.zeros(self.dim, dtype=np.float64)
        self.mid = np.zeros(self.dim, dtype=np.float64)
        self.high_energy = np.zeros(self.dim, dtype=np.float64)
        self.physics_energy = np.zeros(self.dim, dtype=np.float64)
        self.high_persistence = np.zeros(self.dim, dtype=np.float64)
        self.shock_age = np.zeros(self.dim, dtype=np.float64)
        self.n = 0

    def update(self, value: np.ndarray) -> None:
        y = np.asarray(value, dtype=np.float64).reshape(-1)
        if y.size != self.dim:
            raise ValueError(f"expected x dim {self.dim}, got {y.size}")
        self.raw = y.copy()
        if self.n == 0:
            self.low = y.copy()
            self.prev_low = self.low.copy()
            self.prev2_low = self.low.copy()
            self.n = 1
            return

        self.prev2_low = self.prev_low.copy()
        self.prev_low = self.low.copy()
        self.prev_high = self.high.copy()
        predicted_low = self.low + self.slope
        innovation = y - predicted_low
        hidden_input = (
            self.w_in @ innovation
            + self.w_h @ self.hidden
            + self.w_slope @ self.slope
        )
        next_hidden = np.tanh(hidden_input)
        neural_correction = self.w_out @ next_hidden
        neural_correction = np.clip(
            neural_correction,
            -float(self.max_correction),
            float(self.max_correction),
        )
        residual = innovation - neural_correction
        physics_residual = predicted_low - 2.0 * self.prev_low + self.prev2_low
        self.low = (
            predicted_low
            + self.low_alpha * residual
            - self.pinn_gain * physics_residual
        )
        low_delta = self.low - self.prev_low
        self.slope = (1.0 - self.slope_alpha) * self.slope + self.slope_alpha * low_delta
        self.high = (
            (1.0 - self.residual_alpha) * self.high
            + self.residual_alpha * residual
        )
        self.mid = (
            (1.0 - self.mid_alpha) * self.mid
            + self.mid_alpha * (y - self.low - self.high)
        )

        denom = float(np.dot(next_hidden, next_hidden) + self.ridge)
        self.w_out += self.learn_rate * np.outer(residual, next_hidden) / denom
        self.hidden = next_hidden

        self.high_energy = (
            (1.0 - self.energy_alpha) * self.high_energy
            + self.energy_alpha * (self.high ** 2)
        )
        self.physics_energy = (
            (1.0 - self.energy_alpha) * self.physics_energy
            + self.energy_alpha * (physics_residual ** 2)
        )
        active = (np.abs(self.high) >= self.persistence_threshold).astype(np.float64)
        self.high_persistence = (
            (1.0 - self.persistence_alpha) * self.high_persistence
            + self.persistence_alpha * active
        )
        self.shock_age = np.where(active > 0.0, self.shock_age + 1.0, 0.0)
        self.n += 1

    def feature_object(self, timestamp: float, entity_id: Any) -> FrequencyFeatures:
        high_delta = self.high - self.prev_high if self.n > 1 else np.zeros_like(self.high)
        horizon = max(1, int(self.forecast_steps))
        multipliers = np.arange(1, horizon + 1, dtype=np.float64)[:, None]
        forecast = self.low[None, :] + multipliers * self.slope[None, :]
        if np.all(self.raw >= 0.0):
            forecast = np.maximum(forecast, 0.0)
        shock_age = np.clip(self.shock_age / max(self.shock_age_norm, 1.0), 0.0, 1.0)
        uncertainty = np.sqrt(np.maximum(self.high_energy + self.physics_energy, 0.0) + self.ridge)
        return FrequencyFeatures(
            timestamp=float(timestamp),
            entity_id=entity_id,
            x_raw=self.raw,
            x_low=self.low,
            x_low_slope=self.slope,
            x_low_forecast=forecast,
            x_low_uncertainty=uncertainty,
            x_mid=self.mid,
            x_high=self.high,
            x_high_delta=high_delta,
            x_high_energy=self.high_energy,
            x_high_persistence=self.high_persistence,
            shock_age=shock_age,
            metadata={
                "encoder": "causal_neural_state_space",
                "updates": self.n,
                "hidden_dim": int(self.hidden_dim),
                "pinn_residual": float(np.sqrt(np.mean(self.physics_energy))),
                "output_weight_norm": float(np.sqrt(np.mean(self.w_out * self.w_out))),
            },
        )

    def promote_residual(self, strength: float = 1.0, gain: float = 0.10) -> float:
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
        self.high = self.high - absorbed
        self.mid = self.mid - absorbed
        self.high_energy = (
            (1.0 - self.energy_alpha) * self.high_energy
            + self.energy_alpha * (self.high ** 2)
        )
        return float(np.sqrt(np.mean(absorbed * absorbed)))


class CausalNeuralStateSpaceEncoder(CausalSpectralEncoder):
    """Online neural state-space splitter with a causal physics residual penalty.

    The recurrent hidden state and output head learn only from past innovations.
    The PINN-style term penalizes curvature in the low-frequency state, which
    keeps the upper stream smooth while leaving high-frequency innovations to
    the lower controller.
    """

    def __init__(
        self,
        update_interval_s: float = 60.0,
        hidden_dim: int = 8,
        low_period_s: float = 7200.0,
        residual_period_s: float = 300.0,
        mid_period_s: float = 1800.0,
        slope_period_s: float = 1800.0,
        energy_period_s: float = 600.0,
        persistence_period_s: float = 900.0,
        persistence_threshold: float = 1.0,
        forecast_horizon_s: float = 3600.0,
        learn_rate: float = 0.01,
        ridge: float = 1e-5,
        pinn_gain: float = 0.05,
        max_correction: float = 5.0,
    ) -> None:
        self.update_interval_s = max(float(update_interval_s), 1e-9)
        self.hidden_dim = max(1, int(hidden_dim))
        self.low_alpha = alpha_from_period(self.update_interval_s, low_period_s)
        self.residual_alpha = alpha_from_period(self.update_interval_s, residual_period_s)
        self.mid_alpha = alpha_from_period(self.update_interval_s, mid_period_s)
        self.slope_alpha = alpha_from_period(self.update_interval_s, slope_period_s)
        self.energy_alpha = alpha_from_period(self.update_interval_s, energy_period_s)
        self.persistence_alpha = alpha_from_period(self.update_interval_s, persistence_period_s)
        self.persistence_threshold = max(float(persistence_threshold), 0.0)
        self.forecast_steps = max(1, int(round(float(forecast_horizon_s) / self.update_interval_s)))
        self.learn_rate = max(float(learn_rate), 0.0)
        self.ridge = max(float(ridge), 1e-12)
        self.pinn_gain = max(float(pinn_gain), 0.0)
        self.max_correction = max(float(max_correction), 1e-9)
        self._states: dict[Any, _NeuralStateSpaceState] = {}
        self._latest_entity: Any = "global"
        self._latest_t = 0.0

    def reset(self, episode_id: int | None = None) -> None:
        self._states.clear()
        self._latest_entity = "global"
        self._latest_t = 0.0

    def _new_state(self, dim: int) -> _NeuralStateSpaceState:
        return _NeuralStateSpaceState(
            dim=dim,
            hidden_dim=self.hidden_dim,
            low_alpha=self.low_alpha,
            residual_alpha=self.residual_alpha,
            mid_alpha=self.mid_alpha,
            slope_alpha=self.slope_alpha,
            energy_alpha=self.energy_alpha,
            persistence_alpha=self.persistence_alpha,
            persistence_threshold=self.persistence_threshold,
            shock_age_norm=max(1.0, self.forecast_steps),
            forecast_steps=self.forecast_steps,
            learn_rate=self.learn_rate,
            ridge=self.ridge,
            pinn_gain=self.pinn_gain,
            max_correction=self.max_correction,
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
