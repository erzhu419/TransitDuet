"""Shared data contracts for Freq-HRL.

The contracts keep the core package free of transit or trading assumptions.
Everything that crosses the core boundary is a dict-like object with explicit
timestamps, entity identifiers, and numeric arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def as_float_array(value: Any) -> np.ndarray:
    """Return a copied 1-D float array for scalar or vector inputs."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1).copy()


@dataclass
class ExogenousBin:
    """Causal aggregate of an exogenous stream at one time bin."""

    timestamp: float
    x_raw: np.ndarray
    entity_id: Any = "global"
    valid_mask: np.ndarray | None = None
    normalization_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default_t: float | None = None) -> "ExogenousBin":
        timestamp = data.get("timestamp", data.get("t", default_t))
        if timestamp is None:
            raise ValueError("ExogenousBin requires timestamp or t")
        x_raw = as_float_array(data.get("x_raw", data.get("value", data.get("x"))))
        valid_mask = data.get("valid_mask", None)
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
            if valid_mask.size != x_raw.size:
                raise ValueError("valid_mask must match x_raw dimension")
        return cls(
            timestamp=float(timestamp),
            entity_id=data.get("entity_id", "global"),
            x_raw=x_raw,
            valid_mask=valid_mask,
            normalization_context=dict(data.get("normalization_context", {}) or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "entity_id": self.entity_id,
            "x_raw": self.x_raw.copy(),
            "valid_mask": None if self.valid_mask is None else self.valid_mask.copy(),
            "normalization_context": dict(self.normalization_context),
        }


@dataclass
class FrequencyFeatures:
    """Causal frequency features exposed by an encoder."""

    timestamp: float
    entity_id: Any
    x_raw: np.ndarray
    x_low: np.ndarray
    x_low_slope: np.ndarray
    x_low_forecast: np.ndarray
    x_low_uncertainty: np.ndarray
    x_mid: np.ndarray
    x_high: np.ndarray
    x_high_delta: np.ndarray
    x_high_energy: np.ndarray
    x_high_persistence: np.ndarray
    shock_age: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "entity_id": self.entity_id,
            "x_raw": self.x_raw.copy(),
            "x_low": self.x_low.copy(),
            "x_low_slope": self.x_low_slope.copy(),
            "x_low_forecast": self.x_low_forecast.copy(),
            "x_low_uncertainty": self.x_low_uncertainty.copy(),
            "x_mid": self.x_mid.copy(),
            "x_high": self.x_high.copy(),
            "x_high_delta": self.x_high_delta.copy(),
            "x_high_energy": self.x_high_energy.copy(),
            "x_high_persistence": self.x_high_persistence.copy(),
            "shock_age": self.shock_age.copy(),
            **dict(self.metadata),
        }


@dataclass
class PromotionSignal:
    """Output of the high-frequency to low-frequency promotion gate."""

    promote: bool
    promotion_strength: float
    reason: str
    cooldown_remaining: float
    shock_age: float
    score: float
    direction: float = 0.0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "promote": bool(self.promote),
            "promotion_strength": float(self.promotion_strength),
            "reason": self.reason,
            "cooldown_remaining": float(self.cooldown_remaining),
            "shock_age": float(self.shock_age),
            "score": float(self.score),
            "direction": float(self.direction),
        }
