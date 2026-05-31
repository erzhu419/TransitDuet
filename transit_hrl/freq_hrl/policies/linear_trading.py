"""Trainable linear trading policies for the Freq-HRL interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .interfaces import HighLevelDecision, HighLevelPlanner, LowLevelController, LowLevelDecision
from .trading_heuristic import normalize_target


@dataclass
class LinearTradingParams:
    upper_low: float = 1.0
    upper_mid: float = 0.5
    upper_promotion: float = 0.5
    upper_bias: float = 0.0
    lower_base: float = 0.65
    lower_align: float = 0.25
    residual_high: float = 0.0

    @classmethod
    def from_vector(cls, value: np.ndarray) -> "LinearTradingParams":
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 7:
            raise ValueError(f"expected 7 linear trading parameters, got {arr.size}")
        return cls(*[float(v) for v in arr])

    def to_vector(self) -> np.ndarray:
        return np.asarray([
            self.upper_low,
            self.upper_mid,
            self.upper_promotion,
            self.upper_bias,
            self.lower_base,
            self.lower_align,
            self.residual_high,
        ], dtype=np.float64)

    def to_mapping(self) -> dict[str, float]:
        return {
            "upper_low": float(self.upper_low),
            "upper_mid": float(self.upper_mid),
            "upper_promotion": float(self.upper_promotion),
            "upper_bias": float(self.upper_bias),
            "lower_base": float(self.lower_base),
            "lower_align": float(self.lower_align),
            "residual_high": float(self.residual_high),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LinearTradingParams":
        defaults = cls().to_mapping()
        defaults.update({key: float(val) for key, val in dict(value).items()})
        return cls(**defaults)


def _arr(value: Any, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if dim is not None and arr.size != dim:
        arr = np.resize(arr, dim)
    return arr


class LinearFrequencyTradingPlanner(HighLevelPlanner):
    """Asset-shared linear low-frequency planner."""

    def __init__(self, params: LinearTradingParams | None = None, scale: float = 0.0014) -> None:
        self.params = params or LinearTradingParams()
        self.scale = float(scale)

    def plan(
        self,
        observation: Mapping[str, Any],
        upper_features: np.ndarray,
        context: Mapping[str, Any] | None = None,
    ) -> HighLevelDecision:
        context = context or {}
        freq = dict(context.get("frequency", {}) or {})
        raw = _arr(observation.get("raw_signal", []))
        dim = raw.size if raw.size else int(context.get("n_assets", 1))
        x_low = _arr(freq.get("x_low", np.zeros(dim)), dim)
        x_mid = _arr(freq.get("x_mid", np.zeros(dim)), dim)
        promotion = dict(freq.get("promotion", {}) or {})
        promoted = bool(promotion.get("promote", False))
        strength = float(promotion.get("promotion_strength", 0.0)) if promoted else 0.0
        p = self.params
        signal = (
            p.upper_low * x_low
            + (p.upper_mid + p.upper_promotion * strength) * x_mid
            + p.upper_bias * self.scale
        )
        target = normalize_target(signal, scale=self.scale)
        return HighLevelDecision(
            action=target,
            plan={"type": "linear_target_weights", "target": target.copy()},
            metadata={"promotion_used": promoted, "promotion_strength": strength},
        )


class LinearFrequencyTradingController(LowLevelController):
    """Asset-shared linear high-frequency execution controller."""

    def __init__(self, params: LinearTradingParams | None = None, scale: float = 0.0014) -> None:
        self.params = params or LinearTradingParams()
        self.scale = float(scale)

    def act(
        self,
        observation: Mapping[str, Any],
        lower_features: np.ndarray,
        high_level_decision: HighLevelDecision,
        context: Mapping[str, Any] | None = None,
    ) -> LowLevelDecision:
        context = context or {}
        freq = dict(context.get("frequency", {}) or {})
        target = _arr(high_level_decision.action)
        position = _arr(observation.get("position", np.zeros_like(target)), target.size)
        x_high = _arr(freq.get("x_high", np.zeros_like(target)), target.size)
        gap = target - position
        p = self.params
        align = np.sign(gap) * x_high / max(self.scale, 1e-9)
        alpha = np.clip(p.lower_base + p.lower_align * np.tanh(align), 0.05, 1.0)
        residual = np.clip(
            p.residual_high * 0.05 * np.tanh(x_high / max(self.scale, 1e-9)),
            -0.10,
            0.10,
        )
        return LowLevelDecision(
            action={"execution_speed": alpha, "residual_order": residual},
            metadata={"gap_norm": float(np.linalg.norm(gap))},
        )
