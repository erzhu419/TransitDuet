"""Heuristic trading policies implementing the Freq-HRL interfaces."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .interfaces import HighLevelDecision, HighLevelPlanner, LowLevelController, LowLevelDecision


def normalize_target(signal: np.ndarray, scale: float = 0.0014, max_gross: float = 1.0) -> np.ndarray:
    target = np.tanh(np.asarray(signal, dtype=np.float64).reshape(-1) / max(scale, 1e-9))
    gross = float(np.sum(np.abs(target)))
    if gross > max_gross and gross > 1e-12:
        target *= max_gross / gross
    return target


def _arr(value: Any, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if dim is not None and arr.size != dim:
        arr = np.resize(arr, dim)
    return arr


class FrequencyTradingPlanner(HighLevelPlanner):
    """Low-frequency target-weight planner."""

    def __init__(self, promotion_mid_gain: float = 0.5) -> None:
        self.promotion_mid_gain = float(promotion_mid_gain)

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
        strength = float(promotion.get("promotion_strength", 0.0))
        promote = bool(promotion.get("promote", False))
        signal = x_low + (self.promotion_mid_gain * strength * x_mid if promote else 0.0)
        target = normalize_target(signal)
        return HighLevelDecision(
            action=target,
            plan={"type": "target_weights", "target": target.copy()},
            metadata={"promotion_used": promote, "promotion_strength": strength},
        )


class FrequencyTradingController(LowLevelController):
    """High-frequency execution-speed controller."""

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
        align = np.sign(gap) * x_high / 0.0014
        alpha = np.clip(0.65 + 0.25 * np.tanh(align), 0.20, 1.0)
        return LowLevelDecision(
            action={
                "execution_speed": alpha,
                "residual_order": np.zeros(target.size, dtype=np.float64),
            },
            metadata={"gap_norm": float(np.linalg.norm(gap))},
        )
