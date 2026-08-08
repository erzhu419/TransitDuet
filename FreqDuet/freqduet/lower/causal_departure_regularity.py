"""Causal soft regularity cost for stop-level holding actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DepartureRegularityContext:
    """Immutable action-time evidence used when the transition settles."""

    forward_headway_s: float | None
    target_headway_s: float | None
    evidence_source: str
    evidence_valid: bool


@dataclass(frozen=True)
class DepartureRegularityResult:
    cost: float
    predicted_headway_s: float
    normalized_deviation: float
    normalized_overshoot: float
    evidence_valid: bool


class CausalDepartureRegularityCost:
    """Penalize predicted post-hold headway error without changing the action.

    The context is captured after mandatory service, immediately before the
    holding action is returned to the environment. Holding delays the current
    departure, so ``forward_headway + action`` is the causal one-step departure
    headway implied by that action. The frozen context prevents a later bus
    state from leaking into transition settlement.
    """

    EVIDENCE_MODE = "pre_action_departure_v6"
    EVIDENCE_SOURCE = "matched_departure_event"

    def __init__(
        self,
        *,
        enabled: bool = False,
        cost_weight: float = 0.0,
        tolerance_fraction: float = 0.0,
        cost_cap: float = 1.0,
        evidence_mode: str = EVIDENCE_MODE,
    ) -> None:
        self.enabled = bool(enabled)
        self.cost_weight = float(cost_weight)
        self.tolerance_fraction = float(tolerance_fraction)
        self.cost_cap = float(cost_cap)
        self.evidence_mode = str(evidence_mode).strip().lower()
        if self.evidence_mode != self.EVIDENCE_MODE:
            raise ValueError(
                "causal departure regularity requires "
                "evidence_mode=pre_action_departure_v6")
        if not np.isfinite(self.cost_weight) or self.cost_weight < 0.0:
            raise ValueError("cost_weight must be finite and non-negative")
        if self.enabled and self.cost_weight <= 0.0:
            raise ValueError(
                "enabled causal departure regularity requires cost_weight > 0")
        if not np.isfinite(self.tolerance_fraction) or not (
                0.0 <= self.tolerance_fraction < 1.0):
            raise ValueError("tolerance_fraction must lie in [0, 1)")
        if not np.isfinite(self.cost_cap) or self.cost_cap <= 0.0:
            raise ValueError("cost_cap must be finite and positive")

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None
    ) -> "CausalDepartureRegularityCost":
        cfg = dict(config or {})
        return cls(
            enabled=cfg.get("enable", False),
            cost_weight=cfg.get("cost_weight", 0.0),
            tolerance_fraction=cfg.get("tolerance_fraction", 0.0),
            cost_cap=cfg.get("cost_cap", 1.0),
            evidence_mode=cfg.get("evidence_mode", cls.EVIDENCE_MODE),
        )

    def capture(
        self,
        *,
        forward_headway_s: float | None,
        target_headway_s: float | None,
        evidence_source: str | None,
    ) -> DepartureRegularityContext:
        source = str(evidence_source or "unavailable").strip().lower()
        forward = self._optional_finite_nonnegative(forward_headway_s)
        target = self._optional_finite_positive(target_headway_s)
        valid = (
            source == self.EVIDENCE_SOURCE
            and forward is not None
            and target is not None
        )
        return DepartureRegularityContext(
            forward_headway_s=forward,
            target_headway_s=target,
            evidence_source=source,
            evidence_valid=valid,
        )

    def evaluate(
        self,
        context: DepartureRegularityContext | None,
        action_s: float,
    ) -> DepartureRegularityResult:
        if not self.enabled or context is None or not context.evidence_valid:
            return DepartureRegularityResult(0.0, 0.0, 0.0, 0.0, False)

        forward = context.forward_headway_s
        target = context.target_headway_s
        if forward is None or target is None:
            return DepartureRegularityResult(0.0, 0.0, 0.0, 0.0, False)
        action = max(float(action_s), 0.0)
        if not np.isfinite(action):
            raise ValueError("holding action must be finite")
        predicted = forward + action
        deviation = abs(predicted - target) / target
        overshoot = max(predicted - target, 0.0) / target
        excess = max(deviation - self.tolerance_fraction, 0.0)
        raw_cost = min(excess * excess, self.cost_cap)
        return DepartureRegularityResult(
            cost=float(self.cost_weight * raw_cost),
            predicted_headway_s=float(predicted),
            normalized_deviation=float(deviation),
            normalized_overshoot=float(overshoot),
            evidence_valid=True,
        )

    @staticmethod
    def _optional_finite_nonnegative(value: float | None) -> float | None:
        if value is None:
            return None
        result = float(value)
        return result if np.isfinite(result) and result >= 0.0 else None

    @staticmethod
    def _optional_finite_positive(value: float | None) -> float | None:
        if value is None:
            return None
        result = float(value)
        return result if np.isfinite(result) and result > 0.0 else None
