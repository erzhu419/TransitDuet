"""Deployable one-headway feasibility guard for stop-level holding actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CausalHoldingGuardResult:
    requested_s: float
    allowed_s: float
    limit_s: float
    evidence_valid: bool

    @property
    def adjustment_s(self) -> float:
        return max(0.0, self.requested_s - self.allowed_s)

    @property
    def active(self) -> bool:
        return self.adjustment_s > 1e-9


class CausalHoldingActionGuard:
    """Prevent holding beyond the observed gap to the preceding vehicle.

    Holding increases the current bus's forward departure headway by the same
    duration. V6 uses the matched predecessor departure observed after mandatory
    service; the legacy V5 mode uses the same-stop arrival event. Neither mode
    uses follower state or a future trajectory.
    """

    def __init__(
        self,
        enabled: bool = False,
        max_deficit_fraction: float = 1.0,
        minimum_deficit_s: float = 0.0,
        evidence_mode: str = "arrival_event_v5",
    ) -> None:
        self.enabled = bool(enabled)
        self.max_deficit_fraction = float(max_deficit_fraction)
        self.minimum_deficit_s = float(minimum_deficit_s)
        self.evidence_mode = str(evidence_mode).strip().lower()
        if not np.isfinite(self.max_deficit_fraction) or not (
                0.0 <= self.max_deficit_fraction <= 1.0):
            raise ValueError("max_deficit_fraction must lie in [0, 1]")
        if not np.isfinite(self.minimum_deficit_s) or self.minimum_deficit_s < 0.0:
            raise ValueError("minimum_deficit_s must be finite and non-negative")
        if self.evidence_mode not in {
                "arrival_event_v5", "pre_action_departure_v6"}:
            raise ValueError(
                "evidence_mode must be arrival_event_v5 or "
                "pre_action_departure_v6")

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None
    ) -> "CausalHoldingActionGuard":
        cfg = dict(config or {})
        return cls(
            enabled=cfg.get("enable", False),
            max_deficit_fraction=cfg.get("max_deficit_fraction", 1.0),
            minimum_deficit_s=cfg.get("minimum_deficit_s", 0.0),
            evidence_mode=cfg.get("evidence_mode", "arrival_event_v5"),
        )

    def evaluate(
        self,
        requested_s: float,
        *,
        forward_headway_s: float | None,
        target_headway_s: float | None,
        evidence_valid: bool,
    ) -> CausalHoldingGuardResult:
        requested = max(float(requested_s), 0.0)
        if not self.enabled:
            return CausalHoldingGuardResult(
                requested, requested, requested, bool(evidence_valid))

        valid = bool(evidence_valid)
        if forward_headway_s is None or target_headway_s is None:
            valid = False
        if valid:
            forward = float(forward_headway_s)
            target = float(target_headway_s)
            valid = (
                np.isfinite(forward) and np.isfinite(target)
                and forward >= 0.0 and target > 0.0
            )
        if not valid:
            return CausalHoldingGuardResult(requested, 0.0, 0.0, False)

        deficit = max(0.0, target - forward)
        if deficit < self.minimum_deficit_s:
            deficit = 0.0
        limit = self.max_deficit_fraction * deficit
        return CausalHoldingGuardResult(
            requested_s=requested,
            allowed_s=min(requested, limit),
            limit_s=limit,
            evidence_valid=True,
        )
