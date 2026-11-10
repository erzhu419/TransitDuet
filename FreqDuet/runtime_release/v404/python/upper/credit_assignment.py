"""Auditable reward assignment for upper timetable decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


_SYSTEM_REWARD_MODES = {"repeated", "uniform", "terminal", "none"}
_GAP_CREDIT_MODES = {"centered", "absolute", "none"}
_RELIABILITY_REWARD_MODES = {"uniform", "terminal", "none"}


@dataclass(frozen=True)
class UpperCreditAssignment:
    """Assign episode and plan-local outcomes to upper transitions.

    ``repeated`` and ``centered`` reproduce the legacy behavior.  The other
    modes make the reward ownership explicit so structural candidates can be
    evaluated without changing the reference configuration.
    """

    system_reward_mode: str = "repeated"
    system_reward_weight: float = 1.0
    gap_credit_mode: str = "centered"
    gap_credit_weight: float = 0.5
    gap_credit_clip: float = 4.0
    reliability_reward_mode: str = "none"
    reliability_reward_weight: float = 0.0

    def __post_init__(self) -> None:
        system_mode = str(self.system_reward_mode).lower()
        gap_mode = str(self.gap_credit_mode).lower()
        reliability_mode = str(self.reliability_reward_mode).lower()
        if system_mode not in _SYSTEM_REWARD_MODES:
            raise ValueError(
                "upper.credit_assignment.system_reward_mode must be one of "
                f"{sorted(_SYSTEM_REWARD_MODES)}")
        if gap_mode not in _GAP_CREDIT_MODES:
            raise ValueError(
                "upper.credit_assignment.gap_credit_mode must be one of "
                f"{sorted(_GAP_CREDIT_MODES)}")
        if reliability_mode not in _RELIABILITY_REWARD_MODES:
            raise ValueError(
                "upper.credit_assignment.reliability_reward_mode must be "
                f"one of {sorted(_RELIABILITY_REWARD_MODES)}")
        if (not np.isfinite(self.system_reward_weight)
                or self.system_reward_weight < 0):
            raise ValueError(
                "system_reward_weight must be finite and non-negative")
        if not np.isfinite(self.gap_credit_weight) or self.gap_credit_weight < 0:
            raise ValueError("gap_credit_weight must be finite and non-negative")
        if not np.isfinite(self.gap_credit_clip) or self.gap_credit_clip <= 0:
            raise ValueError("gap_credit_clip must be finite and positive")
        if (not np.isfinite(self.reliability_reward_weight)
                or self.reliability_reward_weight < 0):
            raise ValueError(
                "reliability_reward_weight must be finite and non-negative")
        object.__setattr__(self, "system_reward_mode", system_mode)
        object.__setattr__(self, "gap_credit_mode", gap_mode)
        object.__setattr__(self, "reliability_reward_mode", reliability_mode)

    @classmethod
    def from_config(cls, config: Mapping | None) -> "UpperCreditAssignment":
        cfg = dict(config or {})
        return cls(
            system_reward_mode=cfg.get("system_reward_mode", "repeated"),
            system_reward_weight=float(cfg.get("system_reward_weight", 1.0)),
            gap_credit_mode=cfg.get("gap_credit_mode", "centered"),
            gap_credit_weight=float(cfg.get("gap_credit_weight", 0.5)),
            gap_credit_clip=float(cfg.get("gap_credit_clip", 4.0)),
            reliability_reward_mode=cfg.get(
                "reliability_reward_mode", "none"),
            reliability_reward_weight=float(cfg.get(
                "reliability_reward_weight", 0.0)),
        )

    def system_rewards(self, system_reward: float, count: int) -> np.ndarray:
        """Return one global-reward contribution per upper transition."""
        count = int(count)
        if count < 0:
            raise ValueError("transition count must be non-negative")
        if count == 0:
            return np.empty(0, dtype=np.float64)
        reward = float(system_reward) * self.system_reward_weight
        if not np.isfinite(reward):
            raise ValueError("system reward must be finite")
        if self.system_reward_mode == "repeated":
            return np.full(count, reward, dtype=np.float64)
        if self.system_reward_mode == "uniform":
            return np.full(count, reward / count, dtype=np.float64)
        result = np.zeros(count, dtype=np.float64)
        if self.system_reward_mode == "terminal":
            result[-1] = reward
        return result

    def reliability_rewards(
        self, unserved_rate: float, incomplete_rate: float, count: int
    ) -> np.ndarray:
        """Assign endpoint service-failure cost without repeating it."""
        count = int(count)
        if count < 0:
            raise ValueError("transition count must be non-negative")
        if count == 0:
            return np.empty(0, dtype=np.float64)
        values = np.asarray(
            [unserved_rate, incomplete_rate], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "unserved_rate and incomplete_rate must be finite and "
                "non-negative")
        penalty = -self.reliability_reward_weight * float(values.sum())
        if self.reliability_reward_mode == "uniform":
            return np.full(count, penalty / count, dtype=np.float64)
        result = np.zeros(count, dtype=np.float64)
        if self.reliability_reward_mode == "terminal":
            result[-1] = penalty
        return result

    def gap_credits(
        self,
        gap_deviation_by_owner: Mapping[int, float],
        transition_ids: Iterable[int],
    ) -> dict[int, float]:
        """Return plan-owned gap credit keyed by transition id."""
        ids = [int(tid) for tid in transition_ids]
        if not ids or self.gap_credit_mode == "none":
            return {tid: 0.0 for tid in ids}

        observed = np.asarray([
            float(value) for value in gap_deviation_by_owner.values()
            if np.isfinite(value)
        ], dtype=np.float64)
        if observed.size:
            fallback = float(observed.mean())
        else:
            fallback = 0.0

        result: dict[int, float] = {}
        if self.gap_credit_mode == "centered":
            scale = float(max(observed.std(), 1e-6)) if observed.size else 1.0
            for tid in ids:
                value = float(gap_deviation_by_owner.get(tid, fallback))
                result[tid] = float(
                    -self.gap_credit_weight * (value - fallback) / scale)
            return result

        for tid in ids:
            value = float(gap_deviation_by_owner.get(tid, fallback))
            clipped = float(np.clip(value, 0.0, self.gap_credit_clip))
            result[tid] = -self.gap_credit_weight * clipped
        return result
