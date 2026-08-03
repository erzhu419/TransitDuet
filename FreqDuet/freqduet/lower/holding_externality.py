"""Deployable occupancy-weighted holding externality for lower control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class LoadWeightedHoldingPenalty:
    """Normalize onboard passenger delay into a stable reward penalty.

    The load feature is the causal APC occupancy ratio recorded in the state at
    the time the action is selected. Therefore ``action/action_norm * load`` is
    equivalent to holding person-seconds normalized by vehicle capacity and
    ``action_norm``, without reading mutable simulator state at transition end.
    """

    enabled: bool = False
    reward_weight: float = 0.0
    action_norm_s: float = 45.0
    load_clip: float = 1.0
    source: str = "observation_load"

    @classmethod
    def from_config(cls, config: Mapping | None) -> "LoadWeightedHoldingPenalty":
        values = dict(config or {})
        result = cls(
            enabled=bool(values.get("enable", False)),
            reward_weight=max(float(values.get("reward_weight", 0.0)), 0.0),
            action_norm_s=max(float(values.get("action_norm_s", 45.0)), 1e-6),
            load_clip=max(float(values.get("load_clip", 1.0)), 1e-6),
            source=str(values.get("source", "observation_load")).strip().lower(),
        )
        if result.enabled and result.source != "observation_load":
            raise ValueError(
                "lower.load_weighted_holding.source must be observation_load")
        if result.enabled and result.reward_weight <= 0.0:
            raise ValueError(
                "enabled load-weighted holding requires reward_weight > 0")
        return result

    def validate_observation_contract(
        self,
        *,
        observation_mode: str,
        context_features: Iterable[str],
    ) -> None:
        if not self.enabled:
            return
        if str(observation_mode) != "deployable_apc_avl_v4":
            raise ValueError(
                "load-weighted holding requires deployable_apc_avl_v4")
        if "load" not in tuple(str(value) for value in context_features):
            raise ValueError(
                "load-weighted holding requires the causal APC load feature")

    def evaluate(
        self,
        observation,
        action_s: float,
        *,
        base_state_dim: int,
        context_features: Iterable[str],
    ) -> tuple[float, float, float]:
        """Return reward penalty, clipped load ratio and normalized delay."""
        if not self.enabled:
            return 0.0, 0.0, 0.0
        features = tuple(str(value) for value in context_features)
        try:
            load_offset = features.index("load")
        except ValueError as exc:
            raise ValueError(
                "causal APC load is absent from the lower observation") from exc
        raw = np.asarray(observation, dtype=np.float64).reshape(-1)
        load_index = int(base_state_dim) + int(load_offset)
        if load_index >= raw.size:
            raise ValueError(
                "lower observation is too short for its causal APC load slot")
        load_ratio = float(np.clip(raw[load_index], 0.0, self.load_clip))
        action_ratio = max(float(action_s), 0.0) / self.action_norm_s
        normalized_person_delay = action_ratio * load_ratio
        penalty = self.reward_weight * normalized_person_delay
        return float(penalty), load_ratio, float(normalized_person_delay)
