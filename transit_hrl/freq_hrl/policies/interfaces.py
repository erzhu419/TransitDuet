"""Policy interfaces for Freq-HRL planners and controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class HighLevelDecision:
    action: np.ndarray
    plan: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LowLevelDecision:
    action: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class HighLevelPlanner(ABC):
    """Interface for low-frequency planning policies."""

    @abstractmethod
    def plan(
        self,
        observation: Mapping[str, Any],
        upper_features: np.ndarray,
        context: Mapping[str, Any] | None = None,
    ) -> HighLevelDecision:
        raise NotImplementedError


class LowLevelController(ABC):
    """Interface for high-frequency correction policies."""

    @abstractmethod
    def act(
        self,
        observation: Mapping[str, Any],
        lower_features: np.ndarray,
        high_level_decision: HighLevelDecision,
        context: Mapping[str, Any] | None = None,
    ) -> LowLevelDecision:
        raise NotImplementedError
