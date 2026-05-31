"""Linear actor-critic trading policy parameters for Freq-HRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .pg_trading import (
    PolicyGradientTradingController,
    PolicyGradientTradingParams,
    PolicyGradientTradingPlanner,
)


@dataclass
class ActorCriticTradingParams:
    """Actor plus upper/lower linear critics.

    The actor is the same Gaussian frequency-routed policy used by the
    REINFORCE path.  The critics are separated by responsibility: upper critic
    consumes low/mid/promotion state, lower critic consumes execution gap and
    high-frequency energy.
    """

    actor: PolicyGradientTradingParams = field(default_factory=PolicyGradientTradingParams)
    upper_value: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lower_value: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def upper_value_dim(cls) -> int:
        return 6

    @classmethod
    def lower_value_dim(cls) -> int:
        return 5

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActorCriticTradingParams":
        payload = dict(value)
        actor_payload = payload.get("actor", payload.get("params", payload))
        upper = np.asarray(
            payload.get("upper_value", np.zeros(cls.upper_value_dim())),
            dtype=np.float64,
        ).reshape(-1)
        lower = np.asarray(
            payload.get("lower_value", np.zeros(cls.lower_value_dim())),
            dtype=np.float64,
        ).reshape(-1)
        if upper.size != cls.upper_value_dim():
            upper = np.resize(upper, cls.upper_value_dim())
        if lower.size != cls.lower_value_dim():
            lower = np.resize(lower, cls.lower_value_dim())
        return cls(
            actor=PolicyGradientTradingParams.from_mapping(actor_payload),
            upper_value=tuple(float(v) for v in upper),
            lower_value=tuple(float(v) for v in lower),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "actor": self.actor.to_mapping(),
            "upper_value": [float(v) for v in self.upper_value],
            "lower_value": [float(v) for v in self.lower_value],
        }

    def with_vectors(
        self,
        actor_vector: np.ndarray,
        upper_value: np.ndarray,
        lower_value: np.ndarray,
    ) -> "ActorCriticTradingParams":
        return ActorCriticTradingParams(
            actor=PolicyGradientTradingParams.from_vector(actor_vector, template=self.actor),
            upper_value=tuple(float(v) for v in np.asarray(upper_value, dtype=np.float64).reshape(-1)),
            lower_value=tuple(float(v) for v in np.asarray(lower_value, dtype=np.float64).reshape(-1)),
        )

    def actor_vector(self) -> np.ndarray:
        return self.actor.to_vector()

    def upper_value_vector(self) -> np.ndarray:
        return np.asarray(self.upper_value, dtype=np.float64)

    def lower_value_vector(self) -> np.ndarray:
        return np.asarray(self.lower_value, dtype=np.float64)


class ActorCriticTradingPlanner(PolicyGradientTradingPlanner):
    """Deterministic/sample actor wrapper for actor-critic models."""

    def __init__(
        self,
        params: ActorCriticTradingParams | None = None,
        scale: float = 0.0014,
        sample: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            params=(params or ActorCriticTradingParams()).actor,
            scale=scale,
            sample=sample,
            rng=rng,
        )


class ActorCriticTradingController(PolicyGradientTradingController):
    """Deterministic/sample lower actor wrapper for actor-critic models."""

    def __init__(
        self,
        params: ActorCriticTradingParams | None = None,
        scale: float = 0.0014,
        sample: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            params=(params or ActorCriticTradingParams()).actor,
            scale=scale,
            sample=sample,
            rng=rng,
        )
