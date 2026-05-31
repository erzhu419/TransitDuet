"""Abstract interfaces for domain adapters and causal encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class ExogenousStreamAdapter(ABC):
    """Convert domain raw events into causal exogenous bins."""

    @abstractmethod
    def reset(self, episode_id: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def observe(self, raw_event: Mapping[str, Any], t: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_bin(self, t: float) -> dict[str, Any]:
        """Return only events with timestamp <= t."""
        raise NotImplementedError

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        raise NotImplementedError


class CausalSpectralEncoder(ABC):
    """Online exogenous signal splitter.

    Encoders must only use bins that have been passed to ``update``.  Offline
    full-episode transforms can live elsewhere, but they should not implement
    this interface.
    """

    @abstractmethod
    def reset(self, episode_id: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, x_bin: Mapping[str, Any], t: float | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def features(self, t: float | None = None, entity_id: Any = "global") -> dict[str, Any]:
        raise NotImplementedError
