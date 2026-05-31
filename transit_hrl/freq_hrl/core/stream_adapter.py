"""Generic causal exogenous stream binning."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping
import math

import numpy as np

from ..encoders.base import ExogenousStreamAdapter


class BinnedExogenousStreamAdapter(ExogenousStreamAdapter):
    """Aggregate timestamped raw events into causal fixed-width bins.

    The adapter stores only events that have been observed through ``observe``.
    ``get_bin(t)`` never looks ahead; it returns the completed or current bin
    implied by observed events with timestamps up to ``t``.
    """

    def __init__(
        self,
        bin_sec: float = 60.0,
        value_key: str = "value",
        entity_key: str = "entity_id",
        default_entity_id: Any = "global",
        rate_per_sec: float | None = None,
    ) -> None:
        self.bin_sec = max(float(bin_sec), 1e-9)
        self.value_key = str(value_key)
        self.entity_key = str(entity_key)
        self.default_entity_id = default_entity_id
        self.rate_per_sec = None if rate_per_sec is None else float(rate_per_sec)
        self.reset()

    def reset(self, episode_id: int | None = None) -> None:
        self.episode_id = episode_id
        self._bins: dict[tuple[int, Any], np.ndarray] = {}
        self._observed_until = -math.inf

    def _bin_index(self, t: float) -> int:
        return int(math.floor(float(t) / self.bin_sec))

    def observe(self, raw_event: Mapping[str, Any], t: float) -> None:
        event_t = float(raw_event.get("timestamp", raw_event.get("t", t)))
        if event_t > float(t) + 1e-9:
            raise ValueError("observe cannot ingest an event timestamped after t")
        entity_id = raw_event.get(self.entity_key, self.default_entity_id)
        value = np.asarray(raw_event.get(self.value_key, 0.0), dtype=np.float64)
        if value.ndim == 0:
            value = value.reshape(1)
        value = value.reshape(-1)
        key = (self._bin_index(event_t), entity_id)
        if key not in self._bins:
            self._bins[key] = np.zeros_like(value, dtype=np.float64)
        if self._bins[key].shape != value.shape:
            raise ValueError("event value dimension changed within adapter")
        self._bins[key] += value
        self._observed_until = max(self._observed_until, event_t)

    def get_bin(self, t: float, entity_id: Any | None = None) -> dict[str, Any]:
        entity_id = self.default_entity_id if entity_id is None else entity_id
        idx = self._bin_index(float(t))
        value = self._bins.get((idx, entity_id))
        if value is None:
            value = np.zeros(1, dtype=np.float64)
        else:
            value = value.copy()
        if self.rate_per_sec is not None:
            value = value * self.rate_per_sec
        return {
            "timestamp": min((idx + 1) * self.bin_sec, float(t)),
            "entity_id": entity_id,
            "x_raw": value,
            "valid_mask": np.ones_like(value, dtype=bool),
            "normalization_context": {
                "bin_sec": self.bin_sec,
                "observed_until": self._observed_until,
            },
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "timestamp": "float seconds",
            "entity_id": self.entity_key,
            "x_raw": self.value_key,
            "valid_mask": "observed dimensions in bin",
            "normalization_context": {"bin_sec": self.bin_sec},
        }


class MultiEntityBinnedStream:
    """Small helper for domains that need all entity bins at each update."""

    def __init__(self, bin_sec: float = 60.0) -> None:
        self.bin_sec = max(float(bin_sec), 1e-9)
        self.reset()

    def reset(self) -> None:
        self._pending: dict[Any, float] = defaultdict(float)

    def add(self, entity_id: Any, value: float) -> None:
        self._pending[entity_id] += float(value)

    def flush(self, timestamp: float, include_entities: set[Any] | None = None) -> list[dict[str, Any]]:
        keys = set(self._pending.keys())
        if include_entities is not None:
            keys |= set(include_entities)
        out = []
        for key in sorted(keys, key=lambda x: repr(x)):
            out.append({
                "timestamp": float(timestamp),
                "entity_id": key,
                "x_raw": np.asarray([self._pending.get(key, 0.0)], dtype=np.float64),
                "valid_mask": np.asarray([True]),
                "normalization_context": {"bin_sec": self.bin_sec},
            })
        self._pending.clear()
        return out
