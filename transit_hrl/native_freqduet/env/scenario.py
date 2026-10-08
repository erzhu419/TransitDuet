"""Policy-independent exogenous randomness for paired experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np


def _normalise_key(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_normalise_key(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalise_key(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


@dataclass
class ScenarioTape:
    """Counter-based random source keyed only by exogenous scenario fields.

    Scalar scenario draws use a stable key, while fixed-clock exogenous
    processes use independent cached streams. Policy exploration and replay
    sampling therefore cannot shift later demand or traffic draws, which makes
    paired seeds genuine common random numbers without rebuilding a random
    generator at every simulation tick.
    """

    seed: int
    version: str = "freqduet-scenario-v1"
    _streams: dict[tuple[Any, ...], np.random.RandomState] = field(
        default_factory=dict, init=False, repr=False)

    def _seed_for(self, namespace: str, *key: Any) -> int:
        payload = json.dumps(
            [self.version, int(self.seed), str(namespace), _normalise_key(key)],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    def _rng(self, namespace: str, *key: Any) -> np.random.RandomState:
        return np.random.RandomState(self._seed_for(namespace, *key))

    def _stream_rng(self, namespace: str, *key: Any) -> np.random.RandomState:
        stream_key = (str(namespace),) + key
        try:
            hash(stream_key)
        except TypeError:
            stream_key = (str(namespace),) + tuple(_normalise_key(key))
        rng = self._streams.get(stream_key)
        if rng is None:
            rng = self._rng(f"{namespace}:stream", *key)
            self._streams[stream_key] = rng
        return rng

    def poisson(self, lam: float, namespace: str, *key: Any) -> int:
        if lam <= 0.0:
            return 0
        return int(self._rng(namespace, *key).poisson(float(lam)))

    def poisson_stream(self, lam: float, namespace: str, *stream_key: Any) -> int:
        """Draw from an isolated sequential stream for a fixed process.

        This is appropriate when call cadence is policy-independent, as it is
        for fixed-clock passenger bins and route-speed updates.
        """
        if lam <= 0.0:
            return 0
        return int(
            self._stream_rng(namespace, *stream_key).poisson(float(lam))
        )

    def normal(
        self,
        loc: float,
        scale: float,
        namespace: str,
        *key: Any,
    ) -> float:
        if scale <= 0.0:
            return float(loc)
        return float(self._rng(namespace, *key).normal(float(loc), float(scale)))

    def normal_stream(
        self,
        loc: float,
        scale: float,
        namespace: str,
        *stream_key: Any,
    ) -> float:
        if scale <= 0.0:
            return float(loc)
        return float(
            self._stream_rng(namespace, *stream_key).normal(
                float(loc), float(scale))
        )

    def lognormal(
        self,
        mean: float,
        sigma: float,
        namespace: str,
        *key: Any,
    ) -> float:
        if sigma <= 0.0:
            return float(np.exp(mean))
        return float(
            self._rng(namespace, *key).lognormal(float(mean), float(sigma))
        )

    def choice(
        self,
        values: Iterable[Any],
        namespace: str,
        *key: Any,
        probabilities: Optional[Iterable[float]] = None,
    ) -> Any:
        values = list(values)
        if not values:
            raise ValueError("ScenarioTape.choice requires at least one value")
        probs = None
        if probabilities is not None:
            probs = np.asarray(list(probabilities), dtype=np.float64)
            if probs.size != len(values) or np.any(probs < 0.0):
                raise ValueError("choice probabilities must match values and be non-negative")
            total = float(probs.sum())
            if total <= 0.0:
                raise ValueError("choice probabilities must have positive mass")
            probs = probs / total
        index = int(
            self._rng(namespace, *key).choice(len(values), p=probs)
        )
        return values[index]

    @property
    def identifier(self) -> str:
        payload = f"{self.version}:{int(self.seed)}".encode("ascii")
        return hashlib.sha256(payload).hexdigest()[:16]
