"""Causal observation routing and actuation disturbances for MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
    "ood_chirp",
)


@dataclass
class CausalBandDecomposer:
    """Two-rate causal EMA decomposition with no future observations."""

    slow_alpha: float = 0.04
    fast_alpha: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 < float(self.slow_alpha) < float(self.fast_alpha) <= 1.0:
            raise ValueError("band alphas must satisfy 0 < slow < fast <= 1")
        self.slow_alpha = float(self.slow_alpha)
        self.fast_alpha = float(self.fast_alpha)
        self._slow: np.ndarray | None = None
        self._fast: np.ndarray | None = None
        self._previous: np.ndarray | None = None

    def reset(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        current = self._finite_vector(observation)
        self._slow = current.copy()
        self._fast = current.copy()
        self._previous = current.copy()
        return self._bands(current, np.zeros_like(current))

    def update(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        current = self._finite_vector(observation)
        if self._slow is None or self._fast is None or self._previous is None:
            return self.reset(current)
        if current.shape != self._slow.shape:
            raise ValueError("observation dimension changed within an episode")
        delta = current - self._previous
        self._slow += self.slow_alpha * (current - self._slow)
        self._fast += self.fast_alpha * (current - self._fast)
        self._previous = current.copy()
        return self._bands(current, delta)

    @staticmethod
    def _finite_vector(observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float64).reshape(-1)
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("MuJoCo observation must be a finite vector")
        return value

    def _bands(
        self,
        current: np.ndarray,
        delta: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if self._slow is None or self._fast is None:
            raise RuntimeError("decomposer must be reset before use")
        return {
            "raw": current.astype(np.float32, copy=True),
            "slow": self._slow.astype(np.float32, copy=True),
            "mid": (self._fast - self._slow).astype(np.float32, copy=False),
            "high": (current - self._fast).astype(np.float32, copy=False),
            "delta": np.asarray(delta, dtype=np.float32).copy(),
        }


def action_from_unit_box(
    normalized_action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map a clipped action in [-1, 1] to a finite Box action space."""

    unit = np.clip(
        np.asarray(normalized_action, dtype=np.float64).reshape(-1),
        -1.0,
        1.0,
    )
    lower = np.asarray(low, dtype=np.float64).reshape(-1)
    upper = np.asarray(high, dtype=np.float64).reshape(-1)
    if (
        unit.shape != lower.shape
        or lower.shape != upper.shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(upper <= lower)
    ):
        raise ValueError("action bounds must be aligned, finite, and ordered")
    return (lower + 0.5 * (unit + 1.0) * (upper - lower)).astype(
        np.float32
    )


def deterministic_actuation_disturbance(
    *,
    mode: str,
    step: int,
    action_dim: int,
    seed: int,
    horizon: int,
) -> np.ndarray:
    """Return a deterministic normalized-action disturbance for one step."""

    name = str(mode)
    if name not in DISTURBANCE_MODES:
        raise ValueError(f"unknown MuJoCo disturbance mode: {name}")
    if int(step) < 0 or int(action_dim) < 1 or int(horizon) < 1:
        raise ValueError("step, action_dim, and horizon must be valid")
    if name == "standard":
        return np.zeros(int(action_dim), dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=int(action_dim))
    directions = rng.choice((-1.0, 1.0), size=int(action_dim))
    t = float(step)
    low = 0.16 * np.sin(2.0 * np.pi * t / 160.0 + phases)
    high = 0.08 * np.sin(2.0 * np.pi * t / 5.0 + phases) * directions
    if name == "low_frequency":
        value = low
    elif name == "high_frequency":
        value = high
    elif name == "mixed":
        shift = 0.10 * directions if int(step) >= int(horizon) // 2 else 0.0
        value = low + high + shift
    else:
        progress = min(max(t / max(float(horizon - 1), 1.0), 0.0), 1.0)
        cycles = 0.5 * t / 160.0 + 8.0 * progress * progress
        value = 0.14 * np.sin(2.0 * np.pi * cycles + phases)
    return np.asarray(value, dtype=np.float32)
