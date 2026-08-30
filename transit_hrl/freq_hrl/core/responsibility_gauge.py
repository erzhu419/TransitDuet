"""Causal gauge fixing for additive hierarchical action responsibilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size < 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return vector


@dataclass
class CausalGaugeFixer:
    """Choose one causal responsibility split from an additive action total.

    At full strength the returned coordinates depend only on ``upper + lower``:
    the upper responsibility is a causal EMA of that total and the lower
    responsibility is its exact complement. Consequently, any additive gauge
    transform ``(u, l) -> (u + g, l - g)`` produces the same coordinates.
    Partial strength interpolates from the supplied coordinates to this fixed
    gauge while preserving their sum at every step.
    """

    alpha: float = 0.10
    strength: float = 1.0

    def __post_init__(self) -> None:
        self.alpha = float(self.alpha)
        self.strength = float(self.strength)
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("gauge-fixer alpha must be in (0, 1]")
        if not np.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("gauge-fixer strength must be in [0, 1]")
        self._low_pass: np.ndarray | None = None

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("gauge-fixer action_dim must be positive")
        self._low_pass = np.zeros(int(action_dim), dtype=np.float64)

    @property
    def context(self) -> np.ndarray:
        self._require_reset()
        return self._low_pass.astype(np.float32, copy=True)

    def split(
        self,
        upper: Any,
        lower: Any,
        *,
        lower_limit: float | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Return a causal, exactly reconstructing responsibility split."""

        self._require_reset()
        upper_value = _finite_vector(upper, name="upper action")
        lower_value = _finite_vector(lower, name="lower action")
        if upper_value.shape != self._low_pass.shape or lower_value.shape != upper_value.shape:
            raise ValueError("gauge-fixer actions must align with reset action_dim")
        limit = None if lower_limit is None else float(lower_limit)
        if limit is not None and (not np.isfinite(limit) or limit <= 0.0):
            raise ValueError("gauge-fixer lower_limit must be positive and finite")

        total = upper_value + lower_value
        low_before = self._low_pass.copy()
        self._low_pass += self.alpha * (total - self._low_pass)
        canonical_lower_requested = total - self._low_pass
        canonical_lower = (
            canonical_lower_requested
            if limit is None
            else np.clip(canonical_lower_requested, -limit, limit)
        )
        canonical_upper = total - canonical_lower
        canonical_transfer = canonical_upper - upper_value
        transfer = self.strength * canonical_transfer
        fixed_upper = upper_value + transfer
        fixed_lower = lower_value - transfer
        reconstruction_error = fixed_upper + fixed_lower - total
        return {
            "upper": fixed_upper.astype(np.float32, copy=True),
            "lower": fixed_lower.astype(np.float32, copy=True),
            "total": total.astype(np.float32, copy=True),
            "transfer": transfer.astype(np.float32, copy=True),
            "canonical_upper": canonical_upper.astype(np.float32, copy=True),
            "canonical_lower": canonical_lower.astype(np.float32, copy=True),
            "low_pass_before": low_before.astype(np.float32, copy=True),
            "low_pass_after": self._low_pass.astype(np.float32, copy=True),
            "reconstruction_error": reconstruction_error.astype(
                np.float64, copy=True
            ),
            "canonical_lower_clip_rate": float(np.mean(
                np.abs(canonical_lower - canonical_lower_requested) > 1e-12
            )),
            "gauge_fixed": float(self.strength == 1.0),
        }

    def _require_reset(self) -> None:
        if self._low_pass is None:
            raise RuntimeError("gauge fixer must be reset before use")


def canonical_responsibility_trace(
    total_actions: Any,
    *,
    alpha: float,
    lower_limit: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full-strength causal gauge to an action sequence."""

    totals = np.asarray(total_actions, dtype=np.float64)
    if totals.ndim != 2 or totals.shape[0] < 1 or totals.shape[1] < 1:
        raise ValueError("total_actions must be a nonempty matrix")
    if not np.all(np.isfinite(totals)):
        raise ValueError("total_actions must be finite")
    fixer = CausalGaugeFixer(alpha=alpha, strength=1.0)
    fixer.reset(totals.shape[1])
    upper_rows: list[np.ndarray] = []
    lower_rows: list[np.ndarray] = []
    zeros = np.zeros(totals.shape[1], dtype=np.float64)
    for total in totals:
        row = fixer.split(zeros, total, lower_limit=lower_limit)
        upper_rows.append(np.asarray(row["upper"], dtype=np.float64))
        lower_rows.append(np.asarray(row["lower"], dtype=np.float64))
    return np.stack(upper_rows), np.stack(lower_rows)
