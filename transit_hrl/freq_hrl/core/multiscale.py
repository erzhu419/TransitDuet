"""Causal, physical-time multiscale features for the Freq-HRL mainline.

The encoder deliberately operates on a fixed trailing window.  Raw-history
and multiscale policies therefore receive transformations of exactly the same
samples; the multiscale arm does not get an unbounded filter state for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_vector(value: np.ndarray, *, expected_dim: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (int(expected_dim),) or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"sample must contain {int(expected_dim)} finite values"
        )
    return vector


@dataclass(frozen=True)
class PhysicalTimeScaleContract:
    """Resolve algorithm windows in seconds to environment control steps."""

    dt_seconds: float
    upper_period_seconds: float
    history_seconds: float
    fast_period_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.dt_seconds,
            self.upper_period_seconds,
            self.history_seconds,
            self.fast_period_seconds,
        )
        if not all(np.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("physical time scales must be positive and finite")
        if float(self.fast_period_seconds) >= float(self.upper_period_seconds):
            raise ValueError("fast period must be shorter than the upper period")
        if float(self.upper_period_seconds) > float(self.history_seconds):
            raise ValueError("history must cover at least one upper period")
        if self.history_steps < 4 or self.history_steps & (self.history_steps - 1):
            raise ValueError(
                "history_seconds must resolve to a power-of-two window of at least 4 steps"
            )

    @staticmethod
    def _resolve_steps(seconds: float, dt_seconds: float) -> int:
        return max(1, int(round(float(seconds) / float(dt_seconds))))

    @property
    def upper_period_steps(self) -> int:
        return self._resolve_steps(self.upper_period_seconds, self.dt_seconds)

    @property
    def history_steps(self) -> int:
        return self._resolve_steps(self.history_seconds, self.dt_seconds)

    @property
    def fast_period_steps(self) -> int:
        return self._resolve_steps(self.fast_period_seconds, self.dt_seconds)

    @property
    def resolved_upper_period_seconds(self) -> float:
        return float(self.upper_period_steps * self.dt_seconds)

    @property
    def resolved_history_seconds(self) -> float:
        return float(self.history_steps * self.dt_seconds)

    @property
    def resolved_fast_period_seconds(self) -> float:
        return float(self.fast_period_steps * self.dt_seconds)

    def metadata(self, *, response_seconds: float | None = None) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "env_dt_seconds": float(self.dt_seconds),
            "upper_period_steps": int(self.upper_period_steps),
            "upper_period_seconds": self.resolved_upper_period_seconds,
            "history_steps": int(self.history_steps),
            "history_seconds": self.resolved_history_seconds,
            "fast_period_steps": int(self.fast_period_steps),
            "fast_period_seconds": self.resolved_fast_period_seconds,
        }
        if response_seconds is not None:
            response = float(response_seconds)
            if not np.isfinite(response) or response <= 0.0:
                raise ValueError("response_seconds must be positive and finite")
            payload.update({
                "system_response_seconds": response,
                "upper_period_over_response": (
                    self.resolved_upper_period_seconds / response
                ),
                "fast_period_over_response": (
                    self.resolved_fast_period_seconds / response
                ),
            })
        return payload


@dataclass(frozen=True)
class MultiscaleSnapshot:
    """One causal feature snapshot derived from a shared raw-history window."""

    current: np.ndarray
    history: np.ndarray
    filtered: np.ndarray
    multiscale: np.ndarray
    slow: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    slow_energy: np.ndarray


class CausalHaarMultiscaleEncoder:
    """Fixed-window orthonormal Haar representation with physical band labels."""

    def __init__(
        self,
        *,
        feature_dim: int,
        time_scale: PhysicalTimeScaleContract,
    ) -> None:
        if int(feature_dim) < 1:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.time_scale = time_scale
        self._history: np.ndarray | None = None

    @property
    def history_dim(self) -> int:
        return int(self.time_scale.history_steps * self.feature_dim)

    def reset(self, sample: np.ndarray) -> MultiscaleSnapshot:
        current = _finite_vector(sample, expected_dim=self.feature_dim)
        self._history = np.repeat(
            current.reshape(1, -1),
            self.time_scale.history_steps,
            axis=0,
        )
        return self.snapshot()

    def update(self, sample: np.ndarray) -> MultiscaleSnapshot:
        current = _finite_vector(sample, expected_dim=self.feature_dim)
        if self._history is None:
            return self.reset(current)
        self._history[:-1] = self._history[1:]
        self._history[-1] = current
        return self.snapshot()

    def snapshot(self) -> MultiscaleSnapshot:
        if self._history is None:
            raise RuntimeError("encoder must be reset before use")
        blocks = self._haar_blocks(self._history)
        all_coefficients = np.concatenate(
            [coefficients for _, coefficients in blocks], axis=0
        )
        slow_blocks: list[np.ndarray] = []
        mid_blocks: list[np.ndarray] = []
        high_blocks: list[np.ndarray] = []
        for support_steps, coefficients in blocks:
            support_seconds = float(
                support_steps * self.time_scale.dt_seconds
            )
            if support_seconds >= self.time_scale.resolved_upper_period_seconds:
                slow_blocks.append(coefficients)
            elif support_seconds <= self.time_scale.resolved_fast_period_seconds:
                high_blocks.append(coefficients)
            else:
                mid_blocks.append(coefficients)

        def flatten(items: list[np.ndarray]) -> np.ndarray:
            if not items:
                return np.empty(0, dtype=np.float32)
            return np.concatenate(items, axis=0).astype(
                np.float32, copy=False
            ).reshape(-1)

        # A trailing-window RMS exposes slowly varying variance even when the
        # signal mean and the coarse Haar coefficients are near zero.
        slow_energy = np.sqrt(
            np.mean(np.square(self._history), axis=0)
        )
        filtered = np.empty_like(self._history)
        filtered[0] = self._history[0]
        alpha = 1.0 - float(np.exp(
            -self.time_scale.dt_seconds
            / self.time_scale.resolved_upper_period_seconds
        ))
        for index in range(1, filtered.shape[0]):
            filtered[index] = (
                filtered[index - 1]
                + alpha * (self._history[index] - filtered[index - 1])
            )
        return MultiscaleSnapshot(
            current=self._history[-1].astype(np.float32, copy=True),
            history=self._history.astype(np.float32, copy=True).reshape(-1),
            filtered=filtered.astype(np.float32, copy=False).reshape(-1),
            multiscale=all_coefficients.astype(
                np.float32, copy=False
            ).reshape(-1),
            slow=flatten(slow_blocks),
            mid=flatten(mid_blocks),
            high=flatten(high_blocks),
            slow_energy=slow_energy.astype(np.float32, copy=False),
        )

    @staticmethod
    def _haar_blocks(
        history: np.ndarray,
    ) -> list[tuple[int, np.ndarray]]:
        """Return average/coarse-to-fine detail blocks and their support."""

        working = np.asarray(history, dtype=np.float64).copy()
        if working.ndim != 2 or working.shape[0] < 2:
            raise ValueError("Haar history must be a two-dimensional window")
        if working.shape[0] & (working.shape[0] - 1):
            raise ValueError("Haar history length must be a power of two")
        details: list[tuple[int, np.ndarray]] = []
        support_steps = 2
        normalization = float(np.sqrt(2.0))
        while working.shape[0] > 1:
            left = working[0::2]
            right = working[1::2]
            averages = (left + right) / normalization
            differences = (left - right) / normalization
            details.append((support_steps, differences))
            working = averages
            support_steps *= 2
        history_steps = int(history.shape[0])
        return [
            (history_steps, working),
            *reversed(details),
        ]

    def band_layout(self) -> list[dict[str, float | int | str]]:
        """Describe coefficient counts and physical support without state data."""

        placeholder = np.zeros(
            (self.time_scale.history_steps, self.feature_dim),
            dtype=np.float64,
        )
        layout: list[dict[str, float | int | str]] = []
        for support_steps, coefficients in self._haar_blocks(placeholder):
            support_seconds = float(support_steps * self.time_scale.dt_seconds)
            if support_seconds >= self.time_scale.resolved_upper_period_seconds:
                band = "slow"
            elif support_seconds <= self.time_scale.resolved_fast_period_seconds:
                band = "high"
            else:
                band = "mid"
            layout.append({
                "band": band,
                "support_steps": int(support_steps),
                "support_seconds": support_seconds,
                "coefficient_count": int(coefficients.size),
            })
        return layout
