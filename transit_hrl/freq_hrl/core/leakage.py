"""Action-effect frequency leakage utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


def as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def causal_rolling_mean(values: Any, window: int) -> np.ndarray:
    arr = as_2d(values)
    window = max(1, int(window))
    out = np.zeros_like(arr, dtype=np.float64)
    cumsum = np.cumsum(arr, axis=0)
    for i in range(arr.shape[0]):
        start = max(0, i - window + 1)
        total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        out[i] = total / float(i - start + 1)
    return out


def high_pass(values: Any, window: int) -> np.ndarray:
    arr = as_2d(values)
    return arr - causal_rolling_mean(arr, window)


@dataclass
class CausalLowFrequencyEffectProjector:
    """Causally remove the lower controller's slow baseline effect.

    Some domains have nonnegative lower actions, such as transit holding time
    or execution speed.  Treating those raw actions as the lower effect makes
    the lower policy look low-frequency by construction.  This projector keeps
    a causal rolling baseline and returns the residual effect that should be
    attributed to high-frequency lower control.
    """

    window: int = 24
    gain: float = 1.0

    def __post_init__(self) -> None:
        self.window = max(1, int(self.window))
        self.gain = float(np.clip(self.gain, 0.0, 1.0))
        self.reset()

    def reset(self) -> None:
        self._history: list[np.ndarray] = []

    def transform(self, value: Any) -> np.ndarray:
        row = _effect_row(value).reshape(-1)
        self._history.append(row.copy())
        recent = np.asarray(self._history[-self.window:], dtype=np.float64)
        baseline = recent.mean(axis=0)
        return row - self.gain * baseline

    def transform_sequence(self, values: Any) -> np.ndarray:
        self.reset()
        arr = as_2d(values)
        return np.asarray([self.transform(row) for row in arr], dtype=np.float64)


class ActionEffectOperator(ABC):
    """Map action histories to effects whose frequencies can be constrained."""

    @abstractmethod
    def upper_effect(self, upper_action_history: Sequence[Any]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def lower_effect(self, lower_action_history: Sequence[Any]) -> np.ndarray:
        raise NotImplementedError


class CumulativeActionEffectOperator(ActionEffectOperator):
    """Default generic operator.

    Upper effects are the plan/action values themselves.  Lower effects are the
    cumulative sum of residual corrections, matching cases such as transit
    accumulated holding drift or trading inventory drift.
    """

    def upper_effect(self, upper_action_history: Sequence[Any]) -> np.ndarray:
        return as_2d(upper_action_history)

    def lower_effect(self, lower_action_history: Sequence[Any]) -> np.ndarray:
        return np.cumsum(as_2d(lower_action_history), axis=0)


@dataclass
class LeakageRegularizer:
    """Compute upper-HF and lower-LF action-effect penalties."""

    upper_hf_window: int = 6
    lower_lf_window: int = 24
    upper_hf_weight: float = 1.0
    lower_lf_weight: float = 1.0
    eps: float = 1e-9

    def compute(self, upper_effect: Any, lower_effect: Any) -> dict[str, Any]:
        upper = as_2d(upper_effect)
        lower = as_2d(lower_effect)
        upper_hf = high_pass(upper, self.upper_hf_window)
        lower_lf = causal_rolling_mean(lower, self.lower_lf_window)

        upper_power = float(np.mean(upper * upper)) if upper.size else 0.0
        lower_power = float(np.mean(lower * lower)) if lower.size else 0.0
        upper_hf_ratio = float(np.mean(upper_hf * upper_hf) / (upper_power + self.eps))
        lower_lf_ratio = float(np.mean(lower_lf * lower_lf) / (lower_power + self.eps))
        upper_penalty = self.upper_hf_weight * upper_hf_ratio
        lower_penalty = self.lower_lf_weight * lower_lf_ratio
        return {
            "upper_hf_penalty": float(upper_penalty),
            "lower_lf_penalty": float(lower_penalty),
            "leakage_penalty": float(upper_penalty + lower_penalty),
            "UpperHFPower": float(upper_hf_ratio),
            "LowerLFDrift": float(lower_lf_ratio),
            "leakage_feedback": np.asarray([upper_hf_ratio, lower_lf_ratio], dtype=np.float32),
        }


def _effect_row(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(1, -1)
    if arr.size == 0:
        arr = np.zeros((1, 1), dtype=np.float64)
    return arr


@dataclass
class CausalLeakageRewardShaper:
    """Online reward shaper for leakage penalties.

    The shaper stores only action-effect history observed up to the current
    step, computes causal upper-HF and lower-LF leakage metrics, and subtracts
    a scaled penalty from the reward that would be sent to a learner.
    """

    regularizer: LeakageRegularizer = field(default_factory=LeakageRegularizer)
    reward_penalty_scale: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._upper_effects: list[np.ndarray] = []
        self._lower_effects: list[np.ndarray] = []
        self.last_info: dict[str, Any] = {}

    def update(
        self,
        upper_effect: Any,
        lower_effect: Any,
        reward: float | None = None,
    ) -> dict[str, Any]:
        self._upper_effects.append(_effect_row(upper_effect))
        self._lower_effects.append(_effect_row(lower_effect))
        upper = np.concatenate(self._upper_effects, axis=0)
        lower = np.concatenate(self._lower_effects, axis=0)
        metrics = self.regularizer.compute(upper_effect=upper, lower_effect=lower)
        raw_penalty = float(metrics["leakage_penalty"])
        reward_penalty = (
            max(float(self.reward_penalty_scale), 0.0) * raw_penalty
            if self.enabled else 0.0
        )
        shaped_reward = None if reward is None else float(reward) - reward_penalty
        info = {
            **metrics,
            "leakage_enabled": bool(self.enabled),
            "leakage_reward_penalty": float(reward_penalty),
            "raw_reward": None if reward is None else float(reward),
            "shaped_reward": shaped_reward,
            "n": len(self._upper_effects),
        }
        self.last_info = info
        return info
