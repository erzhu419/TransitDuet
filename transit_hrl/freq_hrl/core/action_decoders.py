"""Causal action parameterizations for frequency-separated control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _finite_action(value: Any, *, dimension: int, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float64).reshape(-1)
    if action.shape != (int(dimension),) or not np.all(np.isfinite(action)):
        raise ValueError(f"{name} must be finite and aligned")
    return action


@dataclass
class CausalSmoothstepMacroPlan:
    """Decode one macro target into a continuous primitive-step action path."""

    macro_steps: int

    def __post_init__(self) -> None:
        self.macro_steps = int(self.macro_steps)
        if self.macro_steps < 2:
            raise ValueError("macro plan requires at least two primitive steps")
        self._dimension = 0
        self._start = np.zeros(0, dtype=np.float64)
        self._target = np.zeros(0, dtype=np.float64)
        self._current = np.zeros(0, dtype=np.float64)
        self._phase = 0
        self._active = False

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("macro plan action_dim must be positive")
        self._dimension = int(action_dim)
        zeros = np.zeros(self._dimension, dtype=np.float64)
        self._start = zeros.copy()
        self._target = zeros.copy()
        self._current = zeros.copy()
        self._phase = 0
        self._active = False

    @property
    def current(self) -> np.ndarray:
        self._require_reset()
        return self._current.astype(np.float32, copy=True)

    @property
    def target(self) -> np.ndarray:
        self._require_active()
        return self._target.astype(np.float32, copy=True)

    @property
    def progress(self) -> float:
        self._require_active()
        return float(min(self._phase / self.macro_steps, 1.0))

    def activate(self, target: Any) -> np.ndarray:
        """Activate a target at a macro boundary without a target jump."""

        self._require_reset()
        target_value = _finite_action(
            target, dimension=self._dimension, name="macro target"
        )
        if not self._active:
            self._current = target_value.copy()
        else:
            self._current = self._target.copy()
        self._start = self._current.copy()
        self._target = target_value.copy()
        self._phase = 0
        self._active = True
        return self.current

    def advance(self) -> np.ndarray:
        """Evaluate the frozen active plan at the next primitive step."""

        self._require_active()
        self._phase = min(self._phase + 1, self.macro_steps)
        weight = self._smoothstep(self._phase / self.macro_steps)
        self._current = self._start + weight * (self._target - self._start)
        return self.current

    def peek_advance(self) -> np.ndarray:
        """Return the next primitive value without mutating plan state."""

        self._require_active()
        phase = min(self._phase + 1, self.macro_steps)
        weight = self._smoothstep(phase / self.macro_steps)
        value = self._start + weight * (self._target - self._start)
        return value.astype(np.float32, copy=True)

    @staticmethod
    def _smoothstep(progress: float) -> float:
        value = float(np.clip(progress, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("macro plan must be reset before use")

    def _require_active(self) -> None:
        self._require_reset()
        if not self._active:
            raise RuntimeError("macro plan requires an active target")


@dataclass
class CausalZeroDCMacroProjector:
    """Project fast residuals onto an exactly zero-sum macro action set.

    At each primitive step, the proposal is projected onto the interval that
    leaves enough bounded control authority to repay the accumulated macro debt
    over the remaining steps. The final feasible action is therefore the exact
    negative debt, making every completed macro sum zero without future access.
    """

    macro_steps: int

    def __post_init__(self) -> None:
        self.macro_steps = int(self.macro_steps)
        if self.macro_steps < 2:
            raise ValueError("zero-DC projection requires at least two steps")
        self._dimension = 0
        self._debt = np.zeros(0, dtype=np.float64)
        self._context = np.zeros(0, dtype=np.float64)
        self._phase = 0
        self._active = False

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("zero-DC projector action_dim must be positive")
        self._dimension = int(action_dim)
        self._debt = np.zeros(self._dimension, dtype=np.float64)
        self._context = np.zeros(self._dimension, dtype=np.float64)
        self._phase = 0
        self._active = False

    @property
    def context(self) -> np.ndarray:
        self._require_reset()
        return self._context.astype(np.float32, copy=True)

    @property
    def debt(self) -> np.ndarray:
        self._require_reset()
        return self._debt.astype(np.float32, copy=True)

    def project(
        self,
        proposal: Any,
        *,
        macro_boundary: bool,
        action_limit: float,
    ) -> dict[str, np.ndarray | float]:
        """Return the nearest bounded action that preserves zero-sum feasibility."""

        self._require_reset()
        value = _finite_action(
            proposal, dimension=self._dimension, name="lower proposal"
        )
        limit = float(action_limit)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("zero-DC action_limit must be positive and finite")
        if not isinstance(macro_boundary, (bool, np.bool_)):
            raise ValueError("macro_boundary must be boolean")
        if bool(macro_boundary):
            if self._active and (
                self._phase != self.macro_steps
                or float(np.max(np.abs(self._debt))) > 1e-9
            ):
                raise RuntimeError(
                    "zero-DC macro boundary arrived before debt was repaid"
                )
            self._debt.fill(0.0)
            self._phase = 0
            self._active = True
        elif not self._active:
            raise RuntimeError(
                "zero-DC projector requires a boundary on its first action"
            )
        if self._phase >= self.macro_steps:
            raise RuntimeError("zero-DC macro exceeded its configured horizon")

        remaining = self.macro_steps - self._phase
        debt_before = self._debt.copy()
        future_capacity = float(remaining - 1) * limit
        feasible_low = np.maximum(-limit, -debt_before - future_capacity)
        feasible_high = np.minimum(limit, -debt_before + future_capacity)
        effective = np.clip(value, feasible_low, feasible_high)
        self._debt += effective
        self._phase += 1
        remaining_after = self.macro_steps - self._phase
        self._context = (
            -self._debt / float(remaining_after)
            if remaining_after > 0
            else np.zeros_like(self._debt)
        )
        completed = self._phase == self.macro_steps
        completion_error = (
            float(np.sqrt(np.mean(np.square(self._debt)))) if completed else 0.0
        )
        return {
            "proposal": value.astype(np.float32, copy=True),
            "effective": effective.astype(np.float32, copy=True),
            "debt_before": debt_before.astype(np.float32, copy=True),
            "debt_after": self._debt.astype(np.float32, copy=True),
            "context_after": self._context.astype(np.float32, copy=True),
            "remaining_before": float(remaining),
            "remaining_after": float(remaining_after),
            "projection_rate": float(np.mean(np.abs(effective - value) > 1e-12)),
            "macro_completed": float(completed),
            "macro_completion_error_rms": completion_error,
        }

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("zero-DC projector must be reset before use")
