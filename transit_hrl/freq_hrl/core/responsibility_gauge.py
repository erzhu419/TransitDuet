"""Causal gauge fixing for additive hierarchical action responsibilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .action_decoders import CausalSmoothstepMacroPlan


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


@dataclass
class CausalAuditAlignedGaugeFixer:
    """Adapt a causal complementary split to the registered leakage audit.

    A causal EMA assigns the low-pass total action to the upper controller and
    its exact complement to the lower controller. After every split, the EMA
    cutoff is updated from the normalized HPF(upper) versus LPF(lower) budget
    imbalance measured with the registered audit windows. This feedback avoids
    the unstable horizon-myopia of minimizing only the current rolling-window
    endpoint. At full strength the result depends only on the total action and
    canonical history, so additive gauge transforms of the supplied policy
    outputs are immaterial.
    """

    upper_window: int = 8
    lower_window: int = 32
    upper_rms_budget: float = 0.075
    lower_rms_budget: float = 0.0475
    initial_alpha: float = 0.20
    adaptation_rate: float = 0.03
    minimum_logit: float = -4.0
    maximum_logit: float = 4.0
    strength: float = 1.0

    def __post_init__(self) -> None:
        self.upper_window = int(self.upper_window)
        self.lower_window = int(self.lower_window)
        self.upper_rms_budget = float(self.upper_rms_budget)
        self.lower_rms_budget = float(self.lower_rms_budget)
        self.initial_alpha = float(self.initial_alpha)
        self.adaptation_rate = float(self.adaptation_rate)
        self.minimum_logit = float(self.minimum_logit)
        self.maximum_logit = float(self.maximum_logit)
        self.strength = float(self.strength)
        if self.upper_window < 2 or self.lower_window < 2:
            raise ValueError("audit-aligned gauge windows must be at least two")
        if (
            not np.isfinite(self.upper_rms_budget)
            or self.upper_rms_budget <= 0.0
            or not np.isfinite(self.lower_rms_budget)
            or self.lower_rms_budget <= 0.0
        ):
            raise ValueError("audit-aligned gauge budgets must be positive")
        if (
            not np.isfinite(self.initial_alpha)
            or not 0.0 < self.initial_alpha < 1.0
        ):
            raise ValueError("audit-aligned gauge initial_alpha must be in (0, 1)")
        if (
            not np.isfinite(self.adaptation_rate)
            or self.adaptation_rate < 0.0
        ):
            raise ValueError(
                "audit-aligned gauge adaptation_rate must be non-negative"
            )
        if (
            not np.isfinite(self.minimum_logit)
            or not np.isfinite(self.maximum_logit)
            or self.minimum_logit >= self.maximum_logit
        ):
            raise ValueError("audit-aligned gauge logit bounds are invalid")
        if not np.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("audit-aligned gauge strength must be in [0, 1]")
        self._dimension = 0
        self._low_pass = np.zeros(0, dtype=np.float64)
        self._upper_history: list[np.ndarray] = []
        self._lower_history: list[np.ndarray] = []
        self._context = np.zeros(0, dtype=np.float64)
        self._logit_alpha = 0.0

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("audit-aligned gauge action_dim must be positive")
        self._dimension = int(action_dim)
        self._low_pass = np.zeros(self._dimension, dtype=np.float64)
        self._upper_history = []
        self._lower_history = []
        self._context = np.zeros(self._dimension, dtype=np.float64)
        self._logit_alpha = float(np.log(
            self.initial_alpha / (1.0 - self.initial_alpha)
        ))

    @property
    def context(self) -> np.ndarray:
        self._require_reset()
        return self._context.astype(np.float32, copy=True)

    def split(
        self,
        upper: Any,
        lower: Any,
        *,
        lower_limit: float | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Return the causal audit-aligned responsibility split."""

        self._require_reset()
        upper_value = _finite_vector(upper, name="upper action")
        lower_value = _finite_vector(lower, name="lower action")
        if (
            upper_value.shape != (self._dimension,)
            or lower_value.shape != upper_value.shape
        ):
            raise ValueError(
                "audit-aligned gauge actions must align with reset action_dim"
            )
        limit = None if lower_limit is None else float(lower_limit)
        if limit is not None and (not np.isfinite(limit) or limit <= 0.0):
            raise ValueError(
                "audit-aligned gauge lower_limit must be positive and finite"
            )

        total = upper_value + lower_value
        alpha_before = self._alpha()
        low_pass_before = self._low_pass.copy()
        self._low_pass += alpha_before * (total - self._low_pass)
        canonical_upper = self._low_pass.copy()
        canonical_lower_requested = total - canonical_upper
        canonical_lower = (
            canonical_lower_requested
            if limit is None
            else np.clip(canonical_lower_requested, -limit, limit)
        )
        canonical_upper = total - canonical_lower
        self._low_pass = canonical_upper.copy()
        transfer = self.strength * (canonical_upper - upper_value)
        fixed_upper = upper_value + transfer
        fixed_lower = lower_value - transfer

        self._upper_history.append(fixed_upper.copy())
        self._lower_history.append(fixed_lower.copy())
        self._upper_history = self._upper_history[-self.upper_window:]
        self._lower_history = self._lower_history[-self.lower_window:]
        upper_low = np.mean(self._upper_history, axis=0)
        upper_high = fixed_upper - upper_low
        lower_low = np.mean(self._lower_history, axis=0)
        normalized_upper = float(np.mean(np.square(
            upper_high / self.upper_rms_budget
        )))
        normalized_lower = float(np.mean(np.square(
            lower_low / self.lower_rms_budget
        )))
        objective = normalized_upper + normalized_lower
        normalized_imbalance = normalized_lower - normalized_upper
        self._logit_alpha = float(np.clip(
            self._logit_alpha
            + self.adaptation_rate * np.clip(normalized_imbalance, -1.0, 1.0),
            self.minimum_logit,
            self.maximum_logit,
        ))
        alpha_after = self._alpha()
        reconstruction_error = fixed_upper + fixed_lower - total
        self._context = fixed_upper.copy()
        return {
            "upper": fixed_upper.astype(np.float32, copy=True),
            "lower": fixed_lower.astype(np.float32, copy=True),
            "total": total.astype(np.float32, copy=True),
            "transfer": transfer.astype(np.float32, copy=True),
            "canonical_upper": canonical_upper.astype(np.float32, copy=True),
            "canonical_lower": canonical_lower.astype(np.float32, copy=True),
            "low_pass_before": low_pass_before.astype(np.float32, copy=True),
            "low_pass_after": self._low_pass.astype(np.float32, copy=True),
            "upper_high": upper_high.astype(np.float32, copy=True),
            "lower_low": lower_low.astype(np.float32, copy=True),
            "normalized_upper_hf": normalized_upper,
            "normalized_lower_lf": normalized_lower,
            "normalized_band_imbalance": normalized_imbalance,
            "normalized_local_objective": objective,
            "alpha_before": alpha_before,
            "alpha_after": alpha_after,
            "reconstruction_error": reconstruction_error.astype(
                np.float64, copy=True
            ),
            "canonical_lower_clip_rate": float(np.mean(
                np.abs(canonical_lower - canonical_lower_requested) > 1e-12
            )),
            "gauge_fixed": float(self.strength == 1.0),
        }

    def _alpha(self) -> float:
        return float(1.0 / (1.0 + np.exp(-self._logit_alpha)))

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("audit-aligned gauge fixer must be reset")


@dataclass
class CausalMacroHoldAuditGaugeFixer:
    """Gauge-fix additive responsibilities at upper decision boundaries.

    The primitive-step audit gauge can make the reported upper responsibility
    move at the lower-controller rate. This variant maintains a causal EMA of
    the total action but copies it into the upper responsibility only at an
    explicit macro boundary. Between boundaries the upper responsibility is
    held and the lower responsibility is its exact additive complement. The
    adaptive cutoff uses the same HPF8/LPF32 budget imbalance as the primitive
    gauge while preserving the hierarchy's temporal contract.
    """

    upper_window: int = 8
    lower_window: int = 32
    upper_rms_budget: float = 0.075
    lower_rms_budget: float = 0.0475
    initial_alpha: float = 0.20
    adaptation_rate: float = 0.03
    minimum_logit: float = -4.0
    maximum_logit: float = 4.0
    strength: float = 1.0

    def __post_init__(self) -> None:
        self.upper_window = int(self.upper_window)
        self.lower_window = int(self.lower_window)
        self.upper_rms_budget = float(self.upper_rms_budget)
        self.lower_rms_budget = float(self.lower_rms_budget)
        self.initial_alpha = float(self.initial_alpha)
        self.adaptation_rate = float(self.adaptation_rate)
        self.minimum_logit = float(self.minimum_logit)
        self.maximum_logit = float(self.maximum_logit)
        self.strength = float(self.strength)
        if self.upper_window < 2 or self.lower_window < 2:
            raise ValueError("macro-hold gauge windows must be at least two")
        if (
            not np.isfinite(self.upper_rms_budget)
            or self.upper_rms_budget <= 0.0
            or not np.isfinite(self.lower_rms_budget)
            or self.lower_rms_budget <= 0.0
        ):
            raise ValueError("macro-hold gauge budgets must be positive")
        if (
            not np.isfinite(self.initial_alpha)
            or not 0.0 < self.initial_alpha < 1.0
        ):
            raise ValueError("macro-hold gauge initial_alpha must be in (0, 1)")
        if (
            not np.isfinite(self.adaptation_rate)
            or self.adaptation_rate < 0.0
        ):
            raise ValueError(
                "macro-hold gauge adaptation_rate must be non-negative"
            )
        if (
            not np.isfinite(self.minimum_logit)
            or not np.isfinite(self.maximum_logit)
            or self.minimum_logit >= self.maximum_logit
        ):
            raise ValueError("macro-hold gauge logit bounds are invalid")
        if not np.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("macro-hold gauge strength must be in [0, 1]")
        self._dimension = 0
        self._low_pass = np.zeros(0, dtype=np.float64)
        self._held_upper = np.zeros(0, dtype=np.float64)
        self._upper_history: list[np.ndarray] = []
        self._lower_history: list[np.ndarray] = []
        self._context = np.zeros(0, dtype=np.float64)
        self._logit_alpha = 0.0
        self._has_macro = False

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("macro-hold gauge action_dim must be positive")
        self._dimension = int(action_dim)
        self._low_pass = np.zeros(self._dimension, dtype=np.float64)
        self._held_upper = np.zeros(self._dimension, dtype=np.float64)
        self._upper_history = []
        self._lower_history = []
        self._context = np.zeros(self._dimension, dtype=np.float64)
        self._logit_alpha = float(np.log(
            self.initial_alpha / (1.0 - self.initial_alpha)
        ))
        self._has_macro = False

    @property
    def context(self) -> np.ndarray:
        self._require_reset()
        return self._context.astype(np.float32, copy=True)

    def split(
        self,
        upper: Any,
        lower: Any,
        *,
        macro_boundary: bool,
        lower_limit: float | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Return an exact split whose upper coordinate is macro-held."""

        self._require_reset()
        upper_value = _finite_vector(upper, name="upper action")
        lower_value = _finite_vector(lower, name="lower action")
        if (
            upper_value.shape != (self._dimension,)
            or lower_value.shape != upper_value.shape
        ):
            raise ValueError(
                "macro-hold gauge actions must align with reset action_dim"
            )
        limit = None if lower_limit is None else float(lower_limit)
        if limit is not None and (not np.isfinite(limit) or limit <= 0.0):
            raise ValueError(
                "macro-hold gauge lower_limit must be positive and finite"
            )
        if not isinstance(macro_boundary, (bool, np.bool_)):
            raise ValueError("macro_boundary must be boolean")

        total = upper_value + lower_value
        alpha_before = self._alpha()
        low_pass_before = self._low_pass.copy()
        held_upper_before = self._held_upper.copy()
        if not self._has_macro:
            if not bool(macro_boundary):
                raise RuntimeError(
                    "macro-hold gauge requires a boundary on its first split"
                )
            self._low_pass = total.copy()
        else:
            self._low_pass += alpha_before * (total - self._low_pass)
        if bool(macro_boundary):
            self._held_upper = self._low_pass.copy()
            self._has_macro = True

        canonical_upper_requested = self._held_upper.copy()
        canonical_lower_requested = total - canonical_upper_requested
        canonical_lower = (
            canonical_lower_requested
            if limit is None
            else np.clip(canonical_lower_requested, -limit, limit)
        )
        canonical_upper = total - canonical_lower
        transfer = self.strength * (canonical_upper - upper_value)
        fixed_upper = upper_value + transfer
        fixed_lower = lower_value - transfer

        self._upper_history.append(fixed_upper.copy())
        self._lower_history.append(fixed_lower.copy())
        self._upper_history = self._upper_history[-self.upper_window:]
        self._lower_history = self._lower_history[-self.lower_window:]
        upper_low = np.mean(self._upper_history, axis=0)
        upper_high = fixed_upper - upper_low
        lower_low = np.mean(self._lower_history, axis=0)
        normalized_upper = float(np.mean(np.square(
            upper_high / self.upper_rms_budget
        )))
        normalized_lower = float(np.mean(np.square(
            lower_low / self.lower_rms_budget
        )))
        normalized_imbalance = normalized_lower - normalized_upper
        self._logit_alpha = float(np.clip(
            self._logit_alpha
            + self.adaptation_rate * np.clip(normalized_imbalance, -1.0, 1.0),
            self.minimum_logit,
            self.maximum_logit,
        ))
        alpha_after = self._alpha()
        reconstruction_error = fixed_upper + fixed_lower - total
        self._context = fixed_upper.copy()
        hold_error = canonical_upper - canonical_upper_requested
        return {
            "upper": fixed_upper.astype(np.float32, copy=True),
            "lower": fixed_lower.astype(np.float32, copy=True),
            "total": total.astype(np.float32, copy=True),
            "transfer": transfer.astype(np.float32, copy=True),
            "canonical_upper": canonical_upper.astype(np.float32, copy=True),
            "canonical_lower": canonical_lower.astype(np.float32, copy=True),
            "low_pass_before": low_pass_before.astype(np.float32, copy=True),
            "low_pass_after": self._low_pass.astype(np.float32, copy=True),
            "held_upper_before": held_upper_before.astype(np.float32, copy=True),
            "held_upper_after": self._held_upper.astype(np.float32, copy=True),
            "upper_high": upper_high.astype(np.float32, copy=True),
            "lower_low": lower_low.astype(np.float32, copy=True),
            "normalized_upper_hf": normalized_upper,
            "normalized_lower_lf": normalized_lower,
            "normalized_band_imbalance": normalized_imbalance,
            "normalized_local_objective": normalized_upper + normalized_lower,
            "alpha_before": alpha_before,
            "alpha_after": alpha_after,
            "macro_boundary": float(bool(macro_boundary)),
            "macro_hold_error_rms": float(np.sqrt(np.mean(np.square(
                hold_error
            )))),
            "reconstruction_error": reconstruction_error.astype(
                np.float64, copy=True
            ),
            "canonical_lower_clip_rate": float(np.mean(
                np.abs(canonical_lower - canonical_lower_requested) > 1e-12
            )),
            "gauge_fixed": float(self.strength == 1.0),
        }

    def _alpha(self) -> float:
        return float(1.0 / (1.0 + np.exp(-self._logit_alpha)))

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("macro-hold gauge fixer must be reset")


@dataclass
class CausalSmoothMacroGaugeFixer:
    """Gauge-fix additive responsibilities with a frozen smooth macro plan.

    The canonical target is the prior-step causal low-pass estimate of the
    total action, sampled only at an upper-policy boundary. A smoothstep curve
    connects consecutive targets over the primitive steps of that macro
    interval. The requested curve is projected onto the exact per-step
    component-feasibility interval, and the lower responsibility is its
    additive complement.

    The low-pass state and frozen curve depend only on the total action, never
    on ``strength`` or the supplied responsibility split. Thus a strength-zero
    control and a full-strength gauge follow exactly the same environment path
    while exposing different, identifiable upper/lower coordinates.
    """

    macro_steps: int = 16
    alpha: float = 0.10
    strength: float = 1.0

    def __post_init__(self) -> None:
        self.macro_steps = int(self.macro_steps)
        self.alpha = float(self.alpha)
        self.strength = float(self.strength)
        if self.macro_steps < 2:
            raise ValueError("smooth macro gauge requires at least two steps")
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("smooth macro gauge alpha must be in (0, 1]")
        if not np.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("smooth macro gauge strength must be in [0, 1]")
        self._dimension = 0
        self._low_pass = np.zeros(0, dtype=np.float64)
        self._plan = CausalSmoothstepMacroPlan(self.macro_steps)
        self._has_total = False
        self._has_macro = False

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("smooth macro gauge action_dim must be positive")
        self._dimension = int(action_dim)
        self._low_pass = np.zeros(self._dimension, dtype=np.float64)
        self._plan.reset(self._dimension)
        self._has_total = False
        self._has_macro = False

    @property
    def context(self) -> np.ndarray:
        self._require_reset()
        return self._low_pass.astype(np.float32, copy=True)

    def split(
        self,
        upper: Any,
        lower: Any,
        *,
        macro_boundary: bool,
        upper_limit: float = 1.0,
        lower_limit: float = 1.0,
    ) -> dict[str, np.ndarray | float]:
        """Return a bounded smooth split with exact additive reconstruction."""

        self._require_reset()
        upper_value = _finite_vector(upper, name="upper action")
        lower_value = _finite_vector(lower, name="lower action")
        if (
            upper_value.shape != (self._dimension,)
            or lower_value.shape != upper_value.shape
        ):
            raise ValueError(
                "smooth macro gauge actions must align with reset action_dim"
            )
        if not isinstance(macro_boundary, (bool, np.bool_)):
            raise ValueError("macro_boundary must be boolean")
        upper_bound = float(upper_limit)
        lower_bound = float(lower_limit)
        if not np.isfinite(upper_bound) or upper_bound <= 0.0:
            raise ValueError("smooth macro gauge upper_limit must be positive")
        if not np.isfinite(lower_bound) or lower_bound <= 0.0:
            raise ValueError("smooth macro gauge lower_limit must be positive")
        if not self._has_macro and not bool(macro_boundary):
            raise RuntimeError(
                "smooth macro gauge requires a boundary on its first split"
            )

        total = upper_value + lower_value
        low_pass_before = self._low_pass.copy()
        if bool(macro_boundary):
            smooth_requested = np.asarray(
                self._plan.activate(low_pass_before), dtype=np.float64
            )
            self._has_macro = True
        else:
            smooth_requested = np.asarray(
                self._plan.advance(), dtype=np.float64
            )

        if not self._has_total:
            self._low_pass = total.copy()
            self._has_total = True
        else:
            self._low_pass += self.alpha * (total - self._low_pass)

        feasible_low = np.maximum(-upper_bound, total - lower_bound)
        feasible_high = np.minimum(upper_bound, total + lower_bound)
        if np.any(feasible_low > feasible_high + 1e-10):
            raise RuntimeError(
                "smooth macro gauge has no feasible bounded component split"
            )
        feasible_low = np.minimum(feasible_low, feasible_high)
        canonical_upper = np.clip(
            smooth_requested, feasible_low, feasible_high
        )
        canonical_lower = total - canonical_upper
        canonical_transfer = canonical_upper - upper_value
        transfer = self.strength * canonical_transfer
        fixed_upper = upper_value + transfer
        fixed_lower = lower_value - transfer
        reconstruction_error = fixed_upper + fixed_lower - total
        component_clip = canonical_upper - smooth_requested

        return {
            "upper": fixed_upper.astype(np.float32, copy=True),
            "lower": fixed_lower.astype(np.float32, copy=True),
            "total": total.astype(np.float32, copy=True),
            "transfer": transfer.astype(np.float32, copy=True),
            "canonical_upper": canonical_upper.astype(np.float32, copy=True),
            "canonical_lower": canonical_lower.astype(np.float32, copy=True),
            "low_pass_before": low_pass_before.astype(np.float32, copy=True),
            "low_pass_after": self._low_pass.astype(np.float32, copy=True),
            "smooth_requested": smooth_requested.astype(np.float32, copy=True),
            "smooth_target": self._plan.target,
            "smooth_progress": self._plan.progress,
            "feasible_upper_low": feasible_low.astype(np.float32, copy=True),
            "feasible_upper_high": feasible_high.astype(np.float32, copy=True),
            "alpha_before": self.alpha,
            "alpha_after": self.alpha,
            "normalized_band_imbalance": 0.0,
            "macro_boundary": float(bool(macro_boundary)),
            "reconstruction_error": reconstruction_error.astype(
                np.float64, copy=True
            ),
            "canonical_component_clip_rate": float(np.mean(
                np.abs(component_clip) > 1e-12
            )),
            "canonical_component_clip_rms": float(np.sqrt(np.mean(
                np.square(component_clip)
            ))),
            "canonical_lower_clip_rate": float(np.mean(
                np.abs(component_clip) > 1e-12
            )),
            "gauge_fixed": float(self.strength == 1.0),
        }

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("smooth macro gauge fixer must be reset")


@dataclass
class CausalAuditOptimalMacroGaugeFixer:
    """Choose a causal macro responsibility plan against HPF8/LPF32.

    At each macro boundary, the current total action is known and is used as a
    persistence forecast for the remaining primitive steps. A deterministic
    box-constrained quadratic solve chooses the upper responsibility sequence
    that minimizes the registered normalized upper high-pass and lower low-pass
    residuals over that horizon. Between boundaries the plan is frozen. The
    requested upper responsibility is projected onto the component-feasibility
    interval induced by the realized total, and the lower responsibility is its
    exact complement.

    Planning histories contain canonical responsibilities only. They therefore
    depend on the total action but not on ``strength`` or the supplied additive
    factorization, preserving pathwise function equivalence across interventions.
    """

    macro_steps: int = 16
    upper_window: int = 8
    lower_window: int = 32
    upper_rms_budget: float = 0.075
    lower_rms_budget: float = 0.0475
    low_pass_alpha: float = 0.20
    coordinate_sweeps: int = 128
    strength: float = 1.0

    def __post_init__(self) -> None:
        self.macro_steps = int(self.macro_steps)
        self.upper_window = int(self.upper_window)
        self.lower_window = int(self.lower_window)
        self.upper_rms_budget = float(self.upper_rms_budget)
        self.lower_rms_budget = float(self.lower_rms_budget)
        self.low_pass_alpha = float(self.low_pass_alpha)
        self.coordinate_sweeps = int(self.coordinate_sweeps)
        self.strength = float(self.strength)
        if self.macro_steps < 2:
            raise ValueError("audit-optimal gauge requires at least two steps")
        if self.upper_window < 2 or self.lower_window < 2:
            raise ValueError("audit-optimal gauge windows must be at least two")
        if (
            not np.isfinite(self.upper_rms_budget)
            or self.upper_rms_budget <= 0.0
            or not np.isfinite(self.lower_rms_budget)
            or self.lower_rms_budget <= 0.0
        ):
            raise ValueError("audit-optimal gauge budgets must be positive")
        if (
            not np.isfinite(self.low_pass_alpha)
            or not 0.0 < self.low_pass_alpha <= 1.0
        ):
            raise ValueError("audit-optimal low-pass alpha must be in (0, 1]")
        if self.coordinate_sweeps < 1:
            raise ValueError("audit-optimal coordinate_sweeps must be positive")
        if not np.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("audit-optimal gauge strength must be in [0, 1]")
        self._dimension = 0
        self._low_pass = np.zeros(0, dtype=np.float64)
        self._plan = np.zeros((self.macro_steps, 0), dtype=np.float64)
        self._current = np.zeros(0, dtype=np.float64)
        self._phase = 0
        self._active = False
        self._has_total = False
        self._upper_history: list[np.ndarray] = []
        self._lower_history: list[np.ndarray] = []
        self._predicted_baseline_objective = 0.0
        self._predicted_optimal_objective = 0.0

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("audit-optimal gauge action_dim must be positive")
        self._dimension = int(action_dim)
        self._low_pass = np.zeros(self._dimension, dtype=np.float64)
        self._plan = np.zeros(
            (self.macro_steps, self._dimension), dtype=np.float64
        )
        self._current = np.zeros(self._dimension, dtype=np.float64)
        self._phase = 0
        self._active = False
        self._has_total = False
        self._upper_history = []
        self._lower_history = []
        self._predicted_baseline_objective = 0.0
        self._predicted_optimal_objective = 0.0

    @property
    def context(self) -> np.ndarray:
        self._require_reset()
        return self._low_pass.astype(np.float32, copy=True)

    @property
    def policy_context(self) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
        """Return all compact plan state consumed by a feed-forward policy."""

        self._require_reset()
        target = self._plan[-1] if self._active else np.zeros(self._dimension)
        phase = (
            float(self._phase / (self.macro_steps - 1))
            if self._active else 0.0
        )
        return (
            (
                self._low_pass.astype(np.float32, copy=True),
                self._current.astype(np.float32, copy=True),
                np.asarray(target, dtype=np.float32).copy(),
            ),
            (phase,),
        )

    def split(
        self,
        upper: Any,
        lower: Any,
        *,
        macro_boundary: bool,
        upper_limit: float = 1.0,
        lower_limit: float = 1.0,
    ) -> dict[str, np.ndarray | float]:
        """Return a bounded audit-optimized split with exact reconstruction."""

        self._require_reset()
        upper_value = _finite_vector(upper, name="upper action")
        lower_value = _finite_vector(lower, name="lower action")
        if (
            upper_value.shape != (self._dimension,)
            or lower_value.shape != upper_value.shape
        ):
            raise ValueError(
                "audit-optimal gauge actions must align with reset action_dim"
            )
        if not isinstance(macro_boundary, (bool, np.bool_)):
            raise ValueError("macro_boundary must be boolean")
        upper_bound = float(upper_limit)
        lower_bound = float(lower_limit)
        if not np.isfinite(upper_bound) or upper_bound <= 0.0:
            raise ValueError("audit-optimal upper_limit must be positive")
        if not np.isfinite(lower_bound) or lower_bound <= 0.0:
            raise ValueError("audit-optimal lower_limit must be positive")
        if not self._active and not bool(macro_boundary):
            raise RuntimeError(
                "audit-optimal gauge requires a boundary on its first split"
            )

        total = upper_value + lower_value
        low_pass_before = self._low_pass.copy()
        if bool(macro_boundary):
            forecast = np.repeat(
                total.reshape(1, -1), self.macro_steps, axis=0
            )
            plan_rows: list[np.ndarray] = []
            baseline_objectives: list[float] = []
            optimal_objectives: list[float] = []
            past_upper = self._history_matrix(
                self._upper_history, self.upper_window - 1
            )
            past_lower = self._history_matrix(
                self._lower_history, self.lower_window - 1
            )
            for dimension in range(self._dimension):
                feasible_low = np.maximum(
                    -upper_bound, forecast[:, dimension] - lower_bound
                )
                feasible_high = np.minimum(
                    upper_bound, forecast[:, dimension] + lower_bound
                )
                if np.any(feasible_low > feasible_high + 1e-10):
                    raise RuntimeError(
                        "audit-optimal forecast has no feasible component split"
                    )
                plan, baseline_objective, optimal_objective = (
                    self._solve_dimension(
                        forecast=forecast[:, dimension],
                        past_upper=past_upper[:, dimension],
                        past_lower=past_lower[:, dimension],
                        feasible_low=np.minimum(feasible_low, feasible_high),
                        feasible_high=feasible_high,
                    )
                )
                plan_rows.append(plan)
                baseline_objectives.append(baseline_objective)
                optimal_objectives.append(optimal_objective)
            self._plan = np.stack(plan_rows, axis=1)
            self._phase = 0
            self._active = True
            self._predicted_baseline_objective = float(
                np.mean(baseline_objectives)
            )
            self._predicted_optimal_objective = float(
                np.mean(optimal_objectives)
            )
        else:
            self._phase = min(self._phase + 1, self.macro_steps - 1)

        requested_upper = self._plan[self._phase].copy()
        feasible_low = np.maximum(-upper_bound, total - lower_bound)
        feasible_high = np.minimum(upper_bound, total + lower_bound)
        if np.any(feasible_low > feasible_high + 1e-10):
            raise RuntimeError(
                "audit-optimal gauge has no feasible realized component split"
            )
        feasible_low = np.minimum(feasible_low, feasible_high)
        canonical_upper = np.clip(
            requested_upper, feasible_low, feasible_high
        )
        canonical_lower = total - canonical_upper
        canonical_transfer = canonical_upper - upper_value
        transfer = self.strength * canonical_transfer
        fixed_upper = upper_value + transfer
        fixed_lower = lower_value - transfer
        reconstruction_error = fixed_upper + fixed_lower - total
        component_clip = canonical_upper - requested_upper

        self._current = canonical_upper.copy()
        self._upper_history.append(canonical_upper.copy())
        self._lower_history.append(canonical_lower.copy())
        self._upper_history = self._upper_history[-(self.upper_window - 1):]
        self._lower_history = self._lower_history[-(self.lower_window - 1):]
        if not self._has_total:
            self._low_pass = total.copy()
            self._has_total = True
        else:
            self._low_pass += self.low_pass_alpha * (total - self._low_pass)

        return {
            "upper": fixed_upper.astype(np.float32, copy=True),
            "lower": fixed_lower.astype(np.float32, copy=True),
            "total": total.astype(np.float32, copy=True),
            "transfer": transfer.astype(np.float32, copy=True),
            "canonical_upper": canonical_upper.astype(np.float32, copy=True),
            "canonical_lower": canonical_lower.astype(np.float32, copy=True),
            "low_pass_before": low_pass_before.astype(np.float32, copy=True),
            "low_pass_after": self._low_pass.astype(np.float32, copy=True),
            "audit_requested": requested_upper.astype(np.float32, copy=True),
            "audit_target": self._plan[-1].astype(np.float32, copy=True),
            "audit_progress": float(self._phase / (self.macro_steps - 1)),
            "predicted_baseline_objective": (
                self._predicted_baseline_objective
            ),
            "predicted_optimal_objective": self._predicted_optimal_objective,
            "feasible_upper_low": feasible_low.astype(np.float32, copy=True),
            "feasible_upper_high": feasible_high.astype(np.float32, copy=True),
            "alpha_before": self.low_pass_alpha,
            "alpha_after": self.low_pass_alpha,
            "normalized_band_imbalance": 0.0,
            "macro_boundary": float(bool(macro_boundary)),
            "reconstruction_error": reconstruction_error.astype(
                np.float64, copy=True
            ),
            "canonical_component_clip_rate": float(np.mean(
                np.abs(component_clip) > 1e-12
            )),
            "canonical_component_clip_rms": float(np.sqrt(np.mean(
                np.square(component_clip)
            ))),
            "canonical_lower_clip_rate": float(np.mean(
                np.abs(component_clip) > 1e-12
            )),
            "gauge_fixed": float(self.strength == 1.0),
        }

    def _solve_dimension(
        self,
        *,
        forecast: np.ndarray,
        past_upper: np.ndarray,
        past_lower: np.ndarray,
        feasible_low: np.ndarray,
        feasible_high: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        horizon = self.macro_steps
        zeros = np.zeros(horizon, dtype=np.float64)
        offset = self._audit_residuals(
            zeros, forecast, past_upper, past_lower
        )
        design = np.empty((offset.size, horizon), dtype=np.float64)
        for index in range(horizon):
            basis = zeros.copy()
            basis[index] = 1.0
            design[:, index] = (
                self._audit_residuals(
                    basis, forecast, past_upper, past_lower
                )
                - offset
            )

        reference = (
            float(past_upper[-1]) if past_upper.size else float(forecast[0])
        )
        baseline = np.clip(
            np.full(horizon, reference, dtype=np.float64),
            feasible_low,
            feasible_high,
        )
        gram = design.T @ design
        linear = design.T @ offset
        diagonal_scale = max(float(np.max(np.diag(gram))), 1.0)
        tie_break = 1e-10 * diagonal_scale
        gram = gram + tie_break * np.eye(horizon, dtype=np.float64)
        linear = linear - tie_break * baseline
        candidate = baseline.copy()
        for _ in range(self.coordinate_sweeps):
            largest_change = 0.0
            for index in range(horizon):
                gradient = float(gram[index] @ candidate + linear[index])
                updated = float(np.clip(
                    candidate[index] - gradient / gram[index, index],
                    feasible_low[index],
                    feasible_high[index],
                ))
                largest_change = max(
                    largest_change, abs(updated - candidate[index])
                )
                candidate[index] = updated
            if largest_change <= 1e-11:
                break

        baseline_objective = self._audit_objective(
            baseline, forecast, past_upper, past_lower
        )
        candidate_objective = self._audit_objective(
            candidate, forecast, past_upper, past_lower
        )
        if candidate_objective > baseline_objective:
            return baseline, baseline_objective, baseline_objective
        return candidate, baseline_objective, candidate_objective

    def _audit_objective(
        self,
        upper_plan: np.ndarray,
        forecast: np.ndarray,
        past_upper: np.ndarray,
        past_lower: np.ndarray,
    ) -> float:
        residuals = self._audit_residuals(
            upper_plan, forecast, past_upper, past_lower
        )
        return float(np.mean(np.square(residuals)))

    def _audit_residuals(
        self,
        upper_plan: np.ndarray,
        forecast: np.ndarray,
        past_upper: np.ndarray,
        past_lower: np.ndarray,
    ) -> np.ndarray:
        future_upper = np.asarray(upper_plan, dtype=np.float64).reshape(-1)
        future_total = np.asarray(forecast, dtype=np.float64).reshape(-1)
        if (
            future_upper.shape != (self.macro_steps,)
            or future_total.shape != future_upper.shape
        ):
            raise ValueError("audit-optimal plan and forecast must align")
        upper_values = np.concatenate((past_upper, future_upper))
        lower_values = np.concatenate((past_lower, future_total - future_upper))
        upper_offset = int(past_upper.size)
        lower_offset = int(past_lower.size)
        upper_residuals = []
        lower_residuals = []
        for index in range(self.macro_steps):
            upper_end = upper_offset + index + 1
            upper_start = max(0, upper_end - self.upper_window)
            upper_mean = float(np.mean(
                upper_values[upper_start:upper_end]
            ))
            upper_residuals.append(
                (future_upper[index] - upper_mean) / self.upper_rms_budget
            )
            lower_end = lower_offset + index + 1
            lower_start = max(0, lower_end - self.lower_window)
            lower_residuals.append(
                float(np.mean(lower_values[lower_start:lower_end]))
                / self.lower_rms_budget
            )
        return np.asarray(
            [*upper_residuals, *lower_residuals], dtype=np.float64
        )

    def _history_matrix(
        self, history: list[np.ndarray], maximum_rows: int
    ) -> np.ndarray:
        rows = history[-int(maximum_rows):]
        if not rows:
            return np.empty((0, self._dimension), dtype=np.float64)
        return np.stack(rows, axis=0).astype(np.float64, copy=False)

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("audit-optimal gauge fixer must be reset")


@dataclass
class CausalStreamingAuditProjectionFixer:
    """Project each causal split against the exact streaming audit state.

    The registered HPF and LPF diagnostics are sums of primitive-step rolling
    residuals.  At each step this fixer uses the realized total action as a
    constant-tail forecast, analytically minimizes the normalized audit over one
    receding macro horizon, executes only the first projected coordinate, and
    replans after the next realized total.  This accounts for the delayed effect
    of the current split on the lower rolling mean without freezing a stale
    macro plan.  The physical component bounds are hard.  The current upper HPF
    budget is also hard whenever its intersection with the physical interval is
    nonempty.  If it is physically infeasible, the selected component minimizes
    the unavoidable upper residual.

    Canonical histories depend only on the total action.  They are updated
    independently of ``strength`` and the supplied additive factorization, so
    full-strength coordinates remain gauge invariant and paired interventions
    receive identical policy state.
    """

    upper_window: int = 8
    lower_window: int = 32
    upper_rms_budget: float = 0.075
    lower_rms_budget: float = 0.0475
    planning_horizon: int = 16
    strength: float = 1.0

    def __post_init__(self) -> None:
        self.upper_window = int(self.upper_window)
        self.lower_window = int(self.lower_window)
        self.upper_rms_budget = float(self.upper_rms_budget)
        self.lower_rms_budget = float(self.lower_rms_budget)
        self.planning_horizon = int(self.planning_horizon)
        self.strength = float(self.strength)
        if self.upper_window < 2 or self.lower_window < 2:
            raise ValueError("streaming audit windows must be at least two")
        if (
            not np.isfinite(self.upper_rms_budget)
            or self.upper_rms_budget <= 0.0
            or not np.isfinite(self.lower_rms_budget)
            or self.lower_rms_budget <= 0.0
        ):
            raise ValueError("streaming audit budgets must be positive")
        if self.planning_horizon < 1:
            raise ValueError("streaming audit planning_horizon must be positive")
        if not np.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("streaming audit strength must be in [0, 1]")
        self._dimension = 0
        self._current = np.zeros(0, dtype=np.float64)
        self._upper_history: list[np.ndarray] = []
        self._lower_history: list[np.ndarray] = []

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("streaming audit action_dim must be positive")
        self._dimension = int(action_dim)
        self._current = np.zeros(self._dimension, dtype=np.float64)
        self._upper_history = []
        self._lower_history = []

    @property
    def context(self) -> np.ndarray:
        self._require_reset()
        return self._current.astype(np.float32, copy=True)

    @property
    def policy_context(self) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
        """Return the complete finite-memory state used by the projection."""

        self._require_reset()
        upper_rows = self._padded_history(
            self._upper_history, self.upper_window - 1
        )
        lower_rows = self._padded_history(
            self._lower_history, self.lower_window - 1
        )
        return (
            tuple(
                row.astype(np.float32, copy=True)
                for row in np.concatenate((upper_rows, lower_rows), axis=0)
            ),
            (
                float(len(self._upper_history) / (self.upper_window - 1)),
                float(len(self._lower_history) / (self.lower_window - 1)),
            ),
        )

    def split(
        self,
        upper: Any,
        lower: Any,
        *,
        upper_limit: float = 1.0,
        lower_limit: float = 1.0,
    ) -> dict[str, np.ndarray | float]:
        """Return the exact streaming-audit projection and complement."""

        self._require_reset()
        upper_value = _finite_vector(upper, name="upper action")
        lower_value = _finite_vector(lower, name="lower action")
        if (
            upper_value.shape != (self._dimension,)
            or lower_value.shape != upper_value.shape
        ):
            raise ValueError(
                "streaming audit actions must align with reset action_dim"
            )
        upper_bound = float(upper_limit)
        lower_bound = float(lower_limit)
        if not np.isfinite(upper_bound) or upper_bound <= 0.0:
            raise ValueError("streaming audit upper_limit must be positive")
        if not np.isfinite(lower_bound) or lower_bound <= 0.0:
            raise ValueError("streaming audit lower_limit must be positive")

        total = upper_value + lower_value
        physical_low = np.maximum(-upper_bound, total - lower_bound)
        physical_high = np.minimum(upper_bound, total + lower_bound)
        if np.any(physical_low > physical_high + 1e-10):
            raise RuntimeError(
                "streaming audit has no feasible component split"
            )
        physical_low = np.minimum(physical_low, physical_high)

        upper_count = len(self._upper_history)
        lower_count = len(self._lower_history)
        upper_sum = (
            np.sum(self._upper_history, axis=0)
            if self._upper_history else np.zeros(self._dimension)
        )
        lower_sum = (
            np.sum(self._lower_history, axis=0)
            if self._lower_history else np.zeros(self._dimension)
        )
        upper_denominator = float(upper_count + 1)
        lower_denominator = float(lower_count + 1)
        zeros = np.zeros(self._dimension, dtype=np.float64)
        ones = np.ones(self._dimension, dtype=np.float64)
        offset = self._constant_tail_residuals(
            zeros, total, self._upper_history, self._lower_history
        )
        response = (
            self._constant_tail_residuals(
                ones, zeros, self._zero_history(upper_count),
                self._zero_history(lower_count),
            )
            - self._constant_tail_residuals(
                zeros, zeros, self._zero_history(upper_count),
                self._zero_history(lower_count),
            )
        )
        quadratic = np.sum(np.square(response), axis=0)
        linear = np.sum(response * offset, axis=0)
        unconstrained = -linear / quadratic

        if upper_count:
            upper_audit_low = (
                upper_sum - upper_denominator * self.upper_rms_budget
            ) / float(upper_count)
            upper_audit_high = (
                upper_sum + upper_denominator * self.upper_rms_budget
            ) / float(upper_count)
        else:
            upper_audit_low = np.full(self._dimension, -np.inf)
            upper_audit_high = np.full(self._dimension, np.inf)
        upper_low = np.maximum(physical_low, upper_audit_low)
        upper_high = np.minimum(physical_high, upper_audit_high)
        upper_feasible = upper_low <= upper_high + 1e-12
        upper_low = np.minimum(upper_low, upper_high)

        canonical_upper = np.empty(self._dimension, dtype=np.float64)
        canonical_upper[upper_feasible] = np.clip(
            unconstrained[upper_feasible],
            upper_low[upper_feasible],
            upper_high[upper_feasible],
        )
        upper_infeasible = ~upper_feasible
        upper_zero_residual = (
            upper_sum / float(upper_count)
            if upper_count else unconstrained
        )
        canonical_upper[upper_infeasible] = np.clip(
            upper_zero_residual[upper_infeasible],
            physical_low[upper_infeasible],
            physical_high[upper_infeasible],
        )
        canonical_lower = total - canonical_upper

        upper_mean = (
            upper_sum + canonical_upper
        ) / upper_denominator
        upper_high_residual = canonical_upper - upper_mean
        lower_low_residual = (
            lower_sum + canonical_lower
        ) / lower_denominator
        normalized_upper = float(np.mean(np.square(
            upper_high_residual / self.upper_rms_budget
        )))
        normalized_lower = float(np.mean(np.square(
            lower_low_residual / self.lower_rms_budget
        )))
        raw_residuals = offset + response * upper_value.reshape(1, -1)
        optimal_residuals = offset + response * canonical_upper.reshape(1, -1)
        raw_objective = float(np.mean(np.square(raw_residuals)))
        optimal_objective = float(np.mean(np.square(optimal_residuals)))
        canonical_objective = normalized_upper + normalized_lower

        canonical_transfer = canonical_upper - upper_value
        transfer = self.strength * canonical_transfer
        fixed_upper = upper_value + transfer
        fixed_lower = lower_value - transfer
        reconstruction_error = fixed_upper + fixed_lower - total
        self._current = canonical_upper.copy()
        self._upper_history.append(canonical_upper.copy())
        self._lower_history.append(canonical_lower.copy())
        self._upper_history = self._upper_history[-(self.upper_window - 1):]
        self._lower_history = self._lower_history[-(self.lower_window - 1):]

        upper_violation = np.maximum(
            np.abs(upper_high_residual) - self.upper_rms_budget, 0.0
        )
        lower_violation = np.maximum(
            np.abs(lower_low_residual) - self.lower_rms_budget, 0.0
        )
        return {
            "upper": fixed_upper.astype(np.float32, copy=True),
            "lower": fixed_lower.astype(np.float32, copy=True),
            "total": total.astype(np.float32, copy=True),
            "transfer": transfer.astype(np.float32, copy=True),
            "canonical_upper": canonical_upper.astype(np.float32, copy=True),
            "canonical_lower": canonical_lower.astype(np.float32, copy=True),
            "streaming_upper_high": upper_high_residual.astype(
                np.float32, copy=True
            ),
            "streaming_lower_low": lower_low_residual.astype(
                np.float32, copy=True
            ),
            "normalized_upper_hf": normalized_upper,
            "normalized_lower_lf": normalized_lower,
            "normalized_local_objective": canonical_objective,
            "normalized_band_imbalance": normalized_lower - normalized_upper,
            "predicted_baseline_objective": raw_objective,
            "predicted_optimal_objective": optimal_objective,
            "upper_budget_feasible_rate": float(np.mean(upper_feasible)),
            "lower_budget_satisfied_rate": float(np.mean(
                np.abs(lower_low_residual) <= self.lower_rms_budget + 1e-12
            )),
            "upper_budget_violation_rms": float(np.sqrt(np.mean(
                np.square(upper_violation)
            ))),
            "lower_budget_violation_rms": float(np.sqrt(np.mean(
                np.square(lower_violation)
            ))),
            "feasible_upper_low": physical_low.astype(np.float32, copy=True),
            "feasible_upper_high": physical_high.astype(np.float32, copy=True),
            "alpha_before": 0.0,
            "alpha_after": 0.0,
            "reconstruction_error": reconstruction_error.astype(
                np.float64, copy=True
            ),
            "canonical_component_clip_rate": 0.0,
            "canonical_component_clip_rms": 0.0,
            "canonical_lower_clip_rate": 0.0,
            "gauge_fixed": float(self.strength == 1.0),
        }

    def _constant_tail_residuals(
        self,
        upper_tail: np.ndarray,
        total_tail: np.ndarray,
        upper_history: list[np.ndarray],
        lower_history: list[np.ndarray],
    ) -> np.ndarray:
        future_upper = np.repeat(
            np.asarray(upper_tail, dtype=np.float64).reshape(1, -1),
            self.planning_horizon,
            axis=0,
        )
        future_total = np.repeat(
            np.asarray(total_tail, dtype=np.float64).reshape(1, -1),
            self.planning_horizon,
            axis=0,
        )
        past_upper = (
            np.stack(upper_history, axis=0)
            if upper_history else np.empty((0, self._dimension))
        )
        past_lower = (
            np.stack(lower_history, axis=0)
            if lower_history else np.empty((0, self._dimension))
        )
        upper_values = np.concatenate((past_upper, future_upper), axis=0)
        lower_values = np.concatenate(
            (past_lower, future_total - future_upper), axis=0
        )
        residuals: list[np.ndarray] = []
        for index in range(self.planning_horizon):
            upper_end = len(upper_history) + index + 1
            upper_start = max(0, upper_end - self.upper_window)
            upper_mean = np.mean(
                upper_values[upper_start:upper_end], axis=0
            )
            residuals.append(
                (future_upper[index] - upper_mean) / self.upper_rms_budget
            )
            lower_end = len(lower_history) + index + 1
            lower_start = max(0, lower_end - self.lower_window)
            lower_mean = np.mean(
                lower_values[lower_start:lower_end], axis=0
            )
            residuals.append(lower_mean / self.lower_rms_budget)
        return np.stack(residuals, axis=0)

    def _zero_history(self, row_count: int) -> list[np.ndarray]:
        return [
            np.zeros(self._dimension, dtype=np.float64)
            for _ in range(int(row_count))
        ]

    def _padded_history(
        self, history: list[np.ndarray], maximum_rows: int
    ) -> np.ndarray:
        target = int(maximum_rows)
        rows = history[-target:]
        padding = np.zeros(
            (target - len(rows), self._dimension), dtype=np.float64
        )
        if not rows:
            return padding
        return np.concatenate((padding, np.stack(rows, axis=0)), axis=0)

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("streaming audit fixer must be reset")


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
