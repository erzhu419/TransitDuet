"""Causal high-frequency to low-frequency promotion gate."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from .types import PromotionSignal


def _scalar_norm(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


class CausalPromotionGate:
    """Detect persistent residual shocks without future leakage.

    The gate observes a trailing window of high-frequency residual events.  A
    short isolated spike does not trigger promotion; sustained activity does,
    after which cooldown keeps the promoted regime visible long enough for the
    high-level planner to replan.
    """

    def __init__(
        self,
        update_interval_s: float,
        window_s: float = 900.0,
        residual_threshold: float = 2.0,
        persistence_ratio: float = 0.40,
        cooldown_s: float = 1800.0,
        exit_ratio: float | None = None,
        energy_threshold: float | None = None,
        regime_threshold: float | None = None,
        min_age_s: float = 0.0,
        activation_strength_threshold: float = 0.0,
        startup_strength_age_s: float = 0.0,
        startup_strength_threshold: float = 0.0,
        residual_mode: str = "abs",
    ) -> None:
        update_interval_s = max(float(update_interval_s), 1e-9)
        self.update_interval_s = update_interval_s
        self.window_bins = max(1, int(round(float(window_s) / update_interval_s)))
        self.cooldown_bins = max(0, int(round(float(cooldown_s) / update_interval_s)))
        self.residual_threshold = max(float(residual_threshold), 1e-9)
        self.persistence_ratio = float(np.clip(persistence_ratio, 0.0, 1.0))
        self.exit_ratio = (
            min(self.persistence_ratio, max(float(exit_ratio), 0.0))
            if exit_ratio is not None
            else 0.5 * self.persistence_ratio
        )
        self.energy_threshold = None if energy_threshold is None else max(float(energy_threshold), 1e-9)
        self.regime_threshold = None if regime_threshold is None else max(float(regime_threshold), 1e-12)
        self.min_age_bins = max(0, int(round(float(min_age_s) / update_interval_s)))
        self.activation_strength_threshold = float(np.clip(activation_strength_threshold, 0.0, 1.0))
        self.startup_strength_age_bins = max(0, int(round(float(startup_strength_age_s) / update_interval_s)))
        self.startup_strength_threshold = float(np.clip(startup_strength_threshold, 0.0, 1.0))
        self.residual_mode = str(residual_mode or "abs").lower()
        if self.residual_mode not in {"abs", "positive", "negative", "directional_abs"}:
            self.residual_mode = "abs"
        self.reset()

    def reset(self) -> None:
        self.events: deque[float] = deque(maxlen=self.window_bins)
        self.pos_events: deque[float] = deque(maxlen=self.window_bins)
        self.neg_events: deque[float] = deque(maxlen=self.window_bins)
        self.cooldown = 0
        self.shock_age_bins = 0
        self.flag = False
        self.strength = 0.0
        self.score = 0.0
        self.direction = 0.0
        self.last_reason = "inactive"
        self.updates = 0

    def _directional_residual(self, residual: float) -> float:
        if self.residual_mode == "positive":
            return max(residual, 0.0)
        if self.residual_mode == "negative":
            return max(-residual, 0.0)
        return abs(residual)

    def update(self, freq_features: Mapping[str, Any] | float, t: float | None = None) -> dict[str, Any]:
        if isinstance(freq_features, Mapping):
            residual_raw = freq_features.get("x_high", freq_features.get("residual", 0.0))
            low_raw = freq_features.get("x_low", freq_features.get("low", 0.0))
            mid_raw = freq_features.get("x_mid", freq_features.get("mid", 0.0))
            energy_raw = freq_features.get("x_high_energy", 0.0)
        else:
            residual_raw = freq_features
            low_raw = 0.0
            mid_raw = 0.0
            energy_raw = 0.0

        residual_arr = np.asarray(residual_raw, dtype=np.float64).reshape(-1)
        residual_signed = float(np.mean(residual_arr)) if residual_arr.size else 0.0
        residual = self._directional_residual(residual_signed)
        if self.residual_mode == "directional_abs":
            residual = _scalar_norm(residual_arr)
        low_scale = max(np.sqrt(_scalar_norm(low_raw) + 1.0), 1e-9)
        residual_score = residual / low_scale
        energy_active = False
        if self.energy_threshold is not None:
            energy_active = _scalar_norm(energy_raw) >= self.energy_threshold
        self.score = max(residual_score, _scalar_norm(energy_raw) if energy_active else residual_score)
        active = residual_score >= self.residual_threshold or energy_active

        self.events.append(1.0 if active else 0.0)
        self.pos_events.append(1.0 if active and residual_signed > 0.0 else 0.0)
        self.neg_events.append(1.0 if active and residual_signed < 0.0 else 0.0)
        self.updates += 1
        self.shock_age_bins = self.shock_age_bins + 1 if active else 0

        ratio = float(sum(self.events)) / max(len(self.events), 1)
        pos_ratio = float(sum(self.pos_events)) / max(len(self.pos_events), 1)
        neg_ratio = float(sum(self.neg_events)) / max(len(self.neg_events), 1)
        if self.residual_mode == "directional_abs":
            ratio = max(pos_ratio, neg_ratio)
            direction = 1.0 if pos_ratio >= neg_ratio else -1.0
        elif self.residual_mode == "positive":
            direction = 1.0
        elif self.residual_mode == "negative":
            direction = -1.0
        else:
            direction = float(np.sign(residual_signed)) if active else self.direction

        enough_history = len(self.events) >= self.window_bins and self.updates >= self.min_age_bins
        regime_confirmed = (
            True
            if self.regime_threshold is None
            else _scalar_norm(mid_raw) >= self.regime_threshold
        )
        candidate_strength = 0.0
        if ratio >= self.persistence_ratio:
            denom = max(1.0 - self.persistence_ratio, 1e-9)
            candidate_strength = float(np.clip((ratio - self.persistence_ratio) / denom, 0.0, 1.0))
            if active:
                candidate_strength = max(
                    candidate_strength,
                    float(np.clip(residual_score / self.residual_threshold - 1.0, 0.0, 1.0)),
                )
        required_strength = self.activation_strength_threshold
        if self.updates < self.startup_strength_age_bins:
            required_strength = max(required_strength, self.startup_strength_threshold)
        strength_confirmed = candidate_strength >= required_strength
        if enough_history and ratio >= self.persistence_ratio and regime_confirmed and strength_confirmed:
            self.flag = True
            self.cooldown = self.cooldown_bins
            self.direction = direction
            self.last_reason = "persistent_high_residual"
        elif enough_history and ratio >= self.persistence_ratio and regime_confirmed and not strength_confirmed:
            self.flag = False
            self.last_reason = "candidate_below_activation_strength"
        elif enough_history and ratio >= self.persistence_ratio and not regime_confirmed:
            self.flag = False
            self.last_reason = "candidate_without_regime_buffer"
        elif len(self.events) >= self.window_bins and ratio >= self.persistence_ratio:
            self.flag = False
            self.last_reason = "candidate_before_min_age"
        elif self.flag and ratio >= self.exit_ratio:
            self.flag = True
            self.last_reason = "hysteresis"
        elif self.cooldown > 0:
            self.flag = True
            self.cooldown -= 1
            self.last_reason = "cooldown"
        else:
            self.flag = False
            self.direction = 0.0
            self.last_reason = "inactive"

        if self.flag:
            self.strength = candidate_strength
        else:
            self.strength = 0.0

        return self.signal().to_mapping()

    def signal(self) -> PromotionSignal:
        return PromotionSignal(
            promote=bool(self.flag),
            promotion_strength=float(self.strength),
            reason=self.last_reason,
            cooldown_remaining=float(self.cooldown * self.update_interval_s),
            shock_age=min(float(self.shock_age_bins) / max(self.window_bins, 1), 1.0),
            score=float(self.score),
            direction=float(self.direction),
        )
