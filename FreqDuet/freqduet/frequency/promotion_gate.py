"""Causal high-frequency-to-low-frequency promotion gate.

The gate is intentionally small and deterministic: it observes only past
high-frequency residuals, tracks whether shocks persist over a trailing window,
and exposes a flag/strength/age tuple for the upper and lower policies.
"""

from collections import deque

import numpy as np


class CausalPromotionGate:
    """Detect persistent residual shocks without future leakage."""

    def __init__(
        self,
        update_interval_s,
        window_s=900.0,
        residual_threshold=1.0,
        persistence_ratio=0.35,
        cooldown_s=600.0,
    ):
        update_interval_s = max(float(update_interval_s), 1e-6)
        self.window_bins = max(1, int(round(float(window_s) / update_interval_s)))
        self.cooldown_bins = max(0, int(round(float(cooldown_s) / update_interval_s)))
        self.residual_threshold = max(float(residual_threshold), 1e-6)
        self.persistence_ratio = float(np.clip(persistence_ratio, 0.0, 1.0))
        self.reset()

    def reset(self):
        self.events = deque(maxlen=self.window_bins)
        self.shock_age = 0
        self.cooldown = 0
        self.flag = 0.0
        self.strength = 0.0
        self.score = 0.0

    def update(self, residual, low_level):
        """Update from residual and low component in passengers/minute units."""
        residual = float(residual)
        low_level = max(float(low_level), 0.0)
        self.score = abs(residual) / max(np.sqrt(low_level + 1.0), 1e-6)
        active = 1.0 if self.score >= self.residual_threshold else 0.0
        self.events.append(active)
        self.shock_age = self.shock_age + 1 if active else 0

        ratio = float(sum(self.events)) / max(len(self.events), 1)
        persistent = ratio >= self.persistence_ratio and len(self.events) >= self.window_bins
        if persistent:
            self.flag = 1.0
            self.cooldown = self.cooldown_bins
        elif self.cooldown > 0:
            self.flag = 1.0
            self.cooldown -= 1
        else:
            self.flag = 0.0

        if self.flag:
            denom = max(1.0 - self.persistence_ratio, 1e-6)
            self.strength = float(np.clip((ratio - self.persistence_ratio) / denom, 0.0, 1.0))
            if active:
                self.strength = max(self.strength, min(self.score / self.residual_threshold - 1.0, 1.0))
        else:
            self.strength = 0.0

    def features(self):
        age_norm = min(float(self.shock_age) / max(self.window_bins, 1), 1.0)
        return np.asarray([self.flag, self.strength, age_norm], dtype=np.float32)

    def summary(self):
        return {
            "flag": float(self.flag),
            "strength": float(self.strength),
            "age": min(float(self.shock_age) / max(self.window_bins, 1), 1.0),
            "score": float(self.score),
        }
