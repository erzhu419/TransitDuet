"""Low-frequency timetable/headway curve planner for FreqDuet.

The planner interprets an upper action as smooth headway adjustments over a
future horizon. In the MVP path it writes target_headway values only, so it
remains compatible with the HIRO-style lower goal-conditioning path. In the
terminal-dispatch path it also writes executable scheduled_launch times.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable

import numpy as np


@dataclass
class TimetableCurvePlanner:
    """Map upper action coefficients to a causal rolling headway plan."""

    horizon_s: float = 2700.0
    basis_per_direction: int = 4
    min_headway_s: float = 180.0
    max_headway_s: float = 720.0
    delta_min_s: float = -120.0
    delta_max_s: float = 120.0
    shared_directions: bool = False
    terminal_shift_min_s: float = -180.0
    terminal_shift_max_s: float = 120.0

    @classmethod
    def from_config(cls, cfg, delta_max_s=120.0):
        cfg = cfg or {}
        return cls(
            horizon_s=float(cfg.get("horizon_s", 2700.0)),
            basis_per_direction=int(cfg.get("basis_per_direction", 4)),
            min_headway_s=float(cfg.get("min_headway_s", 180.0)),
            max_headway_s=float(cfg.get("max_headway_s", 720.0)),
            delta_min_s=float(cfg.get("delta_min_s", -float(cfg.get("delta_max_s", delta_max_s)))),
            delta_max_s=float(cfg.get("delta_max_s", delta_max_s)),
            shared_directions=bool(cfg.get("shared_directions", False)),
            terminal_shift_min_s=float(cfg.get("terminal_shift_min_s", -180.0)),
            terminal_shift_max_s=float(cfg.get("terminal_shift_max_s", 120.0)),
        )

    @property
    def action_dim(self) -> int:
        if self.shared_directions:
            return self.basis_per_direction
        return 2 * self.basis_per_direction

    @property
    def action_low(self):
        return [self.delta_min_s] * self.action_dim

    @property
    def action_high(self):
        return [self.delta_max_s] * self.action_dim

    def _basis(self, offset_s: float) -> np.ndarray:
        """Cubic Bernstein basis when basis_per_direction=4."""
        n = max(0, self.basis_per_direction - 1)
        if n == 0:
            return np.ones(1, dtype=np.float64)
        x = float(np.clip(offset_s / max(self.horizon_s, 1.0), 0.0, 1.0))
        vals = [comb(n, i) * (x ** i) * ((1.0 - x) ** (n - i))
                for i in range(n + 1)]
        return np.asarray(vals, dtype=np.float64)

    def _coefficients(self, action: Iterable[float], direction: bool) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        b = self.basis_per_direction
        if a.size == b:
            return a
        if a.size != 2 * b:
            raise ValueError(
                f"Expected {b} shared or {2 * b} directional coefficients, got {a.size}")
        # Env convention: True is up, False is down.
        return a[:b] if bool(direction) else a[b:]

    def delta_at(self, action: Iterable[float], direction: bool,
                 offset_s: float) -> float:
        coeffs = self._coefficients(action, direction)
        return float(np.dot(coeffs, self._basis(offset_s)))

    def target_headway(self, base_headway_s: float, action: Iterable[float],
                       direction: bool, offset_s: float) -> float:
        target = float(base_headway_s) + self.delta_at(action, direction, offset_s)
        return float(np.clip(target, self.min_headway_s, self.max_headway_s))

    @staticmethod
    def _base_headway(tt, fallback=360.0) -> float:
        if not hasattr(tt, "_freqduet_base_target_headway"):
            tt._freqduet_base_target_headway = float(
                getattr(tt, "target_headway", fallback))
        return float(tt._freqduet_base_target_headway)

    def apply(self, timetables, current_trip, action, origin_launch_s=None,
              write_scheduled_launch=False):
        """Write target headways for current and future same-direction trips.

        Returns:
            dict with current target, effective current delta, and plan summary.
        """
        current_launch = float(current_trip.launch_time)
        origin_launch = current_launch if origin_launch_s is None else float(origin_launch_s)
        current_direction = bool(current_trip.direction)
        planned_targets = []
        scheduled_launches = []
        current_seen = False

        same_direction = [
            tt for tt in timetables
            if bool(tt.direction) == current_direction
            and not getattr(tt, "launched", False)
        ]
        same_direction.sort(key=lambda tt: float(tt.launch_time))

        prev_scheduled = None
        for tt in same_direction:
            offset = float(tt.launch_time) - origin_launch
            if offset < -1e-6 or offset > self.horizon_s:
                continue
            base = self._base_headway(tt)
            target = self.target_headway(base, action, current_direction, offset)
            tt.target_headway = target
            tt._freqduet_planned_by = int(current_trip.launch_turn)
            tt._freqduet_plan_offset_s = offset
            planned_targets.append(target)
            if write_scheduled_launch:
                existing = getattr(tt, "_freqduet_scheduled_launch", None)
                if prev_scheduled is None:
                    scheduled = (
                        float(existing) if existing is not None
                        else float(tt.launch_time)
                    )
                else:
                    scheduled = prev_scheduled + target
                scheduled = float(np.clip(
                    scheduled,
                    float(tt.launch_time) + self.terminal_shift_min_s,
                    float(tt.launch_time) + self.terminal_shift_max_s,
                ))
                tt._freqduet_scheduled_launch = int(round(scheduled))
                tt._freqduet_terminal_dispatch = True
                scheduled_launches.append(float(tt._freqduet_scheduled_launch))
                prev_scheduled = float(tt._freqduet_scheduled_launch)
            if tt is current_trip:
                current_seen = True

        base_current = self._base_headway(current_trip)
        current_offset = current_launch - origin_launch
        current_target = self.target_headway(
            base_current, action, current_direction, current_offset)
        current_trip.target_headway = current_target
        if not current_seen:
            planned_targets.append(current_target)
            if write_scheduled_launch:
                scheduled = float(getattr(
                    current_trip, "_freqduet_scheduled_launch", current_launch))
                scheduled = float(np.clip(
                    scheduled,
                    current_launch + self.terminal_shift_min_s,
                    current_launch + self.terminal_shift_max_s,
                ))
                current_trip._freqduet_scheduled_launch = int(round(scheduled))
                current_trip._freqduet_terminal_dispatch = True
                scheduled_launches.append(float(current_trip._freqduet_scheduled_launch))

        targets = np.asarray(planned_targets, dtype=np.float64)
        scheduled = np.asarray(scheduled_launches, dtype=np.float64)
        return {
            "target_headway": current_target,
            "effective_delta": current_target - base_current,
            "base_headway": base_current,
            "planned_n": int(targets.size),
            "planned_mean": float(targets.mean()) if targets.size else current_target,
            "planned_std": float(targets.std()) if targets.size else 0.0,
            "scheduled_n": int(scheduled.size),
            "scheduled_mean": (
                float(scheduled.mean()) if scheduled.size else 0.0),
            "scheduled_std": (
                float(scheduled.std()) if scheduled.size else 0.0),
        }

    def smoothness_penalty(self, action) -> float:
        """Dimensionless coefficient curvature penalty for upper reward shaping."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        b = self.basis_per_direction
        blocks = [a] if a.size == b else [a[:b], a[b:]]
        denom = max(
            max(abs(self.delta_min_s), abs(self.delta_max_s)) ** 2, 1.0)
        vals = []
        for coeffs in blocks:
            if coeffs.size < 3:
                continue
            curvature = np.diff(coeffs, n=2)
            vals.append(float(np.mean(curvature * curvature) / denom))
        return float(np.mean(vals)) if vals else 0.0
