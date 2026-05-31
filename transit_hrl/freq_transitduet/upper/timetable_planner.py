"""Low-frequency timetable/headway curve planner for FreqDuet.

The planner interprets an upper action as smooth headway adjustments over a
future horizon. In the MVP path it writes target_headway values only, so it
remains compatible with the HIRO-style lower goal-conditioning path. In the
terminal-dispatch path it also writes executable scheduled_launch times.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

TRANSIT_HRL_ROOT = Path(__file__).resolve().parents[2]
if str(TRANSIT_HRL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSIT_HRL_ROOT))

from freq_hrl.policies import BernsteinPlanCurve


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
    plan_all_directions: bool = False
    terminal_shift_min_s: float = -180.0
    terminal_shift_max_s: float = 120.0

    def _shared_curve(self) -> BernsteinPlanCurve:
        return BernsteinPlanCurve(
            horizon_s=self.horizon_s,
            basis_dim=self.basis_per_direction,
            min_value=self.min_headway_s,
            max_value=self.max_headway_s,
            delta_min=self.delta_min_s,
            delta_max=self.delta_max_s,
            n_entities=1 if self.shared_directions else 2,
            shared_entities=self.shared_directions,
        )

    def _curve_for_action(self, action: Iterable[float]) -> BernsteinPlanCurve:
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_arr.size == self.basis_per_direction:
            return BernsteinPlanCurve(
                horizon_s=self.horizon_s,
                basis_dim=self.basis_per_direction,
                min_value=self.min_headway_s,
                max_value=self.max_headway_s,
                delta_min=self.delta_min_s,
                delta_max=self.delta_max_s,
                n_entities=1,
                shared_entities=True,
            )
        return self._shared_curve()

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
            plan_all_directions=bool(cfg.get("plan_all_directions", False)),
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
        return self._shared_curve().basis(offset_s)

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
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.shared_directions or action_arr.size == self.basis_per_direction:
            entity_index = 0
        else:
            # Shared core convention maps entity 0/1 to action blocks.
            # Transit env convention is True=up, False=down.
            entity_index = 0 if bool(direction) else 1
        return self._curve_for_action(action_arr).value_at(
            base_headway_s,
            action_arr,
            offset_s,
            entity_index=entity_index,
        )

    @staticmethod
    def _base_headway(tt, fallback=360.0) -> float:
        if not hasattr(tt, "_freqduet_base_target_headway"):
            tt._freqduet_base_target_headway = float(
                getattr(tt, "target_headway", fallback))
        return float(tt._freqduet_base_target_headway)

    def apply(self, timetables, current_trip, action, origin_launch_s=None,
              write_scheduled_launch=False):
        """Write target headways for current and future trips.

        Returns:
            dict with current target, effective current delta, and plan summary.
        """
        current_launch = float(current_trip.launch_time)
        origin_launch = current_launch if origin_launch_s is None else float(origin_launch_s)
        current_direction = bool(current_trip.direction)
        planned_targets = []
        scheduled_launches = []
        current_seen = False

        plan_directions = (
            [True, False] if self.plan_all_directions else [current_direction]
        )
        for plan_direction in plan_directions:
            direction_trips = [
                tt for tt in timetables
                if bool(tt.direction) == bool(plan_direction)
                and not getattr(tt, "launched", False)
            ]
            direction_trips.sort(key=lambda tt: float(tt.launch_time))

            prev_scheduled = None
            for tt in direction_trips:
                offset = float(tt.launch_time) - origin_launch
                if offset < -1e-6 or offset > self.horizon_s:
                    continue
                base = self._base_headway(tt)
                target = self.target_headway(
                    base, action, bool(plan_direction), offset)
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
                    scheduled_launches.append(
                        float(tt._freqduet_scheduled_launch))
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
        return self._curve_for_action(action).smoothness_penalty(action)
