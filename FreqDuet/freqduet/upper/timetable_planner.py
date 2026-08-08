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


_TERMINAL_SCHEDULE_MODES = {
    "bounded_shift_legacy",
    "exact_headway_curve",
}
_HEADWAY_BUDGET_MODES = {"free", "zero_sum_delta_v5"}
_COEFFICIENT_PARAMETERIZATIONS = {"full", "antisymmetric_linear_v5"}


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
    terminal_schedule_mode: str = "bounded_shift_legacy"
    headway_budget_mode: str = "free"
    coefficient_parameterization: str = "full"

    def __post_init__(self) -> None:
        mode = str(self.terminal_schedule_mode).strip().lower()
        if mode not in _TERMINAL_SCHEDULE_MODES:
            raise ValueError(
                "terminal_schedule_mode must be one of "
                f"{sorted(_TERMINAL_SCHEDULE_MODES)}"
            )
        if self.basis_per_direction < 1:
            raise ValueError("basis_per_direction must be positive")
        if self.min_headway_s <= 0.0:
            raise ValueError("min_headway_s must be positive")
        if self.max_headway_s < self.min_headway_s:
            raise ValueError("max_headway_s must not be below min_headway_s")
        self.terminal_schedule_mode = mode
        budget_mode = str(self.headway_budget_mode).strip().lower()
        if budget_mode not in _HEADWAY_BUDGET_MODES:
            raise ValueError(
                "headway_budget_mode must be one of "
                f"{sorted(_HEADWAY_BUDGET_MODES)}"
            )
        self.headway_budget_mode = budget_mode
        parameterization = str(
            self.coefficient_parameterization).strip().lower()
        if parameterization not in _COEFFICIENT_PARAMETERIZATIONS:
            raise ValueError(
                "coefficient_parameterization must be one of "
                f"{sorted(_COEFFICIENT_PARAMETERIZATIONS)}"
            )
        if (parameterization == "antisymmetric_linear_v5"
                and self.basis_per_direction != 2):
            raise ValueError(
                "antisymmetric_linear_v5 requires basis_per_direction=2")
        self.coefficient_parameterization = parameterization

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
            terminal_schedule_mode=str(
                cfg.get("terminal_schedule_mode", "bounded_shift_legacy")
            ),
            headway_budget_mode=str(cfg.get("headway_budget_mode", "free")),
            coefficient_parameterization=str(
                cfg.get("coefficient_parameterization", "full")),
        )

    @property
    def action_dim(self) -> int:
        if self.coefficient_parameterization == "antisymmetric_linear_v5":
            return 1 if self.shared_directions else 2
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
        if self.coefficient_parameterization == "antisymmetric_linear_v5":
            expected = 1 if self.shared_directions else 2
            if a.size != expected:
                raise ValueError(
                    f"Expected {expected} antisymmetric amplitude(s), got {a.size}")
            amplitude = float(a[0] if self.shared_directions else (
                a[0] if bool(direction) else a[1]))
            return np.asarray([-amplitude, amplitude], dtype=np.float64)
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

    @staticmethod
    def _base_launch(tt) -> float:
        if not hasattr(tt, "_freqduet_base_launch"):
            tt._freqduet_base_launch = float(tt.launch_time)
        return float(tt._freqduet_base_launch)

    @staticmethod
    def _effective_launch(tt) -> float:
        actual = getattr(tt, "_freqduet_actual_launch", None)
        if actual is not None:
            return float(actual)
        scheduled = getattr(tt, "_freqduet_scheduled_launch", None)
        if scheduled is not None:
            return float(scheduled)
        return TimetableCurvePlanner._base_launch(tt)

    @staticmethod
    def _project_box_sum(
        raw: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        target_sum: float = 0.0,
    ) -> np.ndarray:
        """Euclidean projection onto box bounds with one sum constraint."""
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        if raw.shape != lower.shape or raw.shape != upper.shape:
            raise ValueError("headway budget projection arrays must align")
        if np.any(lower > upper):
            raise ValueError("headway budget projection has invalid bounds")
        target_sum = float(target_sum)
        tolerance = 1e-8 * max(1.0, raw.size)
        if target_sum < float(lower.sum()) - tolerance or target_sum > float(
                upper.sum()) + tolerance:
            raise ValueError(
                "headway budget target is infeasible under configured bounds")
        if raw.size == 0:
            return raw.copy()

        lambda_low = float(np.min(raw - upper))
        lambda_high = float(np.max(raw - lower))
        for _ in range(100):
            midpoint = 0.5 * (lambda_low + lambda_high)
            projected = np.clip(raw - midpoint, lower, upper)
            if float(projected.sum()) > target_sum:
                lambda_low = midpoint
            else:
                lambda_high = midpoint
        projected = np.clip(
            raw - 0.5 * (lambda_low + lambda_high), lower, upper)
        if abs(float(projected.sum()) - target_sum) > 1e-6:
            raise AssertionError("headway budget projection did not conserve sum")
        return projected

    def _apply_exact_headway_curve(
        self,
        timetables,
        current_trip,
        action,
        origin_launch_s,
        terminal_headway_floor_ratio,
        terminal_headway_floor_min_s,
        terminal_shift_bias_s,
        plan_owner,
    ):
        """Project a headway curve without independently clipping trip phases."""
        if abs(float(terminal_shift_bias_s or 0.0)) > 1e-9:
            raise ValueError(
                "exact_headway_curve does not permit an independent terminal "
                "shift bias; encode the intervention in the headway curve"
            )

        current_direction = bool(current_trip.direction)
        directions = (
            [True, False] if self.plan_all_directions else [current_direction]
        )
        planned_targets = []
        scheduled_launches = []
        phase_displacements = []
        headway_floors = []
        raw_headway_deltas = []
        projected_headway_deltas = []
        current_target = None

        for direction in directions:
            all_direction_trips = sorted(
                (tt for tt in timetables if bool(tt.direction) == direction),
                key=self._base_launch,
            )
            candidates = [
                tt
                for tt in all_direction_trips
                if not getattr(tt, "launched", False)
                and -1e-6
                <= self._base_launch(tt) - float(origin_launch_s)
                <= self.horizon_s
            ]
            if not candidates:
                continue

            anchor = candidates[0]
            anchor_index = all_direction_trips.index(anchor)
            predecessor = (
                all_direction_trips[anchor_index - 1]
                if anchor_index > 0
                else None
            )
            predecessor_launch = (
                self._effective_launch(predecessor)
                if predecessor is not None
                else None
            )
            anchor_launch = self._effective_launch(anchor)
            anchor_base = self._base_headway(anchor)
            existing_target = getattr(
                anchor, "_freqduet_projected_target_headway", None)
            if existing_target is not None:
                anchor_target = float(existing_target)
            elif predecessor_launch is None:
                anchor_target = anchor_base
            else:
                anchor_target = anchor_launch - predecessor_launch
            anchor_lower = max(
                self.min_headway_s, anchor_base + self.delta_min_s)
            anchor_upper = min(
                self.max_headway_s, anchor_base + self.delta_max_s)
            if anchor_lower > anchor_upper:
                raise ValueError(
                    "configured action and absolute headway bounds do not "
                    "admit the anchor baseline")
            anchor_target = float(np.clip(
                anchor_target, anchor_lower, anchor_upper))

            anchor._freqduet_desired_headway_s = float(anchor_target)
            anchor._freqduet_scheduled_launch = float(anchor_launch)
            anchor._freqduet_predecessor_scheduled_launch = predecessor_launch
            anchor._freqduet_projected_target_headway = float(anchor_target)
            anchor._freqduet_phase_displacement_s = float(
                anchor_launch - self._base_launch(anchor)
            )
            anchor._freqduet_projection_mode = "exact_headway_curve"
            anchor._freqduet_plan_offset_s = float(
                self._base_launch(anchor) - float(origin_launch_s)
            )
            anchor._freqduet_terminal_dispatch = True
            if not hasattr(anchor, "_freqduet_planned_by"):
                # The anchor is intentionally unaffected by the new plan.
                anchor._freqduet_planned_by = int(anchor.launch_turn)
            anchor.target_headway = float(anchor_target)

            planned_targets.append(float(anchor_target))
            scheduled_launches.append(float(anchor_launch))
            phase_displacements.append(
                float(anchor._freqduet_phase_displacement_s)
            )
            if anchor is current_trip:
                current_target = float(anchor_target)

            future = candidates[1:]
            desired_headways = []
            bases = []
            floors_for_trip = []
            for tt in future:
                offset = self._base_launch(tt) - float(origin_launch_s)
                base = self._base_headway(tt)
                desired = self.target_headway(base, action, direction, offset)
                floor = 0.0
                ratio = float(terminal_headway_floor_ratio or 0.0)
                if ratio > 0.0:
                    floor = max(floor, base * ratio)
                floor_min = float(terminal_headway_floor_min_s or 0.0)
                if floor_min > 0.0:
                    floor = max(floor, floor_min)
                if floor > 0.0:
                    desired = max(desired, floor)
                    headway_floors.append(float(floor))
                    tt._freqduet_min_dispatch_headway = float(floor)
                bases.append(float(base))
                desired_headways.append(float(desired))
                floors_for_trip.append(float(floor))

            if future:
                bases_arr = np.asarray(bases, dtype=np.float64)
                raw_deltas = (
                    np.asarray(desired_headways, dtype=np.float64) - bases_arr)
                projected_deltas = raw_deltas.copy()
                if self.headway_budget_mode == "zero_sum_delta_v5":
                    lower = np.asarray([
                        max(
                            self.min_headway_s - base,
                            floor - base,
                            self.delta_min_s,
                        )
                        for base, floor in zip(bases, floors_for_trip)
                    ], dtype=np.float64)
                    upper = np.asarray([
                        min(self.max_headway_s - base, self.delta_max_s)
                        for base in bases
                    ], dtype=np.float64)
                    projected_deltas = self._project_box_sum(
                        raw_deltas, lower, upper, target_sum=0.0)
                desired_headways = list(bases_arr + projected_deltas)
                raw_headway_deltas.extend(raw_deltas.tolist())
                projected_headway_deltas.extend(projected_deltas.tolist())

            previous_launch = float(anchor_launch)
            for tt, desired in zip(future, desired_headways):
                offset = self._base_launch(tt) - float(origin_launch_s)
                projected_launch = previous_launch + float(desired)
                projected_target = projected_launch - previous_launch
                if projected_target <= 0.0:
                    raise AssertionError(
                        "exact timetable projection produced a non-positive headway"
                    )

                tt._freqduet_desired_headway_s = float(desired)
                tt._freqduet_scheduled_launch = float(projected_launch)
                tt._freqduet_predecessor_scheduled_launch = float(previous_launch)
                tt._freqduet_projected_target_headway = float(projected_target)
                tt._freqduet_phase_displacement_s = float(
                    projected_launch - self._base_launch(tt)
                )
                tt._freqduet_projection_mode = "exact_headway_curve"
                tt._freqduet_planned_by = int(plan_owner)
                tt._freqduet_plan_offset_s = float(offset)
                tt._freqduet_terminal_dispatch = True
                tt.target_headway = float(projected_target)

                planned_targets.append(float(projected_target))
                scheduled_launches.append(float(projected_launch))
                phase_displacements.append(
                    float(tt._freqduet_phase_displacement_s)
                )
                if tt is current_trip:
                    current_target = float(projected_target)
                previous_launch = float(projected_launch)

        if current_target is None:
            raise ValueError(
                "current trip is outside the exact headway projection horizon"
            )

        targets = np.asarray(planned_targets, dtype=np.float64)
        scheduled = np.asarray(scheduled_launches, dtype=np.float64)
        phases = np.asarray(phase_displacements, dtype=np.float64)
        floors = np.asarray(headway_floors, dtype=np.float64)
        raw_deltas = np.asarray(raw_headway_deltas, dtype=np.float64)
        projected_deltas = np.asarray(
            projected_headway_deltas, dtype=np.float64)
        base_current = self._base_headway(current_trip)
        return {
            "target_headway": float(current_target),
            "effective_delta": float(current_target - base_current),
            "base_headway": float(base_current),
            "planned_n": int(targets.size),
            "planned_mean": float(targets.mean()),
            "planned_std": float(targets.std()),
            "scheduled_n": int(scheduled.size),
            "scheduled_mean": float(scheduled.mean()),
            "scheduled_std": float(scheduled.std()),
            "terminal_shift_min_s": float("nan"),
            "terminal_shift_max_s": float("nan"),
            "terminal_shift_bias_s": 0.0,
            "terminal_headway_floor_n": int(floors.size),
            "terminal_headway_floor_mean": (
                float(floors.mean()) if floors.size else 0.0
            ),
            "phase_displacement_mean_s": float(phases.mean()),
            "phase_displacement_max_abs_s": float(np.max(np.abs(phases))),
            "projection_mode": "exact_headway_curve",
            "headway_budget_mode": self.headway_budget_mode,
            "raw_headway_delta_mean_s": (
                float(raw_deltas.mean()) if raw_deltas.size else 0.0),
            "projected_headway_delta_mean_s": (
                float(projected_deltas.mean())
                if projected_deltas.size else 0.0),
            "projected_headway_delta_sum_s": (
                float(projected_deltas.sum())
                if projected_deltas.size else 0.0),
            "plan_id": int(plan_owner),
        }

    def apply(self, timetables, current_trip, action, origin_launch_s=None,
              write_scheduled_launch=False, terminal_shift_min_s=None,
              terminal_shift_max_s=None, terminal_shift_bias_s=0.0,
              terminal_headway_floor_ratio=0.0,
              terminal_headway_floor_min_s=0.0, plan_id=None,
              terminal_schedule_mode=None):
        """Write target headways for current and future trips.

        Returns:
            dict with current target, effective current delta, and plan summary.
        """
        current_launch = float(current_trip.launch_time)
        origin_launch = current_launch if origin_launch_s is None else float(origin_launch_s)
        current_direction = bool(current_trip.direction)
        plan_owner = int(
            current_trip.launch_turn if plan_id is None else plan_id)
        schedule_mode = str(
            self.terminal_schedule_mode
            if terminal_schedule_mode is None
            else terminal_schedule_mode
        ).strip().lower()
        if schedule_mode not in _TERMINAL_SCHEDULE_MODES:
            raise ValueError(
                "terminal_schedule_mode must be one of "
                f"{sorted(_TERMINAL_SCHEDULE_MODES)}"
            )
        if schedule_mode == "exact_headway_curve":
            if not write_scheduled_launch:
                raise ValueError(
                    "exact_headway_curve requires executable terminal dispatch"
                )
            return self._apply_exact_headway_curve(
                timetables=timetables,
                current_trip=current_trip,
                action=action,
                origin_launch_s=origin_launch,
                terminal_headway_floor_ratio=terminal_headway_floor_ratio,
                terminal_headway_floor_min_s=terminal_headway_floor_min_s,
                terminal_shift_bias_s=terminal_shift_bias_s,
                plan_owner=plan_owner,
            )
        terminal_shift_min = (
            self.terminal_shift_min_s if terminal_shift_min_s is None
            else float(terminal_shift_min_s))
        terminal_shift_max = (
            self.terminal_shift_max_s if terminal_shift_max_s is None
            else float(terminal_shift_max_s))
        terminal_shift_bias = float(terminal_shift_bias_s or 0.0)
        planned_targets = []
        scheduled_launches = []
        headway_floors = []
        current_seen = False
        terminal_attrs = (
            "_freqduet_scheduled_launch",
            "_freqduet_terminal_dispatch",
            "_freqduet_min_dispatch_headway",
        )

        def _clear_terminal_schedule(tt):
            for attr in terminal_attrs:
                if hasattr(tt, attr):
                    delattr(tt, attr)

        def _apply_terminal_headway_floor(tt, base, target):
            floor = 0.0
            ratio = float(terminal_headway_floor_ratio or 0.0)
            if ratio > 0.0:
                floor = max(floor, float(base) * ratio)
            floor_min = float(terminal_headway_floor_min_s or 0.0)
            if floor_min > 0.0:
                floor = max(floor, floor_min)
            if write_scheduled_launch and floor > 0.0:
                target = max(float(target), floor)
                tt._freqduet_min_dispatch_headway = float(floor)
                headway_floors.append(float(floor))
            return float(target), float(floor)

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
                target, min_dispatch_headway = _apply_terminal_headway_floor(
                    tt, base, target)
                tt.target_headway = target
                tt._freqduet_planned_by = plan_owner
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
                    if min_dispatch_headway > 0.0 and prev_scheduled is not None:
                        scheduled = max(
                            scheduled, prev_scheduled + min_dispatch_headway)
                    if terminal_shift_bias > 0.0:
                        scheduled = max(
                            scheduled,
                            float(tt.launch_time) + terminal_shift_bias)
                    scheduled = float(np.clip(
                        scheduled,
                        float(tt.launch_time) + terminal_shift_min,
                        float(tt.launch_time) + terminal_shift_max,
                    ))
                    tt._freqduet_scheduled_launch = int(round(scheduled))
                    tt._freqduet_terminal_dispatch = True
                    scheduled_launches.append(
                        float(tt._freqduet_scheduled_launch))
                    prev_scheduled = float(tt._freqduet_scheduled_launch)
                else:
                    _clear_terminal_schedule(tt)
                if tt is current_trip:
                    current_seen = True

        base_current = self._base_headway(current_trip)
        current_offset = current_launch - origin_launch
        current_target = self.target_headway(
            base_current, action, current_direction, current_offset)
        current_target, current_floor = _apply_terminal_headway_floor(
            current_trip, base_current, current_target)
        current_trip.target_headway = current_target
        if not current_seen:
            planned_targets.append(current_target)
            if write_scheduled_launch:
                scheduled = float(getattr(
                    current_trip, "_freqduet_scheduled_launch", current_launch))
                if terminal_shift_bias > 0.0:
                    scheduled = max(
                        scheduled,
                        current_launch + terminal_shift_bias)
                if current_floor > 0.0:
                    existing = getattr(
                        current_trip, "_freqduet_scheduled_launch", None)
                    if existing is not None:
                        scheduled = max(scheduled, float(existing))
                scheduled = float(np.clip(
                    scheduled,
                    current_launch + terminal_shift_min,
                    current_launch + terminal_shift_max,
                ))
                current_trip._freqduet_scheduled_launch = int(round(scheduled))
                current_trip._freqduet_terminal_dispatch = True
                scheduled_launches.append(float(current_trip._freqduet_scheduled_launch))
            else:
                _clear_terminal_schedule(current_trip)

        targets = np.asarray(planned_targets, dtype=np.float64)
        scheduled = np.asarray(scheduled_launches, dtype=np.float64)
        floors = np.asarray(headway_floors, dtype=np.float64)
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
            "terminal_shift_min_s": float(terminal_shift_min),
            "terminal_shift_max_s": float(terminal_shift_max),
            "terminal_shift_bias_s": float(terminal_shift_bias),
            "terminal_headway_floor_n": int(floors.size),
            "terminal_headway_floor_mean": (
                float(floors.mean()) if floors.size else 0.0),
            "plan_id": plan_owner,
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
            if coeffs.size >= 3:
                curvature = np.diff(coeffs, n=2)
                vals.append(float(np.mean(curvature * curvature) / denom))
            elif coeffs.size == 2:
                slope = np.diff(coeffs)
                vals.append(float(0.25 * np.mean(slope * slope) / denom))
        return float(np.mean(vals)) if vals else 0.0
