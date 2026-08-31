"""Causal projection of hierarchical actions into frequency budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


BUDGET_MODES = ("instantaneous", "prefix_ledger")


@dataclass
class CausalJointFrequencyProjector:
    """Preserve total action when possible, otherwise repair it causally.

    At step ``t`` the exact causal HPF/LPF residuals are affine functions of
    the current upper and lower components. The projector first searches for a
    feasible responsibility split whose sum is the proposed total action. If
    that intersection is empty, it projects each proposed component onto its
    causal frequency-feasible set. The second branch is the only branch that
    may change the total action.
    """

    upper_window: int = 8
    lower_window: int = 32
    upper_rms_budget: float = 0.075
    lower_rms_budget: float = 0.0475
    upper_action_limit: float = 1.0
    lower_action_limit: float = 1.0
    budget_mode: str = "prefix_ledger"
    projection_tolerance: float = 1e-10
    feasibility_tolerance: float = 1e-8
    maximum_projection_iterations: int = 256

    def __post_init__(self) -> None:
        self.upper_window = int(self.upper_window)
        self.lower_window = int(self.lower_window)
        self.upper_rms_budget = float(self.upper_rms_budget)
        self.lower_rms_budget = float(self.lower_rms_budget)
        self.upper_action_limit = float(self.upper_action_limit)
        self.lower_action_limit = float(self.lower_action_limit)
        self.budget_mode = str(self.budget_mode)
        self.projection_tolerance = float(self.projection_tolerance)
        self.feasibility_tolerance = float(self.feasibility_tolerance)
        self.maximum_projection_iterations = int(
            self.maximum_projection_iterations
        )
        if self.upper_window < 2 or self.lower_window < 2:
            raise ValueError("frequency windows must be at least two")
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(f"unknown budget mode: {self.budget_mode}")
        positive = (
            self.upper_rms_budget,
            self.lower_rms_budget,
            self.upper_action_limit,
            self.lower_action_limit,
            self.projection_tolerance,
            self.feasibility_tolerance,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("projector budgets, limits, and tolerances must be positive")
        if self.maximum_projection_iterations < 1:
            raise ValueError("maximum_projection_iterations must be positive")
        self._dimension = 0
        self._upper_history: list[np.ndarray] = []
        self._lower_history: list[np.ndarray] = []
        self._upper_energy = 0.0
        self._lower_energy = 0.0
        self._step_count = 0

    def reset(self, action_dimension: int) -> None:
        dimension = int(action_dimension)
        if dimension < 1:
            raise ValueError("action_dimension must be positive")
        self._dimension = dimension
        self._upper_history = []
        self._lower_history = []
        self._upper_energy = 0.0
        self._lower_energy = 0.0
        self._step_count = 0

    def project(
        self,
        proposed_upper: Any,
        proposed_lower: Any,
    ) -> dict[str, Any]:
        """Return one causal responsibility/action projection."""

        self._require_reset()
        upper_proposal = self._action(proposed_upper, "proposed upper")
        lower_proposal = self._action(proposed_lower, "proposed lower")
        total_proposal = upper_proposal + lower_proposal
        if np.max(np.abs(total_proposal)) > (
            self.upper_action_limit + self.lower_action_limit + 1e-10
        ):
            raise ValueError("proposed total exceeds the component sum box")

        upper_constraint = self._current_constraint(
            history=self._upper_history,
            window=self.upper_window,
            rms_budget=self.upper_rms_budget,
            accumulated_energy=self._upper_energy,
            high_pass=True,
        )
        lower_constraint = self._current_constraint(
            history=self._lower_history,
            window=self.lower_window,
            rms_budget=self.lower_rms_budget,
            accumulated_energy=self._lower_energy,
            high_pass=False,
        )
        physical_low = np.maximum(
            -self.upper_action_limit,
            total_proposal - self.lower_action_limit,
        )
        physical_high = np.minimum(
            self.upper_action_limit,
            total_proposal + self.lower_action_limit,
        )
        fixed_upper, fixed_projection = self._project_fixed_total(
            upper_proposal,
            total=total_proposal,
            physical_low=physical_low,
            physical_high=physical_high,
            upper_constraint=upper_constraint,
            lower_constraint=lower_constraint,
        )
        fixed_lower = total_proposal - fixed_upper
        fixed_feasible = self._components_feasible(
            fixed_upper,
            fixed_lower,
            upper_constraint=upper_constraint,
            lower_constraint=lower_constraint,
        )

        if fixed_feasible:
            upper = fixed_upper
            lower = fixed_lower
            total_changed = False
            projection_iterations = fixed_projection["iterations"]
            projection_converged = fixed_projection["converged"]
        else:
            upper, upper_projection = self._project_component(
                upper_proposal,
                low=np.full(self._dimension, -self.upper_action_limit),
                high=np.full(self._dimension, self.upper_action_limit),
                constraint=upper_constraint,
            )
            lower, lower_projection = self._project_component(
                lower_proposal,
                low=np.full(self._dimension, -self.lower_action_limit),
                high=np.full(self._dimension, self.lower_action_limit),
                constraint=lower_constraint,
            )
            total_changed = True
            projection_iterations = max(
                upper_projection["iterations"],
                lower_projection["iterations"],
            )
            projection_converged = bool(
                upper_projection["converged"]
                and lower_projection["converged"]
            )

        upper_residual = self._residual(upper, upper_constraint)
        lower_residual = self._residual(lower, lower_constraint)
        step_upper_energy = float(np.sum(np.square(upper_residual)))
        step_lower_energy = float(np.sum(np.square(lower_residual)))
        component_feasible = self._components_feasible(
            upper,
            lower,
            upper_constraint=upper_constraint,
            lower_constraint=lower_constraint,
        )
        total = upper + lower
        correction = total - total_proposal
        reconstruction_error = upper + lower - total

        self._upper_energy += step_upper_energy
        self._lower_energy += step_lower_energy
        self._upper_history.append(upper.copy())
        self._lower_history.append(lower.copy())
        self._upper_history = self._upper_history[-(self.upper_window - 1):]
        self._lower_history = self._lower_history[-(self.lower_window - 1):]
        self._step_count += 1

        prefix_denominator = float(self._step_count * self._dimension)
        return {
            "upper": upper.copy(),
            "lower": lower.copy(),
            "total": total.copy(),
            "total_correction": correction.copy(),
            "fixed_total_feasible": bool(fixed_feasible),
            "total_action_changed": bool(total_changed),
            "component_feasible": bool(component_feasible),
            "projection_converged": bool(projection_converged),
            "projection_iterations": int(projection_iterations),
            "upper_residual": upper_residual.copy(),
            "lower_residual": lower_residual.copy(),
            "upper_prefix_power": float(self._upper_energy / prefix_denominator),
            "lower_prefix_power": float(self._lower_energy / prefix_denominator),
            "upper_allowed_step_energy": float(
                upper_constraint["allowed_energy"]
            ),
            "lower_allowed_step_energy": float(
                lower_constraint["allowed_energy"]
            ),
            "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
            "correction_abs_max": float(np.max(np.abs(correction))),
            "component_correction_rms": float(np.sqrt(np.mean(
                np.square(upper - upper_proposal)
                + np.square(lower - lower_proposal)
            ))),
            "reconstruction_error_max": float(
                np.max(np.abs(reconstruction_error))
            ),
        }

    def _current_constraint(
        self,
        *,
        history: list[np.ndarray],
        window: int,
        rms_budget: float,
        accumulated_energy: float,
        high_pass: bool,
    ) -> dict[str, Any]:
        count = min(self._step_count, int(window) - 1)
        rows = history[-count:] if count else []
        past_sum = (
            np.sum(rows, axis=0)
            if rows else np.zeros(self._dimension, dtype=np.float64)
        )
        denominator = float(count + 1)
        if high_pass:
            coefficient = 1.0 - 1.0 / denominator
            offset = -past_sum / denominator
        else:
            coefficient = 1.0 / denominator
            offset = past_sum / denominator
        if self.budget_mode == "instantaneous":
            allowed = self._dimension * float(rms_budget) ** 2
        else:
            allowed = (
                (self._step_count + 1)
                * self._dimension
                * float(rms_budget) ** 2
                - float(accumulated_energy)
            )
        allowed = max(float(allowed), 0.0)
        if abs(coefficient) <= 1e-15:
            center = np.zeros(self._dimension, dtype=np.float64)
            radius = np.inf
        else:
            center = -offset / coefficient
            radius = np.sqrt(allowed) / abs(coefficient)
        return {
            "coefficient": float(coefficient),
            "offset": offset,
            "center": center,
            "radius": float(radius),
            "allowed_energy": float(allowed),
        }

    def _project_fixed_total(
        self,
        proposed_upper: np.ndarray,
        *,
        total: np.ndarray,
        physical_low: np.ndarray,
        physical_high: np.ndarray,
        upper_constraint: dict[str, Any],
        lower_constraint: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        balls: list[tuple[np.ndarray, float]] = []
        if np.isfinite(float(upper_constraint["radius"])):
            balls.append((
                np.asarray(upper_constraint["center"]),
                float(upper_constraint["radius"]),
            ))
        lower_coefficient = float(lower_constraint["coefficient"])
        if abs(lower_coefficient) > 1e-15:
            lower_center_in_upper = (
                total
                + np.asarray(lower_constraint["offset"])
                / lower_coefficient
            )
            lower_radius_in_upper = (
                float(lower_constraint["radius"])
            )
            balls.append((
                lower_center_in_upper,
                lower_radius_in_upper,
            ))
        if not self._ball_box_intersection_possible(
            balls, low=physical_low, high=physical_high
        ):
            return np.clip(proposed_upper, physical_low, physical_high), {
                "converged": False,
                "iterations": 0,
            }
        projectors = [
            self._ball_projector(center, radius)
            for center, radius in balls
        ]
        projectors.append(
            lambda values: np.clip(values, physical_low, physical_high)
        )
        return self._dykstra(proposed_upper, projectors)

    def _project_component(
        self,
        proposed: np.ndarray,
        *,
        low: np.ndarray,
        high: np.ndarray,
        constraint: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        radius = float(constraint["radius"])
        if not np.isfinite(radius):
            return np.clip(proposed, low, high), {
                "converged": True,
                "iterations": 1,
            }
        return self._project_ball_box(
            proposed,
            center=np.asarray(constraint["center"]),
            radius=radius,
            low=low,
            high=high,
        )

    def _project_ball_box(
        self,
        proposed: np.ndarray,
        *,
        center: np.ndarray,
        radius: float,
        low: np.ndarray,
        high: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        point = np.asarray(proposed, dtype=np.float64)
        origin = np.asarray(center, dtype=np.float64)
        bound = float(radius)
        box_projection = np.clip(point, low, high)
        if float(np.linalg.norm(box_projection - origin)) <= (
            bound + self.feasibility_tolerance
        ):
            return box_projection, {"converged": True, "iterations": 1}
        closest = np.clip(origin, low, high)
        if float(np.linalg.norm(closest - origin)) > (
            bound + self.feasibility_tolerance
        ):
            return closest, {"converged": False, "iterations": 1}

        multiplier_low = 0.0
        multiplier_high = 1.0

        def candidate(multiplier: float) -> np.ndarray:
            return np.clip(
                (point + float(multiplier) * origin)
                / (1.0 + float(multiplier)),
                low,
                high,
            )

        projected = candidate(multiplier_high)
        while float(np.linalg.norm(projected - origin)) > bound:
            multiplier_high *= 10.0
            projected = candidate(multiplier_high)
            if multiplier_high > 1e18:
                return closest, {"converged": False, "iterations": 1}
        iterations = 0
        for iterations in range(1, 81):
            midpoint = 0.5 * (multiplier_low + multiplier_high)
            trial = candidate(midpoint)
            if float(np.linalg.norm(trial - origin)) <= bound:
                multiplier_high = midpoint
                projected = trial
            else:
                multiplier_low = midpoint
        return projected, {"converged": True, "iterations": iterations}

    def _ball_box_intersection_possible(
        self,
        balls: list[tuple[np.ndarray, float]],
        *,
        low: np.ndarray,
        high: np.ndarray,
    ) -> bool:
        tolerance = self.feasibility_tolerance
        for center, radius in balls:
            closest = np.clip(center, low, high)
            if float(np.linalg.norm(closest - center)) > radius + tolerance:
                return False
        for left_index, (left_center, left_radius) in enumerate(balls):
            for right_center, right_radius in balls[left_index + 1:]:
                if float(np.linalg.norm(left_center - right_center)) > (
                    left_radius + right_radius + tolerance
                ):
                    return False
        return True

    def _dykstra(
        self,
        start: np.ndarray,
        projectors: list[Callable[[np.ndarray], np.ndarray]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.asarray(start, dtype=np.float64).copy()
        residuals = [np.zeros_like(values) for _ in projectors]
        converged = False
        iteration = 0
        for iteration in range(1, self.maximum_projection_iterations + 1):
            previous = values.copy()
            for index, projector in enumerate(projectors):
                shifted = values + residuals[index]
                projected = np.asarray(projector(shifted), dtype=np.float64)
                residuals[index] = shifted - projected
                values = projected
            if float(np.max(np.abs(values - previous))) <= self.projection_tolerance:
                converged = True
                break
        return values, {"converged": converged, "iterations": iteration}

    @staticmethod
    def _ball_projector(
        center: np.ndarray,
        radius: float,
    ) -> Callable[[np.ndarray], np.ndarray]:
        origin = np.asarray(center, dtype=np.float64)
        bound = float(radius)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("ball radius must be finite and non-negative")

        def project(values: np.ndarray) -> np.ndarray:
            delta = np.asarray(values, dtype=np.float64) - origin
            norm = float(np.linalg.norm(delta))
            if norm <= bound or norm <= 1e-30:
                return np.asarray(values, dtype=np.float64).copy()
            return origin + (bound / norm) * delta

        return project

    def _components_feasible(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        *,
        upper_constraint: dict[str, Any],
        lower_constraint: dict[str, Any],
    ) -> bool:
        tolerance = self.feasibility_tolerance
        upper_residual = self._residual(upper, upper_constraint)
        lower_residual = self._residual(lower, lower_constraint)
        return bool(
            np.max(np.abs(upper)) <= self.upper_action_limit + tolerance
            and np.max(np.abs(lower)) <= self.lower_action_limit + tolerance
            and float(np.sum(np.square(upper_residual)))
            <= float(upper_constraint["allowed_energy"]) + tolerance
            and float(np.sum(np.square(lower_residual)))
            <= float(lower_constraint["allowed_energy"]) + tolerance
        )

    @staticmethod
    def _residual(
        values: np.ndarray,
        constraint: dict[str, Any],
    ) -> np.ndarray:
        return (
            float(constraint["coefficient"])
            * np.asarray(values, dtype=np.float64)
            + np.asarray(constraint["offset"], dtype=np.float64)
        )

    def _action(self, values: Any, role: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != (self._dimension,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{role} must be finite and aligned")
        return array

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("projector must be reset before use")
