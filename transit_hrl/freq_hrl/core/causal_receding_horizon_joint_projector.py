"""Causal receding-horizon projection of hierarchical component actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .receding_horizon_responsibility import future_rolling_mean_system


FORECAST_MODES = ("hold", "damped_velocity")


class AffineQuadraticBallProjector:
    """Euclidean projector onto ``||A x + b||_F^2 <= radius_squared``."""

    def __init__(
        self,
        operator: Any,
        offset: Any,
        radius_squared: float,
        *,
        tolerance: float = 1e-10,
    ) -> None:
        matrix = np.asarray(operator, dtype=np.float64)
        bias = np.asarray(offset, dtype=np.float64)
        radius = float(radius_squared)
        self.tolerance = float(tolerance)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or bias.ndim != 2
            or bias.shape[0] != matrix.shape[0]
        ):
            raise ValueError("affine quadratic projector shapes do not align")
        if (
            not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(bias))
            or not np.isfinite(radius)
            or radius < 0.0
            or not np.isfinite(self.tolerance)
            or self.tolerance <= 0.0
        ):
            raise ValueError("affine quadratic projector values are invalid")
        gram = matrix.T @ matrix
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        self.operator = matrix
        self.offset = bias
        self.radius_squared = radius
        self.eigenvalues = np.maximum(eigenvalues, 0.0)
        self.eigenvectors = eigenvectors
        self.projected_offset = eigenvectors.T @ (matrix.T @ bias)
        least_squares = np.linalg.lstsq(matrix, -bias, rcond=None)[0]
        self.minimum_energy = self.energy(least_squares)
        self.feasible = bool(
            self.minimum_energy <= self.radius_squared + self.tolerance
        )

    def energy(self, values: Any) -> float:
        array = np.asarray(values, dtype=np.float64)
        residual = self.operator @ array + self.offset
        return float(np.sum(np.square(residual)))

    def project(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self.offset.shape or not np.all(np.isfinite(array)):
            raise ValueError("affine quadratic values do not align")
        if self.energy(array) <= self.radius_squared + self.tolerance:
            return array.copy()
        if not self.feasible:
            return np.linalg.lstsq(
                self.operator, -self.offset, rcond=None
            )[0]
        coordinates = self.eigenvectors.T @ array

        def candidate(multiplier: float) -> np.ndarray:
            scale = 1.0 + float(multiplier) * self.eigenvalues
            projected_coordinates = (
                coordinates - float(multiplier) * self.projected_offset
            ) / scale[:, None]
            return self.eigenvectors @ projected_coordinates

        low = 0.0
        high = 1.0
        projected = candidate(high)
        while self.energy(projected) > self.radius_squared:
            high *= 10.0
            projected = candidate(high)
            if high > 1e18:
                raise RuntimeError("affine quadratic multiplier did not bracket")
        for _ in range(80):
            midpoint = 0.5 * (low + high)
            trial = candidate(midpoint)
            if self.energy(trial) <= self.radius_squared:
                high = midpoint
                projected = trial
            else:
                low = midpoint
        return projected


@dataclass
class CausalRecedingHorizonJointProjector:
    """Amortize causal frequency debt over a model-predictive horizon."""

    upper_window: int = 8
    lower_window: int = 32
    upper_rms_budget: float = 0.075
    lower_rms_budget: float = 0.0475
    upper_action_limit: float = 1.0
    lower_action_limit: float = 1.0
    planning_horizon: int = 32
    forecast_mode: str = "damped_velocity"
    velocity_alpha: float = 0.25
    velocity_decay: float = 0.75
    projection_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-8
    maximum_projection_iterations: int = 64

    def __post_init__(self) -> None:
        self.upper_window = int(self.upper_window)
        self.lower_window = int(self.lower_window)
        self.upper_rms_budget = float(self.upper_rms_budget)
        self.lower_rms_budget = float(self.lower_rms_budget)
        self.upper_action_limit = float(self.upper_action_limit)
        self.lower_action_limit = float(self.lower_action_limit)
        self.planning_horizon = int(self.planning_horizon)
        self.forecast_mode = str(self.forecast_mode)
        self.velocity_alpha = float(self.velocity_alpha)
        self.velocity_decay = float(self.velocity_decay)
        self.projection_tolerance = float(self.projection_tolerance)
        self.feasibility_tolerance = float(self.feasibility_tolerance)
        self.maximum_projection_iterations = int(
            self.maximum_projection_iterations
        )
        if self.upper_window < 2 or self.lower_window < 2:
            raise ValueError("frequency windows must be at least two")
        if self.planning_horizon < 2:
            raise ValueError("planning_horizon must be at least two")
        if self.forecast_mode not in FORECAST_MODES:
            raise ValueError(f"unknown forecast mode: {self.forecast_mode}")
        positive = (
            self.upper_rms_budget,
            self.lower_rms_budget,
            self.upper_action_limit,
            self.lower_action_limit,
            self.velocity_alpha,
            self.projection_tolerance,
            self.feasibility_tolerance,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("projector budgets, limits, and rates must be positive")
        if not 0.0 <= self.velocity_decay <= 1.0:
            raise ValueError("velocity_decay must lie in [0, 1]")
        if self.velocity_alpha > 1.0:
            raise ValueError("velocity_alpha must be at most one")
        if self.maximum_projection_iterations < 1:
            raise ValueError("maximum_projection_iterations must be positive")
        self._dimension = 0
        self._upper_history: list[np.ndarray] = []
        self._lower_history: list[np.ndarray] = []
        self._upper_energy = 0.0
        self._lower_energy = 0.0
        self._previous_upper = np.zeros(0, dtype=np.float64)
        self._previous_lower = np.zeros(0, dtype=np.float64)
        self._upper_velocity = np.zeros(0, dtype=np.float64)
        self._lower_velocity = np.zeros(0, dtype=np.float64)
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
        self._previous_upper = np.zeros(dimension, dtype=np.float64)
        self._previous_lower = np.zeros(dimension, dtype=np.float64)
        self._upper_velocity = np.zeros(dimension, dtype=np.float64)
        self._lower_velocity = np.zeros(dimension, dtype=np.float64)
        self._step_count = 0

    def project(
        self,
        proposed_upper: Any,
        proposed_lower: Any,
    ) -> dict[str, Any]:
        self._require_reset()
        upper_proposal = self._action(proposed_upper, "proposed upper")
        lower_proposal = self._action(proposed_lower, "proposed lower")
        if self._step_count:
            self._upper_velocity += self.velocity_alpha * (
                upper_proposal - self._previous_upper - self._upper_velocity
            )
            self._lower_velocity += self.velocity_alpha * (
                lower_proposal - self._previous_lower - self._lower_velocity
            )
        upper_forecast = self._forecast(
            upper_proposal,
            self._upper_velocity,
            limit=self.upper_action_limit,
        )
        lower_forecast = self._forecast(
            lower_proposal,
            self._lower_velocity,
            limit=self.lower_action_limit,
        )
        total_forecast = upper_forecast + lower_forecast
        upper_operator, upper_offset = self._future_residual_system(
            self._upper_history,
            window=self.upper_window,
            high_pass=True,
        )
        lower_operator, lower_offset = self._future_residual_system(
            self._lower_history,
            window=self.lower_window,
            high_pass=False,
        )
        upper_allowed = max(
            (self._step_count + self.planning_horizon)
            * self._dimension
            * self.upper_rms_budget ** 2
            - self._upper_energy,
            0.0,
        )
        lower_allowed = max(
            (self._step_count + self.planning_horizon)
            * self._dimension
            * self.lower_rms_budget ** 2
            - self._lower_energy,
            0.0,
        )
        upper_ball = AffineQuadraticBallProjector(
            upper_operator,
            upper_offset,
            upper_allowed,
            tolerance=self.feasibility_tolerance,
        )
        lower_ball = AffineQuadraticBallProjector(
            lower_operator,
            lower_offset,
            lower_allowed,
            tolerance=self.feasibility_tolerance,
        )
        physical_low = np.maximum(
            -self.upper_action_limit,
            total_forecast - self.lower_action_limit,
        )
        physical_high = np.minimum(
            self.upper_action_limit,
            total_forecast + self.lower_action_limit,
        )
        fixed_lower_ball = AffineQuadraticBallProjector(
            -lower_operator,
            lower_operator @ total_forecast + lower_offset,
            lower_allowed,
            tolerance=self.feasibility_tolerance,
        )
        fixed_upper, fixed_meta = self._dykstra(
            upper_forecast,
            (
                upper_ball.project,
                fixed_lower_ball.project,
                lambda values: np.clip(
                    values, physical_low, physical_high
                ),
            ),
        )
        fixed_lower = total_forecast - fixed_upper
        fixed_feasible = self._forecast_feasible(
            fixed_upper,
            fixed_lower,
            upper_ball=upper_ball,
            lower_ball=lower_ball,
        )
        if fixed_feasible:
            upper_plan = fixed_upper
            lower_plan = fixed_lower
            total_changed = False
            projection_converged = bool(fixed_meta["converged"])
            projection_iterations = int(fixed_meta["iterations"])
        else:
            upper_plan, upper_meta = self._project_ball_box(
                upper_forecast,
                ball=upper_ball,
                low=np.full_like(upper_forecast, -self.upper_action_limit),
                high=np.full_like(upper_forecast, self.upper_action_limit),
            )
            lower_plan, lower_meta = self._project_ball_box(
                lower_forecast,
                ball=lower_ball,
                low=np.full_like(lower_forecast, -self.lower_action_limit),
                high=np.full_like(lower_forecast, self.lower_action_limit),
            )
            total_changed = True
            projection_converged = bool(
                upper_meta["converged"] and lower_meta["converged"]
            )
            projection_iterations = max(
                int(upper_meta["iterations"]),
                int(lower_meta["iterations"]),
            )

        upper = np.asarray(upper_plan[0], dtype=np.float64)
        lower = np.asarray(lower_plan[0], dtype=np.float64)
        total_proposal = upper_proposal + lower_proposal
        total = upper + lower
        correction = total - total_proposal
        upper_residual = upper_operator[0] @ upper_plan + upper_offset[0]
        lower_residual = lower_operator[0] @ lower_plan + lower_offset[0]
        self._upper_energy += float(np.sum(np.square(upper_residual)))
        self._lower_energy += float(np.sum(np.square(lower_residual)))
        self._upper_history.append(upper.copy())
        self._lower_history.append(lower.copy())
        self._upper_history = self._upper_history[-(self.upper_window - 1):]
        self._lower_history = self._lower_history[-(self.lower_window - 1):]
        self._previous_upper = upper_proposal.copy()
        self._previous_lower = lower_proposal.copy()
        self._step_count += 1
        denominator = float(self._step_count * self._dimension)
        forecast_feasible = self._forecast_feasible(
            upper_plan,
            lower_plan,
            upper_ball=upper_ball,
            lower_ball=lower_ball,
        )
        return {
            "upper": upper.copy(),
            "lower": lower.copy(),
            "total": total.copy(),
            "total_correction": correction.copy(),
            "upper_plan": upper_plan.copy(),
            "lower_plan": lower_plan.copy(),
            "total_forecast": total_forecast.copy(),
            "fixed_total_forecast_feasible": bool(fixed_feasible),
            "projected_forecast_feasible": bool(forecast_feasible),
            "total_action_changed": bool(total_changed),
            "projection_converged": projection_converged,
            "projection_iterations": projection_iterations,
            "upper_prefix_power": float(self._upper_energy / denominator),
            "lower_prefix_power": float(self._lower_energy / denominator),
            "upper_forecast_power": float(
                upper_ball.energy(upper_plan) / (
                    self.planning_horizon * self._dimension
                )
            ),
            "lower_forecast_power": float(
                lower_ball.energy(lower_plan) / (
                    self.planning_horizon * self._dimension
                )
            ),
            "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
            "correction_abs_max": float(np.max(np.abs(correction))),
            "reconstruction_error_max": float(
                np.max(np.abs(upper + lower - total))
            ),
        }

    def _future_residual_system(
        self,
        history: list[np.ndarray],
        *,
        window: int,
        high_pass: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = history[-(int(window) - 1):]
        past = (
            np.stack(rows)
            if rows else np.empty((0, self._dimension), dtype=np.float64)
        )
        rolling, offset = future_rolling_mean_system(
            past,
            horizon=self.planning_horizon,
            window=int(window),
        )
        if high_pass:
            return np.eye(self.planning_horizon) - rolling, -offset
        return rolling, offset

    def _forecast(
        self,
        current: np.ndarray,
        velocity: np.ndarray,
        *,
        limit: float,
    ) -> np.ndarray:
        forecast = np.repeat(
            current.reshape(1, -1), self.planning_horizon, axis=0
        )
        if self.forecast_mode == "damped_velocity":
            increment = np.asarray(velocity, dtype=np.float64).copy()
            for index in range(1, self.planning_horizon):
                forecast[index] = forecast[index - 1] + increment
                increment *= self.velocity_decay
        return np.clip(forecast, -float(limit), float(limit))

    def _forecast_feasible(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        *,
        upper_ball: AffineQuadraticBallProjector,
        lower_ball: AffineQuadraticBallProjector,
    ) -> bool:
        tolerance = self.feasibility_tolerance
        return bool(
            np.max(np.abs(upper)) <= self.upper_action_limit + tolerance
            and np.max(np.abs(lower)) <= self.lower_action_limit + tolerance
            and upper_ball.energy(upper)
            <= upper_ball.radius_squared + tolerance
            and lower_ball.energy(lower)
            <= lower_ball.radius_squared + tolerance
        )

    def _project_ball_box(
        self,
        proposed: np.ndarray,
        *,
        ball: AffineQuadraticBallProjector,
        low: np.ndarray,
        high: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not ball.feasible:
            return np.clip(ball.project(proposed), low, high), {
                "converged": False,
                "iterations": 1,
            }
        return self._dykstra(
            proposed,
            (ball.project, lambda values: np.clip(values, low, high)),
        )

    def _dykstra(
        self,
        start: np.ndarray,
        projectors: tuple[Callable[[np.ndarray], np.ndarray], ...],
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

    def _action(self, values: Any, role: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != (self._dimension,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{role} must be finite and aligned")
        return array

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("projector must be reset before use")
