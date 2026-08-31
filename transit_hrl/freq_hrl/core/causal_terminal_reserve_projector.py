"""Causal responsibility projection with a recursive terminal reserve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .receding_horizon_responsibility import future_rolling_mean_system


@dataclass
class _TerminalCertificate:
    """Prefix constraints induced by one finite-memory backup continuation."""

    coefficients: np.ndarray
    offsets: np.ndarray
    allowed_future_energy: np.ndarray
    balls: tuple[tuple[np.ndarray, float], ...]
    impossible: bool
    tolerance: float

    def prefix_energy(self, values: np.ndarray) -> np.ndarray:
        residuals = (
            self.coefficients[:, None] * np.asarray(values, dtype=np.float64)
            + self.offsets
        )
        return np.cumsum(np.sum(np.square(residuals), axis=1))

    def feasible(self, values: np.ndarray) -> bool:
        if self.impossible:
            return False
        return bool(np.all(
            self.prefix_energy(values)
            <= self.allowed_future_energy + self.tolerance
        ))

    def minimum_margin(self, values: np.ndarray) -> float:
        margins = self.allowed_future_energy - self.prefix_energy(values)
        return float(np.min(margins))

    def current_residual(self, values: np.ndarray) -> np.ndarray:
        return (
            float(self.coefficients[0]) * np.asarray(values, dtype=np.float64)
            + self.offsets[0]
        )


@dataclass
class CausalTerminalReserveProjector:
    """Project hierarchical actions into recursively feasible frequency sets.

    The upper backup holds the realized upper action until its causal HPF
    history is constant. The lower backup sets all subsequent lower actions to
    zero until the causal LPF history is zero. Every prefix of those backup
    continuations must satisfy its cumulative RMS budget. Once the histories
    have flushed, both residuals remain zero indefinitely.

    A fixed-total responsibility split is used whenever the proposed physical
    action admits this certificate. Otherwise each component is projected into
    its certified set, which is the only branch allowed to alter total action.
    """

    upper_window: int = 8
    lower_window: int = 32
    upper_rms_budget: float = 0.075
    lower_rms_budget: float = 0.0475
    upper_action_limit: float = 1.0
    lower_action_limit: float = 1.0
    projection_tolerance: float = 1e-10
    feasibility_tolerance: float = 1e-8
    maximum_projection_iterations: int = 512

    def __post_init__(self) -> None:
        self.upper_window = int(self.upper_window)
        self.lower_window = int(self.lower_window)
        self.upper_rms_budget = float(self.upper_rms_budget)
        self.lower_rms_budget = float(self.lower_rms_budget)
        self.upper_action_limit = float(self.upper_action_limit)
        self.lower_action_limit = float(self.lower_action_limit)
        self.projection_tolerance = float(self.projection_tolerance)
        self.feasibility_tolerance = float(self.feasibility_tolerance)
        self.maximum_projection_iterations = int(
            self.maximum_projection_iterations
        )
        if self.upper_window < 2 or self.lower_window < 2:
            raise ValueError("frequency windows must be at least two")
        positive = (
            self.upper_rms_budget,
            self.lower_rms_budget,
            self.upper_action_limit,
            self.lower_action_limit,
            self.projection_tolerance,
            self.feasibility_tolerance,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "projector budgets, limits, and tolerances must be positive"
            )
        if self.maximum_projection_iterations < 1:
            raise ValueError("maximum_projection_iterations must be positive")
        self._dimension = 0
        self._upper_history: list[np.ndarray] = []
        self._lower_history: list[np.ndarray] = []
        self._upper_energy = 0.0
        self._lower_energy = 0.0
        self._previous_upper = np.zeros(0, dtype=np.float64)
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
        self._step_count = 0

    def project(
        self,
        proposed_upper: Any,
        proposed_lower: Any,
    ) -> dict[str, Any]:
        """Return one causal action and its terminal-reserve certificate."""

        self._require_reset()
        upper_proposal = self._action(proposed_upper, "proposed upper")
        lower_proposal = self._action(proposed_lower, "proposed lower")
        total_proposal = upper_proposal + lower_proposal
        if np.max(np.abs(total_proposal)) > (
            self.upper_action_limit + self.lower_action_limit + 1e-10
        ):
            raise ValueError("proposed total exceeds the component sum box")

        upper_certificate = self._terminal_certificate(
            history=self._upper_history,
            window=self.upper_window,
            rms_budget=self.upper_rms_budget,
            accumulated_energy=self._upper_energy,
            high_pass=True,
            backup_coefficients=np.ones(self.upper_window, dtype=np.float64),
        )
        lower_backup = np.zeros(self.lower_window, dtype=np.float64)
        lower_backup[0] = 1.0
        lower_certificate = self._terminal_certificate(
            history=self._lower_history,
            window=self.lower_window,
            rms_budget=self.lower_rms_budget,
            accumulated_energy=self._lower_energy,
            high_pass=False,
            backup_coefficients=lower_backup,
        )

        recursive_upper = self._previous_upper
        recursive_lower = np.zeros(self._dimension, dtype=np.float64)
        if not self._component_feasible(
            recursive_upper,
            certificate=upper_certificate,
            limit=self.upper_action_limit,
        ) or not self._component_feasible(
            recursive_lower,
            certificate=lower_certificate,
            limit=self.lower_action_limit,
        ):
            raise RuntimeError("terminal-reserve invariant was lost")

        physical_low = np.maximum(
            -self.upper_action_limit,
            total_proposal - self.lower_action_limit,
        )
        physical_high = np.minimum(
            self.upper_action_limit,
            total_proposal + self.lower_action_limit,
        )
        fixed_balls = list(upper_certificate.balls)
        fixed_balls.extend(
            (total_proposal - center, radius)
            for center, radius in lower_certificate.balls
        )
        fixed_upper, fixed_meta = self._project_ball_intersection(
            upper_proposal,
            balls=fixed_balls,
            low=physical_low,
            high=physical_high,
        )
        fixed_lower = total_proposal - fixed_upper
        fixed_feasible = bool(
            self._component_feasible(
                fixed_upper,
                certificate=upper_certificate,
                limit=self.upper_action_limit,
            )
            and self._component_feasible(
                fixed_lower,
                certificate=lower_certificate,
                limit=self.lower_action_limit,
            )
        )

        fallback_used = False
        if fixed_feasible:
            upper = fixed_upper
            lower = fixed_lower
            projection_converged = bool(fixed_meta["converged"])
            projection_iterations = int(fixed_meta["iterations"])
        else:
            upper, upper_meta = self._project_component(
                upper_proposal,
                certificate=upper_certificate,
                limit=self.upper_action_limit,
                recursive_fallback=recursive_upper,
            )
            lower, lower_meta = self._project_component(
                lower_proposal,
                certificate=lower_certificate,
                limit=self.lower_action_limit,
                recursive_fallback=recursive_lower,
            )
            fallback_used = bool(
                upper_meta["fallback_used"] or lower_meta["fallback_used"]
            )
            projection_converged = bool(
                upper_meta["converged"] and lower_meta["converged"]
            )
            projection_iterations = max(
                int(upper_meta["iterations"]),
                int(lower_meta["iterations"]),
            )

        upper_feasible = self._component_feasible(
            upper,
            certificate=upper_certificate,
            limit=self.upper_action_limit,
        )
        lower_feasible = self._component_feasible(
            lower,
            certificate=lower_certificate,
            limit=self.lower_action_limit,
        )
        if not upper_feasible or not lower_feasible:
            raise RuntimeError("projector returned an uncertified action")

        upper_residual = upper_certificate.current_residual(upper)
        lower_residual = lower_certificate.current_residual(lower)
        step_upper_energy = float(np.sum(np.square(upper_residual)))
        step_lower_energy = float(np.sum(np.square(lower_residual)))
        upper_margin = upper_certificate.minimum_margin(upper)
        lower_margin = lower_certificate.minimum_margin(lower)
        total = upper + lower
        correction = total - total_proposal

        self._upper_energy += step_upper_energy
        self._lower_energy += step_lower_energy
        self._upper_history.append(upper.copy())
        self._lower_history.append(lower.copy())
        self._upper_history = self._upper_history[-(self.upper_window - 1):]
        self._lower_history = self._lower_history[-(self.lower_window - 1):]
        self._previous_upper = upper.copy()
        self._step_count += 1
        denominator = float(self._step_count * self._dimension)

        return {
            "upper": upper.copy(),
            "lower": lower.copy(),
            "total": total.copy(),
            "total_correction": correction.copy(),
            "fixed_total_feasible": bool(fixed_feasible),
            "total_action_changed": bool(
                np.max(np.abs(correction)) > self.feasibility_tolerance
            ),
            "component_feasible": bool(upper_feasible and lower_feasible),
            "terminal_certificate_feasible": bool(
                upper_feasible and lower_feasible
            ),
            "recursive_backup_feasible_at_entry": True,
            "projection_converged": bool(projection_converged),
            "projection_iterations": int(projection_iterations),
            "recursive_fallback_used": bool(fallback_used),
            "upper_residual": upper_residual.copy(),
            "lower_residual": lower_residual.copy(),
            "upper_prefix_power": float(self._upper_energy / denominator),
            "lower_prefix_power": float(self._lower_energy / denominator),
            "upper_terminal_reserve_min_margin": float(upper_margin),
            "lower_terminal_reserve_min_margin": float(lower_margin),
            "upper_certificate_prefix_count": int(self.upper_window),
            "lower_certificate_prefix_count": int(self.lower_window),
            "upper_allowed_step_energy": float(
                upper_certificate.allowed_future_energy[0]
            ),
            "lower_allowed_step_energy": float(
                lower_certificate.allowed_future_energy[0]
            ),
            "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
            "correction_abs_max": float(np.max(np.abs(correction))),
            "component_correction_rms": float(np.sqrt(np.mean(
                np.square(upper - upper_proposal)
                + np.square(lower - lower_proposal)
            ))),
            "reconstruction_error_max": float(
                np.max(np.abs(upper + lower - total))
            ),
        }

    def _terminal_certificate(
        self,
        *,
        history: list[np.ndarray],
        window: int,
        rms_budget: float,
        accumulated_energy: float,
        high_pass: bool,
        backup_coefficients: np.ndarray,
    ) -> _TerminalCertificate:
        rows = history[-(int(window) - 1):]
        past = (
            np.stack(rows)
            if rows else np.empty((0, self._dimension), dtype=np.float64)
        )
        rolling, rolling_offset = future_rolling_mean_system(
            past,
            horizon=int(window),
            window=int(window),
        )
        coefficients = np.asarray(backup_coefficients, dtype=np.float64)
        mean_coefficients = rolling @ coefficients
        if high_pass:
            residual_coefficients = coefficients - mean_coefficients
            residual_offsets = -rolling_offset
        else:
            residual_coefficients = mean_coefficients
            residual_offsets = rolling_offset

        allowed = (
            (self._step_count + np.arange(1, int(window) + 1))
            * self._dimension
            * float(rms_budget) ** 2
            - float(accumulated_energy)
        )
        balls: list[tuple[np.ndarray, float]] = []
        impossible = False
        coefficient_energy = 0.0
        linear = np.zeros(self._dimension, dtype=np.float64)
        constant = 0.0
        for index in range(int(window)):
            coefficient = float(residual_coefficients[index])
            offset = residual_offsets[index]
            coefficient_energy += coefficient ** 2
            linear += coefficient * offset
            constant += float(np.sum(np.square(offset)))
            radius_numerator = float(allowed[index]) - constant
            if coefficient_energy <= 1e-30:
                if radius_numerator < -self.feasibility_tolerance:
                    impossible = True
                continue
            radius_squared = (
                radius_numerator
                + float(np.dot(linear, linear)) / coefficient_energy
            ) / coefficient_energy
            if radius_squared < -self.feasibility_tolerance:
                impossible = True
                continue
            center = -linear / coefficient_energy
            balls.append((center.copy(), float(np.sqrt(max(radius_squared, 0.0)))))

        return _TerminalCertificate(
            coefficients=residual_coefficients,
            offsets=residual_offsets,
            allowed_future_energy=allowed,
            balls=tuple(balls),
            impossible=bool(impossible),
            tolerance=self.feasibility_tolerance,
        )

    def _project_component(
        self,
        proposed: np.ndarray,
        *,
        certificate: _TerminalCertificate,
        limit: float,
        recursive_fallback: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        low = np.full(self._dimension, -float(limit), dtype=np.float64)
        high = np.full(self._dimension, float(limit), dtype=np.float64)
        clipped = np.clip(proposed, low, high)
        if certificate.feasible(clipped):
            return clipped, {
                "converged": True,
                "iterations": 1,
                "fallback_used": False,
            }
        projected, meta = self._project_ball_intersection(
            proposed,
            balls=list(certificate.balls),
            low=low,
            high=high,
        )
        if certificate.feasible(projected):
            return projected, {
                **meta,
                "fallback_used": False,
            }
        fallback = np.asarray(recursive_fallback, dtype=np.float64).copy()
        if not self._component_feasible(
            fallback,
            certificate=certificate,
            limit=limit,
        ):
            raise RuntimeError("recursive component fallback is infeasible")
        return fallback, {
            "converged": False,
            "iterations": int(meta["iterations"]),
            "fallback_used": True,
        }

    def _project_ball_intersection(
        self,
        proposed: np.ndarray,
        *,
        balls: list[tuple[np.ndarray, float]],
        low: np.ndarray,
        high: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        projectors = [
            self._ball_projector(center, radius)
            for center, radius in balls
        ]
        projectors.append(lambda values: np.clip(values, low, high))
        return self._dykstra(np.clip(proposed, low, high), projectors)

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
            point = np.asarray(values, dtype=np.float64)
            delta = point - origin
            norm = float(np.linalg.norm(delta))
            if norm <= bound or norm <= 1e-30:
                return point.copy()
            return origin + (bound / norm) * delta

        return project

    def _component_feasible(
        self,
        values: np.ndarray,
        *,
        certificate: _TerminalCertificate,
        limit: float,
    ) -> bool:
        return bool(
            np.max(np.abs(values)) <= float(limit) + self.feasibility_tolerance
            and certificate.feasible(values)
        )

    def _action(self, values: Any, role: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != (self._dimension,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{role} must be finite and aligned")
        return array

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError("projector must be reset before use")
