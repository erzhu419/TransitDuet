"""Nearest bounded component pair satisfying registered frequency budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .full_horizon_responsibility_oracle import causal_rolling_operator


@dataclass(frozen=True)
class NearestFeasibleComponentProjectionResult:
    """Euclidean projection of an upper/lower trace onto convex constraints."""

    upper: np.ndarray
    lower: np.ndarray
    status: str
    feasible: bool
    include_total_action_box: bool
    upper_power: float
    lower_power: float
    upper_power_budget: float
    lower_power_budget: float
    component_correction_rms: float
    total_action_correction_rms: float
    upper_correction_rms: float
    lower_correction_rms: float
    upper_bound_violation_max: float
    lower_bound_violation_max: float
    total_action_bound_violation_max: float
    convergence_residual_max: float
    iteration_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "feasible": bool(self.feasible),
            "include_total_action_box": bool(self.include_total_action_box),
            "upper_power": float(self.upper_power),
            "lower_power": float(self.lower_power),
            "upper_power_budget": float(self.upper_power_budget),
            "lower_power_budget": float(self.lower_power_budget),
            "upper_power_budget_excess": float(max(
                self.upper_power - self.upper_power_budget, 0.0
            )),
            "lower_power_budget_excess": float(max(
                self.lower_power - self.lower_power_budget, 0.0
            )),
            "component_correction_rms": float(
                self.component_correction_rms
            ),
            "total_action_correction_rms": float(
                self.total_action_correction_rms
            ),
            "upper_correction_rms": float(self.upper_correction_rms),
            "lower_correction_rms": float(self.lower_correction_rms),
            "upper_bound_violation_max": float(
                self.upper_bound_violation_max
            ),
            "lower_bound_violation_max": float(
                self.lower_bound_violation_max
            ),
            "total_action_bound_violation_max": float(
                self.total_action_bound_violation_max
            ),
            "convergence_residual_max": float(
                self.convergence_residual_max
            ),
            "iteration_count": int(self.iteration_count),
            "trajectory_length": int(self.upper.shape[0]),
            "action_dimension": int(self.upper.shape[1]),
            "projection_contract": (
                "dykstra_euclidean_projection_of_reference_upper_lower_"
                "components_onto_exact_hpf8_lpf32_balls_and_component_"
                "boxes_with_optional_nominal_total_action_box_v1"
            ),
        }


class _QuadraticBallProjector:
    def __init__(self, operator: np.ndarray, radius_squared: float) -> None:
        matrix = np.asarray(operator, dtype=np.float64)
        radius = float(radius_squared)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("frequency operator must be square")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("frequency-ball radius must be positive")
        gram = matrix.T @ matrix
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        self.eigenvalues = np.maximum(eigenvalues, 0.0)
        self.eigenvectors = eigenvectors
        self.radius_squared = radius

    def project(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != self.eigenvectors.shape[0]:
            raise ValueError("frequency-ball values are not aligned")
        coordinates = self.eigenvectors.T @ array
        if self._energy(coordinates, 0.0) <= self.radius_squared:
            return array.copy()
        low = 0.0
        high = 1.0
        while self._energy(coordinates, high) > self.radius_squared:
            high *= 10.0
            if high > 1e18:
                raise RuntimeError("frequency-ball multiplier did not bracket")
        for _ in range(80):
            midpoint = 0.5 * (low + high)
            if self._energy(coordinates, midpoint) <= self.radius_squared:
                high = midpoint
            else:
                low = midpoint
        scale = 1.0 + high * self.eigenvalues
        return self.eigenvectors @ (coordinates / scale[:, None])

    def _energy(self, coordinates: np.ndarray, multiplier: float) -> float:
        scale = 1.0 + float(multiplier) * self.eigenvalues
        return float(np.sum(
            self.eigenvalues[:, None]
            * np.square(coordinates / scale[:, None])
        ))


def project_nearest_feasible_components(
    reference_upper: Any,
    reference_lower: Any,
    *,
    upper_rms_budget: float,
    lower_rms_budget: float,
    upper_action_limit: float = 1.0,
    lower_action_limit: float = 1.0,
    total_action_limit: float = 1.0,
    include_total_action_box: bool = False,
    upper_window: int = 8,
    lower_window: int = 32,
    convergence_tolerance: float = 1e-10,
    feasibility_tolerance: float = 1e-9,
    max_iterations: int = 10000,
) -> NearestFeasibleComponentProjectionResult:
    """Project a component trajectory onto the exact convex feasible set.

    Dykstra's algorithm computes the Euclidean projection onto the intersection
    of exact quadratic frequency balls and pointwise boxes. The optional total
    box is reported separately because the registered responsibility audit is
    pre-saturation, while the environment receives a clipped nominal sum.
    """

    reference_u = _action_trace(reference_upper, role="reference upper")
    reference_l = _action_trace(reference_lower, role="reference lower")
    if reference_u.shape != reference_l.shape:
        raise ValueError("reference component traces must align")
    upper_limit = _positive_finite(upper_action_limit, "upper action limit")
    lower_limit = _positive_finite(lower_action_limit, "lower action limit")
    total_limit = _positive_finite(total_action_limit, "total action limit")
    upper_budget = _positive_finite(upper_rms_budget, "upper RMS budget")
    lower_budget = _positive_finite(lower_rms_budget, "lower RMS budget")
    convergence = _positive_finite(
        convergence_tolerance, "convergence tolerance"
    )
    feasibility = _positive_finite(
        feasibility_tolerance, "feasibility tolerance"
    )
    iterations = int(max_iterations)
    if iterations < 1:
        raise ValueError("maximum projection iterations must be positive")

    length, dimension = reference_u.shape
    upper_operator = (
        np.eye(length, dtype=np.float64)
        - causal_rolling_operator(length, int(upper_window))
    )
    lower_operator = causal_rolling_operator(length, int(lower_window))
    upper_power_budget = upper_budget ** 2
    lower_power_budget = lower_budget ** 2
    initial_metrics = _constraint_metrics(
        reference_u,
        reference_l,
        upper_operator=upper_operator,
        lower_operator=lower_operator,
        upper_limit=upper_limit,
        lower_limit=lower_limit,
        total_limit=total_limit,
        include_total_action_box=bool(include_total_action_box),
    )
    initially_feasible = bool(
        initial_metrics["upper_power"] <= upper_power_budget + feasibility
        and initial_metrics["lower_power"] <= lower_power_budget + feasibility
        and initial_metrics["upper_bound_violation_max"] <= feasibility
        and initial_metrics["lower_bound_violation_max"] <= feasibility
        and initial_metrics["total_action_bound_violation_max"] <= feasibility
    )
    if initially_feasible:
        return NearestFeasibleComponentProjectionResult(
            upper=reference_u.copy(),
            lower=reference_l.copy(),
            status="reference_components_already_feasible",
            feasible=True,
            include_total_action_box=bool(include_total_action_box),
            upper_power=float(initial_metrics["upper_power"]),
            lower_power=float(initial_metrics["lower_power"]),
            upper_power_budget=upper_power_budget,
            lower_power_budget=lower_power_budget,
            component_correction_rms=0.0,
            total_action_correction_rms=0.0,
            upper_correction_rms=0.0,
            lower_correction_rms=0.0,
            upper_bound_violation_max=float(
                initial_metrics["upper_bound_violation_max"]
            ),
            lower_bound_violation_max=float(
                initial_metrics["lower_bound_violation_max"]
            ),
            total_action_bound_violation_max=float(
                initial_metrics["total_action_bound_violation_max"]
            ),
            convergence_residual_max=0.0,
            iteration_count=0,
        )
    upper_ball = _QuadraticBallProjector(
        upper_operator,
        length * dimension * upper_power_budget,
    )
    lower_ball = _QuadraticBallProjector(
        lower_operator,
        length * dimension * lower_power_budget,
    )

    Projector = Callable[
        [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ]
    projectors: list[Projector] = [
        lambda u, l: (np.clip(u, -upper_limit, upper_limit), l.copy()),
        lambda u, l: (u.copy(), np.clip(l, -lower_limit, lower_limit)),
    ]
    if bool(include_total_action_box):
        def project_total_box(
            upper: np.ndarray, lower: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            total = upper + lower
            target = np.clip(total, -total_limit, total_limit)
            correction = 0.5 * (total - target)
            return upper - correction, lower - correction

        projectors.append(project_total_box)
    projectors.extend([
        lambda u, l: (upper_ball.project(u), l.copy()),
        lambda u, l: (u.copy(), lower_ball.project(l)),
    ])

    upper = reference_u.copy()
    lower = reference_l.copy()
    corrections = [
        (np.zeros_like(upper), np.zeros_like(lower))
        for _ in projectors
    ]
    residual = float("inf")
    iteration_count = 0
    for iteration_count in range(1, iterations + 1):
        before_upper = upper.copy()
        before_lower = lower.copy()
        for index, projector in enumerate(projectors):
            correction_upper, correction_lower = corrections[index]
            argument_upper = upper + correction_upper
            argument_lower = lower + correction_lower
            projected_upper, projected_lower = projector(
                argument_upper, argument_lower
            )
            corrections[index] = (
                argument_upper - projected_upper,
                argument_lower - projected_lower,
            )
            upper = projected_upper
            lower = projected_lower
        residual = float(max(
            np.max(np.abs(upper - before_upper)),
            np.max(np.abs(lower - before_lower)),
        ))
        metrics = _constraint_metrics(
            upper,
            lower,
            upper_operator=upper_operator,
            lower_operator=lower_operator,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            total_limit=total_limit,
            include_total_action_box=bool(include_total_action_box),
        )
        constraints_satisfied = bool(
            metrics["upper_power"] <= upper_power_budget + feasibility
            and metrics["lower_power"] <= lower_power_budget + feasibility
            and metrics["upper_bound_violation_max"] <= feasibility
            and metrics["lower_bound_violation_max"] <= feasibility
            and metrics["total_action_bound_violation_max"] <= feasibility
        )
        if residual <= convergence and constraints_satisfied:
            break
    else:
        raise RuntimeError(
            "nearest feasible component projection did not converge"
        )

    metrics = _constraint_metrics(
        upper,
        lower,
        upper_operator=upper_operator,
        lower_operator=lower_operator,
        upper_limit=upper_limit,
        lower_limit=lower_limit,
        total_limit=total_limit,
        include_total_action_box=bool(include_total_action_box),
    )
    feasible = bool(
        metrics["upper_power"] <= upper_power_budget + feasibility
        and metrics["lower_power"] <= lower_power_budget + feasibility
        and metrics["upper_bound_violation_max"] <= feasibility
        and metrics["lower_bound_violation_max"] <= feasibility
        and metrics["total_action_bound_violation_max"] <= feasibility
    )
    upper_delta = upper - reference_u
    lower_delta = lower - reference_l
    total_delta = upper_delta + lower_delta
    component_correction_rms = float(np.sqrt(np.mean(np.concatenate((
        np.square(upper_delta).reshape(-1),
        np.square(lower_delta).reshape(-1),
    )))))
    return NearestFeasibleComponentProjectionResult(
        upper=upper.copy(),
        lower=lower.copy(),
        status=(
            "nearest_feasible_component_projection_complete"
            if feasible else "nearest_feasible_component_projection_invalid"
        ),
        feasible=feasible,
        include_total_action_box=bool(include_total_action_box),
        upper_power=float(metrics["upper_power"]),
        lower_power=float(metrics["lower_power"]),
        upper_power_budget=upper_power_budget,
        lower_power_budget=lower_power_budget,
        component_correction_rms=component_correction_rms,
        total_action_correction_rms=float(np.sqrt(np.mean(np.square(
            total_delta
        )))),
        upper_correction_rms=float(np.sqrt(np.mean(np.square(upper_delta)))),
        lower_correction_rms=float(np.sqrt(np.mean(np.square(lower_delta)))),
        upper_bound_violation_max=float(
            metrics["upper_bound_violation_max"]
        ),
        lower_bound_violation_max=float(
            metrics["lower_bound_violation_max"]
        ),
        total_action_bound_violation_max=float(
            metrics["total_action_bound_violation_max"]
        ),
        convergence_residual_max=residual,
        iteration_count=iteration_count,
    )


def _constraint_metrics(
    upper: np.ndarray,
    lower: np.ndarray,
    *,
    upper_operator: np.ndarray,
    lower_operator: np.ndarray,
    upper_limit: float,
    lower_limit: float,
    total_limit: float,
    include_total_action_box: bool,
) -> dict[str, float]:
    return {
        "upper_power": float(np.mean(np.square(upper_operator @ upper))),
        "lower_power": float(np.mean(np.square(lower_operator @ lower))),
        "upper_bound_violation_max": float(np.max(np.maximum(
            np.abs(upper) - float(upper_limit), 0.0
        ))),
        "lower_bound_violation_max": float(np.max(np.maximum(
            np.abs(lower) - float(lower_limit), 0.0
        ))),
        "total_action_bound_violation_max": (
            float(np.max(np.maximum(
                np.abs(upper + lower) - float(total_limit), 0.0
            )))
            if include_total_action_box else 0.0
        ),
    }


def _action_trace(values: Any, *, role: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{role} action trace must be a nonempty matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{role} action trace must be finite")
    return array


def _positive_finite(value: float, role: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{role} must be positive and finite")
    return numeric
