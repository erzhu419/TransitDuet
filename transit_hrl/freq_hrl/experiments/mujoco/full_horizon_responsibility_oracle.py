"""Convex full-horizon oracle for bounded frequency responsibility splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class _WeightedSolution:
    upper: np.ndarray
    upper_power: float
    lower_power: float
    multiplier: float
    optimality_max: float
    kkt_residual_inf: float
    iteration_count: int


@dataclass(frozen=True)
class FullHorizonResponsibilityOracleResult:
    """Globally optimized split for one frozen total-action trajectory."""

    upper: np.ndarray
    lower: np.ndarray
    status: str
    joint_feasible: bool
    upper_constraint_feasible: bool
    upper_power: float
    lower_power: float
    minimum_upper_power: float
    unconstrained_minimum_lower_power: float
    upper_power_budget: float
    lower_power_budget: float
    lagrange_multiplier: float | None
    solver_optimality_max: float
    kkt_residual_inf: float
    bound_violation_max: float
    reconstruction_error_max: float
    weighted_solve_count: int
    solver_iteration_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "joint_feasible": bool(self.joint_feasible),
            "upper_constraint_feasible": bool(
                self.upper_constraint_feasible
            ),
            "upper_power": float(self.upper_power),
            "lower_power": float(self.lower_power),
            "minimum_upper_power": float(self.minimum_upper_power),
            "unconstrained_minimum_lower_power": float(
                self.unconstrained_minimum_lower_power
            ),
            "upper_power_budget": float(self.upper_power_budget),
            "lower_power_budget": float(self.lower_power_budget),
            "upper_power_budget_excess": float(max(
                self.upper_power - self.upper_power_budget, 0.0
            )),
            "lower_power_budget_excess": float(max(
                self.lower_power - self.lower_power_budget, 0.0
            )),
            "lagrange_multiplier": (
                None
                if self.lagrange_multiplier is None
                else float(self.lagrange_multiplier)
            ),
            "solver_optimality_max": float(self.solver_optimality_max),
            "kkt_residual_inf": float(self.kkt_residual_inf),
            "bound_violation_max": float(self.bound_violation_max),
            "reconstruction_error_max": float(
                self.reconstruction_error_max
            ),
            "weighted_solve_count": int(self.weighted_solve_count),
            "solver_iteration_count": int(self.solver_iteration_count),
            "trajectory_length": int(self.upper.shape[0]),
            "action_dimension": int(self.upper.shape[1]),
            "oracle_contract": (
                "frozen_total_action_box_constrained_full_horizon_"
                "lower_lpf32_minimization_subject_to_upper_hpf8_budget_v1"
            ),
        }


def causal_rolling_operator(length: int, window: int) -> np.ndarray:
    """Return the exact linear operator used by ``causal_rolling_mean``."""

    count = int(length)
    width = int(window)
    if count < 1 or width < 1:
        raise ValueError("rolling operator dimensions must be positive")
    operator = np.zeros((count, count), dtype=np.float64)
    for index in range(count):
        start = max(0, index - width + 1)
        operator[index, start:index + 1] = 1.0 / float(index - start + 1)
    return operator


def responsibility_frequency_powers(
    total_action: Any,
    upper_action: Any,
    *,
    upper_window: int = 8,
    lower_window: int = 32,
) -> tuple[float, float]:
    total = _action_trace(total_action, role="total")
    upper = _action_trace(upper_action, role="upper")
    if upper.shape != total.shape:
        raise ValueError("upper and total action traces must align")
    rolling_upper = causal_rolling_operator(total.shape[0], upper_window)
    rolling_lower = causal_rolling_operator(total.shape[0], lower_window)
    high = (np.eye(total.shape[0]) - rolling_upper) @ upper
    low = rolling_lower @ (total - upper)
    return float(np.mean(np.square(high))), float(np.mean(np.square(low)))


def solve_full_horizon_responsibility_oracle(
    total_action: Any,
    *,
    upper_rms_budget: float,
    lower_rms_budget: float,
    upper_action_limit: float = 1.0,
    lower_action_limit: float = 1.0,
    upper_window: int = 8,
    lower_window: int = 32,
    solver_tolerance: float = 1e-9,
    power_tolerance: float = 1e-8,
    multiplier_bisection_steps: int = 18,
) -> FullHorizonResponsibilityOracleResult:
    """Minimize full-horizon lower LPF power under an upper HPF budget.

    The total action is frozen. Each upper component is box constrained so the
    complementary lower component also remains within its physical action box.
    Convex weighted subproblems are solved exactly enough for an auditable KKT
    and feasibility certificate.
    """

    total = _action_trace(total_action, role="total")
    upper_limit = _positive_finite(upper_action_limit, "upper action limit")
    lower_limit = _positive_finite(lower_action_limit, "lower action limit")
    upper_budget = _positive_finite(upper_rms_budget, "upper RMS budget")
    lower_budget = _positive_finite(lower_rms_budget, "lower RMS budget")
    tolerance = _positive_finite(solver_tolerance, "solver tolerance")
    budget_tolerance = _positive_finite(power_tolerance, "power tolerance")
    bisection_steps = int(multiplier_bisection_steps)
    if bisection_steps < 1:
        raise ValueError("multiplier bisection steps must be positive")
    if np.max(np.abs(total)) > upper_limit + lower_limit + 1e-10:
        raise ValueError("total action exceeds the component reconstruction box")

    length = total.shape[0]
    low_bound = np.maximum(-upper_limit, total - lower_limit)
    high_bound = np.minimum(upper_limit, total + lower_limit)
    if np.any(low_bound > high_bound + 1e-12):
        raise ValueError("total action has no physical responsibility split")
    high_bound = np.maximum(high_bound, low_bound)
    lower_operator = causal_rolling_operator(length, lower_window)
    upper_operator = (
        np.eye(length, dtype=np.float64)
        - causal_rolling_operator(length, upper_window)
    )
    upper_power_budget = upper_budget * upper_budget
    lower_power_budget = lower_budget * lower_budget
    solve_count = 0
    iteration_count = 0

    def weighted(multiplier: float) -> _WeightedSolution:
        nonlocal solve_count, iteration_count
        result = _solve_weighted_split(
            total,
            low_bound=low_bound,
            high_bound=high_bound,
            upper_operator=upper_operator,
            lower_operator=lower_operator,
            multiplier=float(multiplier),
            solver_tolerance=tolerance,
        )
        solve_count += 1
        iteration_count += int(result.iteration_count)
        return result

    lower_minimum = weighted(0.0)
    upper_minimum = _solve_upper_minimum(
        total,
        low_bound=low_bound,
        high_bound=high_bound,
        upper_operator=upper_operator,
        lower_operator=lower_operator,
        solver_tolerance=tolerance,
    )
    solve_count += 1
    iteration_count += int(upper_minimum.iteration_count)
    upper_constraint_feasible = bool(
        upper_minimum.upper_power <= upper_power_budget + budget_tolerance
    )

    if lower_minimum.upper_power <= upper_power_budget + budget_tolerance:
        candidate = lower_minimum
    elif not upper_constraint_feasible:
        candidate = upper_minimum
    else:
        multiplier_low = 0.0
        multiplier_high = 1.0
        candidate = weighted(multiplier_high)
        while (
            candidate.upper_power > upper_power_budget + budget_tolerance
            and multiplier_high < 1e12
        ):
            multiplier_low = multiplier_high
            multiplier_high *= 10.0
            candidate = weighted(multiplier_high)
        if candidate.upper_power > upper_power_budget + budget_tolerance:
            raise RuntimeError(
                "full-horizon oracle could not bracket the upper constraint"
            )
        for _ in range(bisection_steps):
            midpoint = 0.5 * (multiplier_low + multiplier_high)
            trial = weighted(midpoint)
            if trial.upper_power <= upper_power_budget + budget_tolerance:
                multiplier_high = midpoint
                candidate = trial
            else:
                multiplier_low = midpoint

    lower = total - candidate.upper
    upper_power, lower_power = responsibility_frequency_powers(
        total,
        candidate.upper,
        upper_window=upper_window,
        lower_window=lower_window,
    )
    bound_violation = float(max(
        np.max(np.maximum(low_bound - candidate.upper, 0.0)),
        np.max(np.maximum(candidate.upper - high_bound, 0.0)),
        np.max(np.maximum(np.abs(lower) - lower_limit, 0.0)),
    ))
    reconstruction = float(np.max(np.abs(candidate.upper + lower - total)))
    joint_feasible = bool(
        upper_constraint_feasible
        and upper_power <= upper_power_budget + budget_tolerance
        and lower_power <= lower_power_budget + budget_tolerance
    )
    if not upper_constraint_feasible:
        status = "upper_budget_physically_infeasible"
    elif joint_feasible:
        status = "joint_frequency_budgets_feasible"
    else:
        status = "lower_budget_infeasible_at_upper_constrained_floor"
    return FullHorizonResponsibilityOracleResult(
        upper=candidate.upper.copy(),
        lower=lower.copy(),
        status=status,
        joint_feasible=joint_feasible,
        upper_constraint_feasible=upper_constraint_feasible,
        upper_power=upper_power,
        lower_power=lower_power,
        minimum_upper_power=float(upper_minimum.upper_power),
        unconstrained_minimum_lower_power=float(lower_minimum.lower_power),
        upper_power_budget=upper_power_budget,
        lower_power_budget=lower_power_budget,
        lagrange_multiplier=(
            float(candidate.multiplier)
            if np.isfinite(candidate.multiplier) else None
        ),
        solver_optimality_max=float(candidate.optimality_max),
        kkt_residual_inf=float(candidate.kkt_residual_inf),
        bound_violation_max=bound_violation,
        reconstruction_error_max=reconstruction,
        weighted_solve_count=solve_count,
        solver_iteration_count=iteration_count,
    )


def _solve_weighted_split(
    total: np.ndarray,
    *,
    low_bound: np.ndarray,
    high_bound: np.ndarray,
    upper_operator: np.ndarray,
    lower_operator: np.ndarray,
    multiplier: float,
    solver_tolerance: float,
) -> _WeightedSolution:
    weight = float(multiplier)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("oracle multiplier must be finite and non-negative")
    if weight:
        matrix = np.concatenate(
            (lower_operator, np.sqrt(weight) * upper_operator), axis=0
        )
    else:
        matrix = lower_operator
    upper = np.empty_like(total)
    optimality = 0.0
    iterations = 0
    for dimension in range(total.shape[1]):
        target = lower_operator @ total[:, dimension]
        if weight:
            target = np.concatenate((target, np.zeros(total.shape[0])))
        values, dimension_optimality, dimension_iterations = (
            _bounded_least_squares(
                matrix,
                target,
                low_bound[:, dimension],
                high_bound[:, dimension],
                tolerance=solver_tolerance,
            )
        )
        upper[:, dimension] = values
        optimality = max(optimality, dimension_optimality)
        iterations += dimension_iterations
    upper_power, lower_power = _powers_from_operators(
        total, upper, upper_operator, lower_operator
    )
    gradient = 2.0 * (
        lower_operator.T @ (lower_operator @ (upper - total))
        + weight * upper_operator.T @ (upper_operator @ upper)
    )
    kkt = _box_kkt_residual(
        upper, gradient, low_bound=low_bound, high_bound=high_bound
    )
    return _WeightedSolution(
        upper=upper,
        upper_power=upper_power,
        lower_power=lower_power,
        multiplier=weight,
        optimality_max=optimality,
        kkt_residual_inf=kkt,
        iteration_count=iterations,
    )


def _solve_upper_minimum(
    total: np.ndarray,
    *,
    low_bound: np.ndarray,
    high_bound: np.ndarray,
    upper_operator: np.ndarray,
    lower_operator: np.ndarray,
    solver_tolerance: float,
) -> _WeightedSolution:
    upper = np.empty_like(total)
    optimality = 0.0
    iterations = 0
    target = np.zeros(total.shape[0], dtype=np.float64)
    for dimension in range(total.shape[1]):
        values, dimension_optimality, dimension_iterations = (
            _bounded_least_squares(
                upper_operator,
                target,
                low_bound[:, dimension],
                high_bound[:, dimension],
                tolerance=solver_tolerance,
            )
        )
        upper[:, dimension] = values
        optimality = max(optimality, dimension_optimality)
        iterations += dimension_iterations
    upper_power, lower_power = _powers_from_operators(
        total, upper, upper_operator, lower_operator
    )
    gradient = 2.0 * upper_operator.T @ (upper_operator @ upper)
    return _WeightedSolution(
        upper=upper,
        upper_power=upper_power,
        lower_power=lower_power,
        multiplier=float("inf"),
        optimality_max=optimality,
        kkt_residual_inf=_box_kkt_residual(
            upper, gradient, low_bound=low_bound, high_bound=high_bound
        ),
        iteration_count=iterations,
    )


def _bounded_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    low_bound: np.ndarray,
    high_bound: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, float, int]:
    fixed = high_bound - low_bound <= 1e-12
    values = np.empty(low_bound.size, dtype=np.float64)
    values[fixed] = 0.5 * (low_bound[fixed] + high_bound[fixed])
    free = ~fixed
    if not np.any(free):
        return values, 0.0, 0
    adjusted = np.asarray(target, dtype=np.float64).copy()
    if np.any(fixed):
        adjusted -= matrix[:, fixed] @ values[fixed]
    result = lsq_linear(
        matrix[:, free],
        adjusted,
        bounds=(low_bound[free], high_bound[free]),
        method="bvls",
        tol=float(tolerance),
        max_iter=4000,
        verbose=0,
    )
    if not bool(result.success):
        raise RuntimeError(
            "full-horizon bounded least squares failed: "
            f"status={result.status} message={result.message}"
        )
    values[free] = result.x
    return values, float(result.optimality), int(result.nit)


def _powers_from_operators(
    total: np.ndarray,
    upper: np.ndarray,
    upper_operator: np.ndarray,
    lower_operator: np.ndarray,
) -> tuple[float, float]:
    return (
        float(np.mean(np.square(upper_operator @ upper))),
        float(np.mean(np.square(lower_operator @ (total - upper)))),
    )


def _box_kkt_residual(
    values: np.ndarray,
    gradient: np.ndarray,
    *,
    low_bound: np.ndarray,
    high_bound: np.ndarray,
) -> float:
    fixed = high_bound - low_bound <= 1e-10
    at_low = values <= low_bound + 1e-8
    at_high = values >= high_bound - 1e-8
    residual = np.abs(gradient)
    residual = np.where(at_low, np.maximum(-gradient, 0.0), residual)
    residual = np.where(at_high, np.maximum(gradient, 0.0), residual)
    residual = np.where(fixed, 0.0, residual)
    return float(np.max(residual))


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
