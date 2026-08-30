"""Causal receding-horizon responsibility allocation.

The planner operates on the already selected total action. It forecasts only
from the realized total-action prefix, solves a bounded finite-horizon
HPF/LPF allocation, executes the first split, and replans at the next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FORECAST_MODES = ("hold", "damped_velocity")


def future_rolling_mean_system(
    past: Any,
    *,
    horizon: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``matrix, offset`` for future causal rolling means.

    For a future matrix ``values`` with shape ``(horizon, dimension)``, the
    exact rolling means over the fixed past and future prefix are
    ``matrix @ values + offset``.
    """

    history = np.asarray(past, dtype=np.float64)
    if history.ndim != 2:
        raise ValueError("rolling-mean past must be a matrix")
    count = int(horizon)
    width = int(window)
    if count < 1 or width < 1:
        raise ValueError("rolling-mean horizon and window must be positive")
    if not np.all(np.isfinite(history)):
        raise ValueError("rolling-mean past must be finite")
    if history.shape[0] > width - 1:
        history = history[-(width - 1):]

    past_count = int(history.shape[0])
    matrix = np.zeros((count, count), dtype=np.float64)
    offset = np.zeros((count, history.shape[1]), dtype=np.float64)
    for future_index in range(count):
        end = past_count + future_index + 1
        start = max(0, end - width)
        denominator = float(end - start)
        past_start = min(max(start, 0), past_count)
        if past_start < past_count:
            offset[future_index] = (
                np.sum(history[past_start:past_count], axis=0) / denominator
            )
        future_start = max(0, start - past_count)
        matrix[
            future_index,
            future_start:future_index + 1,
        ] = 1.0 / denominator
    return matrix, offset


@dataclass
class CausalRecedingHorizonResponsibilityPlanner:
    """Allocate upper/lower responsibility with causal budget accounting."""

    upper_window: int = 8
    lower_window: int = 32
    upper_rms_budget: float = 0.075
    lower_rms_budget: float = 0.0475
    planning_horizon: int = 32
    forecast_mode: str = "damped_velocity"
    velocity_alpha: float = 0.25
    velocity_decay: float = 0.75
    coordinate_sweeps: int = 48
    multiplier_bisection_steps: int = 12
    power_tolerance: float = 1e-8
    use_budget_ledger: bool = True
    enforce_prefix_upper_budget: bool = True

    def __post_init__(self) -> None:
        self.upper_window = int(self.upper_window)
        self.lower_window = int(self.lower_window)
        self.upper_rms_budget = float(self.upper_rms_budget)
        self.lower_rms_budget = float(self.lower_rms_budget)
        self.planning_horizon = int(self.planning_horizon)
        self.forecast_mode = str(self.forecast_mode)
        self.velocity_alpha = float(self.velocity_alpha)
        self.velocity_decay = float(self.velocity_decay)
        self.coordinate_sweeps = int(self.coordinate_sweeps)
        self.multiplier_bisection_steps = int(
            self.multiplier_bisection_steps
        )
        self.power_tolerance = float(self.power_tolerance)
        self.use_budget_ledger = bool(self.use_budget_ledger)
        self.enforce_prefix_upper_budget = bool(
            self.enforce_prefix_upper_budget
        )
        if self.upper_window < 2 or self.lower_window < 2:
            raise ValueError("receding-horizon windows must be at least two")
        if self.planning_horizon < 2:
            raise ValueError(
                "receding-horizon planning_horizon must be at least two"
            )
        if self.forecast_mode not in FORECAST_MODES:
            raise ValueError(
                f"unknown total-action forecast mode: {self.forecast_mode}"
            )
        if (
            not np.isfinite(self.upper_rms_budget)
            or self.upper_rms_budget <= 0.0
            or not np.isfinite(self.lower_rms_budget)
            or self.lower_rms_budget <= 0.0
        ):
            raise ValueError("receding-horizon budgets must be positive")
        if (
            not np.isfinite(self.velocity_alpha)
            or not 0.0 < self.velocity_alpha <= 1.0
            or not np.isfinite(self.velocity_decay)
            or not 0.0 <= self.velocity_decay <= 1.0
        ):
            raise ValueError("velocity parameters must lie in their unit ranges")
        if self.coordinate_sweeps < 1:
            raise ValueError("coordinate_sweeps must be positive")
        if self.multiplier_bisection_steps < 1:
            raise ValueError("multiplier_bisection_steps must be positive")
        if not np.isfinite(self.power_tolerance) or self.power_tolerance <= 0.0:
            raise ValueError("power_tolerance must be positive")
        self._dimension = 0
        self._upper_history: list[np.ndarray] = []
        self._lower_history: list[np.ndarray] = []
        self._previous_total = np.zeros(0, dtype=np.float64)
        self._previous_forecast_next = np.zeros(0, dtype=np.float64)
        self._velocity = np.zeros(0, dtype=np.float64)
        self._previous_plan = np.zeros((self.planning_horizon, 0))
        self._step_count = 0
        self._upper_energy = 0.0
        self._lower_energy = 0.0

    def reset(self, action_dim: int) -> None:
        dimension = int(action_dim)
        if dimension < 1:
            raise ValueError("receding-horizon action_dim must be positive")
        self._dimension = dimension
        self._upper_history = []
        self._lower_history = []
        self._previous_total = np.zeros(dimension, dtype=np.float64)
        self._previous_forecast_next = np.zeros(
            dimension, dtype=np.float64
        )
        self._velocity = np.zeros(dimension, dtype=np.float64)
        self._previous_plan = np.zeros(
            (self.planning_horizon, dimension), dtype=np.float64
        )
        self._step_count = 0
        self._upper_energy = 0.0
        self._lower_energy = 0.0

    @property
    def context(self) -> np.ndarray:
        self._require_reset()
        if self._upper_history:
            return self._upper_history[-1].astype(np.float32, copy=True)
        return np.zeros(self._dimension, dtype=np.float32)

    @property
    def policy_context(self) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
        """Match the complete finite-memory state shape of the v17.4 router."""

        self._require_reset()
        upper = self._padded_history(
            self._upper_history, self.upper_window - 1
        )
        lower = self._padded_history(
            self._lower_history, self.lower_window - 1
        )
        return (
            tuple(
                row.astype(np.float32, copy=True)
                for row in np.concatenate((upper, lower), axis=0)
            ),
            (
                float(len(self._upper_history) / (self.upper_window - 1)),
                float(len(self._lower_history) / (self.lower_window - 1)),
            ),
        )

    def split(
        self,
        total_action: Any,
        *,
        upper_limit: float = 1.0,
        lower_limit: float = 1.0,
    ) -> dict[str, Any]:
        """Return one causal split and its finite-horizon feasibility floor."""

        self._require_reset()
        total = np.asarray(total_action, dtype=np.float64).reshape(-1)
        upper_bound = float(upper_limit)
        lower_bound = float(lower_limit)
        if total.shape != (self._dimension,) or not np.all(np.isfinite(total)):
            raise ValueError("total action must be finite and aligned")
        if (
            not np.isfinite(upper_bound)
            or upper_bound <= 0.0
            or not np.isfinite(lower_bound)
            or lower_bound <= 0.0
        ):
            raise ValueError("component limits must be positive")
        if np.max(np.abs(total)) > upper_bound + lower_bound + 1e-10:
            raise ValueError("total action exceeds the reconstruction box")

        if self._step_count:
            delta = total - self._previous_total
            self._velocity += self.velocity_alpha * (delta - self._velocity)
            forecast_error = float(np.sqrt(np.mean(np.square(
                total - self._previous_forecast_next
            ))))
        else:
            forecast_error = 0.0
        forecast = self._forecast_total(
            total, total_limit=upper_bound + lower_bound
        )
        physical_low = np.maximum(-upper_bound, forecast - lower_bound)
        physical_high = np.minimum(upper_bound, forecast + lower_bound)
        if np.any(physical_low > physical_high + 1e-12):
            raise RuntimeError("forecast has no bounded responsibility split")
        physical_high = np.maximum(physical_high, physical_low)

        past_upper = self._history_matrix(
            self._upper_history, self.upper_window - 1
        )
        past_lower = self._history_matrix(
            self._lower_history, self.lower_window - 1
        )
        upper_rolling, upper_offset_mean = future_rolling_mean_system(
            past_upper,
            horizon=self.planning_horizon,
            window=self.upper_window,
        )
        lower_rolling, lower_offset = future_rolling_mean_system(
            past_lower,
            horizon=self.planning_horizon,
            window=self.lower_window,
        )
        upper_operator = np.eye(self.planning_horizon) - upper_rolling
        upper_offset = -upper_offset_mean
        lower_operator = -lower_rolling
        lower_offset = lower_rolling @ forecast + lower_offset
        upper_budget_power = self._future_power_budget(
            rms_budget=self.upper_rms_budget,
            accumulated_energy=self._upper_energy,
        )
        lower_budget_power = self._future_power_budget(
            rms_budget=self.lower_rms_budget,
            accumulated_energy=self._lower_energy,
        )
        warm = np.concatenate(
            (self._previous_plan[1:], self._previous_plan[-1:]), axis=0
        )
        warm = np.clip(warm, physical_low, physical_high)
        solution = self._solve_plan(
            upper_operator=upper_operator,
            upper_offset=upper_offset,
            lower_operator=lower_operator,
            lower_offset=lower_offset,
            low_bound=physical_low,
            high_bound=physical_high,
            upper_budget_power=upper_budget_power,
            lower_budget_power=lower_budget_power,
            warm=warm,
        )
        plan = np.asarray(solution["upper"], dtype=np.float64).copy()
        prefix_projection = self._project_prefix_upper_budget(
            planned_upper=plan[0],
            physical_low=physical_low[0],
            physical_high=physical_high[0],
            current_operator_coefficient=float(upper_operator[0, 0]),
            current_offset=upper_offset[0],
        )
        plan[0] = prefix_projection["upper"]
        executed_upper_residuals = upper_operator @ plan + upper_offset
        executed_lower_residuals = lower_operator @ plan + lower_offset
        executed_plan_upper_power = float(np.mean(np.square(
            executed_upper_residuals
        )))
        executed_plan_lower_power = float(np.mean(np.square(
            executed_lower_residuals
        )))
        upper = plan[0].copy()
        lower = total - upper
        if (
            np.max(np.abs(upper)) > upper_bound + 1e-10
            or np.max(np.abs(lower)) > lower_bound + 1e-10
        ):
            raise RuntimeError("planner returned an out-of-bounds component")

        realized_upper_residual = (
            upper_operator[0] @ plan + upper_offset[0]
        )
        realized_lower_residual = (
            lower_operator[0] @ plan + lower_offset[0]
        )
        self._upper_energy += float(np.sum(np.square(
            realized_upper_residual
        )))
        self._lower_energy += float(np.sum(np.square(
            realized_lower_residual
        )))
        self._upper_history.append(upper.copy())
        self._lower_history.append(lower.copy())
        self._upper_history = self._upper_history[-(self.upper_window - 1):]
        self._lower_history = self._lower_history[-(self.lower_window - 1):]
        self._previous_total = total.copy()
        self._previous_forecast_next = forecast[1].copy()
        self._previous_plan = plan.copy()
        self._step_count += 1

        lower_excess = max(
            float(solution["lower_power"]) - lower_budget_power, 0.0
        )
        lower_ratio_excess_squared = max(
            np.sqrt(
                float(solution["lower_power"])
                / (self.lower_rms_budget ** 2)
            ) - 1.0,
            0.0,
        ) ** 2
        return {
            "upper": upper.astype(np.float32, copy=True),
            "lower": lower.astype(np.float32, copy=True),
            "total": total.astype(np.float32, copy=True),
            "upper_plan": plan.astype(np.float32, copy=True),
            "total_forecast": forecast.astype(np.float32, copy=True),
            "status": str(solution["status"]),
            "joint_feasible_forecast": bool(solution["joint_feasible"]),
            "upper_constraint_feasible_forecast": bool(
                solution["upper_constraint_feasible"]
            ),
            "upper_power_forecast": float(solution["upper_power"]),
            "lower_power_at_upper_floor_forecast": float(
                solution["lower_power"]
            ),
            "minimum_upper_power_forecast": float(
                solution["minimum_upper_power"]
            ),
            "unconstrained_minimum_lower_power_forecast": float(
                solution["unconstrained_minimum_lower_power"]
            ),
            "upper_budget_power_forecast": float(upper_budget_power),
            "lower_budget_power_forecast": float(lower_budget_power),
            "actor_floor_power_excess": float(lower_excess),
            "actor_floor_ratio_excess_squared": float(
                lower_ratio_excess_squared
            ),
            "executed_plan_upper_power_forecast": (
                executed_plan_upper_power
            ),
            "executed_plan_lower_power_forecast": (
                executed_plan_lower_power
            ),
            "prefix_upper_budget_feasible": bool(
                prefix_projection["feasible"]
            ),
            "prefix_upper_projection_rms": float(
                prefix_projection["projection_rms"]
            ),
            "prefix_unavoidable_upper_violation_rms": float(
                prefix_projection["unavoidable_violation_rms"]
            ),
            "lagrange_multiplier": (
                None
                if solution["lagrange_multiplier"] is None
                else float(solution["lagrange_multiplier"])
            ),
            "forecast_error_rms": forecast_error,
            "realized_upper_residual": np.asarray(
                realized_upper_residual, dtype=np.float32
            ),
            "realized_lower_residual": np.asarray(
                realized_lower_residual, dtype=np.float32
            ),
            "reconstruction_error": (
                upper + lower - total
            ).astype(np.float64, copy=True),
        }

    def _project_prefix_upper_budget(
        self,
        *,
        planned_upper: np.ndarray,
        physical_low: np.ndarray,
        physical_high: np.ndarray,
        current_operator_coefficient: float,
        current_offset: np.ndarray,
    ) -> dict[str, Any]:
        if not self.enforce_prefix_upper_budget:
            return {
                "upper": planned_upper.copy(),
                "feasible": True,
                "projection_rms": 0.0,
                "unavoidable_violation_rms": 0.0,
            }
        coefficient = float(current_operator_coefficient)
        target = coefficient * planned_upper + current_offset
        allowed_energy = max(
            (self._step_count + 1)
            * self._dimension
            * self.upper_rms_budget ** 2
            - self._upper_energy,
            0.0,
        )
        if abs(coefficient) <= 1e-15:
            residual = np.asarray(current_offset, dtype=np.float64)
            feasible = bool(
                float(np.sum(np.square(residual)))
                <= allowed_energy + self.power_tolerance
            )
            unavoidable = np.sqrt(max(
                float(np.sum(np.square(residual))) - allowed_energy,
                0.0,
            ) / self._dimension)
            return {
                "upper": planned_upper.copy(),
                "feasible": feasible,
                "projection_rms": 0.0,
                "unavoidable_violation_rms": float(unavoidable),
            }

        residual_low = coefficient * physical_low + current_offset
        residual_high = coefficient * physical_high + current_offset
        low = np.minimum(residual_low, residual_high)
        high = np.maximum(residual_low, residual_high)
        minimum = np.clip(np.zeros_like(target), low, high)
        minimum_energy = float(np.sum(np.square(minimum)))
        feasible = bool(
            minimum_energy <= allowed_energy + self.power_tolerance
        )
        if not feasible:
            selected = minimum
        else:
            selected = np.clip(target, low, high)
            if float(np.sum(np.square(selected))) > allowed_energy:
                multiplier_low = 0.0
                multiplier_high = 1.0
                while float(np.sum(np.square(np.clip(
                    target / (1.0 + multiplier_high), low, high
                )))) > allowed_energy:
                    multiplier_high *= 10.0
                for _ in range(48):
                    midpoint = 0.5 * (multiplier_low + multiplier_high)
                    trial = np.clip(target / (1.0 + midpoint), low, high)
                    if float(np.sum(np.square(trial))) <= allowed_energy:
                        multiplier_high = midpoint
                        selected = trial
                    else:
                        multiplier_low = midpoint
        upper = np.clip(
            (selected - current_offset) / coefficient,
            physical_low,
            physical_high,
        )
        realized = coefficient * upper + current_offset
        unavoidable = np.sqrt(max(
            float(np.sum(np.square(realized))) - allowed_energy,
            0.0,
        ) / self._dimension)
        return {
            "upper": upper,
            "feasible": feasible,
            "projection_rms": float(np.sqrt(np.mean(np.square(
                upper - planned_upper
            )))),
            "unavoidable_violation_rms": float(unavoidable),
        }

    def _forecast_total(
        self,
        current: np.ndarray,
        *,
        total_limit: float,
    ) -> np.ndarray:
        forecast = np.repeat(
            current.reshape(1, -1), self.planning_horizon, axis=0
        )
        if self.forecast_mode == "hold":
            return forecast
        value = current.copy()
        velocity = self._velocity.copy()
        for index in range(1, self.planning_horizon):
            velocity *= self.velocity_decay
            value = np.clip(value + velocity, -total_limit, total_limit)
            forecast[index] = value
        return forecast

    def _future_power_budget(
        self,
        *,
        rms_budget: float,
        accumulated_energy: float,
    ) -> float:
        base = float(rms_budget) ** 2
        if not self.use_budget_ledger:
            return base
        future_samples = self.planning_horizon * self._dimension
        target_samples = (
            self._step_count + self.planning_horizon
        ) * self._dimension
        return max(
            (target_samples * base - accumulated_energy) / future_samples,
            0.0,
        )

    def _solve_plan(
        self,
        *,
        upper_operator: np.ndarray,
        upper_offset: np.ndarray,
        lower_operator: np.ndarray,
        lower_offset: np.ndarray,
        low_bound: np.ndarray,
        high_bound: np.ndarray,
        upper_budget_power: float,
        lower_budget_power: float,
        warm: np.ndarray,
    ) -> dict[str, Any]:
        def weighted(weight: float, initial: np.ndarray) -> dict[str, Any]:
            multiplier = float(weight)
            quadratic = (
                lower_operator.T @ lower_operator
                + multiplier * (upper_operator.T @ upper_operator)
            )
            linear = (
                lower_operator.T @ lower_offset
                + multiplier * (upper_operator.T @ upper_offset)
            )
            values = self._coordinate_solve(
                quadratic,
                linear,
                low_bound=low_bound,
                high_bound=high_bound,
                initial=initial,
            )
            upper_residual = upper_operator @ values + upper_offset
            lower_residual = lower_operator @ values + lower_offset
            return {
                "upper": values,
                "upper_power": float(np.mean(np.square(upper_residual))),
                "lower_power": float(np.mean(np.square(lower_residual))),
                "lagrange_multiplier": multiplier,
            }

        lower_minimum = weighted(0.0, warm)
        upper_quadratic = upper_operator.T @ upper_operator
        upper_linear = upper_operator.T @ upper_offset
        upper_values = self._coordinate_solve(
            upper_quadratic,
            upper_linear,
            low_bound=low_bound,
            high_bound=high_bound,
            initial=lower_minimum["upper"],
        )
        upper_residual = upper_operator @ upper_values + upper_offset
        lower_residual = lower_operator @ upper_values + lower_offset
        upper_minimum = {
            "upper": upper_values,
            "upper_power": float(np.mean(np.square(upper_residual))),
            "lower_power": float(np.mean(np.square(lower_residual))),
            "lagrange_multiplier": None,
        }
        tolerance = self.power_tolerance
        upper_feasible = bool(
            upper_minimum["upper_power"] <= upper_budget_power + tolerance
        )
        if lower_minimum["upper_power"] <= upper_budget_power + tolerance:
            candidate = lower_minimum
        elif not upper_feasible:
            candidate = upper_minimum
        else:
            multiplier_low = 0.0
            multiplier_high = 1.0
            candidate = weighted(
                multiplier_high, lower_minimum["upper"]
            )
            while (
                candidate["upper_power"] > upper_budget_power + tolerance
                and multiplier_high < 1e12
            ):
                multiplier_low = multiplier_high
                multiplier_high *= 10.0
                candidate = weighted(multiplier_high, candidate["upper"])
            if candidate["upper_power"] > upper_budget_power + tolerance:
                candidate = upper_minimum
            else:
                feasible_candidate = candidate
                for _ in range(self.multiplier_bisection_steps):
                    midpoint = 0.5 * (multiplier_low + multiplier_high)
                    trial = weighted(midpoint, feasible_candidate["upper"])
                    if trial["upper_power"] <= upper_budget_power + tolerance:
                        multiplier_high = midpoint
                        feasible_candidate = trial
                    else:
                        multiplier_low = midpoint
                candidate = feasible_candidate

        joint_feasible = bool(
            upper_feasible
            and candidate["upper_power"] <= upper_budget_power + tolerance
            and candidate["lower_power"] <= lower_budget_power + tolerance
        )
        if not upper_feasible:
            status = "upper_budget_infeasible_forecast"
        elif joint_feasible:
            status = "joint_frequency_budgets_feasible_forecast"
        else:
            status = "lower_budget_infeasible_at_upper_floor_forecast"
        return {
            **candidate,
            "status": status,
            "joint_feasible": joint_feasible,
            "upper_constraint_feasible": upper_feasible,
            "minimum_upper_power": float(upper_minimum["upper_power"]),
            "unconstrained_minimum_lower_power": float(
                lower_minimum["lower_power"]
            ),
        }

    def _coordinate_solve(
        self,
        quadratic: np.ndarray,
        linear: np.ndarray,
        *,
        low_bound: np.ndarray,
        high_bound: np.ndarray,
        initial: np.ndarray,
    ) -> np.ndarray:
        scale = max(float(np.max(np.diag(quadratic))), 1.0)
        tie_break = 1e-12 * scale
        system = quadratic + tie_break * np.eye(quadratic.shape[0])
        target_linear = linear - tie_break * initial
        values = np.clip(initial.copy(), low_bound, high_bound)
        for _ in range(self.coordinate_sweeps):
            largest_change = 0.0
            for index in range(self.planning_horizon):
                gradient = system[index] @ values + target_linear[index]
                updated = np.clip(
                    values[index] - gradient / system[index, index],
                    low_bound[index],
                    high_bound[index],
                )
                largest_change = max(
                    largest_change,
                    float(np.max(np.abs(updated - values[index]))),
                )
                values[index] = updated
            if largest_change <= 1e-10:
                break
        return values

    def _history_matrix(
        self,
        history: list[np.ndarray],
        maximum_rows: int,
    ) -> np.ndarray:
        rows = history[-int(maximum_rows):]
        if not rows:
            return np.empty((0, self._dimension), dtype=np.float64)
        return np.stack(rows, axis=0).astype(np.float64, copy=False)

    def _padded_history(
        self,
        history: list[np.ndarray],
        maximum_rows: int,
    ) -> np.ndarray:
        target = int(maximum_rows)
        rows = history[-target:]
        padding = np.zeros(
            (target - len(rows), self._dimension), dtype=np.float64
        )
        if not rows:
            return padding
        return np.concatenate((padding, np.stack(rows, axis=0)), axis=0)

    def _require_reset(self) -> None:
        if self._dimension < 1:
            raise RuntimeError(
                "receding-horizon responsibility planner must be reset"
            )
