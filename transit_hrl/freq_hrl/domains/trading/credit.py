"""Auditable frequency-responsibility credit for portfolio execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TradingCreditBreakdown:
    task_reward: float
    upper_task_credit: float
    lower_task_credit: float
    plan_return: float
    execution_deviation_return: float
    transaction_cost: float
    inventory_drift_cost: float
    drawdown_cost: float
    upper_leakage_cost: float
    lower_leakage_cost: float
    plan_smoothness_cost: float
    upper_training_credit: float
    lower_training_credit: float
    reconstructed_task_reward: float
    task_reconstruction_error: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


class TradingCreditAssigner:
    """Split observed task reward without duplicating or dropping PnL.

    The active upper plan owns the return of the planned position. The lower
    controller owns the return of its realized deviation from that plan, plus
    execution and tracking costs. Their task credits exactly reconstruct the
    environment reward. Frequency leakage and plan/promotion regularizers are
    recorded separately and only then subtracted from the corresponding policy
    credit.
    """

    def __init__(self, *, atol: float = 1e-10) -> None:
        if not np.isfinite(float(atol)) or float(atol) < 0.0:
            raise ValueError("atol must be finite and non-negative")
        self.atol = float(atol)

    @staticmethod
    def _nonnegative_cost(info: Mapping[str, Any], key: str) -> float:
        value = float(info.get(key, 0.0))
        if not np.isfinite(value) or value < -1e-12:
            raise ValueError(f"{key} must be finite and non-negative")
        return max(value, 0.0)

    @staticmethod
    def _regularization(value: float, name: str) -> float:
        out = float(value)
        if not np.isfinite(out) or out < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return out

    def assign(
        self,
        info: Mapping[str, Any],
        active_plan: Sequence[float] | np.ndarray,
        *,
        upper_leakage_cost: float = 0.0,
        lower_leakage_cost: float = 0.0,
        plan_smoothness_cost: float = 0.0,
    ) -> TradingCreditBreakdown:
        returns = np.asarray(info["asset_returns"], dtype=np.float64).reshape(-1)
        market_position = np.asarray(
            info["market_position"], dtype=np.float64
        ).reshape(-1)
        plan = np.asarray(active_plan, dtype=np.float64).reshape(-1)
        if returns.size == 0 or market_position.size != returns.size or plan.size != returns.size:
            raise ValueError(
                "asset_returns, market_position, and active_plan must share a non-empty shape"
            )
        if not (
            np.all(np.isfinite(returns))
            and np.all(np.isfinite(market_position))
            and np.all(np.isfinite(plan))
        ):
            raise ValueError("credit inputs must be finite")

        transaction_cost = self._nonnegative_cost(info, "transaction_cost")
        inventory_drift_cost = self._nonnegative_cost(info, "inventory_drift_cost")
        drawdown_cost = self._nonnegative_cost(info, "drawdown_cost")
        task_reward = float(info["task_reward"])
        if not np.isfinite(task_reward):
            raise ValueError("task_reward must be finite")

        plan_return = float(np.dot(plan, returns))
        execution_deviation_return = float(
            np.dot(market_position - plan, returns)
        )
        upper_task_credit = float(plan_return - drawdown_cost)
        lower_task_credit = float(
            execution_deviation_return - transaction_cost - inventory_drift_cost
        )
        reconstructed = float(upper_task_credit + lower_task_credit)
        reconstruction_error = float(reconstructed - task_reward)
        if abs(reconstruction_error) > self.atol:
            raise ValueError(
                "frequency credit does not reconstruct task reward: "
                f"error={reconstruction_error:.12g}"
            )

        upper_leakage = self._regularization(
            upper_leakage_cost, "upper_leakage_cost"
        )
        lower_leakage = self._regularization(
            lower_leakage_cost, "lower_leakage_cost"
        )
        smoothness = self._regularization(
            plan_smoothness_cost, "plan_smoothness_cost"
        )
        return TradingCreditBreakdown(
            task_reward=task_reward,
            upper_task_credit=upper_task_credit,
            lower_task_credit=lower_task_credit,
            plan_return=plan_return,
            execution_deviation_return=execution_deviation_return,
            transaction_cost=transaction_cost,
            inventory_drift_cost=inventory_drift_cost,
            drawdown_cost=drawdown_cost,
            upper_leakage_cost=upper_leakage,
            lower_leakage_cost=lower_leakage,
            plan_smoothness_cost=smoothness,
            upper_training_credit=float(
                upper_task_credit - upper_leakage - smoothness
            ),
            lower_training_credit=float(lower_task_credit - lower_leakage),
            reconstructed_task_reward=reconstructed,
            task_reconstruction_error=reconstruction_error,
        )
