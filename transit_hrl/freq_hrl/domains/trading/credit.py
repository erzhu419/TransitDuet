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


@dataclass(frozen=True)
class TradingTacticalCreditBreakdown:
    """Exact three-way task and training credit for the v5 policy streams."""

    task_reward: float
    upper_task_credit: float
    tracking_task_credit: float
    hf_task_credit: float
    plan_return: float
    tracking_deviation_return: float
    hf_overlay_return: float
    tracking_transaction_cost: float
    tracking_inventory_drift_cost: float
    tracking_drawdown_cost: float
    hf_incremental_transaction_cost: float
    hf_incremental_inventory_drift_cost: float
    hf_incremental_drawdown_cost: float
    upper_leakage_cost: float
    tracking_leakage_cost: float
    hf_leakage_cost: float
    plan_smoothness_cost: float
    upper_training_credit: float
    tracking_training_credit: float
    hf_training_credit: float
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

    def assign_tactical(
        self,
        info: Mapping[str, Any],
        active_plan: Sequence[float] | np.ndarray,
        *,
        upper_leakage_cost: float = 0.0,
        tracking_leakage_cost: float = 0.0,
        hf_leakage_cost: float = 0.0,
        plan_smoothness_cost: float = 0.0,
    ) -> TradingTacticalCreditBreakdown:
        """Assign exact marginal credit to plan, tracking, and HF policies.

        The tracking-only counterfactual is produced by the environment from
        the same pre-step state and action. This avoids estimating the HF
        policy's marginal contribution with a learned critic.
        """

        returns = np.asarray(info["asset_returns"], dtype=np.float64).reshape(-1)
        tracking_market_position = np.asarray(
            info["pre_trade_position"]
            if info.get("mark_to_market_timing") == "pre_trade"
            else info["tracking_only_position"],
            dtype=np.float64,
        ).reshape(-1)
        plan = np.asarray(active_plan, dtype=np.float64).reshape(-1)
        if (
            returns.size == 0
            or tracking_market_position.size != returns.size
            or plan.size != returns.size
        ):
            raise ValueError(
                "asset_returns, tracking market position, and active_plan "
                "must share a non-empty shape"
            )
        if not (
            np.all(np.isfinite(returns))
            and np.all(np.isfinite(tracking_market_position))
            and np.all(np.isfinite(plan))
        ):
            raise ValueError("tactical credit inputs must be finite")

        task_reward = float(info["task_reward"])
        if not np.isfinite(task_reward):
            raise ValueError("task_reward must be finite")
        plan_return = float(np.dot(plan, returns))
        tracking_deviation_return = float(
            np.dot(tracking_market_position - plan, returns)
        )
        tracking_transaction_cost = self._nonnegative_cost(
            info, "tracking_transaction_cost"
        )
        tracking_inventory_drift_cost = self._nonnegative_cost(
            info, "tracking_inventory_drift_cost"
        )
        tracking_drawdown_cost = self._nonnegative_cost(
            info, "tracking_drawdown_cost"
        )

        def finite_increment(key: str) -> float:
            value = float(info[key])
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
            return value

        hf_overlay_return = finite_increment("hf_overlay_return")
        hf_incremental_transaction_cost = finite_increment(
            "hf_overlay_incremental_transaction_cost"
        )
        hf_incremental_inventory_drift_cost = finite_increment(
            "hf_overlay_incremental_inventory_drift_cost"
        )
        hf_incremental_drawdown_cost = finite_increment(
            "hf_overlay_incremental_drawdown_cost"
        )

        upper_task_credit = float(plan_return - tracking_drawdown_cost)
        tracking_task_credit = float(
            tracking_deviation_return
            - tracking_transaction_cost
            - tracking_inventory_drift_cost
        )
        hf_task_credit = float(
            hf_overlay_return
            - hf_incremental_transaction_cost
            - hf_incremental_inventory_drift_cost
            - hf_incremental_drawdown_cost
        )
        reconstructed = float(
            upper_task_credit + tracking_task_credit + hf_task_credit
        )
        reconstruction_error = float(reconstructed - task_reward)
        if abs(reconstruction_error) > self.atol:
            raise ValueError(
                "tactical frequency credit does not reconstruct task reward: "
                f"error={reconstruction_error:.12g}"
            )

        upper_leakage = self._regularization(
            upper_leakage_cost, "upper_leakage_cost"
        )
        tracking_leakage = self._regularization(
            tracking_leakage_cost, "tracking_leakage_cost"
        )
        hf_leakage = self._regularization(hf_leakage_cost, "hf_leakage_cost")
        smoothness = self._regularization(
            plan_smoothness_cost, "plan_smoothness_cost"
        )
        return TradingTacticalCreditBreakdown(
            task_reward=task_reward,
            upper_task_credit=upper_task_credit,
            tracking_task_credit=tracking_task_credit,
            hf_task_credit=hf_task_credit,
            plan_return=plan_return,
            tracking_deviation_return=tracking_deviation_return,
            hf_overlay_return=hf_overlay_return,
            tracking_transaction_cost=tracking_transaction_cost,
            tracking_inventory_drift_cost=tracking_inventory_drift_cost,
            tracking_drawdown_cost=tracking_drawdown_cost,
            hf_incremental_transaction_cost=hf_incremental_transaction_cost,
            hf_incremental_inventory_drift_cost=(
                hf_incremental_inventory_drift_cost
            ),
            hf_incremental_drawdown_cost=hf_incremental_drawdown_cost,
            upper_leakage_cost=upper_leakage,
            tracking_leakage_cost=tracking_leakage,
            hf_leakage_cost=hf_leakage,
            plan_smoothness_cost=smoothness,
            upper_training_credit=float(
                upper_task_credit - upper_leakage - smoothness
            ),
            tracking_training_credit=float(
                tracking_task_credit - tracking_leakage
            ),
            hf_training_credit=float(hf_task_credit - hf_leakage),
            reconstructed_task_reward=reconstructed,
            task_reconstruction_error=reconstruction_error,
        )
