"""Minimal portfolio-planning plus execution environment.

This is deliberately small: it is a testbed for the Freq-HRL interfaces, not a
production market simulator.  A high-level policy sets target weights; a
low-level policy chooses execution speed toward that target at each bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class PortfolioExecutionConfig:
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 1.0
    volume_impact_bps: float = 0.0
    volume_floor: float = 1e-6
    max_leverage: float = 1.0
    inventory_drift_penalty: float = 0.01
    drawdown_penalty: float = 0.0
    mark_to_market_timing: str = "pre_trade"

    def __post_init__(self) -> None:
        if self.mark_to_market_timing not in {"pre_trade", "post_trade"}:
            raise ValueError(
                "mark_to_market_timing must be 'pre_trade' or 'post_trade'"
            )
        for name in (
            "transaction_cost_bps",
            "slippage_bps",
            "volume_impact_bps",
            "inventory_drift_penalty",
            "drawdown_penalty",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(float(self.volume_floor)) or float(self.volume_floor) <= 0.0:
            raise ValueError("volume_floor must be positive and finite")
        if not np.isfinite(float(self.max_leverage)) or float(self.max_leverage) < 0.0:
            raise ValueError("max_leverage must be finite and non-negative")


class PortfolioExecutionEnv:
    """Toy FreqTradeDuet MVP environment."""

    def __init__(
        self,
        returns: Sequence[Sequence[float]] | np.ndarray,
        volumes: Sequence[Sequence[float]] | np.ndarray | None = None,
        config: PortfolioExecutionConfig | None = None,
    ) -> None:
        self.returns = np.asarray(returns, dtype=np.float64)
        if self.returns.ndim == 1:
            self.returns = self.returns.reshape(-1, 1)
        if self.returns.ndim != 2:
            raise ValueError("returns must be T x N")
        self.volumes = None if volumes is None else np.asarray(volumes, dtype=np.float64)
        if self.volumes is not None and self.volumes.shape != self.returns.shape:
            raise ValueError("volumes must match returns shape")
        self.config = config or PortfolioExecutionConfig()
        self.n_assets = self.returns.shape[1]
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.t = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.position = np.zeros(self.n_assets, dtype=np.float64)
        self.target = np.zeros(self.n_assets, dtype=np.float64)
        self.turnover = 0.0
        self.done = False
        return self.state()

    def state(self) -> dict[str, Any]:
        idx = min(self.t, self.returns.shape[0] - 1)
        volume = (
            np.ones(self.n_assets, dtype=np.float64)
            if self.volumes is None else self.volumes[idx]
        )
        return {
            "t": int(self.t),
            "equity": float(self.equity),
            "position": self.position.copy(),
            "target": self.target.copy(),
            "return": self.returns[idx].copy(),
            "volume": volume.copy(),
            "inventory_gap": self.target - self.position,
        }

    def set_target(self, target_weights: Sequence[float], risk_budget: float = 1.0) -> np.ndarray:
        target = np.asarray(target_weights, dtype=np.float64).reshape(-1)
        if target.size != self.n_assets:
            raise ValueError(f"expected {self.n_assets} target weights, got {target.size}")
        budget = float(np.clip(risk_budget, 0.0, self.config.max_leverage))
        gross = float(np.sum(np.abs(target)))
        if gross > budget and gross > 1e-12:
            target = target * (budget / gross)
        self.target = target
        return self.target.copy()

    def lower_step(
        self,
        execution_speed: float | Sequence[float] | Mapping[str, Any],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            return self.state(), 0.0, True, {"reason": "done"}
        residual_order = np.zeros(self.n_assets, dtype=np.float64)
        if isinstance(execution_speed, Mapping):
            residual_order = np.asarray(
                execution_speed.get("residual_order", residual_order),
                dtype=np.float64,
            ).reshape(-1)
            if residual_order.size != self.n_assets:
                residual_order = np.resize(residual_order, self.n_assets)
            execution_speed = execution_speed.get("execution_speed", 1.0)

        alpha = np.asarray(execution_speed, dtype=np.float64)
        if alpha.ndim == 0:
            alpha = np.ones(self.n_assets, dtype=np.float64) * float(alpha)
        alpha = np.clip(alpha.reshape(-1), 0.0, 1.0)
        if alpha.size != self.n_assets:
            alpha = np.resize(alpha, self.n_assets)

        if not np.all(np.isfinite(residual_order)):
            raise ValueError("residual_order must be finite")

        def capped_position(value: np.ndarray) -> np.ndarray:
            out = np.asarray(value, dtype=np.float64).copy()
            gross_value = float(np.sum(np.abs(out)))
            if gross_value > self.config.max_leverage and gross_value > 1e-12:
                out *= self.config.max_leverage / gross_value
            return out

        old_position = self.position.copy()
        requested_tracking_trade = alpha * (self.target - old_position)
        tracking_only_position = capped_position(
            old_position + requested_tracking_trade
        )
        requested_trade = requested_tracking_trade + residual_order
        self.position = capped_position(old_position + requested_trade)
        realized_trade = self.position - old_position
        realized_tracking_trade = tracking_only_position - old_position
        hf_overlay_position_effect = self.position - tracking_only_position

        ret = self.returns[self.t]
        market_position = (
            old_position
            if self.config.mark_to_market_timing == "pre_trade"
            else self.position.copy()
        )
        portfolio_return = float(np.dot(market_position, ret))
        turnover = float(np.sum(np.abs(realized_trade)))
        self.turnover += turnover
        linear_cost = (
            turnover
            * (self.config.transaction_cost_bps + self.config.slippage_bps)
            / 10000.0
        )
        volume = (
            np.ones(self.n_assets, dtype=np.float64)
            if self.volumes is None
            else np.maximum(
                np.asarray(self.volumes[self.t], dtype=np.float64),
                float(self.config.volume_floor),
            )
        )
        impact_cost = float(
            float(self.config.volume_impact_bps)
            / 10000.0
            * np.sum(np.square(realized_trade) / volume)
        )
        cost = float(linear_cost + impact_cost)
        tracking_turnover = float(np.sum(np.abs(realized_tracking_trade)))
        tracking_linear_cost = (
            tracking_turnover
            * (self.config.transaction_cost_bps + self.config.slippage_bps)
            / 10000.0
        )
        tracking_impact_cost = float(
            float(self.config.volume_impact_bps)
            / 10000.0
            * np.sum(np.square(realized_tracking_trade) / volume)
        )
        tracking_transaction_cost = float(
            tracking_linear_cost + tracking_impact_cost
        )
        tracking_market_position = (
            old_position
            if self.config.mark_to_market_timing == "pre_trade"
            else tracking_only_position
        )
        tracking_portfolio_return = float(
            np.dot(tracking_market_position, ret)
        )
        inventory_drift = float(np.mean((self.position - self.target) ** 2))
        inventory_drift_cost = float(
            self.config.inventory_drift_penalty * inventory_drift
        )
        tracking_inventory_drift = float(
            np.mean((tracking_only_position - self.target) ** 2)
        )
        tracking_inventory_drift_cost = float(
            self.config.inventory_drift_penalty * tracking_inventory_drift
        )
        hf_overlay_return = float(
            portfolio_return - tracking_portfolio_return
        )
        hf_overlay_incremental_transaction_cost = float(
            cost - tracking_transaction_cost
        )
        hf_overlay_incremental_inventory_drift_cost = float(
            inventory_drift_cost - tracking_inventory_drift_cost
        )
        pre_step_equity = float(self.equity)
        pre_step_peak_equity = float(self.peak_equity)
        tracking_equity = pre_step_equity * max(
            0.0,
            1.0 + tracking_portfolio_return - tracking_transaction_cost,
        )
        tracking_peak_equity = max(pre_step_peak_equity, tracking_equity)
        tracking_drawdown = float(
            1.0 - tracking_equity / max(tracking_peak_equity, 1e-12)
        )
        tracking_drawdown_cost = float(
            self.config.drawdown_penalty * tracking_drawdown
        )
        self.equity *= max(0.0, 1.0 + portfolio_return - cost)
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = 1.0 - self.equity / max(self.peak_equity, 1e-12)
        drawdown_cost = float(self.config.drawdown_penalty * drawdown)
        hf_overlay_incremental_drawdown_cost = float(
            drawdown_cost - tracking_drawdown_cost
        )
        tracking_task_reward = float(
            tracking_portfolio_return
            - tracking_transaction_cost
            - tracking_inventory_drift_cost
            - tracking_drawdown_cost
        )
        hf_overlay_task_effect = float(
            hf_overlay_return
            - hf_overlay_incremental_transaction_cost
            - hf_overlay_incremental_inventory_drift_cost
            - hf_overlay_incremental_drawdown_cost
        )
        reward = (
            portfolio_return
            - cost
            - inventory_drift_cost
            - drawdown_cost
        )
        tracking_hf_reconstruction_error = float(
            tracking_task_reward + hf_overlay_task_effect - reward
        )

        self.t += 1
        self.done = self.t >= self.returns.shape[0]
        info = {
            "portfolio_return": portfolio_return,
            "transaction_cost": cost,
            "linear_transaction_cost": float(linear_cost),
            "volume_impact_cost": float(impact_cost),
            "turnover": turnover,
            "trade": realized_trade.copy(),
            "requested_tracking_trade": requested_tracking_trade.copy(),
            "requested_residual_order": residual_order.copy(),
            "tracking_only_position": tracking_only_position.copy(),
            "realized_tracking_trade": realized_tracking_trade.copy(),
            "hf_overlay_position_effect": hf_overlay_position_effect.copy(),
            "tracking_portfolio_return": tracking_portfolio_return,
            "tracking_transaction_cost": tracking_transaction_cost,
            "tracking_inventory_drift_cost": tracking_inventory_drift_cost,
            "tracking_equity": tracking_equity,
            "tracking_peak_equity": tracking_peak_equity,
            "tracking_drawdown": tracking_drawdown,
            "tracking_drawdown_cost": tracking_drawdown_cost,
            "tracking_task_reward": tracking_task_reward,
            "hf_overlay_return": hf_overlay_return,
            "hf_overlay_incremental_transaction_cost": (
                hf_overlay_incremental_transaction_cost
            ),
            "hf_overlay_incremental_inventory_drift_cost": (
                hf_overlay_incremental_inventory_drift_cost
            ),
            "hf_overlay_incremental_drawdown_cost": (
                hf_overlay_incremental_drawdown_cost
            ),
            "hf_overlay_task_effect": hf_overlay_task_effect,
            "tracking_hf_task_reconstruction_error": (
                tracking_hf_reconstruction_error
            ),
            "target": self.target.copy(),
            "position": self.position.copy(),
            "pre_trade_position": old_position.copy(),
            "market_position": market_position.copy(),
            "asset_returns": ret.copy(),
            "volume": volume.copy(),
            "inventory_drift": inventory_drift,
            "inventory_drift_cost": inventory_drift_cost,
            "drawdown": drawdown,
            "drawdown_cost": drawdown_cost,
            "task_reward": float(reward),
            "mark_to_market_timing": self.config.mark_to_market_timing,
            "equity": float(self.equity),
        }
        return self.state(), float(reward), self.done, info

    def exogenous_bar(self) -> dict[str, Any]:
        idx = min(self.t, self.returns.shape[0] - 1)
        volume = np.ones(self.n_assets, dtype=np.float64) if self.volumes is None else self.volumes[idx]
        realized_vol = np.abs(self.returns[idx])
        return {
            "timestamp": float(idx),
            "x_raw": np.concatenate([self.returns[idx], volume, realized_vol]),
        }
