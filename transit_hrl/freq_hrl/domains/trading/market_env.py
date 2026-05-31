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
    max_leverage: float = 1.0
    inventory_drift_penalty: float = 0.01
    drawdown_penalty: float = 0.0


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

        old_position = self.position.copy()
        trade = alpha * (self.target - self.position) + residual_order
        self.position = self.position + trade
        gross = float(np.sum(np.abs(self.position)))
        if gross > self.config.max_leverage and gross > 1e-12:
            self.position *= self.config.max_leverage / gross
        realized_trade = self.position - old_position

        ret = self.returns[self.t]
        portfolio_return = float(np.dot(old_position, ret))
        turnover = float(np.sum(np.abs(realized_trade)))
        self.turnover += turnover
        cost = turnover * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000.0
        inventory_drift = float(np.mean((self.position - self.target) ** 2))
        self.equity *= max(0.0, 1.0 + portfolio_return - cost)
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = 1.0 - self.equity / max(self.peak_equity, 1e-12)
        reward = (
            portfolio_return
            - cost
            - self.config.inventory_drift_penalty * inventory_drift
            - self.config.drawdown_penalty * drawdown
        )

        self.t += 1
        self.done = self.t >= self.returns.shape[0]
        info = {
            "portfolio_return": portfolio_return,
            "transaction_cost": cost,
            "turnover": turnover,
            "trade": realized_trade.copy(),
            "target": self.target.copy(),
            "position": self.position.copy(),
            "inventory_drift": inventory_drift,
            "drawdown": drawdown,
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
