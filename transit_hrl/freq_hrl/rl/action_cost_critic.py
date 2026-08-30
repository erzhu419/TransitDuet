"""Action-conditioned cumulative-cost models for deterministic policy updates."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _linear(layer: nn.Linear, *, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=float(gain))
    nn.init.zeros_(layer.bias)
    return layer


class ActionCostCritic(nn.Module):
    """Estimate cumulative constraint cost from a state-action pair."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        state_size = int(state_dim)
        action_size = int(action_dim)
        hidden_size = int(hidden_dim)
        if state_size < 1 or action_size < 1 or hidden_size < 1:
            raise ValueError(
                "action-cost critic dimensions must all be positive"
            )
        self.state_dim = state_size
        self.action_dim = action_size
        self.net = nn.Sequential(
            _linear(
                nn.Linear(state_size + action_size, hidden_size),
                gain=np.sqrt(2.0),
            ),
            nn.Tanh(),
            _linear(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2.0)),
            nn.Tanh(),
            _linear(nn.Linear(hidden_size, 1), gain=0.1),
        )
        if bool(zero_init_output):
            output = self.net[-1]
            if not isinstance(output, nn.Linear):
                raise TypeError("action-cost critic must end in a linear head")
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if (
            state.ndim != 2
            or action.ndim != 2
            or int(state.shape[0]) != int(action.shape[0])
            or int(state.shape[1]) != self.state_dim
            or int(action.shape[1]) != self.action_dim
        ):
            raise ValueError(
                "action-cost critic inputs must be aligned state/action matrices"
            )
        return self.net(torch.cat((state, action), dim=-1)).squeeze(-1)


def discounted_smdp_cost_returns(
    cost: np.ndarray,
    duration: np.ndarray,
    done: np.ndarray,
    *,
    gamma: float,
    max_decisions: int | None = None,
) -> np.ndarray:
    """Compute full or decision-truncated SMDP returns with episode resets."""

    costs = np.asarray(cost, dtype=np.float64).reshape(-1)
    durations = np.asarray(duration).reshape(-1)
    terminals = np.asarray(done).reshape(-1)
    if (
        costs.size < 1
        or durations.size != costs.size
        or terminals.size != costs.size
        or not np.all(np.isfinite(costs))
        or np.any(durations < 1)
        or np.any(durations.astype(np.int64) != durations)
        or not np.all(np.isfinite(terminals))
        or np.any((terminals != 0) & (terminals != 1))
    ):
        raise ValueError("SMDP cost-return inputs are invalid or misaligned")
    discount = float(gamma)
    if not np.isfinite(discount) or not 0.0 < discount <= 1.0:
        raise ValueError("SMDP cost-return gamma must be in (0, 1]")
    horizon = None if max_decisions is None else int(max_decisions)
    if horizon is not None and (
        horizon < 1 or horizon != max_decisions
    ):
        raise ValueError("SMDP cost-return horizon must be a positive integer")
    if horizon is not None:
        returns = np.zeros_like(costs)
        for start in range(costs.size):
            weight = 1.0
            stop = min(costs.size, start + horizon)
            for index in range(start, stop):
                returns[start] += weight * costs[index]
                if bool(terminals[index]):
                    break
                weight *= discount ** int(durations[index])
        return returns.astype(np.float32)
    returns = np.zeros_like(costs)
    continuation = 0.0
    for index in range(costs.size - 1, -1, -1):
        if bool(terminals[index]):
            continuation = 0.0
        returns[index] = costs[index] + (
            discount ** int(durations[index])
        ) * continuation
        continuation = float(returns[index])
    return returns.astype(np.float32)


def transform_latent_action(
    action: torch.Tensor,
    *,
    transform: str,
    scale: float,
) -> torch.Tensor:
    """Map stored latent actions to the action coordinates seen by the plant."""

    if action.ndim != 2:
        raise ValueError("latent action must be a two-dimensional tensor")
    mode = str(transform)
    if mode == "tanh":
        transformed = torch.tanh(action)
    elif mode == "identity":
        transformed = action
    else:
        raise ValueError(f"unknown latent action transform: {mode}")
    numeric_scale = float(scale)
    if not np.isfinite(numeric_scale) or numeric_scale <= 0.0:
        raise ValueError("latent action scale must be positive and finite")
    return numeric_scale * transformed
