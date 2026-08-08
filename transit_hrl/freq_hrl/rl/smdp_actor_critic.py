"""Asynchronous semi-Markov actor-critic core for Freq-HRL.

The upper planner and lower controller operate on different transition
streams.  An upper action is recorded once for the whole macro interval,
whereas lower actions are recorded at the environment rate.  This avoids the
incorrect joint-PPO construction where an upper log probability is repeated
for every lower transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .causal_sequence import CausalGRUGaussianActor, CausalGRUValueNet
from .dual_actor_critic import BernoulliActor, GaussianActor, ValueNet


@dataclass
class SMDPPPOConfig:
    upper_state_dim: int
    lower_state_dim: int
    upper_action_dim: int
    lower_action_dim: int
    hf_state_dim: int = 0
    hf_action_dim: int = 0
    promotion_state_dim: int = 0
    hidden_dim: int = 0
    state_encoder: str = "mlp"
    raw_history_window: int = 0
    raw_feature_dim: int = 0
    upper_learning_rate: float = 3e-3
    lower_learning_rate: float = 3e-3
    hf_learning_rate: float = 0.0
    promotion_learning_rate: float = 0.0
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    cost_value_coef: float = 0.5
    entropy_coef: float = 0.001
    promotion_entropy_coef: float | None = None
    promotion_rate_budget: float = 1.0
    promotion_rate_coef: float = 0.0
    promotion_counterfactual_coef: float = 0.0
    promotion_advantage_learning_rate: float = 0.0
    promotion_advantage_coef: float = 0.0
    promotion_advantage_huber_delta: float = 0.1
    max_grad_norm: float = 1.0
    epochs: int = 4
    minibatch_size: int = 512
    init_log_std: float = -1.0
    promotion_init_logit: float = -2.0
    lower_cost_target: float = 0.0
    lower_dual_lr: float = 0.0
    lower_lambda_init: float = 0.0
    lower_max_lambda: float = 100.0
    device: str = "cpu"


@dataclass
class LevelTrajectoryBatch:
    """One policy level's SMDP transitions.

    ``reward`` is the discounted reward accumulated inside each transition.
    ``duration`` is the number of primitive environment steps represented by
    that transition.  For lower transitions duration is normally one.
    """

    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    duration: np.ndarray
    done: np.ndarray
    old_logp: np.ndarray
    old_value: np.ndarray
    cost: np.ndarray | None = None
    counterfactual_advantage: np.ndarray | None = None

    def validate(self, *, state_dim: int, action_dim: int, level: str) -> None:
        state = np.asarray(self.state)
        action = np.asarray(self.action)
        if state.ndim != 2 or state.shape[1] != int(state_dim):
            raise ValueError(f"{level} state shape must be (n, {state_dim}), got {state.shape}")
        if action.ndim != 2 or action.shape != (state.shape[0], int(action_dim)):
            raise ValueError(
                f"{level} action shape must be ({state.shape[0]}, {action_dim}), got {action.shape}"
            )
        n = int(state.shape[0])
        for name in ("reward", "duration", "done", "old_logp", "old_value"):
            values = np.asarray(getattr(self, name)).reshape(-1)
            if values.size != n:
                raise ValueError(f"{level} {name} length must be {n}, got {values.size}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{level} {name} must be finite")
        duration = np.asarray(self.duration, dtype=np.int64).reshape(-1)
        if np.any(duration < 1):
            raise ValueError(f"{level} duration must be at least one primitive step")
        if self.cost is not None:
            cost = np.asarray(self.cost).reshape(-1)
            if cost.size != n or not np.all(np.isfinite(cost)):
                raise ValueError(f"{level} cost must contain {n} finite values")
        if self.counterfactual_advantage is not None:
            advantage = np.asarray(self.counterfactual_advantage).reshape(-1)
            if advantage.size != n or not np.all(np.isfinite(advantage)):
                raise ValueError(
                    f"{level} counterfactual_advantage must contain {n} finite values"
                )

    @property
    def size(self) -> int:
        return int(np.asarray(self.reward).reshape(-1).size)


@dataclass
class HierarchicalTrajectoryBatch:
    upper: LevelTrajectoryBatch
    lower: LevelTrajectoryBatch
    hf: LevelTrajectoryBatch | None = None
    promotion: LevelTrajectoryBatch | None = None


class TemporalDecisionScheduler:
    """Decide when an upper macro action may be replaced."""

    def __init__(self, upper_period: int, min_upper_duration: int = 1) -> None:
        if int(upper_period) < 1:
            raise ValueError("upper_period must be positive")
        if int(min_upper_duration) < 1 or int(min_upper_duration) > int(upper_period):
            raise ValueError("min_upper_duration must be in [1, upper_period]")
        self.upper_period = int(upper_period)
        self.min_upper_duration = int(min_upper_duration)
        self.last_upper_step: int | None = None

    def reset(self) -> None:
        self.last_upper_step = None

    def decision_reason(self, step: int, *, promotion: bool = False) -> str | None:
        step = int(step)
        if self.last_upper_step is None:
            return "initial"
        elapsed = step - self.last_upper_step
        if elapsed < 0:
            raise ValueError("step must be monotonic")
        if promotion and elapsed >= self.min_upper_duration:
            return "promotion"
        if elapsed >= self.upper_period:
            return "scheduled"
        return None

    def mark_decision(self, step: int) -> None:
        step = int(step)
        if self.last_upper_step is not None and step < self.last_upper_step:
            raise ValueError("step must be monotonic")
        self.last_upper_step = step


class HierarchicalRolloutBuilder:
    """Build separate upper and lower trajectories from one episode."""

    def __init__(self, gamma: float) -> None:
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = float(gamma)
        self._upper: dict[str, list[Any]] = {
            key: [] for key in ("state", "action", "reward", "duration", "done", "old_logp", "old_value", "cost")
        }
        self._lower: dict[str, list[Any]] = {
            key: [] for key in ("state", "action", "reward", "duration", "done", "old_logp", "old_value", "cost")
        }
        self._hf: dict[str, list[Any]] = {
            key: [] for key in ("state", "action", "reward", "duration", "done", "old_logp", "old_value", "cost")
        }
        self._pending_upper: dict[str, Any] | None = None
        self._hf_enabled: bool | None = None

    @property
    def has_pending_upper(self) -> bool:
        return self._pending_upper is not None

    def begin_upper(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        logp: float,
        value: float,
    ) -> None:
        if self._pending_upper is not None:
            self._close_upper(done=False)
        self._pending_upper = {
            "state": np.asarray(state, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "logp": float(logp),
            "value": float(value),
            "rewards": [],
            "costs": [],
        }

    def add_lower(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        logp: float,
        value: float,
        reward: float,
        done: bool,
        cost: float = 0.0,
        upper_reward: float | None = None,
        upper_cost: float | None = None,
        hf_state: np.ndarray | None = None,
        hf_action: np.ndarray | None = None,
        hf_logp: float | None = None,
        hf_value: float | None = None,
        hf_reward: float | None = None,
        hf_cost: float = 0.0,
    ) -> None:
        if self._pending_upper is None:
            raise RuntimeError("begin_upper must be called before add_lower")
        hf_fields = (hf_state, hf_action, hf_logp, hf_value, hf_reward)
        hf_enabled = any(item is not None for item in hf_fields)
        if hf_enabled and not all(item is not None for item in hf_fields):
            raise ValueError(
                "hf_state, hf_action, hf_logp, hf_value, and hf_reward must "
                "be provided together"
            )
        if self._hf_enabled is not None and self._hf_enabled != hf_enabled:
            raise ValueError("HF trajectory presence must be consistent within an episode")
        self._lower["state"].append(np.asarray(state, dtype=np.float32).copy())
        self._lower["action"].append(np.asarray(action, dtype=np.float32).copy())
        self._lower["reward"].append(float(reward))
        self._lower["duration"].append(1)
        self._lower["done"].append(float(bool(done)))
        self._lower["old_logp"].append(float(logp))
        self._lower["old_value"].append(float(value))
        self._lower["cost"].append(float(cost))
        if self._hf_enabled is None:
            self._hf_enabled = hf_enabled
        if hf_enabled:
            self._hf["state"].append(np.asarray(hf_state, dtype=np.float32).copy())
            self._hf["action"].append(np.asarray(hf_action, dtype=np.float32).copy())
            self._hf["reward"].append(float(hf_reward))
            self._hf["duration"].append(1)
            self._hf["done"].append(float(bool(done)))
            self._hf["old_logp"].append(float(hf_logp))
            self._hf["old_value"].append(float(hf_value))
            self._hf["cost"].append(float(hf_cost))
        self._pending_upper["rewards"].append(float(reward if upper_reward is None else upper_reward))
        self._pending_upper["costs"].append(float(cost if upper_cost is None else upper_cost))
        if done:
            self._close_upper(done=True)

    def finish(self, *, terminal: bool = True) -> None:
        if self._pending_upper is not None:
            self._close_upper(done=bool(terminal))
        if self._lower["done"] and terminal:
            self._lower["done"][-1] = 1.0
        if self._hf["done"] and terminal:
            self._hf["done"][-1] = 1.0

    def _close_upper(self, *, done: bool) -> None:
        pending = self._pending_upper
        if pending is None:
            return
        rewards = list(pending["rewards"])
        costs = list(pending["costs"])
        if not rewards:
            raise RuntimeError("an upper macro action must contain at least one lower transition")
        discounts = np.power(self.gamma, np.arange(len(rewards), dtype=np.float64))
        self._upper["state"].append(pending["state"])
        self._upper["action"].append(pending["action"])
        self._upper["reward"].append(float(np.dot(discounts, np.asarray(rewards, dtype=np.float64))))
        self._upper["duration"].append(int(len(rewards)))
        self._upper["done"].append(float(bool(done)))
        self._upper["old_logp"].append(float(pending["logp"]))
        self._upper["old_value"].append(float(pending["value"]))
        self._upper["cost"].append(float(np.dot(discounts, np.asarray(costs, dtype=np.float64))))
        self._pending_upper = None

    @staticmethod
    def _level(data: dict[str, list[Any]]) -> LevelTrajectoryBatch:
        return LevelTrajectoryBatch(
            state=np.asarray(data["state"], dtype=np.float32),
            action=np.asarray(data["action"], dtype=np.float32),
            reward=np.asarray(data["reward"], dtype=np.float32),
            duration=np.asarray(data["duration"], dtype=np.int64),
            done=np.asarray(data["done"], dtype=np.float32),
            old_logp=np.asarray(data["old_logp"], dtype=np.float32),
            old_value=np.asarray(data["old_value"], dtype=np.float32),
            cost=np.asarray(data["cost"], dtype=np.float32),
            counterfactual_advantage=(
                np.asarray(data["counterfactual_advantage"], dtype=np.float32)
                if data.get("counterfactual_advantage") else None
            ),
        )

    def build(self) -> HierarchicalTrajectoryBatch:
        if self._pending_upper is not None:
            raise RuntimeError("finish must be called before build")
        if not self._upper["reward"] or not self._lower["reward"]:
            raise ValueError("rollout must contain upper and lower transitions")
        return HierarchicalTrajectoryBatch(
            upper=self._level(self._upper),
            lower=self._level(self._lower),
            hf=(self._level(self._hf) if self._hf["reward"] else None),
        )


class PromotionRolloutBuilder:
    """Build sparse SMDP transitions for a learned replan/continue gate."""

    def __init__(self, gamma: float) -> None:
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = float(gamma)
        self._data: dict[str, list[Any]] = {
            key: []
            for key in (
                "state",
                "action",
                "reward",
                "duration",
                "done",
                "old_logp",
                "old_value",
                "cost",
                "counterfactual_advantage",
            )
        }
        self._pending: dict[str, Any] | None = None
        self._counterfactual_enabled: bool | None = None

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def begin(
        self,
        *,
        state: np.ndarray,
        action: float,
        logp: float,
        value: float,
    ) -> None:
        self.close(done=False)
        self._pending = {
            "state": np.asarray(state, dtype=np.float32).copy(),
            "action": np.asarray([float(action)], dtype=np.float32),
            "logp": float(logp),
            "value": float(value),
            "rewards": [],
        }

    def add_reward(
        self,
        reward: float,
        *,
        counterfactual_advantage: float | None = None,
        done: bool = False,
    ) -> None:
        if self._pending is None:
            return
        enabled = counterfactual_advantage is not None
        if self._counterfactual_enabled is None:
            self._counterfactual_enabled = enabled
        elif self._counterfactual_enabled != enabled:
            raise ValueError(
                "counterfactual promotion advantages must be present for every reward or none"
            )
        self._pending["rewards"].append(float(reward))
        if enabled:
            self._pending.setdefault("counterfactual_advantages", []).append(
                float(counterfactual_advantage)
            )
        if done:
            self.close(done=True)

    def close(self, *, done: bool) -> None:
        pending = self._pending
        if pending is None:
            return
        rewards = list(pending["rewards"])
        if not rewards:
            raise RuntimeError(
                "a learned promotion decision must own at least one primitive reward"
            )
        discounts = np.power(
            self.gamma,
            np.arange(len(rewards), dtype=np.float64),
        )
        self._data["state"].append(pending["state"])
        self._data["action"].append(pending["action"])
        self._data["reward"].append(
            float(np.dot(discounts, np.asarray(rewards, dtype=np.float64)))
        )
        self._data["duration"].append(int(len(rewards)))
        self._data["done"].append(float(bool(done)))
        self._data["old_logp"].append(float(pending["logp"]))
        self._data["old_value"].append(float(pending["value"]))
        self._data["cost"].append(0.0)
        counterfactual_advantages = list(
            pending.get("counterfactual_advantages", [])
        )
        if counterfactual_advantages:
            if len(counterfactual_advantages) != len(rewards):
                raise RuntimeError(
                    "counterfactual promotion advantages must align with rewards"
                )
            self._data["counterfactual_advantage"].append(float(np.dot(
                discounts,
                np.asarray(counterfactual_advantages, dtype=np.float64),
            )))
        self._pending = None

    def finish(self, *, terminal: bool = True) -> None:
        self.close(done=bool(terminal))
        if self._data["done"] and terminal:
            self._data["done"][-1] = 1.0

    def build(self) -> LevelTrajectoryBatch | None:
        if self._pending is not None:
            raise RuntimeError("finish must be called before build")
        if not self._data["reward"]:
            return None
        return HierarchicalRolloutBuilder._level(self._data)


def concat_level_batches(batches: Iterable[LevelTrajectoryBatch]) -> LevelTrajectoryBatch:
    items = list(batches)
    if not items:
        raise ValueError("at least one level batch is required")
    counterfactual_batches = [
        item.counterfactual_advantage for item in items
    ]
    if any(item is None for item in counterfactual_batches) and not all(
        item is None for item in counterfactual_batches
    ):
        raise ValueError(
            "counterfactual advantages must be present for every batch or none"
        )
    return LevelTrajectoryBatch(
        state=np.concatenate([np.asarray(item.state) for item in items], axis=0),
        action=np.concatenate([np.asarray(item.action) for item in items], axis=0),
        reward=np.concatenate([np.asarray(item.reward).reshape(-1) for item in items], axis=0),
        duration=np.concatenate([np.asarray(item.duration).reshape(-1) for item in items], axis=0),
        done=np.concatenate([np.asarray(item.done).reshape(-1) for item in items], axis=0),
        old_logp=np.concatenate([np.asarray(item.old_logp).reshape(-1) for item in items], axis=0),
        old_value=np.concatenate([np.asarray(item.old_value).reshape(-1) for item in items], axis=0),
        cost=(
            np.concatenate([np.asarray(item.cost).reshape(-1) for item in items], axis=0)
            if all(item.cost is not None for item in items) else None
        ),
        counterfactual_advantage=(
            None
            if all(item is None for item in counterfactual_batches)
            else np.concatenate([
                np.asarray(item).reshape(-1)
                for item in counterfactual_batches if item is not None
            ], axis=0)
        ),
    )


def concat_hierarchical_batches(
    batches: Iterable[HierarchicalTrajectoryBatch],
) -> HierarchicalTrajectoryBatch:
    items = list(batches)
    if not items:
        raise ValueError("at least one hierarchical batch is required")
    promotion_batches = [item.promotion for item in items]
    hf_batches = [item.hf for item in items]
    if any(item is None for item in hf_batches) and not all(
        item is None for item in hf_batches
    ):
        raise ValueError("HF trajectories must be present for every batch or none")
    if any(item is None for item in promotion_batches) and not all(
        item is None for item in promotion_batches
    ):
        raise ValueError(
            "promotion trajectories must be present for every batch or none"
        )
    return HierarchicalTrajectoryBatch(
        upper=concat_level_batches(item.upper for item in items),
        lower=concat_level_batches(item.lower for item in items),
        hf=(
            None
            if all(item is None for item in hf_batches)
            else concat_level_batches(item for item in hf_batches if item is not None)
        ),
        promotion=(
            None
            if all(item is None for item in promotion_batches)
            else concat_level_batches(
                item for item in promotion_batches if item is not None
            )
        ),
    )


class FrequencySeparatedActorCriticPPO:
    """PPO-Lagrangian with independent upper and lower SMDP updates."""

    def __init__(self, config: SMDPPPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        promotion_entropy_coef = (
            float(config.entropy_coef)
            if config.promotion_entropy_coef is None
            else float(config.promotion_entropy_coef)
        )
        if not np.isfinite(promotion_entropy_coef) or promotion_entropy_coef < 0.0:
            raise ValueError("promotion_entropy_coef must be finite and non-negative")
        if (
            not np.isfinite(float(config.promotion_rate_budget))
            or not 0.0 <= float(config.promotion_rate_budget) <= 1.0
        ):
            raise ValueError("promotion_rate_budget must be finite and in [0, 1]")
        if (
            not np.isfinite(float(config.promotion_rate_coef))
            or float(config.promotion_rate_coef) < 0.0
        ):
            raise ValueError("promotion_rate_coef must be finite and non-negative")
        if (
            not np.isfinite(float(config.promotion_counterfactual_coef))
            or float(config.promotion_counterfactual_coef) < 0.0
        ):
            raise ValueError(
                "promotion_counterfactual_coef must be finite and non-negative"
            )
        if (
            not np.isfinite(float(config.promotion_advantage_learning_rate))
            or float(config.promotion_advantage_learning_rate) < 0.0
        ):
            raise ValueError(
                "promotion_advantage_learning_rate must be finite and non-negative"
            )
        if (
            not np.isfinite(float(config.promotion_advantage_coef))
            or float(config.promotion_advantage_coef) < 0.0
        ):
            raise ValueError(
                "promotion_advantage_coef must be finite and non-negative"
            )
        if (
            not np.isfinite(float(config.promotion_advantage_huber_delta))
            or float(config.promotion_advantage_huber_delta) <= 0.0
        ):
            raise ValueError(
                "promotion_advantage_huber_delta must be positive and finite"
            )
        if (
            float(config.promotion_advantage_coef) > 0.0
            and int(config.promotion_state_dim) <= 0
        ):
            raise ValueError(
                "promotion advantage learning requires a promotion state"
            )
        if (int(config.hf_state_dim) > 0) != (int(config.hf_action_dim) > 0):
            raise ValueError(
                "hf_state_dim and hf_action_dim must either both be positive or both be zero"
            )
        self.hf_actor: nn.Module | None = None
        self.hf_value: nn.Module | None = None
        if str(config.state_encoder) == "mlp":
            self.upper_actor = GaussianActor(
                config.upper_state_dim,
                config.upper_action_dim,
                config.hidden_dim,
                config.init_log_std,
            ).to(self.device)
            self.lower_actor = GaussianActor(
                config.lower_state_dim,
                config.lower_action_dim,
                config.hidden_dim,
                config.init_log_std,
            ).to(self.device)
            self.upper_value = ValueNet(
                config.upper_state_dim, config.hidden_dim
            ).to(self.device)
            self.lower_value = ValueNet(
                config.lower_state_dim, config.hidden_dim
            ).to(self.device)
            self.lower_cost_value = ValueNet(
                config.lower_state_dim, config.hidden_dim
            ).to(self.device)
            if int(config.hf_state_dim) > 0:
                self.hf_actor = GaussianActor(
                    config.hf_state_dim,
                    config.hf_action_dim,
                    config.hidden_dim,
                    config.init_log_std,
                ).to(self.device)
                self.hf_value = ValueNet(
                    config.hf_state_dim, config.hidden_dim
                ).to(self.device)
        elif str(config.state_encoder) == "causal_gru":
            actor_kwargs = {
                "history_window": config.raw_history_window,
                "raw_feature_dim": config.raw_feature_dim,
                "hidden_dim": config.hidden_dim,
                "init_log_std": config.init_log_std,
            }
            value_kwargs = {
                "history_window": config.raw_history_window,
                "raw_feature_dim": config.raw_feature_dim,
                "hidden_dim": config.hidden_dim,
            }
            self.upper_actor = CausalGRUGaussianActor(
                state_dim=config.upper_state_dim,
                action_dim=config.upper_action_dim,
                **actor_kwargs,
            ).to(self.device)
            self.lower_actor = CausalGRUGaussianActor(
                state_dim=config.lower_state_dim,
                action_dim=config.lower_action_dim,
                **actor_kwargs,
            ).to(self.device)
            self.upper_value = CausalGRUValueNet(
                state_dim=config.upper_state_dim, **value_kwargs
            ).to(self.device)
            self.lower_value = CausalGRUValueNet(
                state_dim=config.lower_state_dim, **value_kwargs
            ).to(self.device)
            self.lower_cost_value = CausalGRUValueNet(
                state_dim=config.lower_state_dim, **value_kwargs
            ).to(self.device)
            if int(config.hf_state_dim) > 0:
                self.hf_actor = CausalGRUGaussianActor(
                    state_dim=config.hf_state_dim,
                    action_dim=config.hf_action_dim,
                    **actor_kwargs,
                ).to(self.device)
                self.hf_value = CausalGRUValueNet(
                    state_dim=config.hf_state_dim, **value_kwargs
                ).to(self.device)
        else:
            raise ValueError(f"unknown state_encoder: {config.state_encoder}")
        self.promotion_actor: BernoulliActor | None = None
        self.promotion_value: ValueNet | None = None
        self.promotion_actor_optimizer: torch.optim.Optimizer | None = None
        self.promotion_value_optimizer: torch.optim.Optimizer | None = None
        self.promotion_advantage: ValueNet | None = None
        self.promotion_advantage_optimizer: torch.optim.Optimizer | None = None
        if int(config.promotion_state_dim) > 0:
            self.promotion_actor = BernoulliActor(
                state_dim=int(config.promotion_state_dim),
                hidden_dim=int(config.hidden_dim),
                init_logit=float(config.promotion_init_logit),
            ).to(self.device)
            self.promotion_value = ValueNet(
                int(config.promotion_state_dim), int(config.hidden_dim)
            ).to(self.device)
            promotion_lr = (
                float(config.promotion_learning_rate)
                if float(config.promotion_learning_rate) > 0.0
                else float(config.upper_learning_rate)
            )
            self.promotion_actor_optimizer = torch.optim.Adam(
                self.promotion_actor.parameters(), lr=promotion_lr
            )
            self.promotion_value_optimizer = torch.optim.Adam(
                self.promotion_value.parameters(), lr=promotion_lr
            )
            if float(config.promotion_advantage_coef) > 0.0:
                self.promotion_advantage = ValueNet(
                    int(config.promotion_state_dim), int(config.hidden_dim)
                ).to(self.device)
                advantage_lr = (
                    float(config.promotion_advantage_learning_rate)
                    if float(config.promotion_advantage_learning_rate) > 0.0
                    else promotion_lr
                )
                self.promotion_advantage_optimizer = torch.optim.Adam(
                    self.promotion_advantage.parameters(), lr=advantage_lr
                )
        self.upper_actor_optimizer = torch.optim.Adam(
            self.upper_actor.parameters(),
            lr=float(config.upper_learning_rate),
        )
        self.upper_value_optimizer = torch.optim.Adam(
            self.upper_value.parameters(),
            lr=float(config.upper_learning_rate),
        )
        self.lower_actor_optimizer = torch.optim.Adam(
            self.lower_actor.parameters(),
            lr=float(config.lower_learning_rate),
        )
        self.lower_value_optimizer = torch.optim.Adam(
            self.lower_value.parameters(),
            lr=float(config.lower_learning_rate),
        )
        self.lower_cost_value_optimizer = torch.optim.Adam(
            self.lower_cost_value.parameters(),
            lr=float(config.lower_learning_rate),
        )
        self.hf_actor_optimizer: torch.optim.Optimizer | None = None
        self.hf_value_optimizer: torch.optim.Optimizer | None = None
        if self.hf_actor is not None and self.hf_value is not None:
            hf_lr = (
                float(config.hf_learning_rate)
                if float(config.hf_learning_rate) > 0.0
                else float(config.lower_learning_rate)
            )
            self.hf_actor_optimizer = torch.optim.Adam(
                self.hf_actor.parameters(), lr=hf_lr
            )
            self.hf_value_optimizer = torch.optim.Adam(
                self.hf_value.parameters(), lr=hf_lr
            )
        self.constraint_lambda = float(config.lower_lambda_init)

    def state_dict(self) -> dict[str, Any]:
        payload = {
            "config": self.config.__dict__,
            "upper_actor": self.upper_actor.state_dict(),
            "lower_actor": self.lower_actor.state_dict(),
            "upper_value": self.upper_value.state_dict(),
            "lower_value": self.lower_value.state_dict(),
            "lower_cost_value": self.lower_cost_value.state_dict(),
            "upper_actor_optimizer": self.upper_actor_optimizer.state_dict(),
            "upper_value_optimizer": self.upper_value_optimizer.state_dict(),
            "lower_actor_optimizer": self.lower_actor_optimizer.state_dict(),
            "lower_value_optimizer": self.lower_value_optimizer.state_dict(),
            "lower_cost_value_optimizer": self.lower_cost_value_optimizer.state_dict(),
            "constraint_lambda": float(self.constraint_lambda),
        }
        if self.promotion_actor is not None and self.promotion_value is not None:
            payload.update({
                "promotion_actor": self.promotion_actor.state_dict(),
                "promotion_value": self.promotion_value.state_dict(),
                "promotion_actor_optimizer": self.promotion_actor_optimizer.state_dict(),
                "promotion_value_optimizer": self.promotion_value_optimizer.state_dict(),
            })
        if (
            self.promotion_advantage is not None
            and self.promotion_advantage_optimizer is not None
        ):
            payload.update({
                "promotion_advantage": self.promotion_advantage.state_dict(),
                "promotion_advantage_optimizer": (
                    self.promotion_advantage_optimizer.state_dict()
                ),
            })
        if self.hf_actor is not None and self.hf_value is not None:
            payload.update({
                "hf_actor": self.hf_actor.state_dict(),
                "hf_value": self.hf_value.state_dict(),
                "hf_actor_optimizer": self.hf_actor_optimizer.state_dict(),
                "hf_value_optimizer": self.hf_value_optimizer.state_dict(),
            })
        return payload

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.upper_actor.load_state_dict(payload["upper_actor"])
        self.lower_actor.load_state_dict(payload["lower_actor"])
        self.upper_value.load_state_dict(payload["upper_value"])
        self.lower_value.load_state_dict(payload["lower_value"])
        self.lower_cost_value.load_state_dict(payload["lower_cost_value"])
        if self.hf_actor is not None and self.hf_value is not None:
            if "hf_actor" not in payload or "hf_value" not in payload:
                raise ValueError(
                    "checkpoint is missing the configured HF tactical policy"
                )
            self.hf_actor.load_state_dict(payload["hf_actor"])
            self.hf_value.load_state_dict(payload["hf_value"])
        if self.promotion_actor is not None and self.promotion_value is not None:
            if "promotion_actor" not in payload or "promotion_value" not in payload:
                raise ValueError(
                    "checkpoint is missing the configured learned promotion gate"
                )
            self.promotion_actor.load_state_dict(payload["promotion_actor"])
            self.promotion_value.load_state_dict(payload["promotion_value"])
        if self.promotion_advantage is not None:
            if "promotion_advantage" not in payload:
                raise ValueError(
                    "checkpoint is missing the configured promotion advantage head"
                )
            self.promotion_advantage.load_state_dict(
                payload["promotion_advantage"]
            )
        for name in (
            "upper_actor_optimizer",
            "upper_value_optimizer",
            "lower_actor_optimizer",
            "lower_value_optimizer",
            "lower_cost_value_optimizer",
            "hf_actor_optimizer",
            "hf_value_optimizer",
            "promotion_actor_optimizer",
            "promotion_value_optimizer",
            "promotion_advantage_optimizer",
        ):
            optimizer = getattr(self, name, None)
            if name in payload and optimizer is not None:
                optimizer.load_state_dict(payload[name])
        self.constraint_lambda = float(payload.get("constraint_lambda", self.constraint_lambda))

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)

    @torch.no_grad()
    def act_upper(self, state: np.ndarray, sample: bool = True) -> dict[str, np.ndarray | float]:
        tensor = self._state_tensor(state)
        action, logp = self.upper_actor(tensor, sample=sample)
        value = self.upper_value(tensor)
        return {
            "action": action.cpu().numpy().reshape(-1),
            "logp": float(logp.item()),
            "value": float(value.item()),
        }

    @torch.no_grad()
    def act_lower(self, state: np.ndarray, sample: bool = True) -> dict[str, np.ndarray | float]:
        tensor = self._state_tensor(state)
        action, logp = self.lower_actor(tensor, sample=sample)
        value = self.lower_value(tensor)
        cost_value = self.lower_cost_value(tensor)
        return {
            "action": action.cpu().numpy().reshape(-1),
            "logp": float(logp.item()),
            "value": float(value.item()),
            "cost_value": float(cost_value.item()),
        }

    @torch.no_grad()
    def act_hf(
        self,
        state: np.ndarray,
        sample: bool = True,
    ) -> dict[str, np.ndarray | float]:
        if self.hf_actor is None or self.hf_value is None:
            raise RuntimeError("HF tactical policy is not configured")
        tensor = self._state_tensor(state)
        action, logp = self.hf_actor(tensor, sample=sample)
        value = self.hf_value(tensor)
        return {
            "action": action.cpu().numpy().reshape(-1),
            "logp": float(logp.item()),
            "value": float(value.item()),
        }

    @torch.no_grad()
    def act_promotion(
        self,
        state: np.ndarray,
        sample: bool = True,
        deterministic_threshold: float = 0.5,
        deterministic_mode: str = "actor_probability",
        advantage_threshold: float = 0.0,
    ) -> dict[str, float]:
        if self.promotion_actor is None or self.promotion_value is None:
            raise RuntimeError("learned promotion gate is not configured")
        threshold = float(deterministic_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("deterministic_threshold must be finite and in (0, 1)")
        mode = str(deterministic_mode)
        if mode not in {"actor_probability", "counterfactual_advantage"}:
            raise ValueError("unknown deterministic promotion mode")
        if not np.isfinite(float(advantage_threshold)):
            raise ValueError("advantage_threshold must be finite")
        if mode == "counterfactual_advantage" and self.promotion_advantage is None:
            raise RuntimeError(
                "counterfactual-advantage promotion requires its learned head"
            )
        tensor = self._state_tensor(state)
        distribution = self.promotion_actor.distribution(tensor)
        predicted_advantage = (
            float(self.promotion_advantage(tensor).item())
            if self.promotion_advantage is not None else 0.0
        )
        action = (
            distribution.sample()
            if sample
            else (
                (distribution.probs >= threshold).to(tensor.dtype)
                if mode == "actor_probability"
                else torch.as_tensor(
                    [[float(predicted_advantage >= float(advantage_threshold))]],
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            )
        )
        logp = distribution.log_prob(action).sum(dim=-1)
        value = self.promotion_value(tensor)
        probability = distribution.probs
        return {
            "action": float(action.item()),
            "probability": float(probability.item()),
            "logp": float(logp.item()),
            "value": float(value.item()),
            "predicted_counterfactual_advantage": predicted_advantage,
            "advantage_head_enabled": float(
                self.promotion_advantage is not None
            ),
        }

    def _gae(
        self,
        signal: np.ndarray,
        done: np.ndarray,
        duration: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        done = np.asarray(done, dtype=np.float32).reshape(-1)
        duration = np.asarray(duration, dtype=np.float32).reshape(-1)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        advantage = np.zeros_like(signal)
        last = 0.0
        for index in range(signal.size - 1, -1, -1):
            nonterminal = 1.0 - done[index]
            next_value = 0.0 if index == signal.size - 1 else float(values[index + 1])
            discount = float(self.config.gamma) ** float(duration[index])
            trace_discount = discount * (float(self.config.gae_lambda) ** float(duration[index]))
            delta = float(signal[index]) + discount * next_value * nonterminal - float(values[index])
            last = delta + trace_discount * nonterminal * last
            advantage[index] = last
        return advantage, advantage + values

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.size < 2:
            return values
        return (values - float(np.mean(values))) / (float(np.std(values)) + 1e-8)

    def _update_level(
        self,
        *,
        level: str,
        batch: LevelTrajectoryBatch,
        actor: nn.Module,
        value_net: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        cost_value_net: ValueNet | None = None,
        cost_value_optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, float]:
        cfg = self.config
        if level == "upper":
            state_dim = cfg.upper_state_dim
            action_dim = cfg.upper_action_dim
        elif level == "lower":
            state_dim = cfg.lower_state_dim
            action_dim = cfg.lower_action_dim
        elif level == "hf":
            state_dim = cfg.hf_state_dim
            action_dim = cfg.hf_action_dim
        elif level == "promotion":
            state_dim = cfg.promotion_state_dim
            action_dim = 1
        else:
            raise ValueError(f"unknown policy level: {level}")
        batch.validate(state_dim=state_dim, action_dim=action_dim, level=level)
        if batch.size == 0:
            empty = {
                f"{level}_{key}": 0.0
                for key in ("loss", "policy_loss", "value_loss", "entropy")
            }
            return {
                **empty,
                f"{level}_actor_optimizer_steps": 0.0,
                f"{level}_value_optimizer_steps": 0.0,
                f"{level}_cost_value_optimizer_steps": 0.0,
                f"{level}_advantage_optimizer_steps": 0.0,
            }

        state = torch.as_tensor(batch.state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(batch.old_logp, dtype=torch.float32, device=self.device)
        reward_adv, returns = self._gae(batch.reward, batch.done, batch.duration, batch.old_value)
        reward_adv_t = torch.as_tensor(self._normalize(reward_adv), dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        counterfactual_adv_t = None
        counterfactual_target_t = None
        if level == "promotion" and batch.counterfactual_advantage is not None:
            counterfactual_advantage = np.asarray(
                batch.counterfactual_advantage, dtype=np.float32
            ).reshape(-1)
            counterfactual_target_t = torch.as_tensor(
                counterfactual_advantage,
                dtype=torch.float32,
                device=self.device,
            )
            scale = float(np.mean(np.abs(counterfactual_advantage))) + 1e-8
            counterfactual_adv_t = torch.as_tensor(
                np.clip(counterfactual_advantage / scale, -10.0, 10.0),
                dtype=torch.float32,
                device=self.device,
            )
        if (
            level == "promotion"
            and self.promotion_advantage is not None
            and counterfactual_target_t is None
        ):
            raise ValueError(
                "promotion advantage learning requires paired counterfactual targets"
            )

        cost = None
        cost_adv_t = None
        cost_returns_t = None
        if level == "lower" and batch.cost is not None and cost_value_net is not None:
            cost = np.asarray(batch.cost, dtype=np.float32).reshape(-1)
            with torch.no_grad():
                old_cost_value = cost_value_net(state).detach().cpu().numpy()
            cost_adv, cost_returns = self._gae(cost, batch.done, batch.duration, old_cost_value)
            cost_adv_t = torch.as_tensor(self._normalize(cost_adv), dtype=torch.float32, device=self.device)
            cost_returns_t = torch.as_tensor(cost_returns, dtype=torch.float32, device=self.device)

        indices = np.arange(batch.size)
        minibatch = max(1, min(int(cfg.minibatch_size), batch.size))
        rows: list[dict[str, float]] = []
        for _ in range(max(1, int(cfg.epochs))):
            np.random.shuffle(indices)
            for start in range(0, batch.size, minibatch):
                idx_np = indices[start:start + minibatch]
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=self.device)
                logp, entropy = actor.log_prob_entropy(state[idx], action[idx])
                ratio = torch.exp((logp - old_logp[idx]).clamp(-20.0, 20.0))
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                reward_surrogate = torch.minimum(
                    ratio * reward_adv_t[idx], clipped * reward_adv_t[idx]
                ).mean()
                constraint_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if cost_adv_t is not None and self.constraint_lambda > 0.0:
                    cost_surrogate = torch.maximum(
                        ratio * cost_adv_t[idx], clipped * cost_adv_t[idx]
                    ).mean()
                    constraint_loss = float(self.constraint_lambda) * cost_surrogate
                promotion_rate_loss = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
                promotion_probability_mean = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
                promotion_counterfactual_surrogate = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
                promotion_advantage_loss = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
                promotion_advantage_prediction_mean = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
                if level == "promotion":
                    distribution = actor.distribution(state[idx])
                    promotion_probability_mean = distribution.probs.mean()
                    if counterfactual_adv_t is not None:
                        promotion_counterfactual_surrogate = torch.mean(
                            distribution.probs.reshape(-1)
                            * counterfactual_adv_t[idx]
                        )
                    rate_excess = torch.relu(
                        promotion_probability_mean
                        - float(cfg.promotion_rate_budget)
                    )
                    promotion_rate_loss = (
                        float(cfg.promotion_rate_coef) * rate_excess.square()
                    )
                    if (
                        self.promotion_advantage is not None
                        and counterfactual_target_t is not None
                    ):
                        advantage_prediction = self.promotion_advantage(
                            state[idx]
                        )
                        promotion_advantage_prediction_mean = (
                            advantage_prediction.mean()
                        )
                        promotion_advantage_loss = F.smooth_l1_loss(
                            advantage_prediction,
                            counterfactual_target_t[idx],
                            beta=float(cfg.promotion_advantage_huber_delta),
                        )
                policy_loss = (
                    -reward_surrogate
                    - float(cfg.promotion_counterfactual_coef)
                    * promotion_counterfactual_surrogate
                    + constraint_loss
                    + promotion_rate_loss
                )
                value_loss = torch.mean((value_net(state[idx]) - returns_t[idx]) ** 2)
                cost_value_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if cost_returns_t is not None and cost_value_net is not None:
                    cost_value_loss = torch.mean((cost_value_net(state[idx]) - cost_returns_t[idx]) ** 2)
                entropy_mean = entropy.mean()
                entropy_coef = (
                    float(cfg.entropy_coef)
                    if level != "promotion" or cfg.promotion_entropy_coef is None
                    else float(cfg.promotion_entropy_coef)
                )
                actor_loss = policy_loss - entropy_coef * entropy_mean
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(cfg.max_grad_norm))
                actor_optimizer.step()

                value_optimizer.zero_grad()
                (float(cfg.value_coef) * value_loss).backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=float(cfg.max_grad_norm))
                value_optimizer.step()

                if (
                    level == "promotion"
                    and self.promotion_advantage is not None
                    and self.promotion_advantage_optimizer is not None
                ):
                    self.promotion_advantage_optimizer.zero_grad()
                    (
                        float(cfg.promotion_advantage_coef)
                        * promotion_advantage_loss
                    ).backward()
                    nn.utils.clip_grad_norm_(
                        self.promotion_advantage.parameters(),
                        max_norm=float(cfg.max_grad_norm),
                    )
                    self.promotion_advantage_optimizer.step()

                if (
                    cost_returns_t is not None
                    and cost_value_net is not None
                    and cost_value_optimizer is not None
                ):
                    cost_value_optimizer.zero_grad()
                    (float(cfg.cost_value_coef) * cost_value_loss).backward()
                    nn.utils.clip_grad_norm_(
                        cost_value_net.parameters(), max_norm=float(cfg.max_grad_norm)
                    )
                    cost_value_optimizer.step()
                loss = (
                    actor_loss.detach()
                    + float(cfg.value_coef) * value_loss.detach()
                    + float(cfg.cost_value_coef) * cost_value_loss.detach()
                    + float(cfg.promotion_advantage_coef)
                    * promotion_advantage_loss.detach()
                )
                row = {
                    "loss": float(loss.detach().cpu().item()),
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "cost_value_loss": float(cost_value_loss.detach().cpu().item()),
                    "entropy": float(entropy_mean.detach().cpu().item()),
                    "constraint_loss": float(constraint_loss.detach().cpu().item()),
                }
                if level == "promotion":
                    row.update({
                        "rate_loss": float(
                            promotion_rate_loss.detach().cpu().item()
                        ),
                        "probability_mean": float(
                            promotion_probability_mean.detach().cpu().item()
                        ),
                        "counterfactual_surrogate": float(
                            promotion_counterfactual_surrogate.detach().cpu().item()
                        ),
                        "advantage_loss": float(
                            promotion_advantage_loss.detach().cpu().item()
                        ),
                        "advantage_prediction_mean": float(
                            promotion_advantage_prediction_mean.detach().cpu().item()
                        ),
                    })
                rows.append(row)

        out = {
            f"{level}_{key}": float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        out[f"{level}_transitions"] = float(batch.size)
        out[f"{level}_mean_duration"] = float(np.mean(batch.duration))
        out[f"{level}_actor_optimizer_steps"] = float(len(rows))
        out[f"{level}_value_optimizer_steps"] = float(len(rows))
        out[f"{level}_cost_value_optimizer_steps"] = float(
            len(rows) if cost_returns_t is not None and cost_value_optimizer is not None else 0
        )
        out[f"{level}_advantage_optimizer_steps"] = float(
            len(rows)
            if level == "promotion"
            and self.promotion_advantage_optimizer is not None
            else 0
        )
        if cost is not None:
            out[f"{level}_cost_mean"] = float(np.mean(cost))
        return out

    def update(self, batch: HierarchicalTrajectoryBatch) -> dict[str, float]:
        upper_metrics = self._update_level(
            level="upper",
            batch=batch.upper,
            actor=self.upper_actor,
            value_net=self.upper_value,
            actor_optimizer=self.upper_actor_optimizer,
            value_optimizer=self.upper_value_optimizer,
        )
        lower_metrics = self._update_level(
            level="lower",
            batch=batch.lower,
            actor=self.lower_actor,
            value_net=self.lower_value,
            cost_value_net=self.lower_cost_value,
            actor_optimizer=self.lower_actor_optimizer,
            value_optimizer=self.lower_value_optimizer,
            cost_value_optimizer=self.lower_cost_value_optimizer,
        )
        hf_metrics: dict[str, float] = {}
        if batch.hf is not None:
            if (
                self.hf_actor is None
                or self.hf_value is None
                or self.hf_actor_optimizer is None
                or self.hf_value_optimizer is None
            ):
                raise ValueError(
                    "HF trajectory provided to a model without a tactical policy"
                )
            hf_metrics = self._update_level(
                level="hf",
                batch=batch.hf,
                actor=self.hf_actor,
                value_net=self.hf_value,
                actor_optimizer=self.hf_actor_optimizer,
                value_optimizer=self.hf_value_optimizer,
            )
        elif self.hf_actor is not None:
            hf_metrics = {
                "hf_transitions": 0.0,
                "hf_actor_optimizer_steps": 0.0,
                "hf_value_optimizer_steps": 0.0,
                "hf_cost_value_optimizer_steps": 0.0,
                "hf_advantage_optimizer_steps": 0.0,
            }
        promotion_metrics: dict[str, float] = {}
        if batch.promotion is not None:
            if (
                self.promotion_actor is None
                or self.promotion_value is None
                or self.promotion_actor_optimizer is None
                or self.promotion_value_optimizer is None
            ):
                raise ValueError(
                    "promotion trajectory provided to a model without a learned gate"
                )
            promotion_metrics = self._update_level(
                level="promotion",
                batch=batch.promotion,
                actor=self.promotion_actor,
                value_net=self.promotion_value,
                actor_optimizer=self.promotion_actor_optimizer,
                value_optimizer=self.promotion_value_optimizer,
            )
        elif self.promotion_actor is not None:
            promotion_metrics = {
                "promotion_transitions": 0.0,
                "promotion_actor_optimizer_steps": 0.0,
                "promotion_value_optimizer_steps": 0.0,
                "promotion_cost_value_optimizer_steps": 0.0,
                "promotion_advantage_optimizer_steps": 0.0,
            }
        cost_mean = float(np.mean(batch.lower.cost)) if batch.lower.cost is not None else 0.0
        if batch.lower.cost is not None and float(self.config.lower_dual_lr) > 0.0:
            updated = self.constraint_lambda + float(self.config.lower_dual_lr) * (
                cost_mean - float(self.config.lower_cost_target)
            )
            self.constraint_lambda = float(np.clip(updated, 0.0, float(self.config.lower_max_lambda)))
        return {
            **upper_metrics,
            **lower_metrics,
            **hf_metrics,
            **promotion_metrics,
            "constraint_mean": cost_mean,
            "constraint_lambda": float(self.constraint_lambda),
        }
