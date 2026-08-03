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

from .causal_sequence import CausalGRUGaussianActor, CausalGRUValueNet
from .dual_actor_critic import BernoulliActor, GaussianActor, ValueNet


@dataclass
class SMDPPPOConfig:
    upper_state_dim: int
    lower_state_dim: int
    upper_action_dim: int
    lower_action_dim: int
    promotion_state_dim: int = 0
    hidden_dim: int = 0
    state_encoder: str = "mlp"
    raw_history_window: int = 0
    raw_feature_dim: int = 0
    upper_learning_rate: float = 3e-3
    lower_learning_rate: float = 3e-3
    promotion_learning_rate: float = 0.0
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    cost_value_coef: float = 0.5
    entropy_coef: float = 0.001
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

    @property
    def size(self) -> int:
        return int(np.asarray(self.reward).reshape(-1).size)


@dataclass
class HierarchicalTrajectoryBatch:
    upper: LevelTrajectoryBatch
    lower: LevelTrajectoryBatch
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
        self._pending_upper: dict[str, Any] | None = None

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
    ) -> None:
        if self._pending_upper is None:
            raise RuntimeError("begin_upper must be called before add_lower")
        self._lower["state"].append(np.asarray(state, dtype=np.float32).copy())
        self._lower["action"].append(np.asarray(action, dtype=np.float32).copy())
        self._lower["reward"].append(float(reward))
        self._lower["duration"].append(1)
        self._lower["done"].append(float(bool(done)))
        self._lower["old_logp"].append(float(logp))
        self._lower["old_value"].append(float(value))
        self._lower["cost"].append(float(cost))
        self._pending_upper["rewards"].append(float(reward if upper_reward is None else upper_reward))
        self._pending_upper["costs"].append(float(cost if upper_cost is None else upper_cost))
        if done:
            self._close_upper(done=True)

    def finish(self, *, terminal: bool = True) -> None:
        if self._pending_upper is not None:
            self._close_upper(done=bool(terminal))
        if self._lower["done"] and terminal:
            self._lower["done"][-1] = 1.0

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
        )

    def build(self) -> HierarchicalTrajectoryBatch:
        if self._pending_upper is not None:
            raise RuntimeError("finish must be called before build")
        if not self._upper["reward"] or not self._lower["reward"]:
            raise ValueError("rollout must contain upper and lower transitions")
        return HierarchicalTrajectoryBatch(
            upper=self._level(self._upper),
            lower=self._level(self._lower),
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
            )
        }
        self._pending: dict[str, Any] | None = None

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

    def add_reward(self, reward: float, *, done: bool = False) -> None:
        if self._pending is None:
            return
        self._pending["rewards"].append(float(reward))
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
    )


def concat_hierarchical_batches(
    batches: Iterable[HierarchicalTrajectoryBatch],
) -> HierarchicalTrajectoryBatch:
    items = list(batches)
    if not items:
        raise ValueError("at least one hierarchical batch is required")
    promotion_batches = [item.promotion for item in items]
    if any(item is None for item in promotion_batches) and not all(
        item is None for item in promotion_batches
    ):
        raise ValueError(
            "promotion trajectories must be present for every batch or none"
        )
    return HierarchicalTrajectoryBatch(
        upper=concat_level_batches(item.upper for item in items),
        lower=concat_level_batches(item.lower for item in items),
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
        else:
            raise ValueError(f"unknown state_encoder: {config.state_encoder}")
        self.promotion_actor: BernoulliActor | None = None
        self.promotion_value: ValueNet | None = None
        self.promotion_actor_optimizer: torch.optim.Optimizer | None = None
        self.promotion_value_optimizer: torch.optim.Optimizer | None = None
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
        return payload

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.upper_actor.load_state_dict(payload["upper_actor"])
        self.lower_actor.load_state_dict(payload["lower_actor"])
        self.upper_value.load_state_dict(payload["upper_value"])
        self.lower_value.load_state_dict(payload["lower_value"])
        self.lower_cost_value.load_state_dict(payload["lower_cost_value"])
        if self.promotion_actor is not None and self.promotion_value is not None:
            if "promotion_actor" not in payload or "promotion_value" not in payload:
                raise ValueError(
                    "checkpoint is missing the configured learned promotion gate"
                )
            self.promotion_actor.load_state_dict(payload["promotion_actor"])
            self.promotion_value.load_state_dict(payload["promotion_value"])
        for name in (
            "upper_actor_optimizer",
            "upper_value_optimizer",
            "lower_actor_optimizer",
            "lower_value_optimizer",
            "lower_cost_value_optimizer",
            "promotion_actor_optimizer",
            "promotion_value_optimizer",
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
    def act_promotion(
        self,
        state: np.ndarray,
        sample: bool = True,
    ) -> dict[str, float]:
        if self.promotion_actor is None or self.promotion_value is None:
            raise RuntimeError("learned promotion gate is not configured")
        tensor = self._state_tensor(state)
        action, logp = self.promotion_actor(tensor, sample=sample)
        value = self.promotion_value(tensor)
        probability = self.promotion_actor.distribution(tensor).probs
        return {
            "action": float(action.item()),
            "probability": float(probability.item()),
            "logp": float(logp.item()),
            "value": float(value.item()),
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
            }

        state = torch.as_tensor(batch.state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(batch.old_logp, dtype=torch.float32, device=self.device)
        reward_adv, returns = self._gae(batch.reward, batch.done, batch.duration, batch.old_value)
        reward_adv_t = torch.as_tensor(self._normalize(reward_adv), dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

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
                policy_loss = -reward_surrogate + constraint_loss
                value_loss = torch.mean((value_net(state[idx]) - returns_t[idx]) ** 2)
                cost_value_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if cost_returns_t is not None and cost_value_net is not None:
                    cost_value_loss = torch.mean((cost_value_net(state[idx]) - cost_returns_t[idx]) ** 2)
                entropy_mean = entropy.mean()
                actor_loss = policy_loss - float(cfg.entropy_coef) * entropy_mean
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(cfg.max_grad_norm))
                actor_optimizer.step()

                value_optimizer.zero_grad()
                (float(cfg.value_coef) * value_loss).backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=float(cfg.max_grad_norm))
                value_optimizer.step()

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
                )
                rows.append({
                    "loss": float(loss.detach().cpu().item()),
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "cost_value_loss": float(cost_value_loss.detach().cpu().item()),
                    "entropy": float(entropy_mean.detach().cpu().item()),
                    "constraint_loss": float(constraint_loss.detach().cpu().item()),
                })

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
            **promotion_metrics,
            "constraint_mean": cost_mean,
            "constraint_lambda": float(self.constraint_lambda),
        }
