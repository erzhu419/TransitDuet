"""Canonical flat continuous-action PPO used as a learned control baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch import nn

from .causal_sequence import CausalGRUGaussianActor, CausalGRUValueNet
from .dual_actor_critic import GaussianActor, ValueNet


@dataclass
class JointPPOConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    state_encoder: str = "mlp"
    raw_history_window: int = 0
    raw_feature_dim: int = 0
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0
    epochs: int = 4
    minibatch_size: int = 512
    init_log_std: float = -1.0
    device: str = "cpu"


@dataclass
class JointTrajectoryBatch:
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    old_logp: np.ndarray
    old_value: np.ndarray
    next_value: np.ndarray | None = None
    terminal: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(np.asarray(self.reward).reshape(-1).size)

    def validate(self, *, state_dim: int, action_dim: int) -> None:
        state = np.asarray(self.state)
        action = np.asarray(self.action)
        if state.ndim != 2 or state.shape[1] != int(state_dim):
            raise ValueError(
                f"joint state shape must be (n, {state_dim}), got {state.shape}"
            )
        if action.ndim != 2 or action.shape != (state.shape[0], int(action_dim)):
            raise ValueError(
                f"joint action shape must be ({state.shape[0]}, {action_dim}), "
                f"got {action.shape}"
            )
        n = int(state.shape[0])
        for name in ("reward", "done", "old_logp", "old_value"):
            values = np.asarray(getattr(self, name)).reshape(-1)
            if values.size != n:
                raise ValueError(f"joint {name} length must be {n}, got {values.size}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"joint {name} must be finite")
        if (self.next_value is None) != (self.terminal is None):
            raise ValueError(
                "joint next_value and terminal must be provided together"
            )
        for name in ("next_value", "terminal"):
            optional = getattr(self, name)
            if optional is None:
                continue
            values = np.asarray(optional).reshape(-1)
            if values.size != n or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"joint {name} must contain {n} finite values"
                )
        if self.terminal is not None:
            terminal = np.asarray(self.terminal, dtype=np.float32).reshape(-1)
            if np.any((terminal < 0.0) | (terminal > 1.0)):
                raise ValueError("joint terminal must be in [0, 1]")


def concat_joint_batches(
    batches: Iterable[JointTrajectoryBatch],
) -> JointTrajectoryBatch:
    items = list(batches)
    if not items:
        raise ValueError("at least one joint trajectory batch is required")
    explicit_bootstrap = [
        item.next_value is not None and item.terminal is not None
        for item in items
    ]
    if any(explicit_bootstrap) and not all(explicit_bootstrap):
        raise ValueError(
            "joint explicit bootstrap fields must be present for every batch"
        )
    return JointTrajectoryBatch(
        state=np.concatenate([np.asarray(item.state) for item in items], axis=0),
        action=np.concatenate([np.asarray(item.action) for item in items], axis=0),
        reward=np.concatenate([
            np.asarray(item.reward).reshape(-1) for item in items
        ]),
        done=np.concatenate([np.asarray(item.done).reshape(-1) for item in items]),
        old_logp=np.concatenate([
            np.asarray(item.old_logp).reshape(-1) for item in items
        ]),
        old_value=np.concatenate([
            np.asarray(item.old_value).reshape(-1) for item in items
        ]),
        next_value=(
            np.concatenate([
                np.asarray(item.next_value).reshape(-1) for item in items
            ])
            if all(explicit_bootstrap) else None
        ),
        terminal=(
            np.concatenate([
                np.asarray(item.terminal).reshape(-1) for item in items
            ])
            if all(explicit_bootstrap) else None
        ),
    )


class JointActorCriticPPO:
    """Single-state, single-value PPO over one joint continuous action.

    The action may contain multiple semantic components, but PPO computes one
    joint log-probability ratio and one task-return advantage. There is no
    temporal abstraction, upper/lower credit split, or frequency router.
    """

    def __init__(self, config: JointPPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if str(config.state_encoder) == "mlp":
            self.actor = GaussianActor(
                config.state_dim,
                config.action_dim,
                config.hidden_dim,
                config.init_log_std,
            ).to(self.device)
            self.value = ValueNet(
                config.state_dim, config.hidden_dim
            ).to(self.device)
        elif str(config.state_encoder) == "causal_gru":
            self.actor = CausalGRUGaussianActor(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                history_window=config.raw_history_window,
                raw_feature_dim=config.raw_feature_dim,
                hidden_dim=config.hidden_dim,
                init_log_std=config.init_log_std,
            ).to(self.device)
            self.value = CausalGRUValueNet(
                state_dim=config.state_dim,
                history_window=config.raw_history_window,
                raw_feature_dim=config.raw_feature_dim,
                hidden_dim=config.hidden_dim,
            ).to(self.device)
        else:
            raise ValueError(f"unknown state_encoder: {config.state_encoder}")
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(config.learning_rate)
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=float(config.learning_rate)
        )

    def parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor.parameters()
        yield from self.value.parameters()

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.value.load_state_dict(payload["value"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "value_optimizer" in payload:
            self.value_optimizer.load_state_dict(payload["value_optimizer"])
        self.reset_recurrent_inference()

    def reset_recurrent_inference(self) -> None:
        for module in (self.actor, self.value):
            reset = getattr(module, "reset_inference_state", None)
            if reset is not None:
                reset()

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        sample: bool = True,
    ) -> dict[str, np.ndarray | float]:
        tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).view(1, -1)
        if str(self.config.state_encoder) == "causal_gru":
            action, logp = self.actor.forward_incremental(
                tensor, sample=sample
            )
            value = self.value.forward_incremental(tensor)
        else:
            action, logp = self.actor(tensor, sample=sample)
            value = self.value(tensor)
        return {
            "action": action.cpu().numpy().reshape(-1),
            "logp": float(logp.item()),
            "value": float(value.item()),
        }

    def _gae(
        self,
        reward: np.ndarray,
        done: np.ndarray,
        values: np.ndarray,
        next_value: np.ndarray | None = None,
        terminal: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        done = np.asarray(done, dtype=np.float32).reshape(-1)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        explicit = next_value is not None or terminal is not None
        if explicit and (next_value is None or terminal is None):
            raise ValueError(
                "next_value and terminal must be provided together"
            )
        next_values = (
            np.asarray(next_value, dtype=np.float32).reshape(-1)
            if explicit else None
        )
        terminals = (
            np.asarray(terminal, dtype=np.float32).reshape(-1)
            if explicit else None
        )
        if explicit and (
            next_values.size != reward.size or terminals.size != reward.size
        ):
            raise ValueError("explicit bootstrap arrays must match reward")
        advantage = np.zeros_like(reward)
        last = 0.0
        for index in range(reward.size - 1, -1, -1):
            trace_continue = 1.0 - float(done[index])
            if explicit:
                bootstrap_continue = 1.0 - float(terminals[index])
                successor_value = float(next_values[index])
            else:
                bootstrap_continue = trace_continue
                successor_value = (
                    0.0 if index == reward.size - 1
                    else float(values[index + 1])
                )
            delta = (
                float(reward[index])
                + float(self.config.gamma)
                * successor_value
                * bootstrap_continue
                - float(values[index])
            )
            last = delta + (
                float(self.config.gamma)
                * float(self.config.gae_lambda)
                * trace_continue
                * last
            )
            advantage[index] = last
        return advantage, advantage + values

    def update(self, batch: JointTrajectoryBatch) -> dict[str, float]:
        cfg = self.config
        batch.validate(state_dim=cfg.state_dim, action_dim=cfg.action_dim)
        if batch.size == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "actor_optimizer_steps": 0.0,
                "value_optimizer_steps": 0.0,
            }

        state = torch.as_tensor(batch.state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(
            batch.old_logp, dtype=torch.float32, device=self.device
        ).reshape(-1)
        advantage, returns = self._gae(
            batch.reward,
            batch.done,
            batch.old_value,
            batch.next_value,
            batch.terminal,
        )
        if advantage.size > 1:
            advantage = (
                advantage - float(np.mean(advantage))
            ) / (float(np.std(advantage)) + 1e-8)
        advantage_t = torch.as_tensor(
            advantage, dtype=torch.float32, device=self.device
        )
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        indices = np.arange(batch.size)
        minibatch = max(1, min(int(cfg.minibatch_size), batch.size))
        rows: list[dict[str, float]] = []
        for _ in range(max(1, int(cfg.epochs))):
            np.random.shuffle(indices)
            for start in range(0, batch.size, minibatch):
                idx_np = indices[start:start + minibatch]
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=self.device)
                logp, entropy = self.actor.log_prob_entropy(state[idx], action[idx])
                log_ratio = (logp - old_logp[idx]).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(
                    ratio, 1.0 - float(cfg.clip_ratio), 1.0 + float(cfg.clip_ratio)
                )
                policy_loss = -torch.minimum(
                    ratio * advantage_t[idx], clipped * advantage_t[idx]
                ).mean()
                entropy_mean = entropy.mean()
                actor_loss = policy_loss - float(cfg.entropy_coef) * entropy_mean
                value_loss = torch.mean((self.value(state[idx]) - returns_t[idx]) ** 2)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=float(cfg.max_grad_norm)
                )
                self.actor_optimizer.step()

                self.value_optimizer.zero_grad()
                (float(cfg.value_coef) * value_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.value.parameters(), max_norm=float(cfg.max_grad_norm)
                )
                self.value_optimizer.step()

                rows.append({
                    "loss": float(
                        (actor_loss.detach() + float(cfg.value_coef) * value_loss.detach())
                        .cpu()
                        .item()
                    ),
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy_mean.detach().cpu().item()),
                    "approx_kl": float((-log_ratio).mean().detach().cpu().item()),
                    "clip_fraction": float(
                        (torch.abs(ratio - 1.0) > float(cfg.clip_ratio))
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    ),
                })

        out = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        }
        optimizer_steps = float(len(rows))
        return {
            **out,
            "actor_optimizer_steps": optimizer_steps,
            "value_optimizer_steps": optimizer_steps,
        }
