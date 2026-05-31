"""Shared dual-level PPO actor-critic trainer for Freq-HRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class DualPPOConfig:
    upper_state_dim: int
    lower_state_dim: int
    upper_action_dim: int
    lower_action_dim: int
    hidden_dim: int = 0
    learning_rate: float = 3e-3
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
class TrajectoryBatch:
    upper_state: np.ndarray
    lower_state: np.ndarray
    upper_action: np.ndarray
    lower_action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    old_upper_logp: np.ndarray
    old_lower_logp: np.ndarray
    old_upper_value: np.ndarray
    old_lower_value: np.ndarray


def _mlp(in_dim: int, out_dim: int, hidden_dim: int) -> nn.Sequential:
    if int(hidden_dim) <= 0:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, init_log_std: float) -> None:
        super().__init__()
        self.net = _mlp(state_dim, action_dim, hidden_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim, dtype=torch.float32) * float(init_log_std))

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(state)
        std = torch.exp(self.log_std).clamp(1e-4, 3.0)
        return torch.distributions.Normal(mean, std)

    def forward(self, state: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        action = dist.rsample() if sample else dist.mean
        logp = dist.log_prob(action).sum(dim=-1)
        return action, logp

    def log_prob_entropy(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        return dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class DualActorCriticPPO:
    """PPO trainer with separate upper/lower actors and critics.

    The class is domain-agnostic: domains own state construction and action
    squashing.  The trainer only sees continuous latent actions and rewards.
    """

    def __init__(self, config: DualPPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
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
        self.upper_value = ValueNet(config.upper_state_dim, config.hidden_dim).to(self.device)
        self.lower_value = ValueNet(config.lower_state_dim, config.hidden_dim).to(self.device)
        params = (
            list(self.upper_actor.parameters())
            + list(self.lower_actor.parameters())
            + list(self.upper_value.parameters())
            + list(self.lower_value.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=float(config.learning_rate))

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "upper_actor": self.upper_actor.state_dict(),
            "lower_actor": self.lower_actor.state_dict(),
            "upper_value": self.upper_value.state_dict(),
            "lower_value": self.lower_value.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.upper_actor.load_state_dict(payload["upper_actor"])
        self.lower_actor.load_state_dict(payload["lower_actor"])
        self.upper_value.load_state_dict(payload["upper_value"])
        self.lower_value.load_state_dict(payload["lower_value"])

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)

    @torch.no_grad()
    def act_upper(self, upper_state: np.ndarray, sample: bool = True) -> dict[str, np.ndarray | float]:
        upper = self._state_tensor(upper_state)
        action, logp = self.upper_actor(upper, sample=sample)
        value = self.upper_value(upper)
        return {
            "action": action.cpu().numpy().reshape(-1),
            "logp": float(logp.item()),
            "value": float(value.item()),
        }

    @torch.no_grad()
    def act_lower(self, lower_state: np.ndarray, sample: bool = True) -> dict[str, np.ndarray | float]:
        lower = self._state_tensor(lower_state)
        action, logp = self.lower_actor(lower, sample=sample)
        value = self.lower_value(lower)
        return {
            "action": action.cpu().numpy().reshape(-1),
            "logp": float(logp.item()),
            "value": float(value.item()),
        }

    @torch.no_grad()
    def act(
        self,
        upper_state: np.ndarray,
        lower_state: np.ndarray,
        sample: bool = True,
    ) -> dict[str, np.ndarray | float]:
        upper_out = self.act_upper(upper_state, sample=sample)
        lower_out = self.act_lower(lower_state, sample=sample)
        return {
            "upper_action": upper_out["action"],
            "lower_action": lower_out["action"],
            "upper_logp": float(upper_out["logp"]),
            "lower_logp": float(lower_out["logp"]),
            "upper_value": float(upper_out["value"]),
            "lower_value": float(lower_out["value"]),
        }

    def _gae(self, reward: np.ndarray, done: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        done = np.asarray(done, dtype=np.float32).reshape(-1)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        adv = np.zeros_like(reward, dtype=np.float32)
        last_gae = 0.0
        gamma = float(self.config.gamma)
        lam = float(self.config.gae_lambda)
        for t in range(reward.size - 1, -1, -1):
            next_value = 0.0 if t == reward.size - 1 else values[t + 1]
            nonterminal = 1.0 - done[t]
            delta = reward[t] + gamma * next_value * nonterminal - values[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    def update(self, batch: TrajectoryBatch) -> dict[str, float]:
        cfg = self.config
        upper_state = torch.as_tensor(batch.upper_state, dtype=torch.float32, device=self.device)
        lower_state = torch.as_tensor(batch.lower_state, dtype=torch.float32, device=self.device)
        upper_action = torch.as_tensor(batch.upper_action, dtype=torch.float32, device=self.device)
        lower_action = torch.as_tensor(batch.lower_action, dtype=torch.float32, device=self.device)
        old_upper_logp = torch.as_tensor(batch.old_upper_logp, dtype=torch.float32, device=self.device)
        old_lower_logp = torch.as_tensor(batch.old_lower_logp, dtype=torch.float32, device=self.device)
        combined_values = np.asarray(batch.old_upper_value, dtype=np.float32) + np.asarray(batch.old_lower_value, dtype=np.float32)
        adv_np, ret_np = self._gae(batch.reward, batch.done, combined_values)
        if adv_np.size:
            adv_np = (adv_np - float(np.mean(adv_np))) / (float(np.std(adv_np)) + 1e-8)
        advantage = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(ret_np, dtype=torch.float32, device=self.device)

        n = int(upper_state.shape[0])
        if n == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        indices = np.arange(n)
        metrics: list[dict[str, float]] = []
        minibatch = max(1, min(int(cfg.minibatch_size), n))
        for _ in range(max(1, int(cfg.epochs))):
            np.random.shuffle(indices)
            for start in range(0, n, minibatch):
                idx_np = indices[start:start + minibatch]
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=self.device)
                u_logp, u_ent = self.upper_actor.log_prob_entropy(upper_state[idx], upper_action[idx])
                l_logp, l_ent = self.lower_actor.log_prob_entropy(lower_state[idx], lower_action[idx])
                log_ratio = (u_logp + l_logp) - (old_upper_logp[idx] + old_lower_logp[idx])
                ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                policy_loss = -torch.mean(torch.minimum(ratio * advantage[idx], clipped * advantage[idx]))
                u_value = self.upper_value(upper_state[idx])
                l_value = self.lower_value(lower_state[idx])
                value_loss = torch.mean((u_value + l_value - returns[idx]) ** 2)
                entropy = torch.mean(u_ent + l_ent)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.upper_actor.parameters())
                    + list(self.lower_actor.parameters())
                    + list(self.upper_value.parameters())
                    + list(self.lower_value.parameters()),
                    max_norm=float(cfg.max_grad_norm),
                )
                self.optimizer.step()
                metrics.append({
                    "loss": float(loss.detach().cpu().item()),
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                })
        return {
            key: float(np.mean([m[key] for m in metrics]))
            for key in metrics[0]
        }
