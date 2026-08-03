"""Minimal complete SAC/TD3 implementations for controlled flat baselines."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class OffPolicyConfig:
    state_dim: int
    action_dim: int
    algorithm: str = "sac"
    hidden_dim: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    init_alpha: float = 0.2
    target_entropy: float | None = None
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    exploration_noise: float = 0.1
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.algorithm = str(self.algorithm).lower()
        if self.algorithm not in {"sac", "td3"}:
            raise ValueError("algorithm must be 'sac' or 'td3'")
        if int(self.state_dim) <= 0 or int(self.action_dim) <= 0:
            raise ValueError("state_dim and action_dim must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = max(1, int(capacity))
        self.state = np.zeros((self.capacity, int(state_dim)), dtype=np.float32)
        self.action = np.zeros((self.capacity, int(action_dim)), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, int(state_dim)), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.cursor
        self.state[index] = np.asarray(state, dtype=np.float32).reshape(-1)
        self.action[index] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.reward[index, 0] = float(reward)
        self.next_state[index] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.done[index, 0] = float(bool(done))
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("cannot sample an empty replay buffer")
        indices = rng.integers(0, self.size, size=max(1, int(batch_size)))
        return {
            "state": torch.as_tensor(self.state[indices], device=device),
            "action": torch.as_tensor(self.action[indices], device=device),
            "reward": torch.as_tensor(self.reward[indices], device=device),
            "next_state": torch.as_tensor(self.next_state[indices], device=device),
            "done": torch.as_tensor(self.done[indices], device=device),
        }


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    hidden = max(1, int(hidden_dim))
    return nn.Sequential(
        nn.Linear(int(input_dim), hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, int(output_dim)),
    )


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(state_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        hidden = max(1, int(hidden_dim))
        self.body = nn.Sequential(
            nn.Linear(int(state_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, int(action_dim))
        self.log_std = nn.Linear(hidden, int(action_dim))

    def distribution(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(state)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), -5.0, 2.0)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(state)
        std = log_std.exp()
        pre_tanh = mean if deterministic else mean + std * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_prob = torch.zeros((state.shape[0], 1), device=state.device)
        else:
            normal_log_prob = -0.5 * (
                ((pre_tanh - mean) / std).pow(2)
                + 2.0 * log_std
                + float(np.log(2.0 * np.pi))
            )
            correction = torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = (normal_log_prob - correction).sum(dim=-1, keepdim=True)
        return action, log_prob


class TwinQCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = int(state_dim) + int(action_dim)
        self.q1 = _mlp(input_dim, 1, hidden_dim)
        self.q2 = _mlp(input_dim, 1, hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat([state, action], dim=-1)
        return self.q1(joined), self.q2(joined)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1))


class FlatOffPolicyActorCritic(nn.Module):
    """Twin-critic SAC or TD3 with target networks and bounded actions."""

    def __init__(self, config: OffPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        if config.algorithm == "sac":
            self.actor: nn.Module = SquashedGaussianActor(
                config.state_dim, config.action_dim, config.hidden_dim
            )
            self.actor_target = None
        else:
            self.actor = DeterministicActor(
                config.state_dim, config.action_dim, config.hidden_dim
            )
            self.actor_target = copy.deepcopy(self.actor)
        self.critic = TwinQCritic(
            config.state_dim, config.action_dim, config.hidden_dim
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.to(self.device)
        self._freeze_target(self.critic_target)
        if self.actor_target is not None:
            self._freeze_target(self.actor_target)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(config.actor_learning_rate)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(config.critic_learning_rate)
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(
                float(np.log(max(config.init_alpha, 1e-8))),
                device=self.device,
            ),
            requires_grad=(config.algorithm == "sac"),
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=float(config.alpha_learning_rate))
            if config.algorithm == "sac" else None
        )
        self.target_entropy = float(
            -config.action_dim if config.target_entropy is None else config.target_entropy
        )
        self.update_step = 0

    @staticmethod
    def _freeze_target(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def act(self, state: np.ndarray, *, sample: bool) -> np.ndarray:
        state_t = torch.as_tensor(
            np.asarray(state, dtype=np.float32).reshape(1, -1),
            device=self.device,
        )
        with torch.no_grad():
            if self.config.algorithm == "sac":
                action, _ = self.actor.sample(state_t, deterministic=not sample)  # type: ignore[attr-defined]
            else:
                action = self.actor(state_t)
                if sample:
                    action = action + float(self.config.exploration_noise) * torch.randn_like(action)
                action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.update_step += 1
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]
        with torch.no_grad():
            if self.config.algorithm == "sac":
                next_action, next_log_prob = self.actor.sample(next_state)  # type: ignore[attr-defined]
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.minimum(target_q1, target_q2) - self.alpha * next_log_prob
            else:
                if self.actor_target is None:
                    raise RuntimeError("TD3 actor target is missing")
                next_action = self.actor_target(next_state)
                noise = torch.clamp(
                    torch.randn_like(next_action) * float(self.config.target_policy_noise),
                    -float(self.config.target_noise_clip),
                    float(self.config.target_noise_clip),
                )
                next_action = torch.clamp(next_action + noise, -1.0, 1.0)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.minimum(target_q1, target_q2)
            q_target = reward + float(self.config.gamma) * (1.0 - done) * target_q

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        alpha_loss_value = 0.0
        actor_updated = False
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        if self.config.algorithm == "sac":
            policy_action, log_prob = self.actor.sample(state)  # type: ignore[attr-defined]
            policy_q1, policy_q2 = self.critic(state, policy_action)
            actor_loss = (self.alpha * log_prob - torch.minimum(policy_q1, policy_q2)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            if self.alpha_optimizer is None:
                raise RuntimeError("SAC alpha optimizer is missing")
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            actor_loss_value = float(actor_loss.detach().item())
            alpha_loss_value = float(alpha_loss.detach().item())
            actor_updated = True
            self._soft_update(self.critic_target, self.critic)
        elif self.update_step % max(1, int(self.config.policy_delay)) == 0:
            policy_action = self.actor(state)
            actor_loss = -self.critic.q1_value(state, policy_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().item())
            actor_updated = True
            if self.actor_target is None:
                raise RuntimeError("TD3 actor target is missing")
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.item()),
            "actor_updated": float(actor_updated),
            "update_step": float(self.update_step),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        tau = float(self.config.tau)
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)
