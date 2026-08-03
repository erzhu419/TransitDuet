"""
upper/resac_upper.py
====================
RE-SAC for the adaptive target-headway upper policy.

Same ensemble Q + epistemic penalty as the lower RE-SAC, but:
  - No Lagrangian cost constraint (fleet constraint via θ-OGD instead).
  - Action dim is 1 in the main HIRO configuration (per-dispatch
    target-headway shift δ_t ∈ [-120, +120] s); the constructor still accepts
    arbitrary action_dim so the legacy 3-action [H_peak, H_off, H_trans]
    headway-triple variant remains usable for ablations.
  - Bounded action via sigmoid + affine rescale to [action_low, action_high]
    (matches paper Section IV-B; not tanh).
  - Own replay buffer for dispatch-level transitions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random


# ──────────────────── Upper Replay Buffer ────────────────────

class UpperReplayBuffer:
    """Simple replay buffer for (s, a, r, s') dispatch transitions."""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a),
                np.array(r).reshape(-1, 1),
                np.array(ns), np.array(d).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)


# ──────────────────── Networks ────────────────────

class BoundedGaussianPolicy(nn.Module):
    """Gaussian policy with action mapped to [action_low, action_high] via sigmoid."""

    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 action_low=None, action_high=None):
        super().__init__()
        self.action_dim = action_dim

        if action_low is None:
            action_low = [180., 300., 240.]
        if action_high is None:
            action_high = [600., 1200., 900.]

        self.register_buffer('action_low',
                             torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high',
                             torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer('action_range',
                             self.action_high - self.action_low)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Init near sigmoid(0) = 0.5 → mid-range
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -5.0, 0.0)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        """For training: returns action, log_prob."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        # Sigmoid squashing → [0, 1] → scale to [low, high]
        u = torch.sigmoid(z)
        action = self.action_low + u * self.action_range

        # Log-prob with sigmoid correction: log_prob = log_N(z) - log(sigmoid'(z))
        # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) = u * (1 - u)
        log_prob = dist.log_prob(z) - torch.log(u * (1.0 - u) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                u = torch.sigmoid(mean)
            else:
                std = log_std.exp()
                z = Normal(mean, std).sample()
                u = torch.sigmoid(z)
            action = self.action_low + u * self.action_range
        return action.squeeze(0).cpu().numpy()

    def log_prob(self, state, action, epsilon=1e-6):
        """Compute log-prob of a given action under the current policy.
        Used by TPC-Lower for importance-weighted lower training."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        action = action.to(next(self.parameters()).device)
        with torch.no_grad():
            mean, log_std = self.forward(state)
            std = log_std.exp()
            # Invert sigmoid: u = (action - low) / range, z = log(u/(1-u))
            u = ((action - self.action_low) / self.action_range).clamp(epsilon, 1.0 - epsilon)
            z = torch.log(u / (1.0 - u))
            dist = Normal(mean, std)
            lp = dist.log_prob(z) - torch.log(u * (1.0 - u) + epsilon)
            lp = lp.sum(-1)
        return lp.cpu().numpy().squeeze()


class CategoricalPlanPolicy(nn.Module):
    """Categorical policy over a finite library of timetable curves."""

    def __init__(self, state_dim, action_candidates, hidden_dim=64,
                 init_w=3e-3):
        super().__init__()
        candidates = torch.as_tensor(
            action_candidates, dtype=torch.float32)
        if candidates.ndim != 2 or candidates.shape[0] < 2:
            raise ValueError(
                "action_candidates must have shape [n_candidates, action_dim] "
                "with at least two candidates")
        self.register_buffer("action_candidates", candidates)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, candidates.shape[0])
        self.logits.weight.data.uniform_(-init_w, init_w)
        self.logits.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.logits(x)

    def dist_info(self, state):
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs, logits

    def evaluate(self, state):
        probs, log_probs, logits = self.dist_info(state)
        idx = torch.distributions.Categorical(probs=probs).sample()
        action = self.action_candidates[idx]
        log_prob = log_probs.gather(1, idx.view(-1, 1))
        return action, log_prob, idx.view(-1, 1).float(), logits, probs

    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        with torch.no_grad():
            probs, _, _ = self.dist_info(state)
            if deterministic:
                idx = probs.argmax(dim=-1)
            else:
                idx = torch.distributions.Categorical(probs=probs).sample()
            action = self.action_candidates[idx]
        return action.squeeze(0).cpu().numpy()

    def log_prob(self, state, action):
        """Return log probability for actions that belong to the library."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        action = action.to(next(self.parameters()).device)
        with torch.no_grad():
            _, log_probs, _ = self.dist_info(state)
            distances = (
                action.unsqueeze(1)
                - self.action_candidates.unsqueeze(0)
            ).pow(2).sum(dim=-1)
            min_distance, idx = distances.min(dim=-1, keepdim=True)
            if torch.any(min_distance > 1e-6):
                raise ValueError(
                    "categorical policy received an action outside its library")
            result = log_probs.gather(1, idx).squeeze(-1)
        return result.cpu().numpy().squeeze()


class EnsembleQNetwork(nn.Module):
    """Ensemble of K Q-networks (same as lower)."""

    def __init__(self, num_inputs, num_actions, hidden_dim=64,
                 ensemble_size=10, n_layers=3):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_layers = n_layers

        dims = [num_inputs + num_actions] + [hidden_dim] * n_layers + [1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(len(dims) - 1):
            stddev = 1.0 / np.sqrt(dims[i])
            self.weights.append(nn.Parameter(
                torch.randn(ensemble_size, dims[i], dims[i+1]) * stddev))
            self.biases.append(nn.Parameter(
                torch.zeros(ensemble_size, 1, dims[i+1])))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, w) + b
            if i < self.n_layers:
                x = F.relu(x)
        return x.squeeze(-1)  # [K, B]

    def compute_l1_norm(self):
        total = torch.zeros(self.ensemble_size, device=self.weights[0].device)
        for w, b in zip(self.weights, self.biases):
            total = total + w.abs().sum(dim=(1, 2)) + b.abs().sum(dim=(1, 2))
        return total


class IndexedDiscreteEnsembleQNetwork(nn.Module):
    """Ensemble critic with one exact Q output per categorical action."""

    def __init__(self, num_inputs, action_candidates, hidden_dim=64,
                 ensemble_size=10, n_layers=3):
        super().__init__()
        candidates = torch.as_tensor(action_candidates, dtype=torch.float32)
        if candidates.ndim != 2 or candidates.shape[0] < 2:
            raise ValueError(
                "action_candidates must have shape [n_candidates, action_dim]")
        self.register_buffer("action_candidates", candidates)
        self.ensemble_size = int(ensemble_size)
        self.n_layers = int(n_layers)

        dims = [num_inputs] + [hidden_dim] * n_layers + [candidates.shape[0]]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(len(dims) - 1):
            stddev = 1.0 / np.sqrt(dims[i])
            self.weights.append(nn.Parameter(
                torch.randn(ensemble_size, dims[i], dims[i + 1]) * stddev))
            self.biases.append(nn.Parameter(
                torch.zeros(ensemble_size, 1, dims[i + 1])))

    def all_values(self, state):
        x = state.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, weight) + bias
            if i < self.n_layers:
                x = F.relu(x)
        return x  # [K, B, A]

    def action_indices(self, action):
        distances = (
            action.unsqueeze(1)
            - self.action_candidates.unsqueeze(0)
        ).pow(2).sum(dim=-1)
        min_distance, indices = distances.min(dim=-1)
        if torch.any(min_distance > 1e-6):
            raise ValueError(
                "indexed discrete critic received an action outside its library")
        return indices

    def forward(self, state, action):
        values = self.all_values(state)
        indices = self.action_indices(action)
        gather_index = indices.view(1, -1, 1).expand(
            self.ensemble_size, -1, 1)
        return values.gather(dim=-1, index=gather_index).squeeze(-1)

    def compute_l1_norm(self):
        total = torch.zeros(self.ensemble_size, device=self.weights[0].device)
        for weight, bias in zip(self.weights, self.biases):
            total = (
                total
                + weight.abs().sum(dim=(1, 2))
                + bias.abs().sum(dim=(1, 2)))
        return total


# ──────────────────── Trainer ────────────────────

class RESACUpperTrainer:
    """
    RE-SAC for upper-level timetable policy (no Lagrangian).
    Fleet constraint handled externally by θ-OGD reward modulation.
    """

    def __init__(self, state_dim=5, action_dim=3, hidden_dim=64,
                 action_low=None, action_high=None,
                 action_candidates=None,
                 discrete_critic="continuous_action",
                 ensemble_size=10, beta=-2.0, beta_ood=0.01,
                 weight_reg=0.01,
                 lr=3e-4, gamma=0.99, soft_tau=5e-3,
                 auto_entropy=True, maximum_alpha=0.3,
                 replay_capacity=50000, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.ensemble_size = ensemble_size
        self.beta = beta
        self.beta_ood = beta_ood
        self.weight_reg = weight_reg
        self.auto_entropy = auto_entropy
        self.discrete_critic = str(discrete_critic).lower()
        if self.discrete_critic not in {"continuous_action", "indexed"}:
            raise ValueError(
                "discrete_critic must be 'continuous_action' or 'indexed'")

        # Replay buffer for dispatch transitions
        self.replay_buffer = UpperReplayBuffer(replay_capacity)

        self.discrete_actions = None
        if action_candidates is not None:
            candidates = np.asarray(action_candidates, dtype=np.float32)
            if candidates.ndim != 2 or candidates.shape[1] != action_dim:
                raise ValueError(
                    "action_candidates must have shape "
                    f"[n_candidates, {action_dim}]")
            if candidates.shape[0] < 2 or not np.all(np.isfinite(candidates)):
                raise ValueError(
                    "action_candidates must contain at least two finite rows")
            if np.unique(candidates, axis=0).shape[0] != candidates.shape[0]:
                raise ValueError("action_candidates contains duplicate rows")
            self.policy_net = CategoricalPlanPolicy(
                state_dim, candidates, hidden_dim).to(device)
            self.discrete_actions = self.policy_net.action_candidates
        else:
            self.policy_net = BoundedGaussianPolicy(
                state_dim, action_dim, hidden_dim,
                action_low, action_high).to(device)

        if self.discrete_actions is None and self.discrete_critic != "continuous_action":
            raise ValueError(
                "indexed discrete critic requires action_candidates")

        # Ensemble Q
        if self.discrete_actions is not None and self.discrete_critic == "indexed":
            self.q_net = IndexedDiscreteEnsembleQNetwork(
                state_dim, self.discrete_actions, hidden_dim,
                ensemble_size).to(device)
            self.target_q_net = IndexedDiscreteEnsembleQNetwork(
                state_dim, self.discrete_actions, hidden_dim,
                ensemble_size).to(device)
        else:
            self.q_net = EnsembleQNetwork(
                state_dim, action_dim, hidden_dim, ensemble_size).to(device)
            self.target_q_net = EnsembleQNetwork(
                state_dim, action_dim, hidden_dim, ensemble_size).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Entropy
        if auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            if self.discrete_actions is not None:
                self.target_entropy = 0.98 * float(np.log(
                    max(int(self.discrete_actions.shape[0]), 2)))
            else:
                self.target_entropy = -float(action_dim)
        self.alpha = 0.1
        self.maximum_alpha = maximum_alpha

    def _discrete_q_values(self, q_net, state):
        candidates = self.discrete_actions
        if candidates is None:
            raise RuntimeError(
                "discrete Q values requested without action candidates")
        if isinstance(q_net, IndexedDiscreteEnsembleQNetwork):
            return q_net.all_values(state)
        batch = state.shape[0]
        n_actions, action_dim = candidates.shape
        state_rep = (
            state.unsqueeze(1)
            .expand(batch, n_actions, state.shape[-1])
            .reshape(batch * n_actions, state.shape[-1])
        )
        action_rep = (
            candidates.unsqueeze(0)
            .expand(batch, n_actions, action_dim)
            .reshape(batch * n_actions, action_dim)
        )
        q_flat = q_net(state_rep, action_rep)
        return q_flat.view(q_flat.shape[0], batch, n_actions)

    def update(self, batch_size=64):
        """One gradient step from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return {}

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        discrete_policy = self.discrete_actions is not None

        # ── Critic update ──
        with torch.no_grad():
            if discrete_policy:
                next_probs, next_log_probs, _ = self.policy_net.dist_info(
                    next_state)
                target_q_all = self._discrete_q_values(
                    self.target_q_net, next_state)
                target_q_mean = target_q_all.mean(dim=0)
                target_q_mean = (next_probs * (
                    target_q_mean
                    - self.alpha * next_log_probs)).sum(dim=-1)
            else:
                next_action, next_log_prob, _, _, _ = (
                    self.policy_net.evaluate(next_state))
                target_q_all = self.target_q_net(
                    next_state, next_action)
                target_q_mean = target_q_all.mean(dim=0)
                target_q_mean = (
                    target_q_mean - self.alpha
                    * next_log_prob.squeeze(-1))
            r = reward.squeeze(-1)
            d = done.squeeze(-1)
            shared_target = r + (1.0 - d) * self.gamma * target_q_mean
            shared_target = shared_target.clamp(-50.0, 50.0)

        predicted_q = self.q_net(state, action)
        target_value = shared_target.unsqueeze(0).expand(
            predicted_q.shape[0], -1)  # [K, B]
        q_mse = F.mse_loss(predicted_q, target_value)
        ood_loss = predicted_q.std(dim=0).mean()
        l1_norm = self.q_net.compute_l1_norm().mean()
        q_loss = q_mse + self.beta_ood * ood_loss + self.weight_reg * l1_norm

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 50.0)
        self.q_optimizer.step()

        # ── Policy update ──
        if discrete_policy:
            probs, log_probs, _ = self.policy_net.dist_info(state)
            q_all = self._discrete_q_values(self.q_net, state)
            q_mean = q_all.mean(dim=0)
            q_std = q_all.std(dim=0)
            q_lcb = q_mean + self.beta * q_std
            policy_loss = (probs * (
                self.alpha * log_probs - q_lcb)).sum(dim=-1).mean()
            entropy_log_prob = (
                probs * log_probs).sum(dim=-1, keepdim=True)
        else:
            new_action, log_prob, _, _, _ = self.policy_net.evaluate(state)
            q_all = self.q_net(state, new_action)
            q_mean = q_all.mean(dim=0)
            q_std = q_all.std(dim=0)
            q_lcb = q_mean + self.beta * q_std
            policy_loss = (
                self.alpha * log_prob.squeeze(-1) - q_lcb).mean()
            entropy_log_prob = log_prob

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.policy_optimizer.step()

        # ── Alpha update ──
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha *
                           (entropy_log_prob
                            + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = min(self.log_alpha.exp().item(), self.maximum_alpha)

        # ── Soft target update ──
        for tp, p in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.soft_tau) + p.data * self.soft_tau)

        return {
            'upper_q_mean': q_mean.mean().item(),
            'upper_q_std': q_std.mean().item(),
            'upper_policy_loss': policy_loss.item(),
            'upper_q_loss': q_loss.item(),
            'upper_q_mse': q_mse.item(),
            'upper_ood_loss': ood_loss.item(),
            'upper_alpha': self.alpha,
            'upper_pi_grad_norm': pi_grad_norm.item() if isinstance(pi_grad_norm, torch.Tensor) else float(pi_grad_norm),
            'upper_q_grad_norm': q_grad_norm.item() if isinstance(q_grad_norm, torch.Tensor) else float(q_grad_norm),
            'upper_reward_batch_mean': reward.mean().item(),
            'upper_reward_batch_std': reward.std().item(),
            'upper_action_batch_mean': action.mean().item(),
            'upper_action_batch_std': action.std().item(),
        }

    def save(self, path):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'q_net': self.q_net.state_dict(),
            'log_alpha': self.log_alpha.data if self.auto_entropy else None,
            'discrete_critic': self.discrete_critic,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        saved_critic = str(
            ckpt.get('discrete_critic', 'continuous_action')).lower()
        if saved_critic != self.discrete_critic:
            raise ValueError(
                "checkpoint discrete critic does not match configured "
                f"critic: saved={saved_critic}, configured={self.discrete_critic}")
        saved_candidates = ckpt['policy'].get('action_candidates')
        if self.discrete_actions is None and saved_candidates is not None:
            raise ValueError(
                "cannot load a categorical checkpoint into a continuous policy")
        if self.discrete_actions is not None:
            if saved_candidates is None:
                raise ValueError(
                    "cannot load a continuous checkpoint into a categorical policy")
            configured = self.discrete_actions.detach().cpu()
            saved = saved_candidates.detach().cpu()
            if configured.shape != saved.shape or not torch.equal(
                    configured, saved):
                raise ValueError(
                    "checkpoint action library does not match the configured library")
        self.policy_net.load_state_dict(ckpt['policy'])
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_q_net.load_state_dict(ckpt['q_net'])
        if ckpt.get('log_alpha') is not None:
            self.log_alpha.data = ckpt['log_alpha']
