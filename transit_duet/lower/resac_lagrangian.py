"""
lower/resac_lagrangian.py
=========================
RE-SAC (Robust Ensemble SAC) with Lagrangian cost constraint.

Key differences from vanilla SAC (dsac_lagrangian.py):
  - Ensemble Q-networks (K=10) instead of twin-Q
  - Independent targets per ensemble member (no min)
  - Epistemic penalty: policy loss uses mean(Q) + beta*std(Q) with beta<0
  - OOD regularization on critic: penalize cross-ensemble disagreement
  - L1 weight regularization on critic
  - Lagrangian cost constraint (same as before)

Based on RE-SAC paper and /home/erzhu419/mine_code/RE-SAC/ implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


# ──────────────────── Networks ────────────────────

class GaussianPolicy(nn.Module):
    """Gaussian policy for holding time in [0, action_range]."""

    def __init__(self, num_inputs, hidden_dim=64, action_range=60.0, init_w=3e-3):
        super().__init__()
        self.action_range = action_range
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)
        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        action = action * self.action_range
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=False):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean) * self.action_range
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            z = dist.sample()
            action = torch.tanh(z) * self.action_range
        return action.detach().squeeze().cpu().numpy()


class EnsembleQNetwork(nn.Module):
    """
    Ensemble of K Q-networks for RE-SAC.
    Vectorized: all K critics stored as [K, in, out] tensors.
    """

    def __init__(self, num_inputs, num_actions, hidden_dim=64,
                 ensemble_size=10, n_layers=3):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_layers = n_layers

        # Build vectorized layers: weight [K, in, out], bias [K, 1, out]
        dims = [num_inputs + num_actions] + [hidden_dim] * n_layers + [1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(len(dims) - 1):
            stddev = 1.0 / np.sqrt(dims[i])
            w = nn.Parameter(torch.randn(ensemble_size, dims[i], dims[i+1]) * stddev)
            b = nn.Parameter(torch.zeros(ensemble_size, 1, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, state, action):
        """
        Args:
            state: [batch, state_dim]
            action: [batch, action_dim]
        Returns:
            [ensemble_size, batch] Q-values
        """
        x = torch.cat([state, action], dim=-1)  # [B, in]
        x = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)  # [K, B, in]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, w) + b  # [K, B, out]
            if i < self.n_layers:  # ReLU for hidden layers only
                x = F.relu(x)

        return x.squeeze(-1)  # [K, B]

    def compute_l1_norm(self):
        """L1 norm per ensemble member for regularization. Returns [K]."""
        total = torch.zeros(self.ensemble_size, device=self.weights[0].device)
        for w, b in zip(self.weights, self.biases):
            total = total + w.abs().sum(dim=(1, 2)) + b.abs().sum(dim=(1, 2))
        return total


class CostQNetwork(nn.Module):
    """Single Q-network for cost value estimation (not ensembled)."""

    def __init__(self, num_inputs, num_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


# ──────────────────── Trainer ────────────────────

class RESACLagrangianTrainer:
    """
    RE-SAC trainer with Lagrangian cost constraint.

    Key RE-SAC features vs vanilla SAC:
      - Ensemble Q (K=10): independent targets, no min
      - Policy loss: alpha*log_prob - (Q_mean + beta*Q_std) + lambda*Q_cost
        with beta < 0 → pessimistic (penalizes high uncertainty)
      - OOD reg: beta_ood * Q_std on critic loss
      - L1 weight reg on ensemble critic
    """

    def __init__(self, state_dim, action_dim=1, hidden_dim=64,
                 action_range=60.0, cost_limit=0.15,
                 ensemble_size=10, beta=-2.0, beta_ood=0.01,
                 weight_reg=0.01,
                 lr=3e-4, lambda_lr=1e-3, gamma=0.99, soft_tau=5e-3,
                 auto_entropy=True, maximum_alpha=0.3,
                 device='cpu'):
        self.device = device
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.cost_limit = cost_limit
        self.auto_entropy = auto_entropy
        self.ensemble_size = ensemble_size
        self.beta = beta              # LCB coefficient (negative = pessimistic)
        self.beta_ood = beta_ood      # OOD regularization weight
        self.weight_reg = weight_reg  # L1 regularization weight

        # Policy
        self.policy_net = GaussianPolicy(
            state_dim, hidden_dim, action_range).to(device)

        # Ensemble Q-networks
        self.q_net = EnsembleQNetwork(
            state_dim, action_dim, hidden_dim, ensemble_size).to(device)
        self.target_q_net = EnsembleQNetwork(
            state_dim, action_dim, hidden_dim, ensemble_size).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Cost Q-network (single, not ensembled)
        self.cost_q_net = CostQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_cost_q_net = CostQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_cost_q_net.load_state_dict(self.cost_q_net.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.cost_q_optimizer = optim.Adam(self.cost_q_net.parameters(), lr=lr)

        # Entropy temperature alpha
        if auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = -1.0 * action_dim
        self.alpha = 0.1
        self.maximum_alpha = maximum_alpha

        # Lagrangian multiplier lambda
        self.log_lambda = torch.zeros(1, requires_grad=True, device=device)
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=lambda_lr)

    @property
    def lambda_param(self):
        return self.log_lambda.exp().item()

    def update(self, replay_buffer, batch_size, reward_scale=10.0,
               update_policy=True, tap_signal=None):
        """One gradient step for ensemble critic, policy, cost critic, and lambda."""
        state, action, reward, cost, next_state, done, trip_ids = \
            replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device) * reward_scale
        cost = torch.FloatTensor(cost).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # TAP bonus
        if tap_signal is not None:
            tap_bonus = torch.zeros_like(reward)
            for i, tid in enumerate(trip_ids):
                if tid in tap_signal:
                    tap_bonus[i] = tap_signal[tid] * reward_scale
            reward = reward + tap_bonus

        # ──── Ensemble Critic update ────
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
            # Independent targets per ensemble member (NOT min)
            target_q_all = self.target_q_net(next_state, next_action)  # [K, B]
            target_q_all = target_q_all - self.alpha * next_log_prob.squeeze(-1)  # [K, B]
            # reward is [B, 1], done is [B, 1] → squeeze for broadcast
            r = reward.squeeze(-1)   # [B]
            d = done.squeeze(-1)     # [B]
            target_value = r.unsqueeze(0) + (1.0 - d.unsqueeze(0)) * self.gamma * target_q_all  # [K, B]

        predicted_q = self.q_net(state, action)  # [K, B]
        # MSE loss per ensemble member, averaged
        q_mse_loss = F.mse_loss(predicted_q, target_value)

        # OOD regularization: penalize cross-ensemble disagreement
        ood_loss = predicted_q.std(dim=0).mean()

        # L1 weight regularization
        l1_norm = self.q_net.compute_l1_norm().mean()

        q_loss = q_mse_loss + self.beta_ood * ood_loss + self.weight_reg * l1_norm

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_optimizer.step()

        # ──── Cost critic update ────
        with torch.no_grad():
            next_action_c, _, _, _, _ = self.policy_net.evaluate(next_state)
            target_cost_q = self.target_cost_q_net(next_state, next_action_c)
            target_cost_value = cost + (1.0 - done) * self.gamma * target_cost_q

        cost_q = self.cost_q_net(state, action)
        cost_q_loss = F.mse_loss(cost_q, target_cost_value)

        self.cost_q_optimizer.zero_grad()
        cost_q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_q_net.parameters(), 1.0)
        self.cost_q_optimizer.step()

        # ──── Policy update ────
        metrics = {
            'q_loss': q_loss.item(),
            'q_mse': q_mse_loss.item(),
            'ood_loss': ood_loss.item(),
            'cost_q_loss': cost_q_loss.item(),
        }

        if update_policy:
            new_action, log_prob, _, _, _ = self.policy_net.evaluate(state)

            # RE-SAC: ensemble Q statistics
            q_all = self.q_net(state, new_action)  # [K, B]
            q_mean = q_all.mean(dim=0)              # [B]
            q_std = q_all.std(dim=0)                # [B]

            # LCB: pessimistic Q estimate (beta < 0 → subtract uncertainty)
            q_lcb = q_mean + self.beta * q_std      # [B]

            cost_q_new = self.cost_q_net(state, new_action)
            lam = self.log_lambda.exp().detach()

            policy_loss = (self.alpha * log_prob.squeeze(-1)
                           - q_lcb
                           + lam * cost_q_new.squeeze(-1)).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()

            # ──── Alpha update ────
            if self.auto_entropy:
                alpha_loss = -(self.log_alpha *
                               (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = min(self.log_alpha.exp().item(), self.maximum_alpha)

            # ──── Lambda update ────
            lambda_loss = -self.log_lambda.exp() * (
                self.cost_limit - cost_q_new.mean().detach())
            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()
            self.log_lambda.data.clamp_(min=-10.0, max=5.0)

            metrics.update({
                'policy_loss': policy_loss.item(),
                'alpha': self.alpha,
                'lambda': self.lambda_param,
                'q_mean': q_mean.mean().item(),
                'q_std': q_std.mean().item(),
                'cost_q_mean': cost_q_new.mean().item(),
            })

        # ──── Soft target update ────
        for tp, p in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.soft_tau) + p.data * self.soft_tau)
        for tp, p in zip(self.target_cost_q_net.parameters(),
                         self.cost_q_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.soft_tau) + p.data * self.soft_tau)

        return metrics

    def save(self, path):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'q_net': self.q_net.state_dict(),
            'cost_q_net': self.cost_q_net.state_dict(),
            'log_lambda': self.log_lambda.data,
            'log_alpha': self.log_alpha.data if self.auto_entropy else None,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_q_net.load_state_dict(ckpt['q_net'])
        self.cost_q_net.load_state_dict(ckpt['cost_q_net'])
        self.target_cost_q_net.load_state_dict(ckpt['cost_q_net'])
        self.log_lambda.data = ckpt['log_lambda']
        if ckpt.get('log_alpha') is not None:
            self.log_alpha.data = ckpt['log_alpha']
