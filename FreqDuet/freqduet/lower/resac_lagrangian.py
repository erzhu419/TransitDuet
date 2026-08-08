"""
lower/resac_lagrangian.py
=========================
Pessimistic ensemble SAC with a Lagrangian cost constraint.

The historical module and class names are retained for checkpoint compatibility.
This implementation is not claimed to reproduce a named RE-SAC paper exactly.
Its explicit algorithm identifier is
``pessimistic_ensemble_sac_lagrangian_v4``.

Key differences from vanilla SAC (dsac_lagrangian.py):
  - Ensemble Q-networks (K=10) instead of twin-Q
  - Shared ensemble-mean Bellman target instead of twin-min backup
  - Epistemic penalty: policy loss uses mean(Q) + beta*std(Q) with beta<0
  - OOD regularization on critic: penalize cross-ensemble disagreement
  - L1 weight regularization on critic
  - Lagrangian cost constraint (same as before)

The v4 paper describes these equations directly and includes a standard
constrained-SAC ablation instead of relying on a method-name equivalence claim.
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

    def __init__(self, num_inputs, hidden_dim=64, action_range=60.0,
                 init_w=3e-3, entropy_action_coordinates="physical_legacy",
                 sample_seed=None, device="cpu"):
        super().__init__()
        self.action_range = action_range
        self.entropy_action_coordinates = str(
            entropy_action_coordinates).strip().lower()
        if self.entropy_action_coordinates not in {
                "physical_legacy", "normalized_unit_interval"}:
            raise ValueError(
                "entropy_action_coordinates must be physical_legacy or "
                "normalized_unit_interval")
        self._sample_generator = None
        if sample_seed is not None:
            self._sample_generator = torch.Generator(
                device=torch.device(device).type)
            self._sample_generator.manual_seed(int(sample_seed))
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
        if self._sample_generator is None:
            z = dist.rsample()
        else:
            noise = torch.randn(
                mean.shape, dtype=mean.dtype, device=mean.device,
                generator=self._sample_generator)
            z = mean + std * noise
        squashed = torch.tanh(z)
        action = (squashed + 1.0) * 0.5 * self.action_range
        scale = (self.action_range if
                 self.entropy_action_coordinates == "physical_legacy" else 1.0)
        # Entropy is defined either in physical seconds (legacy) or in the
        # dimensionless unit interval used by the v4 optimization contract.
        log_det = torch.log(0.5 * scale * (1 - squashed.pow(2)) + epsilon)
        log_prob = dist.log_prob(z) - log_det
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        mean, log_std = self.forward(state)
        if deterministic:
            action = (torch.tanh(mean) + 1.0) * 0.5 * self.action_range
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            if self._sample_generator is None:
                z = dist.sample()
            else:
                noise = torch.randn(
                    mean.shape, dtype=mean.dtype, device=mean.device,
                    generator=self._sample_generator)
                z = mean + std * noise
            action = (torch.tanh(z) + 1.0) * 0.5 * self.action_range
        return action.detach().squeeze().cpu().numpy()

    def sampling_state(self):
        if self._sample_generator is None:
            return None
        return self._sample_generator.get_state()

    def set_sampling_state(self, state):
        if state is not None:
            if self._sample_generator is None:
                raise ValueError("policy has no isolated sampling generator")
            self._sample_generator.set_state(state)


class CategoricalPolicy(nn.Module):
    """Categorical policy over a configured holding-time alphabet."""

    def __init__(self, num_inputs, action_bins, hidden_dim=64, init_w=3e-3,
                 sample_seed=None, device="cpu",
                 action_limit_feature_index=None):
        super().__init__()
        bins = torch.as_tensor(action_bins, dtype=torch.float32).view(-1, 1)
        if bins.numel() < 2:
            raise ValueError("action_bins must contain at least two values")
        if action_limit_feature_index is not None:
            action_limit_feature_index = int(action_limit_feature_index)
            if not 0 <= action_limit_feature_index < int(num_inputs):
                raise ValueError(
                    "action_limit_feature_index is outside the policy state")
            if not torch.any(torch.isclose(
                    bins.reshape(-1), torch.tensor(0.0))):
                raise ValueError(
                    "a dynamically masked action alphabet must include zero")
        self.register_buffer("action_bins", bins)
        self.action_limit_feature_index = action_limit_feature_index
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, bins.shape[0])
        self.logits.weight.data.uniform_(-init_w, init_w)
        self.logits.bias.data.uniform_(-init_w, init_w)
        self._sample_generator = None
        if sample_seed is not None:
            self._sample_generator = torch.Generator(
                device=torch.device(device).type)
            self._sample_generator.manual_seed(int(sample_seed))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.logits(x)

    def feasible_action_mask(self, state):
        """Return the state-dependent executable action alphabet.

        The configured state feature is the causal holding limit normalized by
        the largest action bin. Keeping the mask inside the policy makes target
        backups, actor optimization, sampling, and deterministic evaluation use
        the same feasible action set.
        """
        batch = int(state.shape[0])
        n_actions = int(self.action_bins.shape[0])
        if self.action_limit_feature_index is None:
            return torch.ones(
                (batch, n_actions), dtype=torch.bool, device=state.device)
        limit_ratio = torch.clamp(
            state[:, self.action_limit_feature_index], 0.0, 1.0)
        max_action = torch.max(self.action_bins).to(
            device=state.device, dtype=state.dtype)
        limit_s = limit_ratio.unsqueeze(-1) * max_action
        bins = self.action_bins.reshape(1, -1).to(
            device=state.device, dtype=state.dtype)
        mask = bins <= limit_s + 1e-6
        zero_idx = torch.argmin(torch.abs(bins), dim=-1).item()
        mask[:, zero_idx] = True
        return mask

    def dist_info(self, state, epsilon=1e-8):
        logits = self.forward(state)
        feasible = self.feasible_action_mask(state)
        masked_logits = logits.masked_fill(
            ~feasible, torch.finfo(logits.dtype).min)
        probs = F.softmax(masked_logits, dim=-1)
        log_probs = torch.where(
            feasible, torch.log(probs + epsilon), torch.zeros_like(probs))
        return probs, log_probs, logits

    def target_entropy(self, state):
        feasible_count = self.feasible_action_mask(state).sum(
            dim=-1, keepdim=True).to(dtype=state.dtype)
        return 0.98 * torch.log(torch.clamp(feasible_count, min=1.0))

    def evaluate(self, state, epsilon=1e-8):
        probs, log_probs, logits = self.dist_info(state, epsilon=epsilon)
        if self._sample_generator is None:
            idx = torch.distributions.Categorical(probs=probs).sample()
        else:
            idx = torch.multinomial(
                probs, 1, generator=self._sample_generator).squeeze(-1)
        action = self.action_bins[idx]
        log_prob = log_probs.gather(1, idx.view(-1, 1))
        return action, log_prob, idx.view(-1, 1).float(), logits, probs

    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        probs, _, _ = self.dist_info(state)
        if deterministic:
            idx = probs.argmax(dim=-1)
        else:
            if self._sample_generator is None:
                idx = torch.distributions.Categorical(probs=probs).sample()
            else:
                idx = torch.multinomial(
                    probs, 1, generator=self._sample_generator).squeeze(-1)
        action = self.action_bins[idx]
        return action.detach().squeeze().cpu().numpy()

    def sampling_state(self):
        if self._sample_generator is None:
            return None
        return self._sample_generator.get_state()

    def set_sampling_state(self, state):
        if state is not None:
            if self._sample_generator is None:
                raise ValueError("policy has no isolated sampling generator")
            self._sample_generator.set_state(state)


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

    def compute_l1_norm(self, mode="sum"):
        """L1 norm per ensemble member for regularization. Returns [K]."""
        total = torch.zeros(self.ensemble_size, device=self.weights[0].device)
        count = 0
        for w, b in zip(self.weights, self.biases):
            total = total + w.abs().sum(dim=(1, 2)) + b.abs().sum(dim=(1, 2))
            count += int(w.shape[1] * w.shape[2] + b.shape[1] * b.shape[2])
        if mode == "mean":
            total = total / max(count, 1)
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
    Pessimistic ensemble SAC trainer with Lagrangian cost constraint.

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
                 weight_reg_mode="sum",
                 lr=3e-4, lambda_lr=1e-3, gamma=0.99, soft_tau=5e-3,
                 auto_entropy=True, maximum_alpha=0.3, action_bins=None,
                 initial_alpha=0.1, minimum_alpha=1e-5,
                 temperature_contract="legacy_capped_scalar",
                 entropy_action_coordinates="physical_legacy",
                 cost_limit_semantics="per_decision_rate",
                 critic_aggregation="ensemble_mean_lcb",
                 policy_sample_seed=None,
                 action_limit_feature_index=None,
                 device='cpu'):
        self.device = device
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.cost_limit = cost_limit
        self.cost_limit_semantics = str(cost_limit_semantics).strip().lower()
        if self.cost_limit_semantics != "per_decision_rate":
            raise ValueError(
                "cost_limit_semantics currently supports only "
                "per_decision_rate")
        self.auto_entropy = auto_entropy
        self.ensemble_size = ensemble_size
        self.critic_aggregation = str(critic_aggregation).strip().lower()
        if self.critic_aggregation not in {
                "ensemble_mean_lcb", "twin_min"}:
            raise ValueError(
                "critic_aggregation must be ensemble_mean_lcb or twin_min")
        if self.critic_aggregation == "twin_min" and int(ensemble_size) != 2:
            raise ValueError("twin_min requires ensemble_size=2")
        self.beta = beta              # LCB coefficient (negative = pessimistic)
        self.beta_ood = beta_ood      # OOD regularization weight
        self.weight_reg = weight_reg  # L1 regularization weight
        self.weight_reg_mode = str(weight_reg_mode).strip().lower()
        if self.weight_reg_mode not in {"sum", "mean"}:
            raise ValueError("weight_reg_mode must be 'sum' or 'mean'")

        self.discrete_actions = None
        self.action_limit_feature_index = (
            None if action_limit_feature_index is None
            else int(action_limit_feature_index))
        if action_bins is not None:
            bins = np.asarray(action_bins, dtype=np.float32).reshape(-1)
            bins = np.unique(np.clip(bins, 0.0, float(action_range)))
            if bins.size < 2:
                raise ValueError("action_bins must contain at least two values")
            self.discrete_actions = torch.as_tensor(
                bins, dtype=torch.float32, device=device).view(-1, 1)
            self.policy_net = CategoricalPolicy(
                state_dim, bins, hidden_dim, sample_seed=policy_sample_seed,
                device=device,
                action_limit_feature_index=self.action_limit_feature_index,
            ).to(device)
        else:
            if self.action_limit_feature_index is not None:
                raise ValueError(
                    "action_limit_feature_index requires categorical actions")
            self.policy_net = GaussianPolicy(
                state_dim, hidden_dim, action_range,
                entropy_action_coordinates=entropy_action_coordinates,
                sample_seed=policy_sample_seed, device=device).to(device)

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
        self.temperature_contract = str(temperature_contract).strip().lower()
        if self.temperature_contract not in {
                "legacy_capped_scalar", "bounded_log_parameter_v4"}:
            raise ValueError(
                "temperature_contract must be legacy_capped_scalar or "
                "bounded_log_parameter_v4")
        self.maximum_alpha = float(maximum_alpha)
        self.minimum_alpha = float(minimum_alpha)
        if not (0.0 < self.minimum_alpha <= self.maximum_alpha):
            raise ValueError("require 0 < minimum_alpha <= maximum_alpha")
        initial_alpha = float(initial_alpha)
        if auto_entropy:
            if self.temperature_contract == "bounded_log_parameter_v4":
                initial_alpha = float(np.clip(
                    initial_alpha, self.minimum_alpha, self.maximum_alpha))
                initial_log_alpha = float(np.log(initial_alpha))
            else:
                initial_log_alpha = 0.0
            self.log_alpha = torch.tensor(
                [initial_log_alpha], dtype=torch.float32,
                requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            if self.discrete_actions is not None:
                self.target_entropy = 0.98 * float(
                    np.log(max(int(self.discrete_actions.shape[0]), 2)))
            else:
                self.target_entropy = -1.0 * action_dim
        self.alpha = (
            initial_alpha
            if self.temperature_contract == "bounded_log_parameter_v4"
            else 0.1)

        # Lagrangian multiplier lambda
        self.log_lambda = torch.zeros(1, requires_grad=True, device=device)
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=lambda_lr)

    @property
    def lambda_param(self):
        return self.log_lambda.exp().item()

    def _discrete_q_values(self, q_net, state):
        """Evaluate a scalar-action critic on every configured action bin."""
        bins = self.discrete_actions
        if bins is None:
            raise RuntimeError("discrete action values requested without bins")
        batch = state.shape[0]
        n_actions = bins.shape[0]
        state_rep = (
            state.unsqueeze(1)
            .expand(batch, n_actions, state.shape[-1])
            .reshape(batch * n_actions, state.shape[-1])
        )
        action_rep = (
            bins.view(1, n_actions, 1)
            .expand(batch, n_actions, 1)
            .reshape(batch * n_actions, 1)
        )
        q_flat = q_net(state_rep, action_rep)
        return q_flat.view(q_flat.shape[0], batch, n_actions)

    def _discrete_cost_values(self, cost_q_net, state):
        bins = self.discrete_actions
        if bins is None:
            raise RuntimeError("discrete cost values requested without bins")
        batch = state.shape[0]
        n_actions = bins.shape[0]
        state_rep = (
            state.unsqueeze(1)
            .expand(batch, n_actions, state.shape[-1])
            .reshape(batch * n_actions, state.shape[-1])
        )
        action_rep = (
            bins.view(1, n_actions, 1)
            .expand(batch, n_actions, 1)
            .reshape(batch * n_actions, 1)
        )
        c_flat = cost_q_net(state_rep, action_rep)
        return c_flat.view(batch, n_actions)

    def _aggregate_target_q(self, q_all):
        if self.critic_aggregation == "twin_min":
            return q_all.min(dim=0).values
        return q_all.mean(dim=0)

    def _policy_q_value(self, q_all):
        q_mean = q_all.mean(dim=0)
        q_std = q_all.std(dim=0)
        if self.critic_aggregation == "twin_min":
            return q_all.min(dim=0).values, q_mean, q_std
        return q_mean + self.beta * q_std, q_mean, q_std

    def update(self, replay_buffer, batch_size, reward_scale=10.0,
               update_policy=True, tap_signal=None, weight_fn=None):
        """One gradient step for ensemble critic, policy, cost critic, and lambda.

        Args:
            weight_fn: optional callable(trip_ids: np.ndarray) -> np.ndarray of shape [B]
                       returning per-sample IS weights for TPC-Lower. Weights should
                       already be clipped + normalised (mean ≈ 1).
        """
        state, action, reward, cost, next_state, done, trip_ids = \
            replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device) * reward_scale
        cost = torch.FloatTensor(cost).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # TPC-Lower: per-sample IS weights for lower SAC losses
        if weight_fn is not None:
            w_np = weight_fn(trip_ids)
            w = torch.FloatTensor(w_np).to(self.device)  # [B]
            ess = float((w.sum() ** 2) / ((w ** 2).sum() + 1e-8))
        else:
            w = torch.ones(state.shape[0], device=self.device)
            ess = float(state.shape[0])

        # TAP bonus
        if tap_signal is not None:
            tap_bonus = torch.zeros_like(reward)
            for i, tid in enumerate(trip_ids):
                if tid in tap_signal:
                    tap_bonus[i] = tap_signal[tid] * reward_scale
            reward = reward + tap_bonus

        discrete_policy = self.discrete_actions is not None

        # ──── Ensemble Critic update ────
        with torch.no_grad():
            if discrete_policy:
                next_probs, next_log_probs, _ = self.policy_net.dist_info(next_state)
                target_q_all = self._discrete_q_values(
                    self.target_q_net, next_state)  # [K, B, A]
                # Use ensemble MEAN for shared target -> prevents member divergence.
                target_q_mean = self._aggregate_target_q(
                    target_q_all)  # [B, A]
                target_q_mean = (next_probs * (
                    target_q_mean - self.alpha * next_log_probs)).sum(dim=-1)
            else:
                next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
                target_q_all = self.target_q_net(next_state, next_action)  # [K, B]
                # Use ensemble MEAN for shared target -> prevents member divergence.
                target_q_mean = self._aggregate_target_q(target_q_all)  # [B]
                target_q_mean = target_q_mean - self.alpha * next_log_prob.squeeze(-1)  # [B]
            r = reward.squeeze(-1)   # [B]
            d = done.squeeze(-1)     # [B]
            # Shared target broadcast to all K members
            shared_target = r + (1.0 - d) * self.gamma * target_q_mean  # [B]
            # Clamp target to prevent runaway values
            shared_target = shared_target.clamp(-100.0, 100.0)

        predicted_q = self.q_net(state, action)  # [K, B]
        target_value = shared_target.unsqueeze(0).expand(
            predicted_q.shape[0], -1)  # [K, B]
        # Weighted Q MSE (per-sample IS weights w broadcast across ensemble axis)
        sq_err = (predicted_q - target_value).pow(2)            # [K, B]
        q_mse_loss = (sq_err * w.unsqueeze(0)).mean()

        # OOD regularization: penalize cross-ensemble disagreement
        ood_loss = predicted_q.std(dim=0).mean()

        # L1 weight regularization
        l1_norm = self.q_net.compute_l1_norm(
            mode=self.weight_reg_mode).mean()

        q_loss = q_mse_loss + self.beta_ood * ood_loss + self.weight_reg * l1_norm

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 50.0)
        self.q_optimizer.step()

        # ──── Cost critic update ────
        with torch.no_grad():
            if discrete_policy:
                next_probs_c, _, _ = self.policy_net.dist_info(next_state)
                target_cost_all = self._discrete_cost_values(
                    self.target_cost_q_net, next_state)
                target_cost_q = (
                    next_probs_c * target_cost_all).sum(dim=-1, keepdim=True)
            else:
                next_action_c, _, _, _, _ = self.policy_net.evaluate(next_state)
                target_cost_q = self.target_cost_q_net(next_state, next_action_c)
            target_cost_value = cost + (1.0 - done) * self.gamma * target_cost_q
            target_cost_value = target_cost_value.clamp(0.0, 50.0)  # cost is non-negative

        cost_q = self.cost_q_net(state, action)
        # Weighted cost-critic MSE
        cost_sq_err = (cost_q - target_cost_value).pow(2)
        cost_q_loss = (cost_sq_err * w.view(-1, 1)).mean()

        self.cost_q_optimizer.zero_grad()
        cost_q_loss.backward()
        cq_grad_norm = torch.nn.utils.clip_grad_norm_(self.cost_q_net.parameters(), 50.0)
        self.cost_q_optimizer.step()

        # ──── Policy update ────
        metrics = {
            'q_loss': q_loss.item(),
            'q_mse': q_mse_loss.item(),
            'ood_loss': ood_loss.item(),
            'q_l1': l1_norm.item(),
            'q_l1_penalty': (self.weight_reg * l1_norm).item(),
            'cost_q_loss': cost_q_loss.item(),
            'q_grad_norm': q_grad_norm.item() if isinstance(q_grad_norm, torch.Tensor) else float(q_grad_norm),
            'cq_grad_norm': cq_grad_norm.item() if isinstance(cq_grad_norm, torch.Tensor) else float(cq_grad_norm),
            'reward_batch_mean': reward.mean().item(),
            'reward_batch_std': reward.std().item(),
            'action_batch_mean': action.mean().item(),
            'action_batch_std': action.std().item(),
        }

        if update_policy:
            new_action, log_prob, _, _, _ = self.policy_net.evaluate(state)

            # RE-SAC: ensemble Q statistics
            if discrete_policy:
                probs, log_probs, _ = self.policy_net.dist_info(state)
                q_all = self._discrete_q_values(self.q_net, state)  # [K, B, A]
                q_lcb, q_mean, q_std = self._policy_q_value(q_all)
                cost_q_new = self._discrete_cost_values(
                    self.cost_q_net, state)             # [B, A]
                entropy_log_prob = (probs * log_probs).sum(
                    dim=-1, keepdim=True)
            else:
                q_all = self.q_net(state, new_action)  # [K, B]
                q_lcb, q_mean, q_std = self._policy_q_value(q_all)
                cost_q_new = self.cost_q_net(state, new_action)
                entropy_log_prob = log_prob

            lam = self.log_lambda.exp().detach()

            # Weighted policy loss (TPC: emphasize transitions consistent with EMA upper)
            if discrete_policy:
                per_action_terms = (
                    self.alpha * log_probs
                    - q_lcb
                    + lam * cost_q_new)
                policy_terms = (probs * per_action_terms).sum(dim=-1)
            else:
                policy_terms = (self.alpha * log_prob.squeeze(-1)
                                - q_lcb
                                + lam * cost_q_new.squeeze(-1))
            policy_loss = (policy_terms * w).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
            self.policy_optimizer.step()

            # ──── Alpha update ────
            if self.auto_entropy:
                target_entropy = self.target_entropy
                if (discrete_policy
                        and self.action_limit_feature_index is not None):
                    target_entropy = self.policy_net.target_entropy(state)
                alpha_loss = -(self.log_alpha *
                               (entropy_log_prob + target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                if self.temperature_contract == "bounded_log_parameter_v4":
                    self.log_alpha.data.clamp_(
                        min=float(np.log(self.minimum_alpha)),
                        max=float(np.log(self.maximum_alpha)))
                    self.alpha = self.log_alpha.exp().item()
                else:
                    self.alpha = min(
                        self.log_alpha.exp().item(), self.maximum_alpha)

            # ──── Lambda update ────
            # Constraint: E_mu[c_t] <= cost_limit under the replay approximation
            # to normalized discounted occupancy. The actor uses Q_c because that
            # is the policy-gradient continuation value; the dual statistic and
            # configured threshold both remain in per-decision cost units.
            #   loss = - λ · (cost - cost_limit)
            # so that ∂loss/∂(log λ) = -λ·(cost - clim);  Adam minimisation gives
            # log λ ← log λ + lr·λ·(cost - clim), i.e. λ INCREASES when violated
            # and DECREASES when slack. (The previous form had this sign reversed.)
            weight_sum = w.sum().clamp_min(1e-8)
            batch_cost_mean = (
                cost.squeeze(-1) * w).sum().div(weight_sum).detach()
            lambda_loss = -self.log_lambda.exp() * (
                batch_cost_mean - self.cost_limit)
            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()
            self.log_lambda.data.clamp_(min=-5.0, max=1.5)  # λ ∈ [e^-5, e^1.5] ≈ [0.007, 4.5]

            metrics.update({
                'policy_loss': policy_loss.item(),
                'alpha': self.alpha,
                'lambda': self.lambda_param,
                'q_mean': q_mean.mean().item(),
                'q_std': q_std.mean().item(),
                'cost_q_mean': cost_q_new.mean().item(),
                'batch_cost_mean': batch_cost_mean.item(),
                'pi_grad_norm': pi_grad_norm.item() if isinstance(pi_grad_norm, torch.Tensor) else float(pi_grad_norm),
                'tpc_ess': ess,
            })

        # ──── Soft target update ────
        for tp, p in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.soft_tau) + p.data * self.soft_tau)
        for tp, p in zip(self.target_cost_q_net.parameters(),
                         self.cost_q_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.soft_tau) + p.data * self.soft_tau)

        return metrics

    def training_state_dict(self):
        return {
            'format': 'freqduet-lower-training-v4',
            'policy': self.policy_net.state_dict(),
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'cost_q_net': self.cost_q_net.state_dict(),
            'target_cost_q_net': self.target_cost_q_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'cost_q_optimizer': self.cost_q_optimizer.state_dict(),
            'log_lambda': self.log_lambda.detach().clone(),
            'lambda_optimizer': self.lambda_optimizer.state_dict(),
            'log_alpha': (
                self.log_alpha.detach().clone() if self.auto_entropy else None),
            'alpha_optimizer': (
                self.alpha_optimizer.state_dict()
                if self.auto_entropy else None),
            'alpha': float(self.alpha),
            'temperature_contract': self.temperature_contract,
            'cost_limit_semantics': self.cost_limit_semantics,
            'critic_aggregation': self.critic_aggregation,
            'policy_sampling_state': self.policy_net.sampling_state(),
            'action_limit_feature_index': self.action_limit_feature_index,
        }

    def load_training_state_dict(self, state):
        if state.get('format') != 'freqduet-lower-training-v4':
            raise ValueError('not a FreqDuet v4 lower training checkpoint')
        if state.get('temperature_contract') != self.temperature_contract:
            raise ValueError('lower temperature contract mismatch')
        if state.get('cost_limit_semantics') != self.cost_limit_semantics:
            raise ValueError('lower cost-limit semantics mismatch')
        if state.get('critic_aggregation') != self.critic_aggregation:
            raise ValueError('lower critic aggregation mismatch')
        if (state.get('action_limit_feature_index')
                != self.action_limit_feature_index):
            raise ValueError('lower action-limit feature contract mismatch')
        self.policy_net.load_state_dict(state['policy'])
        self.q_net.load_state_dict(state['q_net'])
        self.target_q_net.load_state_dict(state['target_q_net'])
        self.cost_q_net.load_state_dict(state['cost_q_net'])
        self.target_cost_q_net.load_state_dict(state['target_cost_q_net'])
        self.policy_optimizer.load_state_dict(state['policy_optimizer'])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
        self.cost_q_optimizer.load_state_dict(state['cost_q_optimizer'])
        self.log_lambda.data.copy_(state['log_lambda'].to(self.device))
        self.lambda_optimizer.load_state_dict(state['lambda_optimizer'])
        if self.auto_entropy:
            if state.get('log_alpha') is None:
                raise ValueError('lower checkpoint is missing entropy state')
            self.log_alpha.data.copy_(state['log_alpha'].to(self.device))
            self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
        self.alpha = float(state['alpha'])
        self.policy_net.set_sampling_state(state.get('policy_sampling_state'))

    def save(self, path):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'q_net': self.q_net.state_dict(),
            'cost_q_net': self.cost_q_net.state_dict(),
            'log_lambda': self.log_lambda.data,
            'log_alpha': self.log_alpha.data if self.auto_entropy else None,
            'temperature_contract': self.temperature_contract,
            'critic_aggregation': self.critic_aggregation,
            'entropy_action_coordinates': getattr(
                self.policy_net, 'entropy_action_coordinates', 'categorical'),
            'policy_sampling_state': self.policy_net.sampling_state(),
            'action_limit_feature_index': self.action_limit_feature_index,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        saved_aggregation = ckpt.get(
            'critic_aggregation', 'ensemble_mean_lcb')
        if saved_aggregation != self.critic_aggregation:
            raise ValueError('lower checkpoint critic aggregation mismatch')
        if (ckpt.get('action_limit_feature_index')
                != self.action_limit_feature_index):
            raise ValueError('lower checkpoint action-limit contract mismatch')
        self.policy_net.load_state_dict(ckpt['policy'])
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_q_net.load_state_dict(ckpt['q_net'])
        self.cost_q_net.load_state_dict(ckpt['cost_q_net'])
        self.target_cost_q_net.load_state_dict(ckpt['cost_q_net'])
        self.log_lambda.data = ckpt['log_lambda']
        if ckpt.get('log_alpha') is not None:
            self.log_alpha.data = ckpt['log_alpha']
            if self.temperature_contract == "bounded_log_parameter_v4":
                self.log_alpha.data.clamp_(
                    min=float(np.log(self.minimum_alpha)),
                    max=float(np.log(self.maximum_alpha)))
                self.alpha = self.log_alpha.exp().item()
            else:
                self.alpha = min(
                    self.log_alpha.exp().item(), self.maximum_alpha)
        self.policy_net.set_sampling_state(
            ckpt.get('policy_sampling_state'))
