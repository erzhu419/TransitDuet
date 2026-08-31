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
                 regularity_policy_objective=None,
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
        self.action_range = float(action_range)
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

        # A separate causal action-regularity constraint bypasses the reward
        # critic. For the two-sided local loss, the action-dependent term is
        # exactly the squared distance from the analytic balancing action.
        # Keeping its dual separate avoids mixing regularity with safety cost.
        regularity_cfg = dict(regularity_policy_objective or {})
        self.regularity_policy_enabled = bool(
            regularity_cfg.get('enable', False))
        self.regularity_policy_contract = {'enabled': False}
        self.regularity_policy_mode = 'disabled'
        self.regularity_constraint_scale_mode = 'raw_cost_v1'
        self.regularity_constraint_cost_scale = 1.0
        self.regularity_initial_lambda = 0.0
        self.log_regularity_lambda = None
        self.regularity_lambda_optimizer = None
        self.regularity_entropy_split_enabled = False
        self.regularity_entropy_target_fraction = 0.98
        self.regularity_alpha_min = self.minimum_alpha
        self.regularity_alpha_max = self.maximum_alpha
        self.log_regularity_alpha = None
        self.regularity_alpha_optimizer = None
        self.regularity_capacity_gain_enabled = False
        self.regularity_capacity_feature_index = None
        self.regularity_capacity_gain_mode = 'disabled'
        self.regularity_capacity_gain_weight = 0.0
        self.regularity_capacity_gain_scale = 1.0
        self.regularity_capacity_exponent = 1.0
        self.regularity_capacity_action_efficiency_penalty = 0.0
        if self.regularity_policy_enabled:
            mode = str(regularity_cfg.get(
                'mode', 'analytic_two_sided_target_dual_v1')).strip().lower()
            if mode not in {
                    'analytic_two_sided_target_dual_v1',
                    'analytic_two_sided_zero_hold_regret_dual_v2',
                    'analytic_two_sided_capacity_gain_regret_dual_v3',
                    'analytic_two_sided_efficiency_gain_regret_dual_v4'}:
                raise ValueError('unknown causal regularity policy objective')
            self.regularity_policy_mode = mode
            if self.discrete_actions is None:
                raise ValueError(
                    'causal regularity policy objective requires action_bins')
            self.regularity_target_feature_index = int(
                regularity_cfg['target_feature_index'])
            self.regularity_valid_feature_index = int(
                regularity_cfg['valid_feature_index'])
            self.regularity_headway_feature_index = int(
                regularity_cfg['target_headway_feature_index'])
            for index in {
                    self.regularity_target_feature_index,
                    self.regularity_valid_feature_index,
                    self.regularity_headway_feature_index}:
                if index < 0 or index >= int(state_dim):
                    raise ValueError(
                        'causal regularity policy feature index is out of range')
            self.regularity_action_target_scale_s = float(
                regularity_cfg['action_target_scale_s'])
            self.regularity_headway_scale_s = float(
                regularity_cfg['target_headway_scale_s'])
            self.regularity_cost_limit = float(
                regularity_cfg.get('cost_limit', 0.001))
            self.regularity_cost_cap = float(
                regularity_cfg.get('cost_cap', 0.25))
            self.regularity_constraint_scale_mode = str(
                regularity_cfg.get(
                    'constraint_scale_mode', 'raw_cost_v1')).strip().lower()
            lambda_lr_regularity = float(
                regularity_cfg.get('lambda_lr', 1e-3))
            self.regularity_lambda_min = float(
                regularity_cfg.get('lambda_min', 1e-3))
            self.regularity_lambda_max = float(
                regularity_cfg.get('lambda_max', 20.0))
            initial_lambda = float(
                regularity_cfg.get('initial_lambda', 1.0))
            if not np.isfinite(self.regularity_action_target_scale_s) or (
                    self.regularity_action_target_scale_s <= 0.0):
                raise ValueError('regularity action-target scale must be positive')
            if not np.isfinite(self.regularity_headway_scale_s) or (
                    self.regularity_headway_scale_s <= 0.0):
                raise ValueError('regularity headway scale must be positive')
            if not np.isfinite(self.regularity_cost_limit) or not (
                    0.0 <= self.regularity_cost_limit < self.regularity_cost_cap):
                raise ValueError(
                    'regularity cost limit must lie in [0, cost_cap)')
            if not np.isfinite(self.regularity_cost_cap) or (
                    self.regularity_cost_cap <= 0.0):
                raise ValueError('regularity cost cap must be positive')
            if self.regularity_constraint_scale_mode not in {
                    'raw_cost_v1', 'cost_limit_ratio_v1'}:
                raise ValueError(
                    'unknown regularity constraint scale mode')
            if (self.regularity_constraint_scale_mode
                    == 'cost_limit_ratio_v1'):
                if self.regularity_cost_limit <= 0.0:
                    raise ValueError(
                        'cost-limit ratio scaling requires a positive limit')
                self.regularity_constraint_cost_scale = (
                    self.regularity_cost_limit)
            if not np.isfinite(lambda_lr_regularity) or (
                    lambda_lr_regularity <= 0.0):
                raise ValueError('regularity lambda_lr must be positive')
            if not (0.0 < self.regularity_lambda_min
                    <= initial_lambda <= self.regularity_lambda_max):
                raise ValueError(
                    'require 0 < lambda_min <= initial_lambda <= lambda_max')
            self.log_regularity_lambda = torch.tensor(
                [float(np.log(initial_lambda))], dtype=torch.float32,
                requires_grad=True, device=device)
            self.regularity_initial_lambda = initial_lambda
            self.regularity_lambda_optimizer = optim.Adam(
                [self.log_regularity_lambda], lr=lambda_lr_regularity)
            entropy_cfg = dict(
                regularity_cfg.get('conditional_entropy', {}) or {})
            self.regularity_entropy_split_enabled = bool(
                entropy_cfg.get('enable', False))
            entropy_contract = {'enabled': False}
            if self.regularity_entropy_split_enabled:
                if not self.auto_entropy:
                    raise ValueError(
                        'conditional regularity entropy requires auto_entropy')
                if self.discrete_actions is None:
                    raise ValueError(
                        'conditional regularity entropy requires action_bins')
                entropy_mode = str(entropy_cfg.get(
                    'mode', 'evidence_split_temperature_v1')).strip().lower()
                if entropy_mode != 'evidence_split_temperature_v1':
                    raise ValueError(
                        'unknown conditional regularity entropy mode')
                target_fraction = float(
                    entropy_cfg.get('target_fraction', 0.5))
                entropy_lr = float(entropy_cfg.get('lr', lr))
                alpha_min = float(entropy_cfg.get(
                    'minimum_alpha', self.minimum_alpha))
                alpha_max = float(entropy_cfg.get(
                    'maximum_alpha', self.maximum_alpha))
                alpha_initial = float(entropy_cfg.get(
                    'initial_alpha', initial_alpha))
                if not (0.0 <= target_fraction < 0.98):
                    raise ValueError(
                        'conditional entropy target_fraction must be in [0, 0.98)')
                if not np.isfinite(entropy_lr) or entropy_lr <= 0.0:
                    raise ValueError(
                        'conditional entropy learning rate must be positive')
                if not (0.0 < alpha_min <= alpha_initial <= alpha_max):
                    raise ValueError(
                        'require 0 < conditional minimum_alpha <= '
                        'initial_alpha <= maximum_alpha')
                self.regularity_entropy_target_fraction = target_fraction
                self.regularity_alpha_min = alpha_min
                self.regularity_alpha_max = alpha_max
                self.log_regularity_alpha = torch.tensor(
                    [float(np.log(alpha_initial))], dtype=torch.float32,
                    requires_grad=True, device=device)
                self.regularity_alpha_optimizer = optim.Adam(
                    [self.log_regularity_alpha], lr=entropy_lr)
                entropy_contract = {
                    'enabled': True,
                    'mode': entropy_mode,
                    'target_fraction': target_fraction,
                    'lr': entropy_lr,
                    'minimum_alpha': alpha_min,
                    'maximum_alpha': alpha_max,
                    'initial_alpha': alpha_initial,
                }
            capacity_gain_cfg = dict(
                regularity_cfg.get('capacity_gated_gain', {}) or {})
            capacity_gain_enabled = bool(
                capacity_gain_cfg.get('enable', False))
            capacity_gain_modes = {
                'analytic_two_sided_capacity_gain_regret_dual_v3': (
                    'positive_zero_hold_gain_v1'),
                'analytic_two_sided_efficiency_gain_regret_dual_v4': (
                    'positive_zero_hold_efficiency_gain_v2'),
            }
            if mode in capacity_gain_modes:
                if not capacity_gain_enabled:
                    raise ValueError(
                        'capacity-gain regularity mode requires an enabled '
                        'capacity_gated_gain contract')
                gain_mode = str(capacity_gain_cfg.get(
                    'mode', capacity_gain_modes[mode])).strip().lower()
                if gain_mode != capacity_gain_modes[mode]:
                    raise ValueError('unknown capacity-gated regularity gain mode')
                capacity_feature_index = int(
                    capacity_gain_cfg['capacity_feature_index'])
                if not 0 <= capacity_feature_index < int(state_dim):
                    raise ValueError(
                        'capacity-gated gain feature index is out of range')
                gain_weight = float(capacity_gain_cfg.get('weight', 0.0))
                gain_scale = float(capacity_gain_cfg.get('gain_scale', 0.002))
                capacity_exponent = float(
                    capacity_gain_cfg.get('capacity_exponent', 1.0))
                action_efficiency_penalty = float(
                    capacity_gain_cfg.get(
                        'action_efficiency_penalty', 0.0))
                if not np.isfinite(gain_weight) or gain_weight <= 0.0:
                    raise ValueError(
                        'capacity-gated gain weight must be positive')
                if not np.isfinite(gain_scale) or gain_scale <= 0.0:
                    raise ValueError(
                        'capacity-gated gain scale must be positive')
                if (not np.isfinite(capacity_exponent)
                        or capacity_exponent <= 0.0):
                    raise ValueError(
                        'capacity-gated gain exponent must be positive')
                if (not np.isfinite(action_efficiency_penalty)
                        or action_efficiency_penalty < 0.0):
                    raise ValueError(
                        'capacity-gated action efficiency penalty must be '
                        'finite and non-negative')
                efficiency_mode = (
                    gain_mode == 'positive_zero_hold_efficiency_gain_v2')
                if efficiency_mode != (action_efficiency_penalty > 0.0):
                    raise ValueError(
                        'V2 efficiency gain requires a positive action '
                        'efficiency penalty and V1 requires zero')
                self.regularity_capacity_gain_enabled = True
                self.regularity_capacity_feature_index = (
                    capacity_feature_index)
                self.regularity_capacity_gain_mode = gain_mode
                self.regularity_capacity_gain_weight = gain_weight
                self.regularity_capacity_gain_scale = gain_scale
                self.regularity_capacity_exponent = capacity_exponent
                self.regularity_capacity_action_efficiency_penalty = (
                    action_efficiency_penalty)
                capacity_gain_contract = {
                    'enabled': True,
                    'mode': gain_mode,
                    'capacity_feature_index': capacity_feature_index,
                    'weight': gain_weight,
                    'gain_scale': gain_scale,
                    'capacity_exponent': capacity_exponent,
                    'action_efficiency_penalty': (
                        action_efficiency_penalty),
                }
            else:
                if capacity_gain_enabled:
                    raise ValueError(
                        'capacity_gated_gain requires the V3 regularity mode')
                capacity_gain_contract = None
            self.regularity_policy_contract = {
                'enabled': True,
                'mode': mode,
                'target_feature_index': self.regularity_target_feature_index,
                'valid_feature_index': self.regularity_valid_feature_index,
                'target_headway_feature_index': (
                    self.regularity_headway_feature_index),
                'action_target_scale_s': self.regularity_action_target_scale_s,
                'target_headway_scale_s': self.regularity_headway_scale_s,
                'cost_limit': self.regularity_cost_limit,
                'cost_cap': self.regularity_cost_cap,
                'constraint_scale_mode': (
                    self.regularity_constraint_scale_mode),
                'lambda_lr': lambda_lr_regularity,
                'lambda_min': self.regularity_lambda_min,
                'lambda_max': self.regularity_lambda_max,
                'initial_lambda': initial_lambda,
                'conditional_entropy': entropy_contract,
            }
            if capacity_gain_contract is not None:
                self.regularity_policy_contract[
                    'capacity_gated_gain'] = capacity_gain_contract

    @property
    def lambda_param(self):
        return self.log_lambda.exp().item()

    @property
    def regularity_lambda_param(self):
        if self.log_regularity_lambda is None:
            return 0.0
        return self.log_regularity_lambda.exp().item()

    @property
    def regularity_alpha_param(self):
        if self.log_regularity_alpha is None:
            return 0.0
        return self.log_regularity_alpha.exp().item()

    @property
    def regularity_scaled_cost_limit(self):
        if not self.regularity_policy_enabled:
            return 0.0
        return (
            self.regularity_cost_limit
            / self.regularity_constraint_cost_scale)

    def _scale_regularity_constraint_cost(self, cost):
        return cost / self.regularity_constraint_cost_scale

    def _regularity_evidence_valid(self, state):
        if not self.regularity_policy_enabled:
            return torch.zeros(state.shape[0], device=state.device)
        return (
            state[:, self.regularity_valid_feature_index] >= 0.5).float()

    def _entropy_alpha_for_state(self, state):
        """Return the causal state-conditional entropy temperature."""
        base = torch.as_tensor(
            self.alpha, dtype=state.dtype, device=state.device
        ).expand(state.shape[0])
        if not self.regularity_entropy_split_enabled:
            return base
        valid = self._regularity_evidence_valid(state)
        regularity_alpha = self.log_regularity_alpha.exp().detach().to(
            dtype=state.dtype)
        return base * (1.0 - valid) + regularity_alpha * valid

    def _maximum_discrete_entropy(self, state):
        feasible_count = self.policy_net.feasible_action_mask(state).sum(
            dim=-1, keepdim=True).to(dtype=state.dtype)
        return torch.log(torch.clamp(feasible_count, min=1.0))

    def _regularity_policy_action_terms(self, state):
        """Return causal validity, absolute action cost, and zero-hold cost."""
        if not self.regularity_policy_enabled:
            raise RuntimeError('causal regularity policy objective is disabled')
        target_norm = state[:, self.regularity_target_feature_index].clamp(
            0.0, 1.0)
        valid = (
            state[:, self.regularity_valid_feature_index] >= 0.5).float()
        target_action_s = (
            target_norm * self.regularity_action_target_scale_s)
        target_headway_s = (
            state[:, self.regularity_headway_feature_index]
            * self.regularity_headway_scale_s).clamp_min(1.0)
        actions_s = self.discrete_actions.view(1, -1)
        absolute_action_costs = (
            (actions_s - target_action_s.unsqueeze(-1))
            / target_headway_s.unsqueeze(-1)).pow(2)
        absolute_action_costs = absolute_action_costs.clamp(
            0.0, self.regularity_cost_cap)
        zero_hold_cost = (
            target_action_s / target_headway_s
        ).pow(2).clamp(
            0.0, self.regularity_cost_cap).unsqueeze(-1)
        return valid, absolute_action_costs, zero_hold_cost

    def _regularity_policy_cost(self, state, action_probs):
        """Return exact conditional action cost for the compact causal target."""
        valid, absolute_action_costs, zero_hold_cost = (
            self._regularity_policy_action_terms(state))
        if self.regularity_policy_mode in {
                'analytic_two_sided_zero_hold_regret_dual_v2',
                'analytic_two_sided_capacity_gain_regret_dual_v3',
                'analytic_two_sided_efficiency_gain_regret_dual_v4'}:
            action_costs = (
                absolute_action_costs - zero_hold_cost).clamp_min(0.0)
        else:
            action_costs = absolute_action_costs
        expected_cost = (action_probs * action_costs).sum(dim=-1)
        return expected_cost, valid, action_costs

    def _regularity_policy_capacity_gain(self, state, action_probs):
        """Reward regularity improvement only where spare capacity is causal."""
        if not self.regularity_capacity_gain_enabled:
            raise RuntimeError('capacity-gated regularity gain is disabled')
        valid, absolute_action_costs, zero_hold_cost = (
            self._regularity_policy_action_terms(state))
        capacity = state[
            :, self.regularity_capacity_feature_index
        ].clamp(0.0, 1.0).pow(self.regularity_capacity_exponent)
        action_gains = (
            zero_hold_cost - absolute_action_costs
        ).clamp_min(0.0) * capacity.unsqueeze(-1)
        action_gains = (
            action_gains
            * self._regularity_policy_action_efficiency_gate().view(1, -1))
        expected_gain = (action_probs * action_gains).sum(dim=-1)
        return expected_gain, valid, action_gains

    def _regularity_policy_action_efficiency_gate(self):
        """Return the fixed per-bin holding-efficiency multiplier."""
        if not self.regularity_capacity_gain_enabled:
            raise RuntimeError('capacity-gated regularity gain is disabled')
        action_fraction = (
            self.discrete_actions / self.regularity_action_target_scale_s
        ).clamp(0.0, 1.0)
        penalty = self.regularity_capacity_action_efficiency_penalty
        return 1.0 / (1.0 + penalty * action_fraction)

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
                next_entropy_alpha = self._entropy_alpha_for_state(
                    next_state).unsqueeze(-1)
                target_q_mean = (next_probs * (
                    target_q_mean
                    - next_entropy_alpha * next_log_probs)).sum(dim=-1)
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
            'regularity_policy_enabled': float(
                self.regularity_policy_enabled),
            'regularity_policy_cost_mean': 0.0,
            'regularity_policy_oracle_cost_mean': 0.0,
            'regularity_policy_excess_cost_mean': 0.0,
            'regularity_policy_valid_fraction': 0.0,
            'regularity_policy_constraint_gap': 0.0,
            'regularity_policy_scaled_cost_mean': 0.0,
            'regularity_policy_scaled_limit': float(
                self.regularity_scaled_cost_limit),
            'regularity_policy_scaled_constraint_gap': 0.0,
            'regularity_policy_penalty': 0.0,
            'regularity_policy_capacity_gain_mean': 0.0,
            'regularity_policy_scaled_capacity_gain_mean': 0.0,
            'regularity_policy_capacity_gain_bonus': 0.0,
            'regularity_policy_capacity_gate_mean': 0.0,
            'regularity_policy_action_efficiency_gate_mean': 0.0,
            'regularity_lambda': self.regularity_lambda_param,
            'regularity_entropy_split_enabled': float(
                self.regularity_entropy_split_enabled),
            'regularity_entropy_target_fraction': float(
                self.regularity_entropy_target_fraction),
            'regularity_entropy_valid_mean': 0.0,
            'regularity_alpha': self.regularity_alpha_param,
        }

        if update_policy:
            new_action, log_prob, _, _, _ = self.policy_net.evaluate(state)

            # RE-SAC: ensemble Q statistics
            regularity_cost_mean = None
            regularity_oracle_cost_mean = None
            regularity_excess_cost_mean = None
            regularity_scaled_cost_mean = None
            regularity_valid_fraction = None
            regularity_capacity_gain_mean = None
            regularity_scaled_capacity_gain_mean = None
            regularity_capacity_gate_mean = None
            regularity_action_efficiency_gate_mean = None
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
                entropy_alpha = self._entropy_alpha_for_state(
                    state).unsqueeze(-1)
                per_action_terms = (
                    entropy_alpha * log_probs
                    - q_lcb
                    + lam * cost_q_new)
                policy_terms = (probs * per_action_terms).sum(dim=-1)
            else:
                policy_terms = (self.alpha * log_prob.squeeze(-1)
                                - q_lcb
                                + lam * cost_q_new.squeeze(-1))
            policy_loss = (policy_terms * w).mean()
            regularity_penalty = torch.zeros((), device=self.device)
            regularity_capacity_gain_bonus = torch.zeros(
                (), device=self.device)
            if self.regularity_policy_enabled:
                regularity_cost, regularity_valid, regularity_action_costs = (
                    self._regularity_policy_cost(state, probs))
                valid_weights = w * regularity_valid
                valid_weight_sum = valid_weights.sum()
                if bool(valid_weight_sum.detach().item() > 0.0):
                    regularity_cost_mean = (
                        regularity_cost * valid_weights
                    ).sum().div(valid_weight_sum)
                    regularity_oracle_cost = regularity_action_costs.min(
                        dim=-1).values
                    regularity_oracle_cost_mean = (
                        regularity_oracle_cost * valid_weights
                    ).sum().div(valid_weight_sum)
                    regularity_excess_cost_mean = (
                        (regularity_cost - regularity_oracle_cost)
                        * valid_weights
                    ).sum().div(valid_weight_sum)
                    regularity_scaled_cost_mean = (
                        self._scale_regularity_constraint_cost(
                            regularity_cost_mean))
                    regularity_valid_fraction = valid_weight_sum.div(
                        w.sum().clamp_min(1e-8))
                    regularity_penalty = (
                        self.log_regularity_lambda.exp().detach()
                        * regularity_scaled_cost_mean)
                    policy_loss = policy_loss + regularity_penalty
                    if self.regularity_capacity_gain_enabled:
                        capacity_gain, gain_valid, _ = (
                            self._regularity_policy_capacity_gain(
                                state, probs))
                        if not torch.equal(gain_valid, regularity_valid):
                            raise RuntimeError(
                                'regularity cost and gain validity diverged')
                        regularity_capacity_gain_mean = (
                            capacity_gain * valid_weights
                        ).sum().div(valid_weight_sum)
                        regularity_scaled_capacity_gain_mean = (
                            regularity_capacity_gain_mean
                            / self.regularity_capacity_gain_scale)
                        capacity_gate = state[
                            :, self.regularity_capacity_feature_index
                        ].clamp(0.0, 1.0).pow(
                            self.regularity_capacity_exponent)
                        regularity_capacity_gate_mean = (
                            capacity_gate * valid_weights
                        ).sum().div(valid_weight_sum)
                        action_efficiency_gate = (
                            self._regularity_policy_action_efficiency_gate())
                        expected_action_efficiency_gate = (
                            probs * action_efficiency_gate.view(1, -1)
                        ).sum(dim=-1)
                        regularity_action_efficiency_gate_mean = (
                            expected_action_efficiency_gate * valid_weights
                        ).sum().div(valid_weight_sum)
                        regularity_capacity_gain_bonus = (
                            self.regularity_capacity_gain_weight
                            * regularity_scaled_capacity_gain_mean)
                        policy_loss = (
                            policy_loss - regularity_capacity_gain_bonus)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
            self.policy_optimizer.step()

            # ──── Alpha update ────
            if self.auto_entropy:
                if self.regularity_entropy_split_enabled:
                    valid = self._regularity_evidence_valid(
                        state).view(-1, 1)
                    invalid = 1.0 - valid
                    max_entropy = self._maximum_discrete_entropy(state)
                    base_target_entropy = 0.98 * max_entropy
                    valid_target_entropy = (
                        self.regularity_entropy_target_fraction * max_entropy)
                    sample_weights = w.view(-1, 1)
                    invalid_weights = sample_weights * invalid
                    invalid_weight_sum = invalid_weights.sum()
                    alpha_loss = torch.zeros((), device=self.device)
                    if bool(invalid_weight_sum.detach().item() > 0.0):
                        alpha_signal = (
                            entropy_log_prob + base_target_entropy).detach()
                        alpha_loss = -(
                            self.log_alpha * alpha_signal * invalid_weights
                        ).sum().div(invalid_weight_sum)
                        self.alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer.step()

                    valid_weights_entropy = sample_weights * valid
                    valid_weight_sum_entropy = valid_weights_entropy.sum()
                    regularity_alpha_loss = torch.zeros(
                        (), device=self.device)
                    if bool(valid_weight_sum_entropy.detach().item() > 0.0):
                        regularity_alpha_signal = (
                            entropy_log_prob + valid_target_entropy).detach()
                        regularity_alpha_loss = -(
                            self.log_regularity_alpha
                            * regularity_alpha_signal
                            * valid_weights_entropy
                        ).sum().div(valid_weight_sum_entropy)
                        self.regularity_alpha_optimizer.zero_grad()
                        regularity_alpha_loss.backward()
                        self.regularity_alpha_optimizer.step()
                        self.log_regularity_alpha.data.clamp_(
                            min=float(np.log(self.regularity_alpha_min)),
                            max=float(np.log(self.regularity_alpha_max)),
                        )
                else:
                    target_entropy = self.target_entropy
                    if (discrete_policy
                            and self.action_limit_feature_index is not None):
                        target_entropy = self.policy_net.target_entropy(state)
                    alpha_loss = -(
                        self.log_alpha
                        * (entropy_log_prob + target_entropy).detach()
                    ).mean()
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
            # to normalized discounted occupancy. The regularity constraint may
            # divide both sides by the positive cost limit; this leaves its
            # feasible set unchanged while putting the dual residual near unit
            # scale. Adam minimisation increases lambda on positive violation.
            weight_sum = w.sum().clamp_min(1e-8)
            batch_cost_mean = (
                cost.squeeze(-1) * w).sum().div(weight_sum).detach()
            lambda_loss = -self.log_lambda.exp() * (
                batch_cost_mean - self.cost_limit)
            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()
            self.log_lambda.data.clamp_(min=-5.0, max=1.5)  # λ ∈ [e^-5, e^1.5] ≈ [0.007, 4.5]

            if regularity_cost_mean is not None:
                regularity_lambda_loss = (
                    -self.log_regularity_lambda.exp()
                    * (regularity_scaled_cost_mean.detach()
                       - self.regularity_scaled_cost_limit))
                self.regularity_lambda_optimizer.zero_grad()
                regularity_lambda_loss.backward()
                self.regularity_lambda_optimizer.step()
                self.log_regularity_lambda.data.clamp_(
                    min=float(np.log(self.regularity_lambda_min)),
                    max=float(np.log(self.regularity_lambda_max)),
                )

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
                'regularity_policy_cost_mean': (
                    regularity_cost_mean.item()
                    if regularity_cost_mean is not None else 0.0),
                'regularity_policy_oracle_cost_mean': (
                    regularity_oracle_cost_mean.item()
                    if regularity_oracle_cost_mean is not None else 0.0),
                'regularity_policy_excess_cost_mean': (
                    regularity_excess_cost_mean.item()
                    if regularity_excess_cost_mean is not None else 0.0),
                'regularity_policy_valid_fraction': (
                    regularity_valid_fraction.item()
                    if regularity_valid_fraction is not None else 0.0),
                'regularity_policy_constraint_gap': (
                    regularity_cost_mean.item() - self.regularity_cost_limit
                    if regularity_cost_mean is not None else 0.0),
                'regularity_policy_scaled_cost_mean': (
                    regularity_scaled_cost_mean.item()
                    if regularity_scaled_cost_mean is not None else 0.0),
                'regularity_policy_scaled_limit': float(
                    self.regularity_scaled_cost_limit),
                'regularity_policy_scaled_constraint_gap': (
                    regularity_scaled_cost_mean.item()
                    - self.regularity_scaled_cost_limit
                    if regularity_scaled_cost_mean is not None else 0.0),
                'regularity_policy_penalty': regularity_penalty.item(),
                'regularity_policy_capacity_gain_mean': (
                    regularity_capacity_gain_mean.item()
                    if regularity_capacity_gain_mean is not None else 0.0),
                'regularity_policy_scaled_capacity_gain_mean': (
                    regularity_scaled_capacity_gain_mean.item()
                    if regularity_scaled_capacity_gain_mean is not None
                    else 0.0),
                'regularity_policy_capacity_gain_bonus': (
                    regularity_capacity_gain_bonus.item()),
                'regularity_policy_capacity_gate_mean': (
                    regularity_capacity_gate_mean.item()
                    if regularity_capacity_gate_mean is not None else 0.0),
                'regularity_policy_action_efficiency_gate_mean': (
                    regularity_action_efficiency_gate_mean.item()
                    if regularity_action_efficiency_gate_mean is not None
                    else 0.0),
                'regularity_lambda': self.regularity_lambda_param,
                'regularity_entropy_valid_mean': (
                    float((
                        -entropy_log_prob.squeeze(-1) * valid_weights
                    ).sum().div(valid_weight_sum).item())
                    if (self.regularity_policy_enabled
                        and valid_weight_sum is not None
                        and bool(valid_weight_sum.detach().item() > 0.0))
                    else 0.0),
                'regularity_alpha': self.regularity_alpha_param,
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
            'format': 'freqduet-lower-training-v7',
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
            'regularity_policy_contract': self.regularity_policy_contract,
            'log_regularity_lambda': (
                self.log_regularity_lambda.detach().clone()
                if self.regularity_policy_enabled else None),
            'regularity_lambda_optimizer': (
                self.regularity_lambda_optimizer.state_dict()
                if self.regularity_policy_enabled else None),
            'log_regularity_alpha': (
                self.log_regularity_alpha.detach().clone()
                if self.regularity_entropy_split_enabled else None),
            'regularity_alpha_optimizer': (
                self.regularity_alpha_optimizer.state_dict()
                if self.regularity_entropy_split_enabled else None),
        }

    def load_training_state_dict(self, state):
        if state.get('format') not in {
                'freqduet-lower-training-v4',
                'freqduet-lower-training-v5',
                'freqduet-lower-training-v6',
                'freqduet-lower-training-v7'}:
            raise ValueError('not a FreqDuet lower training checkpoint')
        if state.get('temperature_contract') != self.temperature_contract:
            raise ValueError('lower temperature contract mismatch')
        if state.get('cost_limit_semantics') != self.cost_limit_semantics:
            raise ValueError('lower cost-limit semantics mismatch')
        if state.get('critic_aggregation') != self.critic_aggregation:
            raise ValueError('lower critic aggregation mismatch')
        if (state.get('action_limit_feature_index')
                != self.action_limit_feature_index):
            raise ValueError('lower action-limit feature contract mismatch')
        saved_regularity_contract = state.get(
            'regularity_policy_contract', {'enabled': False})
        if (saved_regularity_contract.get('enabled')
                and 'conditional_entropy' not in saved_regularity_contract):
            saved_regularity_contract = dict(saved_regularity_contract)
            saved_regularity_contract['conditional_entropy'] = {
                'enabled': False}
        if (saved_regularity_contract.get('enabled')
                and 'constraint_scale_mode'
                not in saved_regularity_contract):
            saved_regularity_contract = dict(saved_regularity_contract)
            saved_regularity_contract['constraint_scale_mode'] = 'raw_cost_v1'
        if saved_regularity_contract != self.regularity_policy_contract:
            raise ValueError('lower regularity-policy contract mismatch')
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
        if self.regularity_policy_enabled:
            if state.get('log_regularity_lambda') is None:
                raise ValueError(
                    'lower checkpoint is missing regularity dual state')
            self.log_regularity_lambda.data.copy_(
                state['log_regularity_lambda'].to(self.device))
            self.regularity_lambda_optimizer.load_state_dict(
                state['regularity_lambda_optimizer'])
        if self.regularity_entropy_split_enabled:
            if state.get('log_regularity_alpha') is None:
                raise ValueError(
                    'lower checkpoint is missing regularity entropy state')
            self.log_regularity_alpha.data.copy_(
                state['log_regularity_alpha'].to(self.device))
            self.regularity_alpha_optimizer.load_state_dict(
                state['regularity_alpha_optimizer'])
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
            'regularity_policy_contract': self.regularity_policy_contract,
            'log_regularity_lambda': (
                self.log_regularity_lambda.data
                if self.regularity_policy_enabled else None),
            'log_regularity_alpha': (
                self.log_regularity_alpha.data
                if self.regularity_entropy_split_enabled else None),
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
        saved_regularity_contract = ckpt.get(
            'regularity_policy_contract', {'enabled': False})
        if (saved_regularity_contract.get('enabled')
                and 'conditional_entropy' not in saved_regularity_contract):
            saved_regularity_contract = dict(saved_regularity_contract)
            saved_regularity_contract['conditional_entropy'] = {
                'enabled': False}
        if (saved_regularity_contract.get('enabled')
                and 'constraint_scale_mode'
                not in saved_regularity_contract):
            saved_regularity_contract = dict(saved_regularity_contract)
            saved_regularity_contract['constraint_scale_mode'] = 'raw_cost_v1'
        if saved_regularity_contract != self.regularity_policy_contract:
            raise ValueError('lower checkpoint regularity-policy mismatch')
        self.policy_net.load_state_dict(ckpt['policy'])
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_q_net.load_state_dict(ckpt['q_net'])
        self.cost_q_net.load_state_dict(ckpt['cost_q_net'])
        self.target_cost_q_net.load_state_dict(ckpt['cost_q_net'])
        self.log_lambda.data = ckpt['log_lambda']
        if self.regularity_policy_enabled:
            if ckpt.get('log_regularity_lambda') is None:
                raise ValueError(
                    'lower checkpoint is missing regularity dual state')
            self.log_regularity_lambda.data.copy_(
                ckpt['log_regularity_lambda'].to(self.device))
        if self.regularity_entropy_split_enabled:
            if ckpt.get('log_regularity_alpha') is None:
                raise ValueError(
                    'lower checkpoint is missing regularity entropy state')
            self.log_regularity_alpha.data.copy_(
                ckpt['log_regularity_alpha'].to(self.device))
            self.log_regularity_alpha.data.clamp_(
                min=float(np.log(self.regularity_alpha_min)),
                max=float(np.log(self.regularity_alpha_max)),
            )
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
