"""Causal sequence networks for raw-history actor-critic baselines."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _initialize_linear(layer: nn.Linear, *, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=float(gain))
    nn.init.zeros_(layer.bias)
    return layer


def _validate_contract(
    *,
    state_dim: int,
    history_window: int,
    raw_feature_dim: int,
    hidden_dim: int,
) -> int:
    state_size = int(state_dim)
    window = int(history_window)
    features = int(raw_feature_dim)
    hidden = int(hidden_dim)
    if window < 1 or features < 1 or hidden < 1:
        raise ValueError(
            "causal GRU requires positive history_window, raw_feature_dim, "
            "and hidden_dim"
        )
    sequence_size = window * features
    if state_size < sequence_size:
        raise ValueError(
            f"state_dim={state_size} is smaller than the raw sequence prefix "
            f"{window}*{features}={sequence_size}"
        )
    context_dim = state_size - sequence_size
    if context_dim < 1:
        raise ValueError(
            "causal GRU state must place history coverage immediately after "
            "the raw sequence prefix"
        )
    return context_dim


class CausalGRUStateEncoder(nn.Module):
    """Encode valid raw history only, followed by static context.

    The first context coordinate is the fraction of the fixed window that
    contains observed samples. It is converted to an exact sequence length so
    left padding cannot affect the recurrent state.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        history_window: int,
        raw_feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.history_window = int(history_window)
        self.raw_feature_dim = int(raw_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = _validate_contract(
            state_dim=self.state_dim,
            history_window=self.history_window,
            raw_feature_dim=self.raw_feature_dim,
            hidden_dim=self.hidden_dim,
        )
        self.gru = nn.GRU(
            input_size=self.raw_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        if self.context_dim > 0:
            self.context_projection = _initialize_linear(
                nn.Linear(self.context_dim, self.hidden_dim), gain=np.sqrt(2.0)
            )
            self.fusion = _initialize_linear(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                gain=np.sqrt(2.0),
            )
        else:
            self.context_projection = None
            self.fusion = None
        self._initialize_gru()

    def _initialize_gru(self) -> None:
        with torch.no_grad():
            for name, parameter in self.gru.named_parameters():
                if "weight_ih" in name:
                    for gate in parameter.chunk(3, dim=0):
                        nn.init.xavier_uniform_(gate)
                elif "weight_hh" in name:
                    for gate in parameter.chunk(3, dim=0):
                        nn.init.orthogonal_(gate)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"causal GRU state must have shape (batch, {self.state_dim}), "
                f"got {tuple(state.shape)}"
            )
        sequence_size = self.history_window * self.raw_feature_dim
        sequence = state[:, :sequence_size].reshape(
            state.shape[0], self.history_window, self.raw_feature_dim
        )
        coverage = state[:, sequence_size].detach().clamp(
            1.0 / float(self.history_window), 1.0
        )
        lengths = torch.round(coverage * float(self.history_window)).to(
            dtype=torch.long
        ).clamp(1, self.history_window)
        starts = self.history_window - lengths
        positions = torch.arange(
            self.history_window, device=state.device, dtype=torch.long
        ).view(1, -1)
        source_indices = (starts.view(-1, 1) + positions).clamp(
            max=self.history_window - 1
        )
        ordered_sequence = torch.gather(
            sequence,
            dim=1,
            index=source_indices.unsqueeze(-1).expand(
                -1, -1, self.raw_feature_dim
            ),
        )
        packed = pack_padded_sequence(
            ordered_sequence,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        sequence_features = hidden[-1]
        if self.context_projection is None or self.fusion is None:
            return sequence_features
        context_features = torch.tanh(
            self.context_projection(state[:, sequence_size:])
        )
        return torch.tanh(
            self.fusion(torch.cat([sequence_features, context_features], dim=-1))
        )


class CausalGRUGaussianActor(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        history_window: int,
        raw_feature_dim: int,
        hidden_dim: int,
        init_log_std: float,
    ) -> None:
        super().__init__()
        self.encoder = CausalGRUStateEncoder(
            state_dim=state_dim,
            history_window=history_window,
            raw_feature_dim=raw_feature_dim,
            hidden_dim=hidden_dim,
        )
        self.mean = _initialize_linear(
            nn.Linear(int(hidden_dim), int(action_dim)), gain=0.01
        )
        self.log_std = nn.Parameter(
            torch.full((int(action_dim),), float(init_log_std), dtype=torch.float32)
        )

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean(self.encoder(state))
        std = torch.exp(self.log_std).clamp(1e-4, 3.0)
        return torch.distributions.Normal(mean, std)

    def forward(
        self, state: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(state)
        action = distribution.rsample() if sample else distribution.mean
        return action, distribution.log_prob(action).sum(dim=-1)

    def log_prob_entropy(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(state)
        return (
            distribution.log_prob(action).sum(dim=-1),
            distribution.entropy().sum(dim=-1),
        )


class CausalGRUValueNet(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        history_window: int,
        raw_feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = CausalGRUStateEncoder(
            state_dim=state_dim,
            history_window=history_window,
            raw_feature_dim=raw_feature_dim,
            hidden_dim=hidden_dim,
        )
        self.value = _initialize_linear(nn.Linear(int(hidden_dim), 1), gain=0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value(self.encoder(state)).squeeze(-1)


def causal_gru_encoder_parameter_count(
    *,
    state_dim: int,
    history_window: int,
    raw_feature_dim: int,
    hidden_dim: int,
) -> int:
    context_dim = _validate_contract(
        state_dim=state_dim,
        history_window=history_window,
        raw_feature_dim=raw_feature_dim,
        hidden_dim=hidden_dim,
    )
    hidden = int(hidden_dim)
    features = int(raw_feature_dim)
    count = 3 * hidden * features + 3 * hidden * hidden + 6 * hidden
    if context_dim > 0:
        count += context_dim * hidden + hidden
        count += 2 * hidden * hidden + hidden
    return int(count)


def causal_gru_actor_parameter_count(
    *,
    state_dim: int,
    action_dim: int,
    history_window: int,
    raw_feature_dim: int,
    hidden_dim: int,
) -> int:
    hidden = int(hidden_dim)
    action = int(action_dim)
    return int(
        causal_gru_encoder_parameter_count(
            state_dim=state_dim,
            history_window=history_window,
            raw_feature_dim=raw_feature_dim,
            hidden_dim=hidden_dim,
        )
        + hidden * action
        + 2 * action
    )


def causal_gru_value_parameter_count(
    *,
    state_dim: int,
    history_window: int,
    raw_feature_dim: int,
    hidden_dim: int,
) -> int:
    hidden = int(hidden_dim)
    return int(
        causal_gru_encoder_parameter_count(
            state_dim=state_dim,
            history_window=history_window,
            raw_feature_dim=raw_feature_dim,
            hidden_dim=hidden_dim,
        )
        + hidden
        + 1
    )
