"""An optional mean residual that cannot rewrite its frozen feedback donor."""
import copy

import torch
from torch import nn


class OptionalActionResidual(nn.Module):
    def __init__(self, base, *, feedback_dim, advice_dim):
        super().__init__()
        self.feedback_dim = feedback_dim
        self.base = copy.deepcopy(base).requires_grad_(False)
        self.readout = nn.Linear(feedback_dim+advice_dim, base.log_std.numel())
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def flat_input(self, state):
        return torch.cat((state[..., :self.feedback_dim], torch.zeros_like(state[..., self.feedback_dim:])), -1)

    def distribution(self, state):
        base = self.base.distribution(self.flat_input(state))
        return torch.distributions.Normal(base.mean + self.readout(state), base.stddev)

    def forward_with_mean(self, state, sample=True):
        distribution = self.distribution(state)
        action = distribution.rsample() if sample else distribution.mean
        return action, distribution.log_prob(action).sum(-1), distribution.mean

    def forward(self, state, sample=True):
        action, logp, _ = self.forward_with_mean(state, sample=sample)
        return action, logp
