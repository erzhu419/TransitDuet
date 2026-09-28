"""Causal state prediction with action-independent external and physical heads."""

from __future__ import annotations

import math

import torch
from torch import nn

from .causal_sequence import CausalGRUStateEncoder


class ActionConditionedStatePredictor(nn.Module):
    def __init__(self, *, physical_dim, external_dim, action_dim, history_window=64,
                 hidden_dim=64, latent_dim=16):
        super().__init__()
        self.physical_dim, self.external_dim = physical_dim, external_dim
        self.action_dim, self.history_window = action_dim, history_window
        self.feature_dim = physical_dim + external_dim + action_dim
        self.encoder = CausalGRUStateEncoder(
            state_dim=history_window*self.feature_dim+1, history_window=history_window,
            raw_feature_dim=self.feature_dim, hidden_dim=hidden_dim)
        self.latent = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.Tanh())
        self.physical = nn.Sequential(nn.Linear(latent_dim+action_dim, hidden_dim), nn.Tanh(),
                                      nn.Linear(hidden_dim, 2*physical_dim))
        self.external = nn.Linear(latent_dim, 2*external_dim)

    def forward(self, history, action):
        if history.shape[1:] != (self.history_window, self.feature_dim):
            raise ValueError("state predictor requires a complete observed history window")
        state = torch.cat((history.flatten(1), history.new_ones((history.shape[0], 1))), dim=1)
        latent = self.latent(self.encoder(state))
        physical_mean, physical_logvar = self.physical(torch.cat((latent, action), dim=1)).chunk(2, dim=1)
        external_mean, external_logvar = self.external(latent).chunk(2, dim=1)
        return (torch.cat((physical_mean, external_mean), dim=1),
                torch.cat((physical_logvar, external_logvar), dim=1).clamp(-8., 4.))


def gaussian_prediction_loss(mean, log_variance, target):
    return .5*((target-mean).square()*torch.exp(-log_variance)
               + log_variance + math.log(2.*math.pi)).mean()
