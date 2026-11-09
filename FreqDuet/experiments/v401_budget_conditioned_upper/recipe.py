"""Matched training inputs; no collection or optimization is launched here."""

import random

import numpy as np
import torch

from conditioned_integration import ObservedUpperTrainer, STATE_DIM


def training_cells(protocol, arm):
    assert arm in protocol['arms']
    return [dict(arm=arm, batch=i//protocol['days_per_batch'],
                 day=protocol['train_day_start']+i,
                 rho=protocol['rho_by_batch_position'][i % protocol['days_per_batch']])
            for i in range(protocol['train_days'])]


def divisors(base_config):
    result = np.asarray([*base_config['state_divisors'], 1.], dtype=np.float32)
    assert result.shape == (STATE_DIM,)
    return result


def make_trainer(base_config, seed):
    # Identical streams for both arms; observation, not initialization, differs.
    streams = np.random.SeedSequence(seed).spawn(3)
    initial, policy, replay = [int(s.generate_state(1)[0]) for s in streams]
    random.seed(initial)
    np.random.seed(initial)
    torch.manual_seed(initial)
    settings = dict(base_config['trainer_settings'])
    settings.update(policy_sample_seed=policy, replay_seed=replay)
    return ObservedUpperTrainer(**settings)
