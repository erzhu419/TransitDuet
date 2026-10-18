"""Independent finite-day levels and seven-action system-cost contrasts."""

import copy

import numpy as np
from scipy.optimize import linprog
import torch
from torch import nn


GOALS = ('passenger_person_s', 'cv_squared', 'readiness_delay_s')
PAIRS = np.asarray([(a, b) for a in range(7) for b in range(a + 1, 7)])


def center(values, probabilities):
    return values - (values * probabilities[..., :, None]).sum(-2)[..., None, :]


def differences(values):
    return values[..., PAIRS[:, 0], :] - values[..., PAIRS[:, 1], :]


class GoalHeads(nn.Module):
    def __init__(self, dimension, outputs):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(dimension, 16), nn.Tanh(),
                                                nn.Linear(16, outputs)) for _ in GOALS])
        self.double()

    def forward(self, states):
        return torch.stack([head(states) for head in self.heads], -1)


def scale(values, axes):
    result = np.std(values, axis=axes)
    return np.where(result > 0, result, 1.)


def fit(states, probabilities, outcomes, steps=256):
    """Fit only development prefixes; no scalarized goal or actor/dual update."""
    x, p, y = [np.asarray(a, np.float64) for a in (states, probabilities, outcomes)]
    y = y.mean(1)  # Four CRN futures are replicates, not independent states.
    level = (y * p[..., :, None]).sum(1)
    advantage = center(y, p)
    norms = dict(x_mean=x.mean(0), x_scale=scale(x, 0), y_mean=y.mean((0, 1)),
                 q_scale=scale(y, (0, 1)), a_scale=scale(advantage, (0, 1)),
                 v_mean=level.mean(0), v_scale=scale(level, 0))
    xt = torch.from_numpy((x - norms['x_mean']) / norms['x_scale'])
    pt = torch.from_numpy(p)
    torch.manual_seed(104)
    action = GoalHeads(x.shape[1], 7)
    networks = {'direct_q': copy.deepcopy(action), 'separated_a': copy.deepcopy(action),
                'level_v': GoalHeads(x.shape[1], 1)}
    targets = {'direct_q': (y - norms['y_mean']) / norms['q_scale'],
               'separated_a': advantage / norms['a_scale'],
               'level_v': ((level - norms['v_mean']) / norms['v_scale'])[:, None, :]}
    losses = {}
    for name, network in networks.items():
        target = torch.from_numpy(targets[name])
        optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
        for _ in range(steps):
            pred = network(xt)
            if name == 'separated_a':
                pred = center(pred, pt)
            loss = (pred - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses[name] = float(loss.detach())
    return dict(norms=norms, dimension=x.shape[1], networks={k: n.state_dict() for k, n in networks.items()},
                steps=steps, prefixes=len(x), final_normalized_losses=losses)


def predict(artifact, states, probabilities):
    norms = artifact['norms']
    x = torch.as_tensor((np.asarray(states) - norms['x_mean']) / norms['x_scale'], dtype=torch.float64)
    p = np.asarray(probabilities, np.float64)
    output = {}
    with torch.no_grad():
        for name, weights in artifact['networks'].items():
            network = GoalHeads(artifact['dimension'], 1 if name == 'level_v' else 7)
            network.load_state_dict(weights)
            value = network(x).numpy()
            if name == 'level_v':
                output[name] = value[:, 0] * norms['v_scale'] + norms['v_mean']
            elif name == 'direct_q':
                raw = value * norms['q_scale'] + norms['y_mean']
                output['direct_q_level'] = (raw * p[..., :, None]).sum(1)
                output[name] = center(raw, p)
            else:
                output[name] = center(value * norms['a_scale'], p)
    output['zero'] = np.zeros_like(output['separated_a'])
    return output


def proposal(advantage, native_p, training_scale):
    """Single-decision cost LP, not a full-policy gradient or a service guarantee."""
    a, p = np.asarray(advantage, float), np.asarray(native_p, float)
    a = center(a, p) / np.asarray(training_scale)
    if not np.any(a):
        return p.copy()
    result = linprog(a[:, 0], A_ub=a[:, 1:].T, b_ub=np.zeros(2),
                     A_eq=np.ones((1, 7)), b_eq=[1.], bounds=(0, None), method='highs')
    if not result.success:
        raise AssertionError('system action proposal failed: ' + result.message)
    return result.x if result.fun < 0 else p.copy()


def errors(prediction, outcomes, probabilities):
    """Keep value-level fit, contrast error, and finite-future noise separate."""
    y, p = np.asarray(outcomes), np.asarray(probabilities)
    mean = y.mean(1)
    target = differences(mean)
    noise = differences(y).var(1, ddof=1).mean(1) / y.shape[1]
    result = {}
    for name in ('separated_a', 'direct_q', 'zero'):
        error = differences(prediction[name]) - target
        result[name] = dict(difference_mse=np.mean(error ** 2, 1),
                            difference_mae=np.mean(np.abs(error), 1))
    level = (mean * p[..., :, None]).sum(1)
    result['level_v'] = dict(level_squared_error=(prediction['level_v'] - level) ** 2)
    result['direct_q_level'] = dict(level_squared_error=(prediction['direct_q_level'] - level) ** 2)
    result['signal'] = dict(difference_noise_mse=noise,
                           difference_energy_minus_noise=np.mean(target ** 2, 1) - noise)
    return result
