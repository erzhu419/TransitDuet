"""Small state/policy-conditioned models of native-anchored action effects."""

import numpy as np
import torch

import freqduet_action_increment as baseline


HIDDEN = 8
STEPS = 2048
LR = 0.003
WEIGHT_DECAY = 0.001
INIT_SEED = 77000
ACTORS = 4


def features(data, conditioned):
    state = np.column_stack((data["states"], data["resources"]))
    return np.column_stack((state, np.eye(ACTORS)[data["cases"]])) if conditioned else state


def action_basis(native):
    actions = np.asarray(baseline.ACTIONS, dtype=float) / max(baseline.ACTIONS)
    reference = actions[native, None]
    return np.stack((actions[None, :] - reference,
                     actions[None, :] ** 2 - reference ** 2), axis=-1)


class ConditionalResponse(torch.nn.Module):
    def __init__(self, center, scale, target_scale):
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float64).clone())
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float64).clone())
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float64).clone())
        self.net = torch.nn.Sequential(torch.nn.Linear(len(center), HIDDEN), torch.nn.Tanh(),
                                       torch.nn.Linear(HIDDEN, 2 * len(baseline.METRICS))).double()
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, states, basis):
        coefficients = self.net((states - self.center) / self.scale).reshape(-1, 2, len(baseline.METRICS))
        return torch.einsum("nai,nim->nam", basis, coefficients) * self.target_scale

    def predict(self, states, native):
        with torch.no_grad():
            return self(torch.as_tensor(states, dtype=torch.float64),
                        torch.as_tensor(action_basis(native), dtype=torch.float64)).numpy()


def fit_model(states, native, targets, days, steps=STEPS):
    weights = baseline.day_weights(days)
    center = (weights[:, None] * states).sum(axis=0)
    scale = np.sqrt((weights[:, None] * (states - center) ** 2).sum(axis=0))
    scale[scale < 1e-6] = 1.
    valid = np.arange(len(baseline.ACTIONS))[None, :] != native[:, None]
    action_weights = weights[:, None] * valid / (len(baseline.ACTIONS) - 1)
    target_scale = np.sqrt((action_weights[:, :, None] * targets ** 2).sum(axis=(0, 1)))
    target_scale[target_scale < 1e-12] = 1.
    with torch.random.fork_rng():
        torch.manual_seed(INIT_SEED)
        fitted = ConditionalResponse(center, scale, target_scale)
    optimizer = torch.optim.Adam(fitted.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x = torch.as_tensor(states, dtype=torch.float64)
    basis = torch.as_tensor(action_basis(native), dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.float64)
    w = torch.as_tensor(action_weights, dtype=torch.float64)

    def loss():
        return (w[:, :, None] * ((fitted(x, basis) - y) / fitted.target_scale) ** 2).sum() / len(baseline.METRICS)

    initial = float(loss().detach())
    for _ in range(steps):
        optimizer.zero_grad()
        objective = loss()
        objective.backward()
        optimizer.step()
    final = float(loss().detach())
    if not np.isfinite(final) or final > initial + 1e-12:
        raise ValueError("conditional response fit did not reduce its training loss")
    return fitted.eval(), dict(steps=steps, initial_normalized_mse=initial, final_normalized_mse=final,
                              parameters=sum(p.numel() for p in fitted.parameters()))
