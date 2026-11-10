"""KL projection onto an observable, state-local regularity moment constraint."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
import torch

import freqduet_system_policy_step as step


TOLERANCE = 1e-12


def project_distribution(base, candidate, gains):
    base, candidate = np.asarray(base, dtype=float), np.asarray(candidate, dtype=float)
    base, candidate = base / base.sum(), candidate / candidate.sum()
    gains = np.asarray(gains, dtype=float)
    if base @ gains - candidate @ gains <= TOLERANCE:
        return candidate, False
    span = np.ptp(gains)
    scaled = (gains - gains.min()) / span
    target = base @ scaled - TOLERANCE / span
    log_q = np.full_like(candidate, -np.inf)
    np.log(candidate, out=log_q, where=candidate > 0)

    def tilted(weight):
        logits = log_q + weight * scaled
        return np.exp(logits - logsumexp(logits))

    upper = 1.
    while tilted(upper) @ scaled < target:
        upper *= 2.
    # The original law is feasible; a single increasing moment determines the root.
    weight = brentq(lambda w: tilted(w) @ scaled - target, 0., upper, xtol=1e-12)
    return tilted(weight), True


class RegularityProjection:
    def __init__(self):
        self.calls = self.active = 0
        self.raw_deficit = self.projection_kl = self.modal_gain_change = 0.
        self.max_projected_deficit = 0.

    def __call__(self, policy, states, features, theta):
        adjusted = step.probabilities(policy, states, features, theta)
        base = step.probabilities(policy, states, features, np.zeros(len(step.FEATURES)))[0].numpy().astype(float)
        candidate = adjusted[0].numpy().astype(float)
        base, candidate = base / base.sum(), candidate / candidate.sum()
        gains = features[:, 0]
        projected, active = project_distribution(base, candidate, gains)
        self.calls += 1
        self.active += int(active)
        self.raw_deficit += max(float((base - candidate) @ gains), 0.)
        self.max_projected_deficit = max(self.max_projected_deficit, float((base - projected) @ gains))
        support = projected > 0
        self.projection_kl += float(projected[support] @ np.log(projected[support] / candidate[support]))
        self.modal_gain_change += float(gains[np.argmax(projected)] - gains[np.argmax(base)])
        if not active:
            return adjusted
        return torch.from_numpy(projected[None, :])

    def summary(self):
        return dict(decisions=self.calls, active=self.active, active_fraction=self.active / self.calls,
                    mean_raw_gain_deficit=self.raw_deficit / self.calls,
                    max_projected_gain_deficit=self.max_projected_deficit,
                    mean_kl_to_raw=self.projection_kl / self.calls,
                    mean_modal_gain_change=self.modal_gain_change / self.calls,
                    production_dual_updates=0)
