"""Four-parameter on-policy step using complete-day costs, without a Q model."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch


ACTIONS = np.asarray((0, 5, 10, 15, 20, 30, 45), dtype=float)
FEATURES = ("two_sided_gain", "onboard_holding", "return_backlog_holding", "cumulative_trip_holding")
KL_LIMIT = .001
PASSENGER_SCALE = 3600. * 1000.


def action_features(context, load_fraction, ready_return, due_return, spent_holding_s):
    target = float(context.target_headway_s)
    fraction = ACTIONS / ACTIONS[-1]
    gain = np.zeros(len(ACTIONS))
    if context.evidence_valid and context.follower_evidence_valid:
        forward, follower = float(context.forward_headway_s), float(context.follower_departure_gap_s)
        baseline = ((forward - target) ** 2 + (follower - target) ** 2) / (2. * target ** 2)
        gain = baseline - ((forward + ACTIONS - target) ** 2 +
                           (np.maximum(follower - ACTIONS, 0.) - target) ** 2) / (2. * target ** 2)
    pressure = float(due_return) / (1. + float(ready_return))
    return np.column_stack((gain, -float(load_fraction) * fraction, -pressure * fraction,
                            -float(spent_holding_s) / target * fraction))


def probabilities(policy, states, features, theta):
    with torch.no_grad():
        logits = policy.forward(states)
        bias = torch.as_tensor(np.asarray(features) @ np.asarray(theta), dtype=logits.dtype)
        feasible = policy.feasible_action_mask(states)
        return torch.softmax((logits + bias).masked_fill(~feasible, torch.finfo(logits.dtype).min), -1)


def remaining_cost_at(day, times):
    # This is an other-day training baseline, not a deployment forecast.
    return np.interp(times, np.append(day["times"], day["horizon_s"]),
                     np.append(day["remaining_passenger"], 0.))


def gradient_days(days):
    if len(days) < 2 or len({int(d["scenario_seed"]) for d in days}) != len(days):
        raise ValueError("Monte Carlo baselines require distinct independent demand days")
    gradients = []
    for i, day in enumerate(days):
        others = [d for j, d in enumerate(days) if i != j]
        mean_features = np.einsum("na,nai->ni", day["probabilities"], day["features"])
        score = day["features"][np.arange(len(day["actions"])), day["actions"]] - mean_features
        passenger_baseline = np.mean([remaining_cost_at(d, day["times"]) for d in others], axis=0)
        passenger_advantage = (day["remaining_passenger"] - passenger_baseline) / PASSENGER_SCALE
        cv_advantage = float(day["cv_squared"]) - np.mean([float(d["cv_squared"]) for d in others])
        ready_advantage = float(day["readiness_delay_s"]) - np.mean([float(d["readiness_delay_s"]) for d in others])
        gradients.append(np.stack((score.T @ passenger_advantage, score.sum(0) * cv_advantage,
                                   score.sum(0) * ready_advantage)))
    return np.asarray(gradients)


def empirical_kl(theta, days):
    values, gradients = [], []
    for day in days:
        p, f = day["probabilities"], day["features"]
        bias = f @ theta
        log_p = np.full_like(p, -np.inf)
        np.log(p, out=log_p, where=p > 0)
        normalizer = logsumexp(log_p + bias, axis=1)
        q = np.exp(log_p + bias - normalizer[:, None])
        values.append(float(np.mean(normalizer - (p * bias).sum(1))))
        gradients.append(np.einsum("na,nai->i", q - p, f) / len(p))
    return float(np.mean(values)), np.mean(gradients, axis=0)


def fit_step(days):
    daily = gradient_days(days)
    means = daily.mean(0)
    lengths = np.linalg.norm(means, axis=1)
    normalized = means / np.where(lengths > 0, lengths, 1.)[:, None]
    objective, costs = normalized[0], normalized[1:]
    zero = np.zeros(len(FEATURES))
    result = minimize(lambda theta: objective @ theta, zero, jac=lambda theta: objective,
        bounds=[(0., 1.)] * len(FEATURES), method="SLSQP",
        constraints=[dict(type="ineq", fun=lambda theta: -costs @ theta, jac=lambda theta: -costs),
                     dict(type="ineq", fun=lambda theta: KL_LIMIT - empirical_kl(theta, days)[0],
                          jac=lambda theta: -empirical_kl(theta, days)[1])],
        options=dict(maxiter=200, ftol=1e-10))
    theta = result.x
    kl, _ = empirical_kl(theta, days)
    identified = bool(result.success and objective @ theta < -1e-12 and
                      np.all(costs @ theta <= 1e-9) and kl <= KL_LIMIT + 1e-9)
    if not identified:
        theta = zero
        kl = 0.
    return dict(theta=theta.tolist(), update_identified=identified, solver_success=bool(result.success),
        solver_message=str(result.message), solver_iterations=int(result.nit), training_mean_kl=kl,
        kl_limit=KL_LIMIT, coefficient_bounds=[0., 1.], features=FEATURES,
        gradient_means=means.tolist(), directional_cost_changes=(means @ theta).tolist(),
        daily_gradients=daily.tolist(), training_days=[int(d["scenario_seed"]) for d in days],
        baseline="other-demand-day time-profile / terminal means; no paired-reference labels",
        policy_step="one Monte Carlo direction; no fitted Q, critic, replay or dual update")
