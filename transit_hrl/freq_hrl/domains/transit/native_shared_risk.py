"""Matched causal risk heads: physical MSE, two-head MSE, and signed events."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from freq_hrl.domains.transit.native_signed_risk import ZERO

ARMS = ("raw_reference", "direct_mse", "signed_mse", "signed_mse_simplex", "signed_hurdle")
PENALTIES = (.001, .01, .1)
PRIMARY_PENALTY = .01
# Orthonormal contrasts for none, added fleet, and saved fleet.
CONTRAST = np.array([[1/np.sqrt(2), 1/np.sqrt(6)],
    [-1/np.sqrt(2), 1/np.sqrt(6)], [0., -2/np.sqrt(6)]])


def state_weights(eligible):
    n = eligible.sum(axis=1)
    return eligible/np.maximum(n[:, None], 1)


def event_targets(y):
    return np.stack((np.abs(y) <= ZERO, y > ZERO, y < -ZERO), axis=-1).mean(axis=1)


def amplitudes(y, weights):
    indicators = np.stack((y > ZERO, y < -ZERO), axis=-1)
    w = weights[:, None, :, None]*indicators
    count = w.sum(axis=(0, 1, 2))
    if np.any(count == 0): raise ValueError("Conditional amplitude unidentified in this training subset")
    return (w*np.abs(y)[..., None]).sum(axis=(0, 1, 2))/count


def ridge(z, target, weights, penalty):
    w = weights/weights.sum()
    regularizer = np.eye(z.shape[1])*penalty;regularizer[0, 0] = 0.
    matrix = z.T@(w[:, None]*z)+regularizer
    rhs = z.T@(w[:, None]*target)
    beta = np.linalg.solve(matrix, rhs)
    error = float(np.max(np.abs(matrix@beta-rhs)))
    if error > 1e-9: raise ValueError("Risk MSE solve failed normal equations")
    return beta, error


def event_loss(beta, z, counts, weights, penalty):
    b = beta.reshape(z.shape[1], 2)
    logits = z@b@CONTRAST.T
    p = softmax(logits, axis=1)
    w = weights/weights.sum()
    # Scaling gives unit contrast curvature at uniform probabilities, as in MSE.
    loss = 3*np.sum(w*(logsumexp(logits, axis=1)-np.sum(counts*logits, axis=1)))+penalty*np.sum(b[1:]**2)/2
    grad = 3*z.T@(w[:, None]*(p-counts)@CONTRAST)
    grad[1:] += penalty*b[1:]
    return float(loss), grad.ravel()


def fit(z, y, eligible, penalty):
    """Only this subset's assigned 16 futures enter any target or amplitude."""
    weights = state_weights(eligible)
    amp = amplitudes(y, weights)
    mask = eligible
    design, w = z[mask], weights[mask]
    direct, normal_error = ridge(design, y.mean(axis=1)[mask, None], w, penalty)
    positive = np.maximum(y, 0.).mean(axis=1)/amp[0]
    negative = np.maximum(-y, 0.).mean(axis=1)/amp[1]
    pseudo = np.stack((1-positive-negative, positive, negative), axis=-1)
    signed, signed_error = ridge(design, pseudo[mask]@CONTRAST, w, penalty)
    counts = event_targets(y)[mask]
    initial = np.zeros((design.shape[1], 2))
    initial[0] = np.log(np.average(counts, weights=w, axis=0))@CONTRAST
    solved = minimize(event_loss, initial.ravel(), args=(design, counts, w, penalty), jac=True,
        method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9})
    optimality = float(np.max(np.abs(event_loss(solved.x, design, counts, w, penalty)[1])))
    if optimality > 1e-6: raise ValueError(f"Signed event fit failed optimality: {optimality}")
    d = design.shape[1]
    return {"amplitudes": amp, "direct_mse": direct, "signed_mse": signed,
        "signed_hurdle": solved.x.reshape(d, 2),
        "checks": {"normal_error": max(normal_error, signed_error), "event_gradient": optimality,
            "event_iterations": int(solved.nit), "training_states_with_actions": int(np.count_nonzero(eligible.sum(axis=1))),
            "parameter_counts": {"direct_mse": d, "signed_mse": 2*d+2, "signed_mse_simplex": 2*d+2, "signed_hurdle": 2*d+2}}}


def simplex(p):
    flat = p.reshape(-1, 3)
    order = np.sort(flat, axis=1)[:, ::-1]
    threshold = (np.cumsum(order, axis=1)-1)/np.arange(1, 4)
    rho = (order > threshold).sum(axis=1)-1
    theta = threshold[np.arange(len(flat)), rho]
    return np.maximum(flat-theta[:, None], 0.).reshape(p.shape)


def predict(model, z, originals, raw):
    linear = 1/3+z@model["signed_mse"]@CONTRAST.T
    probabilities = {"signed_mse": linear, "signed_mse_simplex": simplex(linear),
        "signed_hurdle": softmax(z@model["signed_hurdle"]@CONTRAST.T, axis=-1)}
    amp = model["amplitudes"]
    costs = {"raw_reference": raw.copy(), "direct_mse": (z@model["direct_mse"])[..., 0],
        **{a: p[..., 1]*amp[0]-p[..., 2]*amp[1] for a, p in probabilities.items()}}
    for x in costs.values(): x[np.arange(len(z)), originals] = 0.
    return costs, probabilities


def metrics(cost, probability, y, eligible, amp):
    weights = state_weights(eligible)
    mean = y.mean(axis=1)
    error = cost-mean
    oracle = event_targets(y)[..., 1]*amp[0]-event_targets(y)[..., 2]*amp[1]
    out = {"state_exposures": int(np.count_nonzero(eligible.sum(axis=1))), "action_exposures": int(eligible.sum()),
        "state_balanced_squared_error_sum": float(np.sum(weights*error**2)),
        "action_squared_error_sum": float(np.sum(error[eligible]**2)),
        "state_balanced_signed_error_sum": float(np.sum(weights*error)),
        "amplitude_oracle_squared_error_sum": float(np.sum(weights*(oracle-mean)**2)),
        "nonzero_target_actions": int(np.count_nonzero(eligible&(np.abs(mean) > ZERO))),
        "wrong_sign_actions": int(np.count_nonzero(eligible&(np.abs(mean) > ZERO)&(cost*mean <= 0)))}
    if probability is not None:
        target = event_targets(y)
        out.update(state_balanced_brier_sum=float(np.sum(weights[..., None]*(probability-target)**2)),
            outside_simplex_actions=int(np.count_nonzero(eligible&np.any((probability < -1e-12)|(probability > 1+1e-12), axis=-1))))
    return out


def summarize(rows):
    if not rows: return {}
    out = {k: sum(x[k] for x in rows) for k in rows[0]}
    n, m = out["state_exposures"], out["action_exposures"]
    out.update(MSE=out["state_balanced_squared_error_sum"]/n,
        action_MSE=out["action_squared_error_sum"]/m,
        bias=out["state_balanced_signed_error_sum"]/n,
        amplitude_oracle_MSE=out["amplitude_oracle_squared_error_sum"]/n)
    if "state_balanced_brier_sum" in out: out["Brier"] = out["state_balanced_brier_sum"]/n
    return out
