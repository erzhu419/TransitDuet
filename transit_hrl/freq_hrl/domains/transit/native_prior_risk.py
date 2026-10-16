"""Fixed-prior residual regression and its exact span/shrinkage decomposition."""

import numpy as np
from scipy.optimize import brentq

from freq_hrl.domains.transit import native_shared_risk as base

PRIORS = ("raw", "free_old", "shrunk")
ARMS = ("cold_mse", *(f"{p}_{a}" for p in PRIORS for a in ("prior", "residual", "bounded", "outside_span")))


def project(z, prior, eligible):
    w = base.state_weights(eligible)[eligible]
    x = z[eligible]*np.sqrt(w[:, None])
    return np.linalg.lstsq(x, prior[eligible, None]*np.sqrt(w[:, None]), rcond=None)[0]


def fit(z, y, prior, eligible, penalty, radius):
    w = base.state_weights(eligible)[eligible];w = w/w.sum()
    x = z[eligible]
    target = (y.mean(axis=1)-prior)[eligible, None]
    beta, error = base.ridge(x, target, w, penalty)
    gram = x.T@(w[:, None]*x)
    regularizer = np.eye(x.shape[1])*penalty;regularizer[0, 0] = 0.
    rhs = x.T@(w[:, None]*target)
    norm = lambda b: float(np.sqrt(max(0., float((b.T@gram@b).item()))))
    unbounded = norm(beta)
    mu, calls = 0., 0
    if radius == 0:
        bounded = np.zeros_like(beta)
        mu, kkt = None, None
    elif unbounded <= radius:
        bounded = beta.copy();kkt = error
    else:
        def solution(value):
            nonlocal calls
            calls += 1
            return np.linalg.solve((1+value)*gram+regularizer, rhs)
        high = 1.
        while norm(solution(high)) > radius: high *= 2
        mu = brentq(lambda value: norm(solution(value))-radius, 0., high, xtol=1e-12)
        bounded = solution(mu)
        kkt = float(np.max(np.abs(((1+mu)*gram+regularizer)@bounded-rhs)))
    bounded_norm = norm(bounded)
    if (kkt is not None and kkt > 1e-9) or bounded_norm > radius+1e-10: raise ValueError("Residual RMS constraint failed")
    return {"unbounded": beta, "bounded": bounded, "radius": radius,
        "checks": {"normal_error": error, "kkt": kkt, "unbounded_RMS": unbounded,
            "bounded_RMS": bounded_norm, "active": unbounded > radius, "zero_radius": radius == 0,
            "multiplier": mu, "constraint_linear_solves": calls}}


def predict(z, prior, cold, fitted, projection, originals):
    cold_cost = (z@cold)[..., 0]
    outside = prior-(z@projection)[..., 0]
    shrink = (z@(projection-(cold-fitted["unbounded"])))[..., 0]
    residual = prior+(z@fitted["unbounded"])[..., 0]
    bounded = prior+(z@fitted["bounded"])[..., 0]
    ix = np.arange(len(z)), originals
    for x in (cold_cost, outside, shrink, residual, bounded): x[ix] = 0.
    gap = residual-cold_cost
    error = float(np.max(np.abs(gap-outside-shrink)))
    if error > 1e-10: raise ValueError("Prior response span/shrinkage identity failed")
    return {"prior": prior.copy(), "residual": residual, "bounded": bounded, "outside_span": cold_cost+outside}, {
        "outside": outside, "shrink": shrink, "gap": gap, "identity_error": error}


def response_stats(parts, prior, bounded, eligible, radius):
    w = base.state_weights(eligible)
    delta = bounded-prior
    return {"state_exposures": int(np.count_nonzero(eligible.sum(axis=1))),
        "outside_squared_sum": float(np.sum(w*parts["outside"]**2)),
        "shrink_squared_sum": float(np.sum(w*parts["shrink"]**2)),
        "gap_squared_sum": float(np.sum(w*parts["gap"]**2)),
        "signed_cross_sum": float(np.sum(2*w*parts["outside"]*parts["shrink"])),
        "bounded_delta_squared_sum": float(np.sum(w*delta**2)),
        "radius_squared_exposure_sum": float(np.sum(w)*radius**2)}


def summarize_response(rows):
    sums = {k: sum(r[k] for r in rows) for k in rows[0]}
    n = sums["state_exposures"]
    return {"outside_RMS": float(np.sqrt(sums["outside_squared_sum"]/n)),
        "shrink_RMS": float(np.sqrt(sums["shrink_squared_sum"]/n)),
        "gap_RMS": float(np.sqrt(sums["gap_squared_sum"]/n)),
        "signed_cross_mean": sums["signed_cross_sum"]/n,
        "bounded_delta_over_training_radius_RMS": float(np.sqrt(sums["bounded_delta_squared_sum"]/sums["radius_squared_exposure_sum"])) if sums["radius_squared_exposure_sum"] else None}
