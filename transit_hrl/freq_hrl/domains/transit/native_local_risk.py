"""Equal-parameter linear/observable-context residuals with fixed local bounds."""

import numpy as np

from freq_hrl.domains.transit import native_prior_risk as previous

base = previous.base
PRIORS = previous.PRIORS
MAPS = ("linear", "interaction")
SUFFIXES = ("prior", "residual", "bounded", "residual_clip", "bounded_clip",
    "linear", "linear_clip", "interaction", "interaction_clip")
ARMS = ("cold_mse", *(f"{p}_{a}" for p in PRIORS for a in SUFFIXES))
CLIPPED = ("residual_clip", "bounded_clip", "linear_clip", "interaction_clip")


def features(z, eligible, kind):
    x = z[..., 1:]
    if kind == "linear": return x.copy()
    if kind != "interaction": raise ValueError(kind)
    w = base.state_weights(eligible)
    context = np.sum(w[..., None]*x, axis=1)
    return x*np.tanh(context[:, None])


def normalization(x, eligible):
    w = base.state_weights(eligible);w = w/w.sum()
    center = np.sum(w[..., None]*x, axis=(0, 1))
    scale = np.sqrt(np.sum(w[..., None]*(x-center)**2, axis=(0, 1)))
    # Constant training coordinates carry no slope information.
    scale = np.where(scale > 0, scale, 1.)
    return {"center": center, "scale": scale}


def transform(x, stats):
    return np.concatenate((np.ones((*x.shape[:2], 1)), (x-stats["center"])/stats["scale"]), axis=-1)


def fit(z, y, priors, eligible, penalty):
    """Share the design solve, not coefficients, among the three frozen priors."""
    target = y.mean(axis=1)[..., None]-priors
    w = base.state_weights(eligible)[eligible]
    models, maximum = {}, 0.
    for kind in MAPS:
        x = features(z, eligible, kind)
        stats = normalization(x, eligible)
        design = transform(x, stats)
        beta, error = base.ridge(design[eligible], target[eligible], w, penalty)
        models[kind] = {**stats, "weights": beta}
        maximum = max(maximum, error)
    return {"maps": models, "normal_error": maximum,
        "parameters_per_prior_per_map": z.shape[-1], "head_fits": len(PRIORS)*len(MAPS), "ridge_solver_calls": len(MAPS)}


def predict(model, z, priors, cached, originals, eligible, radius):
    costs = {"cold_mse": cached["cold_mse"].copy()};clips = {}
    mapped = {kind: transform(features(z, eligible, kind), model["maps"][kind])@model["maps"][kind]["weights"] for kind in MAPS}
    ix = np.arange(len(z)), originals
    for j, prior in enumerate(PRIORS):
        anchor = priors[..., j]
        costs[prior+"_prior"] = anchor.copy()
        for suffix in ("residual", "bounded", *MAPS):
            key = prior+"_"+suffix
            delta = cached[key]-anchor if suffix in ("residual", "bounded") else mapped[suffix][..., j].copy()
            delta[ix] = 0.
            costs[key] = anchor+delta
            clips[key+"_clip"] = np.abs(delta) > radius
            costs[key+"_clip"] = anchor+np.clip(delta, -radius, radius)
        for suffix in SUFFIXES: costs[prior+"_"+suffix][ix] = 0.
    return costs, clips


def metrics(cost, prior, y, eligible, radius, clipped=None):
    w = base.state_weights(eligible)
    target = y.mean(axis=1)
    error, delta = cost-target, cost-prior
    max_delta = float(np.max(np.abs(delta[eligible]), initial=0.))
    excess = max(0., max_delta-radius)
    if clipped is not None and excess > 1e-12: raise ValueError("Query-local correction bound failed")
    return {"state_exposures": int(np.count_nonzero(eligible.sum(axis=1))), "action_exposures": int(eligible.sum()),
        "squared_error_sum": float(np.sum(w*error**2)), "signed_error_sum": float(np.sum(w*error)),
        "delta_squared_sum": float(np.sum(w*delta**2)), "alignment_sum": float(np.sum(w*(target-prior)*delta)),
        "radius_squared_exposure_sum": float(w.sum()*radius**2),
        "max_delta_over_radius": max_delta/radius if radius else None,
        "clipped_actions": int(np.count_nonzero(eligible&clipped)) if clipped is not None else 0,
        "checked_bound_excess": excess if clipped is not None else 0.}


def summarize(rows, detailed):
    n = sum(r["state_exposures"] for r in rows);a = sum(r["action_exposures"] for r in rows)
    result = {"MSE": sum(r["squared_error_sum"] for r in rows)/n,
        "bias": sum(r["signed_error_sum"] for r in rows)/n}
    if detailed:
        radius = sum(r["radius_squared_exposure_sum"] for r in rows)
        result.update(delta_RMS_over_radius=float(np.sqrt(sum(r["delta_squared_sum"] for r in rows)/radius)) if radius else None,
            alignment_mean=sum(r["alignment_sum"] for r in rows)/n,
            max_delta_over_radius=max((r["max_delta_over_radius"] for r in rows if r["max_delta_over_radius"] is not None), default=None),
            clipped_action_fraction=sum(r["clipped_actions"] for r in rows)/a if a else None)
    return result
