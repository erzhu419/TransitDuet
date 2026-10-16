"""Matched residual lifts using observed state or action-derived context."""

import numpy as np

from freq_hrl.domains.transit import native_conditional_wait as state
from freq_hrl.domains.transit import native_local_risk as base

MAPS = ("action_context", "observed_context")
CONTEXT_NAMES = ("clock_hours", "same_queue", "other_queue", "same_onboard",
    "same_active", "other_active", "same_remaining_mean", "lf_forecast")
ARMS = ("raw_prior", "raw_bounded_clip", "raw_linear_clip",
    *(a+s for a in MAPS for s in ("", "_clip")))


def observed_context(rows):
    return np.stack([np.r_[r["time_s"]/3600., state.condition_features(r)[[0, 2, 3, 7, 8, 9, 11]]] for r in rows])


def context(z, observed, eligible, kind):
    if kind == "observed_context": return observed
    if kind == "action_context":
        return np.sum(base.base.state_weights(eligible)[..., None]*z[..., 1:], axis=1)
    raise ValueError(kind)


def lift(z, c, stats):
    g = np.tanh((c-np.asarray(stats["center"]))/np.asarray(stats["scale"]))
    x = z[..., 1:]
    return np.concatenate((x, (x[..., :, None]*g[:, None, None, :]).reshape(*x.shape[:2], 64)), axis=-1)


def fit(z, observed, target, prior, eligible, penalty):
    models, error = {}, 0.
    w = base.base.state_weights(eligible)[eligible]
    for kind in MAPS:
        c = context(z, observed, eligible, kind)
        stats = base.normalization(np.broadcast_to(c[:, None], (*z.shape[:2], 8)), eligible)
        x = lift(z, c, stats)
        normalizer = base.normalization(x, eligible)
        design = base.transform(x, normalizer)
        beta, err = base.base.ridge(design[eligible], (target-prior)[eligible, None], w, penalty)
        models[kind] = {"context_stats": stats, **normalizer, "weights": beta}
        error = max(error, err)
    return {"maps": models, "parameters_per_map": 73, "head_fits": 2, "normal_error": error}


def predict(model, z, observed, prior, eligible, originals, radius):
    costs, clips = {}, {}
    for kind, m in model["maps"].items():
        x = lift(z, context(z, observed, eligible, kind), m["context_stats"])
        delta = (base.transform(x, m)@np.asarray(m["weights"]))[..., 0]
        delta[np.arange(len(z)), originals] = 0.
        costs[kind] = prior+delta
        costs[kind+"_clip"] = prior+np.clip(delta, -radius, radius)
        clips[kind+"_clip"] = np.abs(delta) > radius
    return costs, clips
