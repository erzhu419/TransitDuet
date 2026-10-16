"""Feature-only ridge complexity matching for frozen residual representations."""

import numpy as np
from scipy.optimize import brentq

from freq_hrl.domains.transit import native_state_risk as previous

MAPS = previous.MAPS
ARMS = (*previous.ARMS, *(kind+"_df"+suffix for kind in MAPS for suffix in ("", "_clip")))


def spectrum(design, eligible):
    w = previous.base.base.state_weights(eligible)[eligible];w /= w.sum()
    slopes = design[eligible][:, 1:]
    if np.max(np.abs(w@slopes), initial=0.) > 1e-9:
        raise ValueError("Effective-df calculation requires fit-centered slopes")
    return np.maximum(np.linalg.eigvalsh(slopes.T@(w[:, None]*slopes)), 0.)


def effective_df(eigenvalues, penalty):
    return float(1.+np.sum(eigenvalues/(eigenvalues+penalty)))


def matched_penalty(eigenvalues, target, minimum=.01):
    if effective_df(eigenvalues, minimum) < target-1e-8:
        raise ValueError("Lift is less complex than reference at the frozen penalty")
    if abs(effective_df(eigenvalues, minimum)-target) <= 1e-8: return minimum
    high = minimum*2
    while effective_df(eigenvalues, high) > target: high *= 2
    return float(brentq(lambda p: effective_df(eigenvalues, p)-target, minimum, high, xtol=1e-12))


def fit(cached, linear, z, observed, target, prior, eligible):
    base = previous.base
    reference = base.transform(z[..., 1:], linear)
    target_df = effective_df(spectrum(reference, eligible), .01)
    maps, checks, error = {}, {}, 0.
    weights = base.base.state_weights(eligible)[eligible]
    for kind in MAPS:
        m = cached["maps"][kind]
        c = previous.context(z, observed, eligible, kind)
        design = base.transform(previous.lift(z, c, m["context_stats"]), m)
        eigenvalues = spectrum(design, eligible)
        penalty = matched_penalty(eigenvalues, target_df)
        beta, err = base.base.ridge(design[eligible], (target-prior)[eligible, None], weights, penalty)
        fitted_df = effective_df(eigenvalues, penalty)
        if abs(fitted_df-target_df) > 1e-8: raise ValueError("Effective-df match failed")
        maps[kind] = {**m, "weights": beta}
        checks[kind] = {"target_df": target_df, "old_df": effective_df(eigenvalues, .01),
            "matched_df": fitted_df, "penalty": penalty}
        error = max(error, err)
    return {"maps": maps, "head_fits": 2, "parameters_per_map": 73, "normal_error": error, "complexity": checks}


def predict(model, *args):
    costs, clips = previous.predict(model, *args)
    rename = lambda k: k[:-5]+"_df_clip" if k.endswith("_clip") else k+"_df"
    return {rename(k): v for k, v in costs.items()}, {rename(k): v for k, v in clips.items()}
