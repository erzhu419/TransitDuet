"""Frozen-head MSE attribution on natural and common-action query populations."""

import numpy as np

from freq_hrl.domains.transit.native_shared_risk import state_weights

FIELDS = ("raw_MSE", "model_MSE", "correction_squared", "alignment", "delta_MSE")


def components(prediction, raw, target, weights, units):
    delta, residual = prediction-raw, raw-target
    raw_error, model_error = residual**2, (prediction-target)**2
    square, alignment = delta**2, 2*delta*residual
    difference = model_error-raw_error
    error = float(np.max(np.abs(difference-square-alignment)[weights > 0], initial=0.))
    if error > 1e-12: raise ValueError("Risk error decomposition failed")
    return {"units": int(units), **{k: float(np.sum(weights*v)) for k, v in zip(FIELDS,
        (raw_error, model_error, square, alignment, difference))}}, error


def common_pairs(excluded, held, excluded_eligible, held_eligible):
    """One unit per supported probe; average held replicas before pooling probes."""
    by_probe = {r["probe_index"]: i for i, r in enumerate(excluded)}
    if len(by_probe) != len(excluded): raise ValueError("Repeated excluded probe")
    pairs = [(by_probe[r["probe_index"]], j) for j, r in enumerate(held) if r["probe_index"] in by_probe]
    ex, he = np.asarray(pairs, dtype=int).T
    eligible = excluded_eligible[ex]&held_eligible[he]
    weights = state_weights(eligible)
    valid = eligible.any(axis=1)
    probes = np.asarray([held[i]["probe_index"] for i in he])
    units = 0
    for probe in sorted(set(probes.tolist())):
        ids = probes == probe;n = int(np.count_nonzero(ids&valid))
        if n: weights[ids] /= n;units += 1
    return ex, he, weights, {"units": units, "attempted_state_pairs": len(pairs),
        "retained_state_pairs": int(valid.sum()), "shared_action_exposures": int(eligible.sum())}


def summarize(records):
    n = sum(r["units"] for r in records)
    return {"units": n, **{k: sum(r[k] for r in records)/n if n else None for k in FIELDS}}


def contrast(excluded, held):
    return {k: excluded[k]-held[k] if excluded[k] is not None and held[k] is not None else None
        for k in ("correction_squared", "alignment", "delta_MSE")}
