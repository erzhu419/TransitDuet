"""State-balanced paired GLS under frozen labels, subspace and RMS budget."""

import copy
from functools import partial

import numpy as np

from freq_hrl.domains.transit import native_pair_critic as previous

base, joint, strength, optimizer = previous.base, previous.joint, previous.strength, previous.optimizer
NEW_OBJECTIVES = ("state_gls", "state_head_gls")
OBJECTIVES = (*previous.OBJECTIVES, *NEW_OBJECTIVES)
FILTERS = previous.FILTERS
VARIANTS = ("raw", *(f"{o}_{f}" for o in OBJECTIVES for f in FILTERS))
candidates, predict, metrics, compare_cached = previous.candidates, previous.predict, previous.metrics, previous.compare_cached
select = partial(previous.previous.select, objectives=OBJECTIVES)
aggregate = partial(previous.previous.aggregate, variants=VARIANTS)


def balanced_precision(precision, eligible, *, per_head=False):
    mask = eligible[:, previous.LO] & eligible[:, previous.HI]
    q = mask/np.maximum(mask.sum(axis=1, keepdims=True), 1)
    mass = (q[..., None]*precision).sum(axis=1, keepdims=True)
    denominator = mass if per_head else mass.mean(axis=-1, keepdims=True)
    normalized = np.divide(precision, denominator, out=np.zeros_like(precision), where=denominator > 0)
    effective = (q[..., None]*normalized).sum(axis=1)
    active = mask.any(axis=1)
    expected = np.broadcast_to(active[:, None], effective.shape)
    deviation = abs(effective-expected) if per_head else abs(effective.sum(axis=-1)-2*active)
    return normalized, {"active_states": int(active.sum()), "mass_max_abs_difference": float(deviation.max()),
        "mean_service_risk_mass": effective.mean(axis=0).tolist()}


def fit_paths(rows, data, draws, cached, cached_fit):
    models = copy.deepcopy(cached)
    source = cached["all_mse"]
    models.update({o: copy.deepcopy(source) for o in NEW_OBJECTIVES})
    scale, anchor = source["scale"], data["originals"]
    residual, _, reliability, _ = base.noise_targets(rows, data, draws, scale)
    target = residual*reliability
    precision, noise = previous.pair_precision(rows, draws, scale)
    eligible = data["raw_logits"] <= 0
    balanced = {o: balanced_precision(precision, eligible, per_head=o == "state_head_gls") for o in NEW_OBJECTIVES}
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)
    _, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    numerical = int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum())
    radii = {f["level"]: f["radius"] for f in cached_fit["fits"] if f["objective"] == "pair_gls"}
    ledger, maximum = [], 0.
    for level in strength.lowrank.LEVELS[1:]:
        k = min(int(level), numerical)
        basis = vt[:k].T*np.sqrt(mask.sum())/s[:k]
        z = x@basis
        old = np.asarray(source["weights"][level])
        heads = (vt[:k]@old.T).T*s[:k]/np.sqrt(mask.sum())
        initial, difference = heads.sum(axis=0), heads[0]-heads[1]
        maximum = max(maximum, float(np.max(np.abs((x@(basis@heads.T)-x@old.T)*scale))))
        for objective in NEW_OBJECTIVES:
            loss, hessian = previous.pair_loss(z, target, balanced[objective][0], difference, eligible)
            theta, fit = optimizer.quadratic_fit(loss, hessian, initial, radius=radii[level])
            models[objective]["weights"][level] = (basis@np.stack(((theta+difference)/2, (theta-difference)/2), axis=-1)).T.tolist()
            ledger.append({"objective": objective, "level": level, "retained_rank": k,
                "total_cost_function_df": k, "new_parameters": k, **fit})
    return models, {"fits": ledger, "subspace_max_abs_difference": maximum, "pair_noise": noise,
        "balance": {o: b[1] for o, b in balanced.items()}}
