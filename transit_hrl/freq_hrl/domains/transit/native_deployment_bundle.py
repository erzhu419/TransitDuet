"""Incremental cross-fitting in the frozen deployment reference's coordinates."""

import copy
from functools import partial

import numpy as np

from freq_hrl.domains.transit import native_selector_information as source

MODELS, FILTERS = source.MODELS, source.FILTERS
SELECTORS = ("matched2", "expanded8")
ARMS = tuple(f"{m}_{s}" for m in MODELS for s in SELECTORS)
BUNDLES = ("prefix_refit", "fixed_bundle")
VARIANTS = ("raw", *(f"{b}_{a}_{f}" for b in BUNDLES for a in ARMS for f in FILTERS))
base, joint, strength = source.base, source.joint, source.strength
select = partial(source.previous.pair.previous.select, objectives=ARMS)
aggregate = partial(source.previous.pair.previous.aggregate, variants=VARIANTS)


def deployment_geometry(data):
    """Whitening is defined on the full past training design, not each inner fold."""
    x = joint.relative(data["x"][..., :198], data["originals"])
    mask = np.arange(6)[None, :] != data["originals"][:, None]
    _, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    numerical = int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum())
    return {level: (vt[:min(int(level), numerical)].T*np.sqrt(mask.sum())/s[:min(int(level), numerical)],
        vt[:min(int(level), numerical)]*s[:min(int(level), numerical), None]/np.sqrt(mask.sum()))
        for level in strength.lowrank.LEVELS[1:]}


def fit_inner(rows, data, draws, prior, full_fit, geometry):
    """Use existing GLS and shrinkage rules with fixed full basis, prior and cap."""
    previous = source.previous
    scale, anchor = prior["scale"], data["originals"]
    residual, _, reliability, _ = base.noise_targets(rows, data, draws, scale)
    target = residual*reliability
    precision, _ = previous.pair.pair_precision(rows, draws, scale)
    eligible = data["raw_logits"] <= 0
    precision, balance = previous.balanced.balanced_precision(precision, eligible)
    x = joint.relative(data["x"][..., :198], anchor)
    radii = {f["level"]: f["radius"] for f in full_fit["fits"] if f["objective"] == "pair_gls"}
    models, ledger, maximum = {m: copy.deepcopy(prior) for m in MODELS}, [], 0.
    for level, (basis, coordinates) in geometry.items():
        old = np.asarray(prior["weights"][level])
        heads = old@coordinates.T
        initial, difference_prior = heads.sum(axis=0), heads[0]-heads[1]
        maximum = max(maximum, float(np.max(np.abs((x@(basis@heads.T)-x@old.T)*scale))))
        z = x@basis
        free_loss, free_hessian, free_difference, _ = previous.previous.profiled_pair_loss(z, target, precision, eligible, difference_prior)
        free_total, free_fit = source.optimizer.quadratic_fit(free_loss, free_hessian, initial, radius=radii[level])
        free_d = free_difference(free_total)
        models["free_old"]["weights"][level] = (basis@np.stack(((free_total+free_d)/2, (free_total-free_d)/2), axis=-1)).T.tolist()
        noise = previous.difference_noise(z, draws, precision, eligible, reliability, scale)
        free_delta = free_d-difference_prior
        signal, penalty = previous.penalty_rule(float(free_delta@free_delta), noise)
        if penalty is None:
            loss, hessian = previous.pair.pair_loss(z, target, precision, difference_prior, eligible)
            difference, optimality = lambda total: difference_prior, lambda total: 0.
        else:
            loss, hessian, difference, optimality = previous.profiled_shrink_loss(z, target, precision, eligible, difference_prior, penalty)
        total, fit = source.optimizer.quadratic_fit(loss, hessian, initial, radius=radii[level])
        d = difference(total)
        models["shrunk"]["weights"][level] = (basis@np.stack(((total+d)/2, (total-d)/2), axis=-1)).T.tolist()
        if not free_fit["success"] or free_fit["optimality"] > 1e-8 or free_fit["norm"] > radii[level]+1e-10:
            raise ValueError("Fixed-bundle free GLS failed its full RMS cap or KKT check")
        ledger.append({"level": level, "retained_rank": basis.shape[1], "old_radius": float(radii[level]),
            "cached_initial_norm": float(np.linalg.norm(initial)), "penalty": penalty, "signal_energy": signal, **noise,
            "free_difference_change_rms": float(np.linalg.norm(free_delta)),
            "difference_change_rms": float(np.linalg.norm(d-difference_prior)), "difference_optimality": optimality(total),
            "auxiliary_free_fit": free_fit, **fit})
    fit = {"fits": ledger, "subspace_max_abs_difference": maximum, "balance": balance}
    previous.qualify_fit(fit)
    return models, fit


def predict(paths, selections):
    return {"raw": paths["free_old"]["zero"], **{f"{b}_{m}_{s}_{f}": paths[m][selections[b][f"{m}_{s}"][f] or "zero"]
        for b in BUNDLES for m in MODELS for s in SELECTORS for f in FILTERS}}
