"""Training-only noise calibration of new service-minus-risk corrections."""

import copy
from functools import partial

import numpy as np

from freq_hrl.domains.transit import native_head_budget as previous

base, joint, strength, optimizer = previous.base, previous.joint, previous.strength, previous.optimizer
balanced, pair = previous.previous, previous.previous.previous
ARMS = ("frozen_old", "free_old", "shrunk")
FILTERS = previous.FILTERS
VARIANTS = ("raw", *(f"{a}_{f}" for a in ("old", *ARMS) for f in FILTERS))
select = partial(pair.previous.select, objectives=("shrunk",))
aggregate = partial(pair.previous.aggregate, variants=VARIANTS)


def geometry(z, precision, eligible):
    dz = z[:, pair.HI]-z[:, pair.LO]
    mask = eligible[:, pair.LO] & eligible[:, pair.HI]
    weights = (mask/np.maximum(mask.sum(axis=1, keepdims=True), 1))[..., None]*precision
    matrix = .5*np.einsum("np,npk,npl->kl", weights.sum(axis=-1), dz, dz)/len(z)
    cross = .5*np.einsum("np,npk,npl->kl", weights[..., 0]-weights[..., 1], dz, dz)/len(z)
    return dz, weights, matrix, cross


def difference_noise(z, draws, precision, eligible, reliability, scale):
    """Conditional score covariance: retain action and service/risk covariance."""
    dz, weights, matrix, _ = geometry(z, precision, eligible)
    values = draws[..., (2, 1)]/scale
    margins = values[:, :, pair.HI]-values[:, :, pair.LO]
    scores = np.einsum("nph,npk,ndph->ndk", weights*reliability, dz,
        margins*np.array([1., -1.]))/len(z)
    centered = scores-scores.mean(axis=1, keepdims=True)
    d = draws.shape[1]
    covariance = np.einsum("ndk,ndl->kl", centered, centered)/(d*(d-1))
    inverse = np.linalg.pinv(matrix, rcond=z.shape[-1]*np.finfo(float).eps, hermitian=True)
    conditional = inverse@covariance@inverse
    return {"noise_energy": max(float(np.trace(conditional)), 0.),
        "weighted_noise_energy": max(float(np.trace(matrix@conditional)), 0.),
        "draws": d}


def penalty_rule(observed_energy, noise):
    signal = max(observed_energy-noise["noise_energy"], 0.)
    return signal, noise["weighted_noise_energy"]/signal if signal > 0 else None


def profiled_shrink_loss(z, target, precision, eligible, initial_difference, penalty):
    dz, weights, matrix, cross = geometry(z, precision, eligible)
    dt = target[:, pair.HI]-target[:, pair.LO]
    rhs = np.einsum("nph,nph,npk->hk", weights, dt, dz)/len(z)
    regularized = matrix+penalty*np.eye(z.shape[-1])
    inverse = np.linalg.pinv(regularized, rcond=z.shape[-1]*np.finfo(float).eps, hermitian=True)
    def increment(total):
        return inverse@(rhs[0]-rhs[1]-cross@total-matrix@initial_difference)
    def difference(total): return initial_difference+increment(total)
    def loss(total):
        delta = increment(total)
        error = (dz@total)[..., None]/2+(dz@(initial_difference+delta))[..., None]*np.array([.5, -.5])-dt
        value = np.sum(weights*error**2)/len(z)+.5*penalty*(delta@delta)
        return float(value), matrix@total+cross@(initial_difference+delta)-rhs.sum(axis=0)
    def optimality(total):
        delta = increment(total)
        gradient = matrix@(initial_difference+delta)+cross@total-rhs[0]+rhs[1]+penalty*delta
        return float(np.linalg.norm(gradient, ord=np.inf)/max(1., np.max(np.abs(rhs))))
    hessian = matrix-cross@inverse@cross
    return loss, lambda total: (hessian+hessian.T)/2, difference, optimality


def fit_paths(rows, data, draws, source, cached_fit):
    scale, anchor = source["scale"], data["originals"]
    residual, _, reliability, _ = base.noise_targets(rows, data, draws, scale)
    target = residual*reliability
    precision, _ = pair.pair_precision(rows, draws, scale)
    eligible = data["raw_logits"] <= 0
    precision, balance = balanced.balanced_precision(precision, eligible)
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)
    _, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    numerical = int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum())
    radii = {f["level"]: f["radius"] for f in cached_fit["fits"] if f["objective"] == "pair_gls"}
    model, ledger, maximum = copy.deepcopy(source), [], 0.
    for level in strength.lowrank.LEVELS[1:]:
        k = min(int(level), numerical)
        basis = vt[:k].T*np.sqrt(mask.sum())/s[:k]
        z = x@basis
        old = np.asarray(source["weights"][level])
        heads = (vt[:k]@old.T).T*s[:k]/np.sqrt(mask.sum())
        initial, prior = heads.sum(axis=0), heads[0]-heads[1]
        maximum = max(maximum, float(np.max(np.abs((x@(basis@heads.T)-x@old.T)*scale))))
        free_loss, free_hessian, free_difference, _ = previous.profiled_pair_loss(z, target, precision, eligible, prior)
        free_total, free_fit = optimizer.quadratic_fit(free_loss, free_hessian, initial, radius=radii[level])
        free_delta = free_difference(free_total)-prior
        noise = difference_noise(z, draws, precision, eligible, reliability, scale)
        signal, penalty = penalty_rule(float(free_delta@free_delta), noise)
        if penalty is None:
            loss, hessian = pair.pair_loss(z, target, precision, prior, eligible)
            difference, optimality = lambda total: prior, lambda total: 0.
        else:
            loss, hessian, difference, optimality = profiled_shrink_loss(z, target, precision, eligible, prior, penalty)
        total, fit = optimizer.quadratic_fit(loss, hessian, initial, radius=radii[level])
        d = difference(total)
        coefficients = np.stack(((total+d)/2, (total-d)/2))
        model["weights"][level] = (basis@coefficients.T).T.tolist()
        data_loss, _ = pair.pair_loss(z, target, precision, d, eligible)
        ledger.append({"level": level, "retained_rank": k, "total_cost_function_df": k,
            "component_function_df_max": 2*k, "old_radius": float(radii[level]),
            "penalty": penalty, "signal_energy": signal, **noise,
            "free_difference_change_rms": float(np.linalg.norm(free_delta)),
            "difference_change_rms": float(np.linalg.norm(d-prior)), "difference_optimality": optimality(total),
            "data_loss_after": data_loss(total)[0], "free_data_loss_after": free_loss(free_total)[0],
            "auxiliary_free_fit": free_fit, **fit})
    return {"shrunk": model}, {"fits": ledger, "subspace_max_abs_difference": maximum, "balance": balance}


def qualify_fit(fit):
    if len(fit["fits"]) != 4 or {f["level"] for f in fit["fits"]} != set(strength.lowrank.LEVELS[1:]):
        raise ValueError("Dropped shrinkage rank")
    if fit["subspace_max_abs_difference"] > 1e-12 or fit["balance"]["mass_max_abs_difference"] > 1e-12:
        raise ValueError("Changed subspace or state balance")
    for f in fit["fits"]:
        if (not f["success"] or max(f["optimality"], f["difference_optimality"]) > 1e-8
                or f["norm"] > f["old_radius"]+1e-10 or f["radius"] != f["old_radius"]
                or f["difference_change_rms"] > f["free_difference_change_rms"]+1e-8
                or f["loss_after"] > f["loss_before"]+1e-10*max(1., f["loss_before"])):
            raise ValueError("Shrinkage KKT, old total RMS or difference shrinkage failed")


def predict(paths, selection):
    return {"raw": paths["frozen_old"]["zero"], **{f"{a}_{f}": paths[a][selection[a][f] or "zero"]
        for a in ("old", *ARMS) for f in FILTERS}}
