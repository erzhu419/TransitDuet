"""Cross head-difference freedom with training-calibrated total RMS budgets."""

import copy
from functools import partial

import numpy as np

from freq_hrl.domains.transit import native_balanced_critic as previous

base, joint, strength, optimizer = previous.base, previous.joint, previous.strength, previous.optimizer
ARMS = ("frozen_old", "free_old", "frozen_train", "free_train")
NEW_ARMS = ARMS[1:]
FILTERS = previous.FILTERS
VARIANTS = ("raw", *(f"{a}_{f}" for a in ("old", *ARMS) for f in FILTERS))
select = partial(previous.previous.previous.select, objectives=NEW_ARMS)
aggregate = partial(previous.previous.previous.aggregate, variants=VARIANTS)


def profiled_pair_loss(z, target, precision, eligible, initial_difference):
    """Eliminate the free difference, leaving the existing sum-ball optimizer."""
    lo, hi = previous.previous.LO, previous.previous.HI
    dz, dt = z[:, hi]-z[:, lo], target[:, hi]-target[:, lo]
    mask = eligible[:, lo] & eligible[:, hi]
    weights = (mask/np.maximum(mask.sum(axis=1, keepdims=True), 1))[..., None]*precision
    n, k = len(z), z.shape[-1]
    matrix = .5*np.einsum("np,npk,npl->kl", weights.sum(axis=-1), dz, dz)/n
    cross = .5*np.einsum("np,npk,npl->kl", weights[..., 0]-weights[..., 1], dz, dz)/n
    rhs = np.einsum("nph,nph,npk->hk", weights, dt, dz)/n
    inverse = np.linalg.pinv(matrix, rcond=k*np.finfo(float).eps, hermitian=True)
    # Retain the cached difference in unsupported directions; no extra prior/grid.
    def difference(total):
        return initial_difference+inverse@(rhs[0]-rhs[1]-cross@total-matrix@initial_difference)
    def loss(total):
        d = difference(total)
        error = (dz@total)[..., None]/2+(dz@d)[..., None]*np.array([.5, -.5])-dt
        return float(np.sum(weights*error**2)/n), matrix@total+cross@d-rhs.sum(axis=0)
    def difference_optimality(total):
        gradient = matrix@difference(total)+cross@total-rhs[0]+rhs[1]
        return float(np.linalg.norm(gradient, ord=np.inf)/max(1., np.max(np.abs(rhs))))
    hessian = matrix-cross@inverse@cross
    return loss, lambda total: (hessian+hessian.T)/2, difference, difference_optimality


def fit_paths(rows, data, draws, source, cached_fit):
    scale, anchor = source["scale"], data["originals"]
    residual, _, reliability, shrinkage = base.noise_targets(rows, data, draws, scale)
    target = residual*reliability
    precision, noise = previous.previous.pair_precision(rows, draws, scale)
    eligible = data["raw_logits"] <= 0
    precision, balance = previous.balanced_precision(precision, eligible)
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)
    _, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    numerical = int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum())
    radii = {f["level"]: f["radius"] for f in cached_fit["fits"] if f["objective"] == "pair_gls"}
    models = {a: copy.deepcopy(source) for a in NEW_ARMS}
    ledger, maximum = [], 0.
    for level in strength.lowrank.LEVELS[1:]:
        k = min(int(level), numerical)
        basis = vt[:k].T*np.sqrt(mask.sum())/s[:k]
        z = x@basis
        old = np.asarray(source["weights"][level])
        heads = (vt[:k]@old.T).T*s[:k]/np.sqrt(mask.sum())
        initial, fixed_difference = heads.sum(axis=0), heads[0]-heads[1]
        maximum = max(maximum, float(np.max(np.abs((x@(basis@heads.T)-x@old.T)*scale))))
        projected = z[mask].T@target.sum(axis=-1)[mask]/mask.sum()
        training_radius = max(float(radii[level]), float(np.linalg.norm(projected)))
        frozen = previous.previous.pair_loss(z, target, precision, fixed_difference, eligible)
        free_loss, free_hessian, free_difference, free_optimality = profiled_pair_loss(z, target, precision, eligible, fixed_difference)
        for arm in NEW_ARMS:
            free = arm.startswith("free")
            radius = radii[level] if arm.endswith("old") else training_radius
            loss, hessian = (free_loss, free_hessian) if free else frozen
            total, fit = optimizer.quadratic_fit(loss, hessian, initial, radius=radius)
            difference = free_difference(total) if free else fixed_difference
            coefficients = np.stack(((total+difference)/2, (total-difference)/2))
            models[arm]["weights"][level] = (basis@coefficients.T).T.tolist()
            ledger.append({"arm": arm, "level": level, "retained_rank": k,
                "total_cost_function_df": k, "component_function_df": (2 if free else 1)*k,
                "old_radius": float(radii[level]), "projected_training_radius": float(np.linalg.norm(projected)),
                "training_radius": training_radius, "difference_change_rms": float(np.linalg.norm(difference-fixed_difference)),
                "difference_optimality": free_optimality(total) if free else 0.,
                "head_rms": np.linalg.norm(coefficients, axis=1).tolist(), **fit})
    return models, {"fits": ledger, "subspace_max_abs_difference": maximum,
        "pair_noise": noise, "shrinkage": shrinkage, "balance": balance}


def predict(paths, selection):
    return {"raw": paths["frozen_old"]["zero"], **{f"{arm}_{f}": paths[arm][selection[arm][f] or "zero"]
        for arm in ("old", *ARMS) for f in FILTERS}}


def qualify_fit(fit):
    expected = {(a, l) for a in NEW_ARMS for l in strength.lowrank.LEVELS[1:]}
    if len(fit["fits"]) != len(expected) or {(f["arm"], f["level"]) for f in fit["fits"]} != expected:
        raise ValueError("Dropped factorial arm or rank")
    if fit["subspace_max_abs_difference"] > 1e-12 or fit["balance"]["mass_max_abs_difference"] > 1e-12:
        raise ValueError("Changed subspace or state-balanced precision")
    for f in fit["fits"]:
        expected_radius = f["old_radius"] if f["arm"].endswith("old") else f["training_radius"]
        if (not f["success"] or max(f["optimality"], f["difference_optimality"]) > 1e-8 or f["norm"] > f["radius"]+1e-10
                or f["radius"] != expected_radius or f["training_radius"] < f["old_radius"]
                or f["loss_after"] > f["loss_before"]+1e-10*max(1., f["loss_before"])):
            raise ValueError("Factorial optimization or budget failed")
