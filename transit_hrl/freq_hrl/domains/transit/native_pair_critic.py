"""Feasible paired-margin regressions with common-future covariance retained."""

import copy
from functools import partial

import numpy as np

from freq_hrl.domains.transit import native_feasible_critic as previous

base, joint, strength, optimizer = previous.base, previous.joint, previous.strength, previous.optimizer
NEW_OBJECTIVES = ("pair_mse", "pair_gls")
OBJECTIVES = (*previous.OBJECTIVES, *NEW_OBJECTIVES)
FILTERS = previous.FILTERS
VARIANTS = ("raw", *(f"{o}_{f}" for o in OBJECTIVES for f in FILTERS))
LO, HI = np.triu_indices(6, 1)
candidates, predict = previous.candidates, previous.predict
select = partial(previous.select, objectives=OBJECTIVES)
aggregate = partial(previous.aggregate, variants=VARIANTS)


def pair_precision(rows, draws, scale):
    values = draws[..., (2, 1)]/scale
    paired = values[:, :, HI]-values[:, :, LO]
    variance = paired.var(axis=1, ddof=1)/draws.shape[1]
    keys = sorted({(r["root"], r["scene_seed"]) for r in rows})
    membership = np.array([keys.index((r["root"], r["scene_seed"])) for r in rows])
    group_variance = np.stack([variance[membership == i].mean(axis=(0, 1)) for i in range(len(keys))])
    pooled = group_variance.mean(axis=0)
    prior = (group_variance+pooled)/2
    smoothed = (variance+prior[membership, None])/2
    precision = np.ones_like(smoothed)
    observed = pooled > 0
    precision[..., observed] = 1/smoothed[..., observed]
    precision /= precision.mean(axis=(0, 1))
    return precision, {"pooled_variance_of_mean": pooled.tolist(),
        "zero_observed_variance_heads": [h for i, h in enumerate(base.HEADS) if not observed[i]],
        "groups": [{"group": list(k), "variance_of_mean": group_variance[i].tolist()} for i, k in enumerate(keys)]}


def pair_loss(z, target, precision, difference, eligible):
    dz, dt = z[:, HI]-z[:, LO], target[:, HI]-target[:, LO]
    mask = eligible[:, LO] & eligible[:, HI]
    weights = (mask/np.maximum(mask.sum(axis=1, keepdims=True), 1))[..., None]*precision
    fixed = (dz@difference)[..., None]*np.array([.5, -.5])
    n = len(z)
    def loss(theta):
        error = (dz@theta)[..., None]/2+fixed-dt
        return float(np.sum(weights*error**2)/n), np.einsum("nph,npk->k", weights*error, dz)/n
    matrix = .5*np.einsum("np,npk,npl->kl", weights.sum(axis=-1), dz, dz)/n
    return loss, lambda theta: matrix


def fit_paths(rows, data, draws, cached, cached_fit):
    models = copy.deepcopy(cached)
    source = cached["all_mse"]
    models.update({o: copy.deepcopy(source) for o in NEW_OBJECTIVES})
    scale, anchor = source["scale"], data["originals"]
    residual, _, reliability, _ = base.noise_targets(rows, data, draws, scale)
    target = residual*reliability
    precision, noise = pair_precision(rows, draws, scale)
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)
    _, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    numerical = int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum())
    radii = {f["level"]: f["radius"] for f in cached_fit["fits"] if f["objective"] == "feasible_mse"}
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
            weights = precision if objective == "pair_gls" else np.ones_like(precision)
            loss, hessian = pair_loss(z, target, weights, difference, data["raw_logits"] <= 0)
            theta, fit = optimizer.quadratic_fit(loss, hessian, initial, radius=radii[level])
            models[objective]["weights"][level] = (basis@np.stack(((theta+difference)/2, (theta-difference)/2), axis=-1)).T.tolist()
            ledger.append({"objective": objective, "level": level, "retained_rank": k,
                "total_cost_function_df": k, "new_parameters": k, **fit})
    return models, {"fits": ledger, "subspace_max_abs_difference": maximum, "pair_noise": noise}


def metrics(data, draws, predictions):
    result = previous.previous.metrics(data, draws, predictions)
    mean = draws.mean(axis=1)
    truth = mean[:, HI, 0]-mean[:, LO, 0]
    feasible = data["raw_logits"] <= 0
    mask = feasible[:, LO] & feasible[:, HI]
    weights = mask/np.maximum(mask.sum(axis=1, keepdims=True), 1)
    for name, p in predictions.items():
        margin = p["cost"][:, HI]-p["cost"][:, LO]
        error = (weights*(margin-truth)**2).sum(axis=1)
        signs = mask & (truth != 0)
        wrong = (signs*(np.sign(margin) != np.sign(truth))).sum(axis=1)/np.maximum(signs.sum(axis=1), 1)
        ix = np.arange(len(mean))
        harm = (p["chosen"] != data["raw_choices"]) & (mean[ix, p["chosen"], 0] > mean[ix, data["raw_choices"], 0])
        for piece, indices in [(result["all"], ix), *((v, np.flatnonzero(data["roots"] == int(r))) for r, v in result["by_root"].items())]:
            piece["variants"][name].update(feasible_margin_mse=float(error[indices].mean()),
                feasible_pair_sign_error=float(wrong[indices].mean()), changed_harm_rate=float(harm[indices].mean()))
    return result


def compare_cached(matrix, expected):
    maximum, records = 0., 0
    for scope, labels in expected.items():
        for label, stats in labels.items():
            actual = matrix[scope][label]
            for a, b in [(actual["all"], stats["all"]), *((actual["by_root"][r], v) for r, v in stats["by_root"].items())]:
                if a["comparisons"] != b["comparisons"] or a["mean_cases"] != b["mean_cases"]: raise ValueError("Dropped cached states")
                for name, fields in b["variants"].items():
                    for field, value in fields.items():
                        maximum = max(maximum, abs(a["variants"][name][field]-value))
                        records += 1
    if maximum > 1e-12: raise ValueError("Stage245 frozen objectives changed")
    return {"records": records, "max_abs_difference": maximum}
