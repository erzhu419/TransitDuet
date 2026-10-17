"""Matched state/future budgets with different independent-prefix coverage."""

import copy

import numpy as np

from freq_hrl.domains.transit import native_balanced_critic as base

PROBES = (8, 9, 22, 23, 36, 37)
VARIANTS = ("raw", "stage247", "dense", "wide")
FIELDS = ("cost", "fleet", "service", "wait_min")


def assignment(rows, arm, rotation):
    if arm == "dense":
        return np.array([i for i, r in enumerate(rows) if r["replica"] == rotation])
    if arm == "wide":
        return np.array([i for i, r in enumerate(rows)
            if r["probe_index"] == PROBES[(r["replica"]+rotation) % 6]])
    raise ValueError(arm)


def geometry(rows, data, model, selection, radius):
    result = {"model": model, "selection": selection}
    if selection is None:
        return result
    level = base.strength.CANDIDATES[selection]["level"]
    anchor = data["originals"]
    x = base.joint.relative(data["x"][..., :198], anchor)
    mask = np.arange(6)[None] != anchor[:, None]
    _, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    k = min(int(level), int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum()))
    basis = vt[:k].T*np.sqrt(mask.sum())/s[:k]
    old = np.asarray(model["weights"][level])
    heads = (vt[:k]@old.T).T*s[:k]/np.sqrt(mask.sum())
    error = float(np.max(abs((x@(basis@heads.T)-x@old.T)*model["scale"])))
    predicted = base.base.candidates(data, model)[selection]
    if error > 1e-12 or not np.allclose(predicted["cost"], predicted["components"].sum(axis=-1), rtol=0, atol=1e-12):
        raise ValueError("Frozen geometry or scored head path does not reconstruct Stage247")
    if np.linalg.norm(heads.sum(axis=0)) > radius+1e-10:
        raise ValueError("Cached Stage247 is outside its original RMS budget")
    result.update(level=level, basis=basis.tolist(), initial=heads.sum(axis=0).tolist(),
        difference=(heads[0]-heads[1]).tolist(), radius=float(radius), reconstruction_error=error)
    return result


def fit(rows, data, draws, frozen):
    model = copy.deepcopy(frozen["model"])
    if frozen["selection"] is None:
        return model, {"active": False}
    scale = model["scale"]
    residual, _, reliability, _ = base.base.noise_targets(rows, data, draws, scale)
    precision, noise = base.previous.pair_precision(rows, draws, scale)
    precision, balance = base.balanced_precision(precision, data["raw_logits"] <= 0)
    x = base.joint.relative(data["x"][..., :198], data["originals"])
    basis, difference = np.asarray(frozen["basis"]), np.asarray(frozen["difference"])
    loss, hessian = base.previous.pair_loss(x@basis, residual*reliability, precision,
        difference, data["raw_logits"] <= 0)
    theta, ledger = base.optimizer.quadratic_fit(loss, hessian, np.asarray(frozen["initial"]), radius=frozen["radius"])
    model["weights"][frozen["level"]] = (basis@np.stack(((theta+difference)/2, (theta-difference)/2), axis=-1)).T.tolist()
    return model, {"active": True, "parameters": len(theta), "balance": balance, "noise": noise, **ledger}


def predict(data, model, selection):
    return base.base.candidates(data, model)[selection or "zero"]


def margin_errors(data, draws, prediction):
    """Per-state paired MSE and its independent label-mean variance."""
    lo, hi = base.previous.LO, base.previous.HI
    eligible = data["raw_logits"] <= 0
    mask = eligible[:, lo] & eligible[:, hi]
    weights = mask/np.maximum(mask.sum(axis=1, keepdims=True), 1)
    truth = draws[..., (0, 1, 2)]
    margins = truth[:, :, hi]-truth[:, :, lo]
    predicted = np.stack((prediction["cost"], prediction["components"][..., 1], prediction["components"][..., 0]), axis=-1)
    difference = predicted[:, hi]-predicted[:, lo]-margins.mean(axis=1)
    mse = np.einsum("np,npf->nf", weights, difference**2)
    noise = np.einsum("np,npf->nf", weights, margins.var(axis=1, ddof=1)/draws.shape[1])
    return mse, noise


def paired_controls(choices, draws):
    """Collapse fixed profiles/rotations before retaining shared future noise."""
    frequencies = np.stack([(choices == a).mean(axis=(0, 1)).T for a in range(6)], axis=-1)
    return np.einsum("nva,ndaf->ndvf", frequencies-frequencies[:, :1], draws)


def paired_prefix_summary(records, *, fields=FIELDS, bootstrap=5000):
    means = np.asarray([r["mean"] for r in records])
    rng = np.random.default_rng(288)
    samples = np.zeros((bootstrap, *means.shape[1:]))
    strata = sorted({(r["root"], r["scenario"]) for r in records})
    for group in strata:
        ix = np.flatnonzero([(r["root"], r["scenario"]) == group for r in records])
        if len(ix) != 4:
            raise ValueError("Expected four fresh held prefixes per root/regime")
        samples += means[rng.choice(ix, (bootstrap, len(ix)), replace=True)].mean(axis=1)/len(strata)
    pairs = [(v, "raw") for v in VARIANTS[1:]]+[("wide", "dense"), ("wide", "stage247")]
    output = {}
    for after, before in pairs:
        a, b = VARIANTS.index(after), VARIANTS.index(before)
        bounds = np.quantile(samples[:, a]-samples[:, b], [.025, .975], axis=0)
        output[after+"_vs_"+before] = {"mean": dict(zip(fields, map(float, (means[:, a]-means[:, b]).mean(axis=0)))),
            "prefix_CI95": {f: list(map(float, bounds[:, j])) for j, f in enumerate(fields)}}
    return output
