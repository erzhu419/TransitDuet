"""Fixed-model OOF selectors with matched and expanded validation futures."""

import copy
from functools import partial

import numpy as np

from freq_hrl.domains.transit import native_head_shrinkage as previous
from freq_hrl.domains.transit.native_training_information import draw_assignment

MODELS = ("free_old", "shrunk")
SELECTORS = ("old_A2", "matched2", "expanded8")
ARMS = tuple(f"{m}_{s}" for m in MODELS for s in SELECTORS)
FILTERS = previous.FILTERS
VARIANTS = ("raw", *(f"{a}_{f}" for a in ("old", "frozen_old", *ARMS) for f in FILTERS))
base, joint, strength, optimizer = previous.base, previous.joint, previous.strength, previous.optimizer
select = partial(previous.pair.previous.select, objectives=ARMS)
aggregate = partial(previous.pair.previous.aggregate, variants=VARIANTS)


def validation_draws(bank, half, selector):
    if selector not in SELECTORS[1:]: raise ValueError("New validation assignment requires matched2 or expanded8")
    draws, _ = draw_assignment(bank, "expanded", 1-half)
    return draws[:2] if selector == "matched2" else draws


def summary_difference(actual, expected):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys(): raise ValueError("OOF summary fields changed")
        return max(summary_difference(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        if len(actual) != len(expected): raise ValueError("OOF summary halves changed")
        return max(summary_difference(a, b) for a, b in zip(actual, expected))
    return abs(actual-expected)


def rebuild_inner(rows, data, draws, source, cached_fit):
    """Replay Stage251 inner fits, retaining the already-required free solution."""
    scale, anchor = source["scale"], data["originals"]
    residual, _, reliability, _ = base.noise_targets(rows, data, draws, scale)
    target = residual*reliability
    precision, _ = previous.pair.pair_precision(rows, draws, scale)
    eligible = data["raw_logits"] <= 0
    precision, balance = previous.balanced.balanced_precision(precision, eligible)
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)
    _, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    numerical = int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum())
    radii = {f["level"]: f["radius"] for f in cached_fit["fits"] if f["objective"] == "pair_gls"}
    models, ledger, maximum = {a: copy.deepcopy(source) for a in MODELS}, [], 0.
    for level in strength.lowrank.LEVELS[1:]:
        k = min(int(level), numerical)
        basis = vt[:k].T*np.sqrt(mask.sum())/s[:k]
        z = x@basis
        old = np.asarray(source["weights"][level])
        heads = (vt[:k]@old.T).T*s[:k]/np.sqrt(mask.sum())
        initial, prior = heads.sum(axis=0), heads[0]-heads[1]
        maximum = max(maximum, float(np.max(np.abs((x@(basis@heads.T)-x@old.T)*scale))))
        free_loss, free_hessian, free_difference, _ = previous.previous.profiled_pair_loss(z, target, precision, eligible, prior)
        free_total, free_fit = optimizer.quadratic_fit(free_loss, free_hessian, initial, radius=radii[level])
        free_d = free_difference(free_total)
        free_coefficients = np.stack(((free_total+free_d)/2, (free_total-free_d)/2))
        models["free_old"]["weights"][level] = (basis@free_coefficients.T).T.tolist()
        free_delta = free_d-prior
        noise = previous.difference_noise(z, draws, precision, eligible, reliability, scale)
        signal, penalty = previous.penalty_rule(float(free_delta@free_delta), noise)
        if penalty is None:
            loss, hessian = previous.pair.pair_loss(z, target, precision, prior, eligible)
            difference, optimality = lambda total: prior, lambda total: 0.
        else:
            loss, hessian, difference, optimality = previous.profiled_shrink_loss(z, target, precision, eligible, prior, penalty)
        total, fit = optimizer.quadratic_fit(loss, hessian, initial, radius=radii[level])
        d = difference(total)
        coefficients = np.stack(((total+d)/2, (total-d)/2))
        models["shrunk"]["weights"][level] = (basis@coefficients.T).T.tolist()
        data_loss, _ = previous.pair.pair_loss(z, target, precision, d, eligible)
        ledger.append({"level": level, "retained_rank": k, "old_radius": float(radii[level]),
            "penalty": penalty, "signal_energy": signal, **noise,
            "free_difference_change_rms": float(np.linalg.norm(free_delta)),
            "difference_change_rms": float(np.linalg.norm(d-prior)), "difference_optimality": optimality(total),
            "data_loss_after": data_loss(total)[0], "free_data_loss_after": free_loss(free_total)[0],
            "auxiliary_free_fit": free_fit, **fit})
    fit = {"fits": ledger, "subspace_max_abs_difference": maximum, "balance": balance}
    previous.qualify_fit(fit)
    return models, fit


def predict(paths, selection):
    return {"raw": paths["free_old"]["zero"],
        **{f"{a}_{f}": paths[a][selection[a][f] or "zero"] for a in ("old", "frozen_old") for f in FILTERS},
        **{f"{m}_{s}_{f}": paths[m][selection[f"{m}_{s}"][f] or "zero"]
            for m in MODELS for s in SELECTORS for f in FILTERS}}
