"""Prefix deletion under frozen full-training GLS and shrinkage calibration."""

import copy

import numpy as np

from freq_hrl.domains.transit import native_deployment_bundle as source

shrink = source.source.previous
PAIRS = {"fit_population": ("fixed", "full"), "recalibration": ("refit", "fixed"),
    "total": ("refit", "full")}


def full_calibration(rows, data, draws, prior, full_fit):
    residual, _, reliability, _ = source.base.noise_targets(rows, data, draws, prior["scale"])
    precision, _ = shrink.pair.pair_precision(rows, draws, prior["scale"])
    precision, balance = shrink.balanced.balanced_precision(precision, data["raw_logits"] <= 0)
    return {"target": residual*reliability, "precision": precision, "balance": balance,
        "reliability": reliability, "penalties": {f["level"]: f["penalty"] for f in full_fit["fits"]}}


def objectives(z, target, precision, eligible, difference, penalty):
    free = shrink.previous.profiled_pair_loss(z, target, precision, eligible, difference)
    if penalty is None:
        loss, hessian = shrink.pair.pair_loss(z, target, precision, difference, eligible)
        regularized = loss, hessian, lambda total: difference, lambda total: 0.
    else:
        regularized = shrink.profiled_shrink_loss(z, target, precision, eligible, difference, penalty)
    return {"free_old": free, "shrunk": regularized}


def check_cached_full(data, calibration, prior, cap, geometry, models):
    """Evaluate cached coefficients' KKT conditions; do not solve/refit full60."""
    x = source.joint.relative(data["x"][..., :198], data["originals"])
    radii = {f["level"]: f["radius"] for f in cap["fits"] if f["objective"] == "pair_gls"}
    maximum, optimality, head_gap = 0., 0., 0.
    for level, (basis, coordinates) in geometry.items():
        initial_heads = np.asarray(prior["weights"][level])@coordinates.T
        initial, difference = initial_heads.sum(axis=0), initial_heads[0]-initial_heads[1]
        z = x@basis
        losses = objectives(z, calibration["target"], calibration["precision"], data["raw_logits"] <= 0,
            difference, calibration["penalties"][level])
        for model, (loss, _, profiled, _) in losses.items():
            coefficients = np.asarray(models[model]["weights"][level])
            heads = coefficients@coordinates.T
            total, cached_difference = heads.sum(axis=0), heads[0]-heads[1]
            if np.linalg.norm(total) > radii[level]+1e-10: raise ValueError("Cached full RMS cap changed")
            _, fit = source.source.optimizer.finish_fit(loss, initial, total, 0, radius=radii[level])
            optimality = max(optimality, fit["optimality"])
            maximum = max(maximum, float(np.max(np.abs((x@(basis@heads.T)-x@coefficients.T)*prior["scale"]))))
            head_gap = max(head_gap, float(np.max(np.abs(z@(profiled(total)-cached_difference)*prior["scale"]))))
    if maximum > 1e-12 or head_gap > 1e-10: raise ValueError("Full calibration does not reproduce cached full function")
    return {"full_KKT_checks": len(geometry)*2, "max_kkt": optimality,
        "subspace_max_abs_difference": maximum, "profiled_head_max_abs_difference": head_gap}


def fit_fixed(data, calibration, indices, prior, cap, geometry):
    """Only the fit population changes; sliced weights/targets and lambda stay fixed."""
    x = source.joint.relative(data["x"][indices, ..., :198], data["originals"][indices])
    target, precision = calibration["target"][indices], calibration["precision"][indices]
    eligible = data["raw_logits"][indices] <= 0
    radii = {f["level"]: f["radius"] for f in cap["fits"] if f["objective"] == "pair_gls"}
    models, fits, maximum = {m: copy.deepcopy(prior) for m in source.MODELS}, [], 0.
    for level, (basis, coordinates) in geometry.items():
        old = np.asarray(prior["weights"][level])
        heads = old@coordinates.T
        initial, difference = heads.sum(axis=0), heads[0]-heads[1]
        maximum = max(maximum, float(np.max(np.abs((x@(basis@heads.T)-x@old.T)*prior["scale"]))))
        losses = objectives(x@basis, target, precision, eligible, difference, calibration["penalties"][level])
        fitted = {}
        for model, (loss, hessian, profiled, optimality) in losses.items():
            total, fit = source.source.optimizer.quadratic_fit(loss, hessian, initial, radius=radii[level])
            d = profiled(total)
            models[model]["weights"][level] = (basis@np.stack(((total+d)/2, (total-d)/2), axis=-1)).T.tolist()
            fitted[model] = {**fit, "difference_optimality": optimality(total),
                "difference_change_rms": float(np.linalg.norm(d-difference))}
        fits.append({"level": level, "old_radius": float(radii[level]), "penalty": calibration["penalties"][level],
            **fitted["shrunk"], "free_difference_change_rms": fitted["free_old"]["difference_change_rms"],
            "auxiliary_free_fit": fitted["free_old"]})
    fit = {"fits": fits, "balance": calibration["balance"], "subspace_max_abs_difference": maximum}
    shrink.qualify_fit(fit)
    for f in fits:
        if f["auxiliary_free_fit"]["optimality"] > 1e-8: raise ValueError("Frozen-calibration free fit failed KKT")
    return models, fit


def action_transitions(full, fixed, refit):
    a, b, c = map(np.asarray, (full, fixed, refit))
    return {"unchanged": int(((a == b) & (b == c)).sum()),
        "fit_only": int(((a != b) & (b == c)).sum()),
        "calibration_only": int(((a == b) & (b != c)).sum()),
        "cancelled": int(((a != b) & (a == c)).sum()),
        "both_distinct": int(((a != b) & (b != c) & (a != c)).sum())}


def check_telescope(comparisons, fields):
    error = max(abs(comparisons["fit_population"][f]+comparisons["recalibration"][f]-comparisons["total"][f]) for f in fields)
    if error > 1e-12: raise ValueError("Calibration path does not telescope")
    return error
