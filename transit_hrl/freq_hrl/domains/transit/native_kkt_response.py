"""Attribute cached quadratic responses to data and RMS multiplier changes."""

from itertools import combinations

import numpy as np

from freq_hrl.domains.transit import native_curvature_response as source

PARTS = ("data", "dual", "null", "residual")
CROSSES = tuple(combinations(PARTS, 2))
FIELDS = ("total", *PARTS, *(f"cross_{a}_{b}" for a, b in CROSSES))


def squared_parts(total, parts):
    sums = {"total": float(np.sum(total**2)),
        **{name: float(np.sum(parts[name]**2)) for name in PARTS},
        **{f"cross_{a}_{b}": float(2*np.sum(parts[a]*parts[b])) for a, b in CROSSES}}
    error = abs(sum(v for k, v in sums.items() if k != "total")-sums["total"])
    if error > 1e-10*max(1., sum(sums[name] for name in PARTS)):
        raise ValueError("KKT response squared decomposition does not add")
    return sums, error


def split_response(hessian, full, prefix, force, full_dual, prefix_dual):
    h = (hessian+hessian.T)/2
    a = h+prefix_dual*np.eye(len(full))
    values, vectors = np.linalg.eigh(a)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    tolerance = len(values)*np.finfo(float).eps*scale
    if values.min() < -1e-8*scale: raise ValueError("Negative KKT curvature")
    keep = values > tolerance
    supported = vectors[:, keep]
    inverse = (supported/values[keep])@supported.T
    delta = prefix-full
    dual_force = (prefix_dual-full_dual)*full
    stationarity = a@delta+force+dual_force
    error = float(np.linalg.norm(stationarity, ord=np.inf)/max(1., np.linalg.norm(force, ord=np.inf)))
    if error > 1e-8: raise ValueError("Cached KKT response equation failed")
    parts = {"data": -inverse@force, "dual": -inverse@dual_force,
        "null": delta-supported@(supported.T@delta)}
    parts["residual"] = delta-sum(parts.values())
    low, spectrum = source.low_subspace(h)
    squared, decomposition_error = squared_parts(delta, parts)
    return {"delta": delta, "parts": parts, "low_basis": low,
        "diagnostics": {**spectrum, "full_dual": full_dual, "prefix_dual": prefix_dual,
            "dual_change": prefix_dual-full_dual, "supported_rank": int(keep.sum()),
            "data_force_norm": float(np.linalg.norm(force)), "dual_force_norm": float(np.linalg.norm(dual_force)),
            "unsupported_force_norm": float(np.linalg.norm(force+dual_force-supported@(supported.T@(force+dual_force)))),
            "influence_equation_residual": error, "parameter_squared_response": squared,
            "decomposition_residual": decomposition_error}}


def full_context(z, calibration, eligible, prior_heads, heads, model, radius):
    loss, _, _, _ = source.source.objectives(z, calibration["target"], calibration["precision"], eligible,
        prior_heads[0]-prior_heads[1], calibration["penalty"])[model]
    total = heads.sum(axis=0)
    return {"loss": loss, "total": total, "dual": source.multiplier(loss, total, radius)}


def prefix_context(z, calibration, eligible, prior_heads, heads, indices, model, radius, full):
    loss, hessian, _, _ = source.source.objectives(z[indices], calibration["target"][indices],
        calibration["precision"][indices], eligible[indices], prior_heads[0]-prior_heads[1], calibration["penalty"])[model]
    total = heads.sum(axis=0)
    force = loss(full["total"])[1]-full["loss"](full["total"])[1]
    return split_response(hessian(total), full["total"], total, force,
        full["dual"], source.multiplier(loss, total, radius))


def region_stats(z, mask, feasible, context, physical):
    x = z[mask & feasible]*physical
    total = x@context["delta"]
    parts = {name: x@value for name, value in context["parts"].items()}
    squared, error = squared_parts(total, parts)
    low_x = (x@context["low_basis"])@context["low_basis"].T
    bands = {}
    for name, design in (("low", low_x), ("rest", x-low_x)):
        bands[name], band_error = squared_parts(design@context["delta"],
            {name: design@value for name, value in context["parts"].items()})
        error = max(error, band_error)
    return {"feasible_action_cases": len(x), "squared_response": squared,
        "bands": bands, "decomposition_residual": error}
