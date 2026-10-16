"""Cached future-target and GLS-weight contrasts at fixed model/KKT geometry."""

from itertools import combinations

import numpy as np

from freq_hrl.domains.transit import native_kkt_response as source

PARTS = ("design_prior", "target_mean", "target_contrast")
CROSSES = tuple(combinations(PARTS, 2))
NAMES = ("anchor", "other", *PARTS, "target_swap", "precision_swap", "interaction", "joint_swap")
FIELDS = (*NAMES, "cross_anchor_other", *(f"cross_{a}_{b}" for a, b in CROSSES))


def residual_target(draws, data, scale):
    anchor = data["originals"]
    values = draws[..., (2, 1)]
    paired = values-values[np.arange(len(values)), :, anchor][:, :, None]
    return paired.mean(axis=1)/scale-source.source.source.source.joint.relative(data["components"], anchor)/scale


def alignment(a, b):
    aa, bb, ab = float(np.sum(a*a)), float(np.sum(b*b)), float(np.sum(a*b))
    return {"anchor_squared": aa, "other_squared": bb, "dot": ab,
        "cosine": ab/np.sqrt(aa*bb) if aa > 0 and bb > 0 else None}


def forcing(z, target, precision, eligible, difference, penalty, model, indices, full):
    losses = source.source.source.objectives
    lf, _, _, _ = losses(z, target, precision, eligible, difference, penalty)[model]
    lp, hp, _, _ = losses(z[indices], target[indices], precision[indices], eligible[indices], difference, penalty)[model]
    return lp(full)[1]-lf(full)[1], lf, lp, hp(full)


def context(z, calibrations, residuals, eligible, prior_heads, full_heads, prefix_heads, indices, model, radius, bank, level):
    c, other_c = calibrations[bank], calibrations[1-bank]
    full, prefix = full_heads.sum(axis=0), prefix_heads.sum(axis=0)
    difference, penalty = prior_heads[0]-prior_heads[1], c["penalties"][level]
    targets = [residuals[b]*np.asarray(c["reliability"]) for b in (bank, 1-bank)]
    precisions = [np.asarray(v["precision"]) for v in (c, other_c)]
    cases = {(t, w): forcing(z, target, precision, eligible, difference, penalty, model, indices, full)
        for t, target in enumerate(targets) for w, precision in enumerate(precisions)}
    q, lf, lp, h = cases[0, 0]
    mf, mp = source.source.multiplier(lf, full, radius), source.source.multiplier(lp, prefix, radius)
    cached = source.split_response(h, full, prefix, q, mf, mp)
    values, vectors = np.linalg.eigh((h+h.T)/2+mp*np.eye(len(full)))
    keep = values > len(values)*np.finfo(float).eps*max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    inverse = (vectors[:, keep]/values[keep])@vectors[:, keep].T
    q0 = forcing(z, np.zeros_like(targets[0]), precisions[0], eligible, difference, penalty, model, indices, full)[0]
    qa, qb, qw, qbw = (cases[k][0] for k in ((0, 0), (1, 0), (0, 1), (1, 1)))
    forces = {"anchor": qa, "other": qb, "design_prior": q0,
        "target_mean": (qa+qb)/2-q0, "target_contrast": (qa-qb)/2,
        "target_swap": qb-qa, "precision_swap": qw-qa,
        "interaction": qbw-qb-qw+qa, "joint_swap": qbw-qa}
    responses = {name: -inverse@value for name, value in forces.items()}
    response_error = float(np.max(np.abs(responses["anchor"]-cached["parts"]["data"])))
    split_error = max(float(np.max(np.abs(responses["anchor"]-sum(responses[n] for n in PARTS)))),
        float(np.max(np.abs(responses["joint_swap"]-sum(responses[n] for n in ("target_swap", "precision_swap", "interaction"))))))
    if response_error > 1e-12 or split_error > 1e-10*max(1., np.linalg.norm(responses["anchor"])):
        raise ValueError("Cached forcing identities failed")
    return {"responses": responses, "delta": cached["delta"], "low_basis": cached["low_basis"],
        "diagnostics": {"response_reproduction_error": response_error, "split_error": split_error,
            "influence_equation_residual": cached["diagnostics"]["influence_equation_residual"],
            "force_alignment": alignment(qa, qb), "target_force_alignment": alignment(qa-q0, qb-q0),
            "full_dual": mf, "prefix_dual": mp}}


def squares(predictions):
    out = {name: float(np.sum(predictions[name]**2)) for name in NAMES}
    out["cross_anchor_other"] = float(2*np.sum(predictions["anchor"]*predictions["other"]))
    out.update({f"cross_{a}_{b}": float(2*np.sum(predictions[a]*predictions[b])) for a, b in CROSSES})
    error = abs(out["anchor"]-sum(out[n] for n in PARTS)-sum(out[f"cross_{a}_{b}"] for a, b in CROSSES))
    if error > 1e-10*max(1., sum(out[n] for n in PARTS)): raise ValueError("Forcing prediction decomposition failed")
    return out, error


def region_stats(z, mask, feasible, context, physical):
    x = z[mask & feasible]*physical
    low = (x@context["low_basis"])@context["low_basis"].T
    bands, maximum = {}, 0.
    for name, design in (("full", x), ("low", low), ("rest", x-low)):
        bands[name], error = squares({n: design@v for n, v in context["responses"].items()})
        maximum = max(maximum, error)
    return {"feasible_action_cases": len(x), "bands": bands, "decomposition_residual": maximum}
