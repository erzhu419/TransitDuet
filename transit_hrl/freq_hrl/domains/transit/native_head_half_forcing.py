"""Head/population attribution and cached future-half stability at fixed KKT."""

from itertools import combinations

import numpy as np

from freq_hrl.domains.transit import native_calibration_ablation as calibration
from freq_hrl.domains.transit import native_future_forcing as previous
from freq_hrl.domains.transit import native_kkt_response as kkt

HEADS = ("joint", "service", "risk")
CONTRASTS = ("cross16", "first8", "last8", "within_anchor", "within_other")
PARTS = ("excluded_service", "excluded_risk", "retained_service", "retained_risk")
CROSSES = tuple(combinations(PARTS, 2))
NAMES = (*(f"{c}_{h}" for c in CONTRASTS for h in HEADS),
    "cross16_excluded_labels", "cross16_retained_labels", *(f"cross16_{p}" for p in PARTS))
FIELDS = (*NAMES, *(f"cross16_cross_{a}_{b}" for a, b in CROSSES),
    *(f"{c}_cross_service_risk" for c in CONTRASTS), *(f"first_last_dot_{h}" for h in HEADS))


def label_gradient(z, precision, eligible, model, penalty):
    """Linear target part of the existing profiled gradient; no parameter fit."""
    dz, weights, matrix, cross = calibration.shrink.geometry(z, precision, eligible)
    k = z.shape[-1]
    if model == "shrunk" and penalty is None:
        coupling = np.zeros_like(matrix)
    else:
        lam = 0. if model == "free_old" else penalty
        inverse = np.linalg.pinv(matrix+lam*np.eye(k), rcond=k*np.finfo(float).eps, hermitian=True)
        coupling = cross@inverse
    def gradient(target):
        dt = target[:, calibration.shrink.pair.HI]-target[:, calibration.shrink.pair.LO]
        rhs = np.einsum("nph,nph,npk->hk", weights, dt, dz)/len(z)
        return coupling@(rhs[0]-rhs[1])-rhs.sum(axis=0)
    return gradient


def context(z, c, residuals, halves, eligible, prior_heads, full_heads, prefix_heads, indices, model, radius, bank, level):
    full, prefix = full_heads.sum(axis=0), prefix_heads.sum(axis=0)
    precision, rho = np.asarray(c["precision"]), np.asarray(c["reliability"])
    penalty, difference = c["penalties"][level], prior_heads[0]-prior_heads[1]
    q, lf, lp, h = previous.forcing(z, np.asarray(c["target"]), precision, eligible, difference, penalty, model, indices, full)
    mf, mp = kkt.source.multiplier(lf, full, radius), kkt.source.multiplier(lp, prefix, radius)
    cached = kkt.split_response(h, full, prefix, q, mf, mp)
    values, vectors = np.linalg.eigh((h+h.T)/2+mp*np.eye(len(full)))
    keep = values > len(values)*np.finfo(float).eps*max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    inverse = (vectors[:, keep]/values[keep])@vectors[:, keep].T
    gf = label_gradient(z, precision, eligible, model, penalty)
    gp = label_gradient(z[indices], precision[indices], eligible[indices], model, penalty)
    def response(target): return -inverse@(gp(target[indices])-gf(target))
    other = 1-bank
    changes = {"cross16": residuals[other]-residuals[bank],
        "first8": halves[other, 0]-halves[bank, 0], "last8": halves[other, 1]-halves[bank, 1],
        "within_anchor": halves[bank, 1]-halves[bank, 0],
        "within_other": halves[other, 1]-halves[other, 0]}
    responses = {}
    maximum = 0.
    for contrast, change in changes.items():
        target = change*rho
        for head, index in (("service", 0), ("risk", 1)):
            selected = np.zeros_like(target);selected[..., index] = target[..., index]
            responses[f"{contrast}_{head}"] = response(selected)
        joint = responses[f"{contrast}_service"]+responses[f"{contrast}_risk"]
        maximum = max(maximum, float(np.max(np.abs(joint-response(target)))))
        responses[f"{contrast}_joint"] = joint
    retained = np.zeros(len(z), bool);retained[indices] = True
    target = changes["cross16"]*rho
    for population, mask in (("excluded", ~retained), ("retained", retained)):
        for head, index in (("service", 0), ("risk", 1)):
            selected = np.zeros_like(target);selected[mask, :, index] = target[mask, :, index]
            responses[f"cross16_{population}_{head}"] = response(selected)
        responses[f"cross16_{population}_labels"] = sum(responses[f"cross16_{population}_{head}"] for head in ("service", "risk"))
    maximum = max(maximum, float(np.max(np.abs(responses["cross16_joint"]-sum(responses[f"cross16_{p}"] for p in PARTS)))))
    for head in HEADS:
        maximum = max(maximum, float(np.max(np.abs(responses[f"cross16_{head}"]-(responses[f"first8_{head}"]+responses[f"last8_{head}"])/2))),
            float(np.max(np.abs(responses[f"first8_{head}"]-responses[f"last8_{head}"]-responses[f"within_anchor_{head}"]+responses[f"within_other_{head}"]))))
    qb = previous.forcing(z, residuals[other]*rho, precision, eligible, difference, penalty, model, indices, full)[0]
    reference = -inverse@(qb-q)
    reproduction = float(np.max(np.abs(reference-responses["cross16_joint"])))
    if max(maximum, reproduction) > 1e-10*max(1., np.linalg.norm(reference)):
        raise ValueError("Head/half target-response identities failed")
    return {"responses": responses, "delta": cached["delta"], "low_basis": cached["low_basis"],
        "diagnostics": {"identity_max_residual": maximum, "reference_max_abs_difference": reproduction,
            "influence_equation_residual": cached["diagnostics"]["influence_equation_residual"]}}


def squares(predictions):
    out = {n: float(np.sum(predictions[n]**2)) for n in NAMES}
    out.update({f"cross16_cross_{a}_{b}": float(2*np.sum(predictions[f"cross16_{a}"]*predictions[f"cross16_{b}"])) for a, b in CROSSES})
    out.update({f"{c}_cross_service_risk": float(2*np.sum(predictions[f"{c}_service"]*predictions[f"{c}_risk"])) for c in CONTRASTS})
    out.update({f"first_last_dot_{h}": float(np.sum(predictions[f"first8_{h}"]*predictions[f"last8_{h}"])) for h in HEADS})
    errors = [abs(out["cross16_joint"]-sum(out[f"cross16_{p}"] for p in PARTS)-sum(out[f"cross16_cross_{a}_{b}"] for a, b in CROSSES))]
    for c in CONTRASTS:
        errors.append(abs(out[f"{c}_joint"]-out[f"{c}_service"]-out[f"{c}_risk"]-out[f"{c}_cross_service_risk"]))
    for h in HEADS:
        errors.append(abs(4*out[f"cross16_{h}"]-out[f"first8_{h}"]-out[f"last8_{h}"]-2*out[f"first_last_dot_{h}"]))
    error = max(errors)
    if error > 1e-10*max(1., *(out[n] for n in NAMES)): raise ValueError("Head/half squared decomposition failed")
    return out, error


def region_stats(z, mask, feasible, context, physical):
    x = z[mask & feasible]*physical
    low = (x@context["low_basis"])@context["low_basis"].T
    bands, maximum = {}, 0.
    for band, design in (("full", x), ("low", low), ("rest", x-low)):
        bands[band], error = squares({n: design@v for n, v in context["responses"].items()})
        maximum = max(maximum, error)
    return {"feasible_action_cases": len(x), "bands": bands, "decomposition_residual": maximum}


def raw_label_stats(paired, feasible, mask):
    values = paired[mask]
    x = values[np.broadcast_to(feasible[mask, None, :], values.shape[:-1])]
    if not len(x): return {"draw_state_action_exposures": 0, "service_mean": None, "risk_mean": None, "risk_nonzero": 0, "risk_positive": 0, "risk_negative": 0}
    risk = x[:, 1]
    return {"draw_state_action_exposures": len(x), "service_mean": float(x[:, 0].mean()), "risk_mean": float(risk.mean()),
        "risk_nonzero": int((np.abs(risk)>1e-12).sum()), "risk_positive": int((risk>1e-12).sum()), "risk_negative": int((risk < -1e-12).sum())}
