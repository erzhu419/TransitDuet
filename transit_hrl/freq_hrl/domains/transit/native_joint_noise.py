"""Frozen-operator plug-in MC noise for paired service/risk future labels."""

import numpy as np

from freq_hrl.domains.transit import native_head_half_forcing as previous

POSITIVE = ("joint", "service", "risk", "excluded_joint", "excluded_service", "excluded_risk",
    "retained_joint", "retained_service", "retained_risk")
CROSSES = ("cross_sr", "excluded_cross_sr", "retained_cross_sr")
COVARIANCES = (*POSITIVE, *CROSSES)
OLD_NAMES = {name: f"cross16_{name}" for name in POSITIVE}
OLD_NAMES.update(excluded_joint="cross16_excluded_labels", retained_joint="cross16_retained_labels")


def state_scores(z, target_draws, precision, eligible, model, penalty):
    """Per-state/draw gradient contributions with the loss's original 1/n factor."""
    dz, weights, matrix, cross = previous.calibration.shrink.geometry(z, precision, eligible)
    k = z.shape[-1]
    if model == "shrunk" and penalty is None:
        coupling = np.zeros_like(matrix)
    else:
        lam = 0. if model == "free_old" else penalty
        coupling = cross@np.linalg.pinv(matrix+lam*np.eye(k), rcond=k*np.finfo(float).eps, hermitian=True)
    pair = previous.calibration.shrink.pair
    dt = target_draws[:, :, pair.HI]-target_draws[:, :, pair.LO]
    rhs = np.einsum("nph,ndph,npk->ndhk", weights, dt, dz)/len(z)
    transforms = np.stack((coupling-np.eye(k), -coupling-np.eye(k)))
    return np.einsum("ndhk,hjk->ndhj", rhs, transforms)


def noise_covariances(scores, retained):
    """Sum independent state contributions; preserve action/head covariance."""
    m = scores.shape[1]//2
    centered = np.concatenate([q-q.mean(axis=1, keepdims=True) for q in (scores[:, :m], scores[:, m:])], axis=1)
    output = {}
    for population, mask in (("", np.ones(len(scores), bool)), ("excluded_", ~retained), ("retained_", retained)):
        s, r = centered[mask, :, 0], centered[mask, :, 1]
        gram = lambda a, b: np.einsum("ndk,ndl->kl", a, b)/(m*(m-1))
        sr = gram(s, r)
        output.update({population+"service": gram(s, s), population+"risk": gram(r, r),
            population+"cross_sr": sr+sr.T, population+"joint": gram(s+r, s+r)})
    return output


def covariance_error(covariances):
    errors = []
    for population in ("", "excluded_", "retained_"):
        errors.append(np.max(np.abs(covariances[population+"joint"]-covariances[population+"service"]
            -covariances[population+"risk"]-covariances[population+"cross_sr"])))
    for head in ("joint", "service", "risk", "cross_sr"):
        errors.append(np.max(np.abs(covariances[head]-covariances["excluded_"+head]-covariances["retained_"+head])))
    return float(max(errors)/max(1., *(np.linalg.norm(c, ord=np.inf) for c in covariances.values())))


def context(z, c, paired, eligible, prior_heads, full_heads, prefix_heads, indices, model, radius, bank, level, scale):
    full, prefix = full_heads.sum(axis=0), prefix_heads.sum(axis=0)
    precision, rho = np.asarray(c["precision"]), np.asarray(c["reliability"])
    penalty, difference = c["penalties"][level], prior_heads[0]-prior_heads[1]
    q, lf, lp, h = previous.previous.forcing(z, np.asarray(c["target"]), precision, eligible,
        difference, penalty, model, indices, full)
    mf, mp = previous.kkt.source.multiplier(lf, full, radius), previous.kkt.source.multiplier(lp, prefix, radius)
    cached = previous.kkt.split_response(h, full, prefix, q, mf, mp)
    values, vectors = np.linalg.eigh((h+h.T)/2+mp*np.eye(len(full)))
    keep = values > len(values)*np.finfo(float).eps*max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    inverse = (vectors[:, keep]/values[keep])@vectors[:, keep].T
    targets = np.concatenate([paired[b] for b in (0, 1)], axis=1)*rho[:, None]/scale
    scores = -state_scores(z, targets, precision, eligible, model, penalty)
    scores[indices] += state_scores(z[indices], targets[indices], precision[indices], eligible[indices], model, penalty)
    retained = np.zeros(len(z), bool);retained[indices] = True
    force_covariance = noise_covariances(scores, retained)
    parameter_covariance = {n: inverse@c@inverse.T for n, c in force_covariance.items()}
    m = paired[0].shape[1]
    gap = (scores[:, m:].mean(axis=1)-scores[:, :m].mean(axis=1))*(1 if bank == 0 else -1)
    responses = {}
    for population, mask in (("", np.ones(len(z), bool)), ("excluded_", ~retained), ("retained_", retained)):
        for head, index in (("service", 0), ("risk", 1)):
            responses[population+head] = -inverse@gap[mask, index].sum(axis=0)
        responses[population+"joint"] = responses[population+"service"]+responses[population+"risk"]
    change = (paired[1-bank].mean(axis=1)-paired[bank].mean(axis=1))*rho/scale
    qb = previous.previous.forcing(z, np.asarray(c["target"])+change, precision, eligible,
        difference, penalty, model, indices, full)[0]
    reference = -inverse@(qb-q)
    reproduction = float(np.max(np.abs(reference-responses["joint"])))
    covariance_residual = max(covariance_error(force_covariance), covariance_error(parameter_covariance))
    cq, ct = force_covariance["joint"], parameter_covariance["joint"]
    force_diagonal = np.einsum("ki,kl,li->i", vectors, cq, vectors)
    theta_diagonal = np.einsum("ki,kl,li->i", vectors, ct, vectors)
    predicted_diagonal = np.zeros_like(values);predicted_diagonal[keep] = force_diagonal[keep]/values[keep]**2
    spectral_residual = float(np.max(np.abs(theta_diagonal-predicted_diagonal))/max(1., np.max(np.abs(theta_diagonal))))
    if reproduction > 1e-10*max(1., np.linalg.norm(reference)) or covariance_residual > 1e-10 or spectral_residual > 1e-8:
        raise ValueError("Frozen joint-noise propagation failed its identities")
    low = cached["low_basis"]
    force_trace, theta_trace = max(0., float(np.trace(cq))), max(0., float(np.trace(ct)))
    eigenvalues = np.linalg.eigvalsh((ct+ct.T)/2)
    return {"responses": responses, "delta": cached["delta"], "low_basis": low,
        "force_covariance": force_covariance, "parameter_covariance": parameter_covariance,
        "diagnostics": {"reference_max_abs_difference": reproduction, "covariance_identity_residual": covariance_residual,
            "spectral_propagation_residual": spectral_residual, "influence_equation_residual": cached["diagnostics"]["influence_equation_residual"],
            "force_noise_energy": force_trace, "parameter_noise_energy": theta_trace,
            "low_parameter_noise_energy": max(0., float(np.trace(low.T@ct@low))), "low_dimension_fraction": low.shape[1]/len(full),
            "unsupported_force_noise_energy": max(0., force_trace-float(force_diagonal[keep].sum())),
            "dominant_covariance_fraction": max(0., float(eigenvalues[-1]))/theta_trace if theta_trace else None,
            "dominant_KKT_mode_fraction": max(0., float(theta_diagonal.max()))/theta_trace if theta_trace else None,
            "directional_profile": {"KKT_eigenvalues": values.tolist(), "supported": keep.tolist(),
                "force_noise_diagonal": force_diagonal.tolist(), "parameter_noise_diagonal": theta_diagonal.tolist()}}}


def region_stats(z, mask, feasible, context, physical):
    x = z[mask & feasible]*physical
    low = (x@context["low_basis"])@context["low_basis"].T
    bands, maximum = {}, 0.
    for band, design in (("full", x), ("low", low), ("rest", x-low)):
        gram = design.T@design
        expected = {n: float(np.einsum("ij,ji->", gram, c)) for n, c in context["parameter_covariance"].items()}
        predictions = {n: design@v for n, v in context["responses"].items()}
        observed = {n: float(v@v) for n, v in predictions.items()}
        observed.update({p+"cross_sr": float(2*predictions[p+"service"]@predictions[p+"risk"]) for p in ("", "excluded_", "retained_")})
        observed["cross_population_joint"] = float(2*predictions["excluded_joint"]@predictions["retained_joint"])
        for p in ("", "excluded_", "retained_"):
            maximum = max(maximum, abs(expected[p+"joint"]-expected[p+"service"]-expected[p+"risk"]-expected[p+"cross_sr"]),
                abs(observed[p+"joint"]-observed[p+"service"]-observed[p+"risk"]-observed[p+"cross_sr"]))
        maximum = max(maximum, abs(expected["joint"]-expected["excluded_joint"]-expected["retained_joint"]))
        for n in POSITIVE: expected[n] = max(0., expected[n])
        bands[band] = {"noise_squared_sum": expected, "observed_squared_sum": observed}
    if maximum > 1e-10*max(1., *(abs(v) for b in bands.values() for v in b["noise_squared_sum"].values())):
        raise ValueError("Joint-noise query covariance decomposition failed")
    return {"feasible_action_cases": len(x), "bands": bands,
        "low_rest_noise_cross": {n: bands["full"]["noise_squared_sum"][n]-bands["low"]["noise_squared_sum"][n]-bands["rest"]["noise_squared_sum"][n] for n in POSITIVE},
        "decomposition_residual": maximum}
