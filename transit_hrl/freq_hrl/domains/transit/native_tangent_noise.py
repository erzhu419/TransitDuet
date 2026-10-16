"""Local active-ball KKT response to the same cached gradient-force noise."""

import numpy as np

from freq_hrl.domains.transit import native_joint_noise as previous


def operators(hessian, total, dual, radius):
    a = (hessian+hessian.T)/2+dual*np.eye(len(total))
    values, vectors = np.linalg.eigh(a)
    keep = values > len(values)*np.finfo(float).eps*max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    inverse = (vectors[:, keep]/values[keep])@vectors[:, keep].T
    slack = 1-float(total@total)/radius**2 if radius > 0 else None
    tolerance = previous.previous.calibration.source.source.optimizer.GTOL
    if radius <= 0 or (dual == 0 and slack <= tolerance):
        raise ValueError("Cached case lacks a smooth interior/strictly-active RMS derivative")
    if dual > 0:
        if abs(slack) > tolerance: raise ValueError("Positive multiplier away from RMS boundary")
        bu = inverse@total
        denominator = float(total@bu)
        projection = np.eye(len(total))-np.outer(bu, total)/denominator
        tangent = inverse-np.outer(bu, bu)/denominator
        dual_response = -bu/denominator
        stationarity = np.linalg.norm(-a@tangent+np.outer(total, dual_response)+np.eye(len(total)), ord=np.inf)
        tangency = np.linalg.norm(total@tangent, ord=np.inf)/max(1., np.linalg.norm(total)*np.linalg.norm(tangent, ord=np.inf))
        state = "strictly_active"
    else:
        projection, tangent, dual_response = np.eye(len(total)), inverse.copy(), np.zeros(len(total))
        stationarity, tangency, state = 0., 0., "interior"
    if max(stationarity, tangency) > 1e-8: raise ValueError("Active RMS KKT tangent equation failed")
    return {"inverse": inverse, "tangent": tangent, "projection": projection, "dual_response": dual_response,
        "diagnostics": {"constraint_state": state, "prefix_dual": dual, "radius": radius,
            "squared_relative_slack": slack, "stationarity_residual": float(stationarity), "tangency_residual": float(tangency)}}


def context(z, c, paired, eligible, prior_heads, full_heads, prefix_heads, indices, model, radius, bank, level, scale):
    fixed = previous.context(z, c, paired, eligible, prior_heads, full_heads, prefix_heads, indices, model, radius, bank, level, scale)
    _, _, loss, h = previous.previous.previous.forcing(z, np.asarray(c["target"]), np.asarray(c["precision"]),
        eligible, prior_heads[0]-prior_heads[1], c["penalties"][level], model, indices, full_heads.sum(axis=0))
    total = prefix_heads.sum(axis=0)
    dual = previous.previous.kkt.source.multiplier(loss, total, radius)
    op = operators(h, total, dual, radius)
    p, q, t = op["projection"], np.eye(len(total))-op["projection"], op["tangent"]
    tangent = {**fixed, "responses": {n: p@v for n, v in fixed["responses"].items()},
        "parameter_covariance": {n: t@c@t.T for n, c in fixed["force_covariance"].items()}}
    radial = {**fixed, "responses": {n: q@v for n, v in fixed["responses"].items()},
        "parameter_covariance": {n: q@c@q.T for n, c in fixed["parameter_covariance"].items()}}
    cross = {n: p@c@q.T+q@c@p.T for n, c in fixed["parameter_covariance"].items()}
    errors = [previous.covariance_error(tangent["parameter_covariance"]), previous.covariance_error(radial["parameter_covariance"])]
    for n, ctheta in fixed["parameter_covariance"].items():
        difference = ctheta-tangent["parameter_covariance"][n]-radial["parameter_covariance"][n]-cross[n]
        errors.append(np.linalg.norm(difference, ord=np.inf)/max(1., np.linalg.norm(ctheta, ord=np.inf)))
    covariance_error = float(max(errors))
    if covariance_error > 1e-8: raise ValueError("Tangent/radial joint covariance identity failed")
    energy = lambda c: max(0., float(np.trace(c)))
    diagnostics = {**op["diagnostics"], "covariance_residual": covariance_error,
        "fixed_parameter_noise_energy": energy(fixed["parameter_covariance"]["joint"]),
        "tangent_parameter_noise_energy": energy(tangent["parameter_covariance"]["joint"]),
        "radial_parameter_noise_energy": energy(radial["parameter_covariance"]["joint"]),
        "parameter_noise_cross": float(np.trace(cross["joint"])),
        "dual_noise_variance": max(0., float(op["dual_response"]@fixed["force_covariance"]["joint"]@op["dual_response"]))}
    return {"fixed": fixed, "tangent": tangent, "radial": radial, "cross_covariance": cross, "diagnostics": diagnostics}


def region_stats(z, mask, feasible, contexts, physical):
    out = {n: previous.region_stats(z, mask, feasible, contexts[n], physical) for n in ("fixed", "tangent", "radial")}
    x = z[mask & feasible]*physical
    low = (x@contexts["fixed"]["low_basis"])@contexts["fixed"]["low_basis"].T
    crosses, maximum = {}, 0.
    for band, design in (("full", x), ("low", low), ("rest", x-low)):
        gram = design.T@design
        crosses[band] = {n: float(np.einsum("ij,ji->", gram, c)) for n, c in contexts["cross_covariance"].items()}
        for n in previous.POSITIVE:
            b, t, r = (out[k]["bands"][band]["noise_squared_sum"][n] for k in ("fixed", "tangent", "radial"))
            maximum = max(maximum, abs(b-t-r-crosses[band][n])/max(1., abs(b), abs(t), abs(r)))
    if maximum > 1e-8: raise ValueError("Query tangent/radial covariance identity failed")
    return {"operators": out, "cross": crosses, "decomposition_residual": maximum}
