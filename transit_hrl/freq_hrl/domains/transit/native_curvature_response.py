"""Conditional curvature and block leverage of cached prefix-deletion responses."""

import numpy as np

from freq_hrl.domains.transit import native_calibration_ablation as source


def multiplier(loss, total, radius):
    gradient = loss(total)[1]
    norm2 = float(total@total)
    active = radius > 0 and norm2/radius**2 >= 1-source.source.source.optimizer.GTOL
    return max(0., float(-total@gradient/norm2)) if active else 0.


def low_subspace(hessian):
    values, vectors = np.linalg.eigh((hessian+hessian.T)/2)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if values.min() < -1e-8*scale: raise ValueError("Profiled curvature is materially negative")
    tolerance = len(values)*np.finfo(float).eps*scale
    positive = values > tolerance
    n = int(positive.sum())
    cut = values[positive][int(np.ceil(n/4))-1] if n else 0.
    low = values <= cut+tolerance
    return vectors[:, low], {"dimension": len(values), "positive_modes": n,
        "null_modes": int((~positive).sum()), "low_modes": int(low.sum()),
        "low_dimension_fraction": float(low.mean()),
        "minimum_positive_to_maximum": float(values[positive].min()/values.max()) if n else None,
        "low_cut_to_maximum": float(cut/values.max()) if n else None}


def factor(z, precision, eligible, penalty, dual):
    """SVD avoids inversion of a squared-condition Gram for block leverage."""
    dz, weights, _, _ = source.shrink.geometry(z, precision, eligible)
    n, _, k = dz.shape
    if penalty is None:
        rows = dz*np.sqrt(weights.sum(axis=-1)/(2*n))[..., None]
    else:
        heads = np.stack((np.concatenate((dz,dz),axis=-1),np.concatenate((dz,-dz),axis=-1)),axis=2)
        rows = (heads*np.sqrt(weights/(2*n))[..., None]).reshape(n,-1,2*k)
    data = rows.reshape(-1,rows.shape[-1])
    extras = []
    if penalty is not None and penalty > 0: extras.append(np.c_[np.zeros((k,k)),np.sqrt(penalty)*np.eye(k)])
    if dual > 0:
        extras.append(np.sqrt(dual)*np.eye(k) if penalty is None else np.c_[np.sqrt(dual)*np.eye(k),np.zeros((k,k))])
    design = np.vstack((data,*extras)) if extras else data
    u, s, vt = np.linalg.svd(design,full_matrices=False)
    keep = s**2 > design.shape[1]*np.finfo(float).eps*(s[0]**2 if len(s) else 0.)
    return {"inverse_basis": vt[keep].T/s[keep], "supported": vt[keep].T,
        "data_u": u[:len(data),keep].reshape(n,rows.shape[1],int(keep.sum())),
        "dimension": design.shape[1], "rank": int(keep.sum()),
        "condition": float((s[0]/s[keep][-1])**2) if keep.any() else None}


def leverage(full_factor, removed):
    block = full_factor["data_u"][removed].reshape(len(removed)*full_factor["data_u"].shape[1],full_factor["rank"])
    singular = np.linalg.svd(block,compute_uv=False) if block.size else np.array([])
    trace, total = float(np.sum(block**2)),float(np.sum(full_factor["data_u"]**2))
    return {"maximum": float(singular[0]**2) if len(singular) else 0., "trace": trace,
        "data_trace_fraction": trace/total if total else None}


def full_context(z, calibration, eligible, prior_heads, heads, model, radius):
    loss, _, _, _ = source.objectives(z,calibration["target"],calibration["precision"],eligible,
        prior_heads[0]-prior_heads[1],calibration["penalty"])[model]
    total = heads.sum(axis=0)
    dual = multiplier(loss,total,radius)
    penalty = 0. if model == "free_old" else calibration["penalty"]
    return {"loss":loss,"total":total,"dual":dual,
        "factor":factor(z,calibration["precision"],eligible,penalty,dual),"penalty":penalty}


def prefix_context(z, calibration, eligible, prior_heads, heads, indices, removed, model, radius, full):
    loss, hessian, _, _ = source.objectives(z[indices],calibration["target"][indices],calibration["precision"][indices],eligible[indices],
        prior_heads[0]-prior_heads[1],calibration["penalty"])[model]
    total = heads.sum(axis=0)
    dual = multiplier(loss,total,radius)
    h = hessian(total)
    delta = total-full["total"]
    force = loss(full["total"])[1]-full["loss"](full["total"])[1]
    residual = (h+dual*np.eye(len(total)))@delta+force+(dual-full["dual"])*full["total"]
    error = float(np.linalg.norm(residual,ord=np.inf)/max(1.,np.linalg.norm(force,ord=np.inf)))
    if error > 1e-8: raise ValueError("Cached prefix response fails its quadratic KKT influence equation")
    basis, spectrum = low_subspace(h)
    low = basis@(basis.T@delta)
    energy = float(delta@delta)
    prefix_factor = factor(z[indices],calibration["precision"][indices],eligible[indices],full["penalty"],dual)
    return {"delta":delta,"low":low,"full_factor":full["factor"],
        "prefix_factor":prefix_factor,
        "diagnostics":{**spectrum,"parameter_energy":energy,"low_parameter_energy":float(low@low),
            "low_parameter_fraction":float(low@low)/energy if energy else None,
            "full_dual":full["dual"],"prefix_dual":dual,
            "full_effective_condition":full["factor"]["condition"],
            "prefix_effective_condition":prefix_factor["condition"],
            "full_effective_rank":full["factor"]["rank"],"prefix_effective_rank":prefix_factor["rank"],
            "block_leverage":leverage(full["factor"],removed),"influence_equation_residual":error}}


def region_stats(z, mask, feasible, context, physical):
    x = z[mask & feasible]
    total, low = (x@context[name]*physical for name in ("delta","low"))
    rest = total-low
    sums = {"total":float(total@total),"low":float(low@low),"rest":float(rest@rest),"cross":float(2*low@rest)}
    error = abs(sums["low"]+sums["rest"]+sums["cross"]-sums["total"])
    if error > 1e-12*max(1.,sums["total"]): raise ValueError("Spectral prediction decomposition does not add")
    exposures = {}
    for name in ("full_factor","prefix_factor"):
        f = context[name]
        query = x if f["dimension"] == x.shape[1] else np.c_[x,np.zeros_like(x)]
        inverse = query@f["inverse_basis"]
        unsupported = query-(query@f["supported"])@f["supported"].T
        exposures[name] = {"inverse_curvature_sum":float(np.sum(inverse**2))*physical**2,
            "unsupported_squared_sum":float(np.sum(unsupported**2)),"design_squared_sum":float(np.sum(query**2))}
    return {"feasible_action_cases":len(x),"squared_response":sums,"exposure":exposures,"decomposition_residual":error}
