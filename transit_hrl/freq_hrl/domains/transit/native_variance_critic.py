"""Training-only paired variance and prefix shrinkage for service/risk residuals."""

import numpy as np
from scipy.optimize import brentq

from freq_hrl.domains.transit import native_objective_oof as previous

joint, strength = previous.joint, previous.strength
ARMS = ("combined", "split", "variance", "group", "variance_group")
FILTERS = previous.FILTERS
VARIANTS = ("raw", *(f"{a}_{f}" for a in ARMS for f in FILTERS))
HEADS = ("service", "risk")


def causal_data(rows, reference, *, batch=None):
    batch = joint.binding.learned.neural.batches(rows) if batch is None else batch
    data = joint.causal_data(rows, reference, batch=batch)
    ridge = reference["ridge"]
    heads = (joint.binding.learned.centered_cost(batch[0], rows, ridge)
        @ np.asarray(ridge["coefficients"]))*ridge["target_scale"]
    data["components"] = heads[..., (1, 0)]
    return data


def training_draws(rows, panels, indices):
    return np.stack([panels[joint.binding.learned.state_key(r)]["values"][list(indices)] for r in rows])


def noise_targets(rows, data, draws, scale):
    """Paired MC variance; leave-group-out signal energy sets raw-centred shrinkage."""
    anchor = data["originals"]
    mask = np.arange(6)[None, :] != anchor[:, None]
    values = draws[..., (2, 1)]
    paired = values-values[np.arange(len(rows)), :, anchor][:, :, None]
    truth = paired.mean(axis=1)/scale
    residual = truth-joint.relative(data["components"], anchor)/scale
    variance = paired.var(axis=1, ddof=1)/(draws.shape[1]*scale**2)
    keys = sorted({(r["root"], r["scene_seed"]) for r in rows})
    membership = np.array([keys.index((r["root"], r["scene_seed"])) for r in rows])
    group_variance = np.stack([variance[(membership == i)[:, None] & mask].mean(axis=0) for i in range(len(keys))])
    energy = np.stack([(residual[(membership == i)[:, None] & mask]**2).mean(axis=0) for i in range(len(keys))])
    pooled = group_variance.mean(axis=0)
    # One group's degrees of freedom is also the prior strength: no tuning grid.
    group_prior = (group_variance+pooled)/2
    smoothed = (variance+group_prior[membership, None])/2
    precision = np.ones_like(smoothed)
    observed = pooled > 0
    precision[..., observed] = 1/smoothed[..., observed]
    precision /= precision[mask].mean(axis=0)
    signal = np.maximum(((energy-group_variance).sum(axis=0)-(energy-group_variance))/(len(keys)-1), 0.)
    reliability = np.divide(signal, signal+group_prior, out=np.zeros_like(signal), where=(signal+group_prior) > 0)
    groups = [{"group": list(k), "states": int((membership == i).sum()),
        "variance_of_mean": group_variance[i].tolist(), "prior_signal_energy": signal[i].tolist(),
        "reliability": reliability[i].tolist()} for i, k in enumerate(keys)]
    return residual, precision, reliability[membership, None], {"groups": groups,
        "draws": draws.shape[1], "pooled_variance_of_mean": pooled.tolist(),
        "zero_observed_variance_heads": [h for h, ok in zip(HEADS, observed) if not ok],
        "precision_min": precision[mask].min(axis=0).tolist(), "precision_max": precision[mask].max(axis=0).tolist()}


def solve_components(z, target, weights, initial, radius):
    """Convex two-head GLS subject to the cached total-residual RMS ball."""
    n, k = z.shape
    matrices = [z.T@(weights[:, h, None]*z)/n for h in range(2)]
    rhs = [z.T@(weights[:, h]*target[:, h])/n for h in range(2)]
    unrestricted = np.stack([np.linalg.solve(m, b) for m, b in zip(matrices, rhs)])
    multiplier, iterations = 0., 0
    if radius == 0:
        first = np.linalg.solve(matrices[0]+matrices[1], rhs[0]-rhs[1])
        theta = np.stack((first, -first))
        optimality = np.max(np.abs(matrices[0]@theta[0]-rhs[0]-(matrices[1]@theta[1]-rhs[1])))
    else:
        inverse = [np.linalg.solve(m, np.eye(k)) for m in matrices]
        eigenvalues, vectors = np.linalg.eigh(inverse[0]+inverse[1])
        beta = vectors.T@unrestricted.sum(axis=0)
        def total(lam): return vectors@(beta/(1+lam*eigenvalues))
        if np.linalg.norm(total(0)) > radius:
            high = max(float(np.linalg.norm(beta)/radius/eigenvalues.min()), 1.)
            multiplier, solved = brentq(lambda lam: np.linalg.norm(total(lam))-radius, 0., high,
                xtol=1e-14, full_output=True)
            iterations = int(solved.iterations)
        summed = total(multiplier)
        theta = np.stack([u-multiplier*inv@summed for u, inv in zip(unrestricted, inverse)])
        # Evaluate the actual summed solution, not the algebraic dual shortcut.
        optimality = max(np.max(np.abs(m@t-b+multiplier*theta.sum(axis=0))) for m, t, b in zip(matrices, theta, rhs))
    def loss(t): return float(np.mean(np.sum(weights*(z@t.T-target)**2, axis=1)))
    norm = float(np.linalg.norm(theta.sum(axis=0)))
    normalized_kkt = float(optimality/max(1., *(np.linalg.norm(b, ord=np.inf) for b in rhs)))
    if normalized_kkt > 1e-8 or norm > radius+1e-10 or loss(theta) > loss(initial)+1e-10*max(1., loss(initial)):
        raise ValueError("Component GLS failed optimality, total RMS or objective check")
    return theta, {"optimality": normalized_kkt, "radius": float(radius), "norm": norm,
        "dual_multiplier": float(multiplier), "iterations": iterations,
        "loss_before": loss(initial), "loss_after": loss(theta)}


def fit_paths(rows, data, draws, cached):
    if cached["anchor"] != "original": raise ValueError("Stage243 uses the fixed original anchor")
    if not np.allclose(draws[..., 0], draws[..., 1]+draws[..., 2], rtol=0, atol=1e-12):
        raise ValueError("Cached service/risk do not sum to physical total cost")
    anchor = data["originals"]
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)[mask]
    scale = cached["scales"][0]
    residual, precision, reliability, noise = noise_targets(rows, data, draws, scale)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    numerical = int((s > s[0]*max(x.shape)*np.finfo(float).eps).sum())
    models = {a: {"scale": scale, "weights": {"zero": np.zeros((2, 198)).tolist()}} for a in ARMS}
    for arm in ("combined", "split"): models[arm]["scalar_path"] = cached
    ledger, maximum = [], 0.
    for level in strength.lowrank.LEVELS[1:]:
        k = min(int(level), numerical)
        basis = vt[:k].T*np.sqrt(len(x))/s[:k]
        z = x@basis
        initial = u[:, :k].T@residual[mask]/np.sqrt(len(x))
        old = np.asarray(cached["weights"][level][0])
        difference = float(np.max(np.abs(x@(basis@initial.sum(axis=1)-old)*scale)))
        maximum = max(maximum, difference)
        if difference > 1e-12: raise ValueError("Split OLS changed cached combined-MSE cost")
        radius = np.linalg.norm((vt[:k]@old)*s[:k]/np.sqrt(len(x)))
        for arm in ARMS:
            target = residual*reliability if arm in ("group", "variance_group") else residual
            weight = precision if arm in ("variance", "variance_group") else np.ones_like(precision)
            theta, fit = solve_components(z, target[mask], weight[mask], initial.T, radius)
            coefficients = (basis@theta.T).T
            if arm == "combined":
                # Retain the exact frozen scalar predictor, with a split OLS diagnostic decomposition.
                coefficients[0] = old-coefficients[1]
            models[arm]["weights"][level] = coefficients.tolist()
            ledger.append({"arm": arm, "level": level, "retained_rank": k,
                "total_cost_function_df": k, "component_function_df": 2*k,
                "residual_rms_physical": fit["norm"]*scale, "residual_rms_limit_physical": radius*scale, **fit})
    return models, {"noise": noise, "fits": ledger, "combined_max_abs_difference": maximum}


def candidates(data, model):
    output = {}
    anchor = data["originals"]
    base = joint.relative(data["components"], anchor)
    x = joint.relative(data["x"][..., :198], anchor)
    for key, p in strength.CANDIDATES.items():
        parts = base+p["alpha"]*(x@np.asarray(model["weights"][p["level"]]).T)*model["scale"]
        cost = parts.sum(axis=-1)
        if "scalar_path" in model: cost = strength.base.costs(data, model["scalar_path"], **p)
        if key == "zero": cost = joint.relative(data["heads"], anchor)[..., 0]
        chosen = data["raw_choices"] if key == "zero" else joint.binding.policy.selected(cost, data["raw_logits"], anchor)
        output[key] = {"chosen": chosen, "cost": cost, "components": parts}
    return output


def curve(data, draws, prediction):
    v = draws.mean(axis=1)
    ix = np.arange(len(v))
    delta = v[ix, prediction["chosen"]]-v[ix, data["raw_choices"]]
    unsafe = (v[ix, prediction["chosen"], 3] > 0).astype(int)-(v[ix, data["raw_choices"], 3] > 0).astype(int)
    return {"changed_count": int((prediction["chosen"] != data["raw_choices"]).sum()),
        "by_root": {str(r): {"cases": int((data["roots"] == r).sum()),
            "cost_sum": float(delta[data["roots"] == r, 0].sum()), "wait_sum_min": float(delta[data["roots"] == r, 3].sum()),
            "unsafe_delta_count": int(unsafe[data["roots"] == r].sum())} for r in sorted(set(data["roots"]))}}


def select(blocks):
    roots = sorted({r for b in blocks for r in b["curves"]["combined"]["zero"]["by_root"]})
    scales = {r: np.mean([b["scales"] for b in blocks if str(b["held_group"][0]) == r], axis=0) for r in roots}
    summaries, selections = {}, {}
    for arm in ARMS:
        summaries[arm] = {c: {"pooled": previous.curve_summary([b["curves"][arm][c] for b in blocks], scales),
            "halves": [previous.curve_summary([b["curves"][arm][c] for b in blocks if b["half"] == h], scales) for h in (0, 1)]}
            for c in strength.CANDIDATES}
        selections[arm] = {}
        for mode in FILTERS:
            eligible = [c for c in strength.CANDIDATES if c != "zero" and summaries[arm][c]["pooled"]["score"] < -previous.TOL
                and previous.nonharm(summaries[arm][c]["pooled"])
                and (mode != "half_safe" or all(previous.nonharm(h) for h in summaries[arm][c]["halves"]))]
            selections[arm][mode] = min(eligible, key=lambda c: summaries[arm][c]["pooled"]["score"]) if eligible else None
    return selections, summaries


def predict(data, models, selection):
    paths = {a: candidates(data, models[a]) for a in ARMS}
    return {"raw": paths["combined"]["zero"],
        **{f"{a}_{f}": paths[a][selection[a][f] or "zero"] for a in ARMS for f in FILTERS}}


def metrics(data, draws, predictions):
    mean = draws.mean(axis=1)
    ix, anchor = np.arange(len(draws)), data["originals"]
    mask = np.arange(6)[None, :] != anchor[:, None]
    paired = mean-mean[ix, anchor][:, None]
    result = {}
    for group, indices in [("all", ix), *((str(r), np.flatnonzero(data["roots"] == r)) for r in sorted(set(data["roots"])) )]:
        pieces = {}
        for name, p in predictions.items():
            actual, raw = mean[ix, p["chosen"]], mean[ix, data["raw_choices"]]
            delta = actual-raw
            unsafe = (actual[:, 3] > 0).astype(int)-(raw[:, 3] > 0).astype(int)
            error = (p["cost"]-paired[..., 0])**2
            head_error = (p["components"]-paired[..., (2, 1)])**2
            pieces[name] = {"delta_cost": float(delta[indices, 0].mean()), "delta_wait_min": float(delta[indices, 3].mean()),
                "delta_service": float(delta[indices, 2].mean()), "delta_risk": float(delta[indices, 1].mean()),
                "delta_unsafe": float(unsafe[indices].mean()), "changed_rate": float((p["chosen"] != data["raw_choices"])[indices].mean()),
                "cost_mse": float(error[indices][mask[indices]].mean()),
                **{f"{h}_mse": float(head_error[indices][mask[indices]][:, j].mean()) for j, h in enumerate(HEADS)}}
        result[group] = {"cases": len(indices), "variants": pieces}
    return {"all": result.pop("all"), "by_root": result}


def average(pieces):
    return {"comparisons": len(pieces), "mean_cases": float(np.mean([p["cases"] for p in pieces])),
        "variants": {v: {f: float(np.mean([p["variants"][v][f] for p in pieces])) for f in pieces[0]["variants"][v]} for v in VARIANTS}}


def aggregate(pieces):
    roots = sorted({r for p in pieces for r in p["by_root"]})
    return {"all": average([p["all"] for p in pieces]),
        "by_root": {r: average([p["by_root"][r] for p in pieces if r in p["by_root"]]) for r in roots}}
