"""Cost objectives in the same cached TSVD subspace and residual RMS budget."""

import copy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from freq_hrl.domains.transit import native_refit_matrix as previous

joint, strength = previous.joint, previous.strength
OBJECTIVES = ("all_mse", "feasible_mse", "rank_regret")
VARIANTS = ("raw", *(f"{o}_{a}_{m}" for o in OBJECTIVES for a in joint.ANCHORS for m in previous.diagnostic.FIELDS))
MAX_ITERATIONS, FTOL = 200, 1e-10


def losses(z, offset, truth, eligible, anchor):
    """Equal-state MSE and regret-gap-weighted logistic ranking losses."""
    mask = np.arange(6)[None, :] != anchor[:, None]
    feasible = mask & eligible
    weights = feasible / np.maximum(feasible.sum(axis=1, keepdims=True), 1)
    lo, hi = np.triu_indices(6, 1)
    gap = truth[:, hi]-truth[:, lo]
    sign = np.sign(gap)
    pair_weights = (eligible[:, lo] & eligible[:, hi]) * np.abs(gap)
    pair_weights /= np.maximum(((eligible[:, lo] & eligible[:, hi]) & (gap != 0)).sum(axis=1, keepdims=True), 1)
    dz = (z[:, hi]-z[:, lo])*sign[..., None]
    delta = (offset[:, hi]-offset[:, lo])*sign
    n = len(z)
    def mse(theta):
        error = offset+z@theta-truth
        return float((weights*error**2).sum()/n), 2*np.einsum("na,nak->k", weights*error, z)/n
    def rank(theta):
        score = delta+dz@theta
        return float((pair_weights*np.logaddexp(0., -score)).sum()/n), -np.einsum("na,nak->k", pair_weights*expit(-score), dz)/n
    return {"feasible_mse": mse, "rank_regret": rank}


def constrained_fit(loss, initial):
    radius = float(np.linalg.norm(initial))
    start = loss(initial)[0]
    if radius == 0:
        return initial.copy(), {"iterations": 0, "success": True, "loss_before": start, "loss_after": start, "radius": 0., "norm": 0.}
    normalization = max(start, 1e-12)
    def objective(unit):
        value, gradient = loss(radius*unit)
        return value/normalization, gradient*radius/normalization
    result = minimize(objective, initial/radius, jac=True, method="SLSQP",
        constraints={"type": "ineq", "fun": lambda u: 1-u@u, "jac": lambda u: -2*u},
        options={"maxiter": MAX_ITERATIONS, "ftol": FTOL})
    unit = result.x/max(1., np.linalg.norm(result.x))
    theta, end = radius*unit, loss(radius*unit)[0]
    if not result.success or end > start+1e-8*max(1., start):
        raise ValueError(f"Cost objective solve failed: {result.message}; loss {start} -> {end}")
    return theta, {"iterations": int(result.nit), "success": bool(result.success), "loss_before": start,
        "loss_after": end, "radius": radius, "norm": float(np.linalg.norm(theta))}


def fit_objectives(data, values, cached, levels):
    anchor = joint.anchors(data, cached["anchor"])
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)
    scale = cached["scales"][0]
    target = joint.relative(values, anchor)[..., 0]/scale
    offset = joint.relative(data["heads"], anchor)[..., 0]/scale
    u, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    numerical = int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum())
    models = {o: copy.deepcopy(cached) for o in OBJECTIVES}
    ledger, maximum = [], 0.
    for level in sorted(levels, key=int):
        k = min(int(level), numerical)
        expected = vt[:k].T@((u[:, :k].T@(target-offset)[mask])/s[:k])
        old = np.asarray(cached["weights"][level][0])
        difference = float(np.max(np.abs(x@(expected-old)*scale)))
        maximum = max(maximum, difference)
        if difference > 1e-12: raise ValueError("Cached all-action MSE subspace reconstruction changed")
        # Whiten only within the frozen subspace: ||theta|| equals fit-pair residual RMS.
        basis = vt[:k].T*(np.sqrt(mask.sum())/s[:k])
        z = x@basis
        initial = (vt[:k]@old)*s[:k]/np.sqrt(mask.sum())
        objectives = losses(z, offset, target, data["raw_logits"] <= 0, anchor)
        for name, loss in objectives.items():
            theta, fit = constrained_fit(loss, initial)
            models[name]["weights"][level][0] = (basis@theta).tolist()
            ledger.append({"objective": name, "level": level, "requested_rank": int(level), "retained_rank": k,
                "parameter_count": k, "residual_rms_limit_physical": float(np.linalg.norm(initial)*scale),
                "residual_rms_physical": float(np.linalg.norm(theta)*scale), **fit})
    return models, ledger, maximum


def predict(data, models, selection):
    output = {"raw": {"chosen": data["raw_choices"], "cost": data["heads"][..., 0]}}
    for objective in OBJECTIVES:
        for anchor in joint.ANCHORS:
            for mode, field in previous.diagnostic.FIELDS.items():
                key = selection[anchor][field]
                if key in (None, "zero"):
                    cost = joint.relative(data["heads"], joint.anchors(data, anchor))[..., 0]
                    choice = data["raw_choices"]
                else:
                    c = strength.CANDIDATES[key]
                    cost = strength.base.costs(data, models[anchor][objective], c["level"], c["alpha"])
                    choice = joint.binding.policy.selected(cost, data["raw_logits"], joint.anchors(data, anchor))
                output[f"{objective}_{anchor}_{mode}"] = {"chosen": choice, "cost": cost}
    return output


def metrics(data, values, predictions):
    result = previous.metrics(data, data, values, predictions)
    mask = (np.arange(6)[None, :] != data["originals"][:, None]) & (data["raw_logits"] <= 0)
    ix = np.arange(len(values))
    for name, p in predictions.items():
        error = (p["cost"]-p["cost"][ix, data["originals"]][:, None]-joint.relative(values, data["originals"])[..., 0])**2
        weighted = (error*mask).sum(axis=1)/np.maximum(mask.sum(axis=1), 1)
        result["all"]["variants"][name]["feasible_cost_mse"] = float(weighted.mean())
        for root, stats in result["by_root"].items():
            stats["variants"][name]["feasible_cost_mse"] = float(weighted[data["roots"] == int(root)].mean())
    return result


def compare_cell(train, held, panels, reference, full_paths, source, cached):
    data = {n: joint.causal_data(rs, reference) for n, rs in (("train", train), ("held", held))}
    full = source["trials"][0]
    a = tuple(full["training_draws"])
    draws = {tuple(t["training_draws"]) for t in source["trials"]}
    b = tuple(i for i in range(8) if i not in a)
    draws.add(b)
    values = {(n, ds): joint.physical_values(rs, panels, ds) for n, rs in (("train", train), ("held", held)) for ds in draws}
    trials, cache = [], {}
    maximum, metric_max, changed, choices = 0., 0., 0, 0
    for index, old in enumerate(source["trials"]):
        group = tuple(old["removed_group"]) if old["removed_group"] else None
        indices = np.array([i for i, r in enumerate(train) if (r["root"], r["scene_seed"]) != group])
        fit_data = previous.sliced(data["train"], indices)
        ds = tuple(old["training_draws"])
        paths = full_paths if index == 0 else cached["paths"][str(index)]["fixed"]
        models, fits = {}, {}
        for kind in joint.ANCHORS:
            selected = [source["selection"][kind][f] for f in previous.diagnostic.FIELDS.values()]
            levels = {strength.CANDIDATES[k]["level"] for k in selected if k not in (None, "zero")}
            models[kind], fits[kind], difference = fit_objectives(fit_data, values["train", ds][indices], paths[kind], levels)
            maximum = max(maximum, difference)
        scopes = {"fit_training": (fit_data, values["train", ds][indices])}
        scopes.update({scope: (data[n], values[n, a if scope.endswith("same") else b])
            for scope, n in (("train_same", "train"), ("train_other", "train"), ("held_same", "held"), ("held_other", "held"))})
        outcome = {}
        for scope, (d, v) in scopes.items():
            prediction = predict(d, models, source["selection"])
            control = previous.predict(d, paths, source["selection"])
            for name, p in control.items():
                new = prediction["raw" if name == "raw" else name.replace("cost_strength_", "all_mse_")]
                changed += int((new["chosen"] != p["chosen"]).sum())
                choices += len(v)
            outcome[scope] = metrics(d, v, prediction)
            old_stats = old["arms"]["fixed"]["scopes"][scope]
            for old_piece, piece in [(old_stats["all"], outcome[scope]["all"]), *((old_stats["by_root"][r], p) for r, p in outcome[scope]["by_root"].items())]:
                for name, xs in old_piece["variants"].items():
                    current = piece["variants"]["raw" if name == "raw" else name.replace("cost_strength_", "all_mse_")]
                    metric_max = max(metric_max, *(abs(current[f]-v) for f, v in xs.items()))
        trials.append({k: old[k] for k in ("removed_group", "half", "training_states", "training_draws")}|{"fits": fits, "scopes": outcome})
        cache[str(index)] = models
        print(f"COST_OBJECTIVES_DESIGN_COMPLETE {index}", flush=True)
    if changed or metric_max > 1e-12: raise ValueError("Stage237 all-MSE controls changed")
    return {"selection": source["selection"], "trials": trials, "blocks": source["blocks"], "reproduction": {
        "choice_comparisons": choices, "changed_choices": changed, "metric_max_abs_difference": metric_max,
        "subspace_prediction_max_abs_difference": maximum}}, cache


def averaged(pieces):
    return {"matched_blocks": len(pieces), "mean_cases": float(np.mean([p["cases"] for p in pieces])),
        "variants": {name: {f: float(np.mean([p["variants"][name][f] for p in pieces])) for f in pieces[0]["variants"][name]} for name in VARIANTS}}
