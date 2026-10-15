"""Matched target/calibration future interventions under frozen raw references."""

import copy

import numpy as np

from freq_hrl.domains.transit import native_variance_critic as base

MODES = {"same": ("A", "A"), "calibration_only": ("A", "B"),
    "target_only": ("B", "A"), "disjoint": ("B", "B")}
VARIANTS = ("raw", *(f"{m}_{a}_{f}" for m in MODES for a in base.ARMS for f in base.FILTERS))
joint, strength = base.joint, base.strength


def fit_paths(rows, data, same, disjoint, cached_models):
    if same.shape != disjoint.shape: raise ValueError("Target/calibration draw counts must match")
    for draws in (same, disjoint):
        if not np.allclose(draws[..., 0], draws[..., 1]+draws[..., 2], rtol=0, atol=1e-12):
            raise ValueError("Service/risk do not sum to total cost")
    cached = cached_models["combined"]["scalar_path"]
    scale = cached["scales"][0]
    noise = {label: base.noise_targets(rows, data, draws, scale) for label, draws in (("A", same), ("B", disjoint))}
    anchor = data["originals"]
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)[mask]
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    numerical = int((s > s[0]*max(x.shape)*np.finfo(float).eps).sum())
    models = {"same": cached_models, **{m: {a: {"scale": scale, "weights": {"zero": np.zeros((2, 198)).tolist()}}
        for a in base.ARMS} for m in MODES if m != "same"}}
    fits, maximum = [], 0.
    for level in strength.lowrank.LEVELS[1:]:
        k = min(int(level), numerical)
        basis = vt[:k].T*np.sqrt(len(x))/s[:k]
        z = x@basis
        initials = {label: (u[:, :k].T@n[0][mask]/np.sqrt(len(x))).T for label, n in noise.items()}
        old = np.asarray(cached["weights"][level][0])
        difference = float(np.max(np.abs(x@(basis@initials["A"].sum(axis=0)-old)*scale)))
        maximum = max(maximum, difference)
        if difference > 1e-12: raise ValueError("Matched A targets do not reproduce cached MSE")
        radius = float(np.linalg.norm((vt[:k]@old)*s[:k]/np.sqrt(len(x))))
        for mode, (target_set, calibration_set) in MODES.items():
            if mode == "same": continue
            initial = initials[target_set].copy()
            total = initial.sum(axis=0)
            unconstrained = float(np.linalg.norm(total))
            # Euclidean projection onto the same SUM ball, preserving opposing head components.
            initial -= ((1-min(1., radius/max(unconstrained, 1e-30)))*total/2)[None]
            residual = noise[target_set][0]
            _, precision, reliability, _ = noise[calibration_set]
            for arm in base.ARMS:
                target = residual*reliability if arm in ("group", "variance_group") else residual
                weights = precision if arm in ("variance", "variance_group") else np.ones_like(precision)
                theta, fit = base.solve_components(z, target[mask], weights[mask], initial, radius)
                models[mode][arm]["weights"][level] = (basis@theta.T).T.tolist()
                fits.append({"mode": mode, "arm": arm, "level": level, "retained_rank": k,
                    "total_cost_function_df": k, "component_function_df": 2*k,
                    "unconstrained_OLS_sum_norm": unconstrained, "OLS_budget_clipped": unconstrained > radius+1e-12, **fit})
            for arm in ("combined", "split"):
                if "scalar_path" not in models[mode][arm]: models[mode][arm]["scalar_path"] = copy.deepcopy(cached)
                coefficients = np.asarray(models[mode][arm]["weights"][level])
                scalar = old if target_set == "A" else coefficients.sum(axis=0)
                models[mode][arm]["scalar_path"]["weights"][level][0] = scalar.tolist()
    return models, {"fits": fits, "noise": {k: v[3] for k, v in noise.items()}, "cached_MSE_max_abs_difference": maximum}


def candidates(data, models):
    return {m: {a: base.candidates(data, model) for a, model in paths.items()} for m, paths in models.items()}


def predict(paths, selection):
    return {"raw": paths["same"]["combined"]["zero"], **{f"{m}_{a}_{f}": paths[m][a][selection[m][a][f] or "zero"]
        for m in MODES for a in base.ARMS for f in base.FILTERS}}


def metrics(data, draws, predictions):
    result = base.metrics(data, draws, predictions)
    mean = draws.mean(axis=1)
    anchor, ix = data["originals"], np.arange(len(draws))
    truth = joint.relative(mean, anchor)
    feasible = data["raw_logits"] <= 0
    mask = feasible & (np.arange(6)[None, :] != anchor[:, None])
    weights = mask/np.maximum(mask.sum(axis=1, keepdims=True), 1)
    oracle = np.where(feasible, mean[..., 0], np.inf).min(axis=1)
    for name, p in predictions.items():
        error = (weights*(p["cost"]-truth[..., 0])**2).sum(axis=1)
        regret = mean[ix, p["chosen"], 0]-oracle
        groups = [(result["all"], ix), *((piece, np.flatnonzero(data["roots"] == int(root))) for root, piece in result["by_root"].items())]
        for piece, indices in groups:
            piece["variants"][name].update(feasible_cost_mse=float(error[indices].mean()), gate_cost_regret=float(regret[indices].mean()))
    return result


def aggregate(pieces):
    def average(xs):
        return {"comparisons": len(xs), "mean_cases": float(np.mean([p["cases"] for p in xs])),
            "variants": {v: {f: float(np.mean([p["variants"][v][f] for p in xs])) for f in xs[0]["variants"][v]} for v in VARIANTS}}
    roots = sorted({r for p in pieces for r in p["by_root"]})
    return {"all": average([p["all"] for p in pieces]),
        "by_root": {r: average([p["by_root"][r] for p in pieces if r in p["by_root"]]) for r in roots}}


def compare_cached(matrix, expected):
    records, maximum = 0, 0.
    for scope, labels in expected.items():
        for label, stats in labels.items():
            actual = matrix[scope][label]
            for a, b in [(actual["all"], stats["all"]), *((actual["by_root"][r], p) for r, p in stats["by_root"].items())]:
                if a["mean_cases"] != b["mean_cases"] or a["comparisons"] != b["comparisons"]:
                    raise ValueError("Cached control dropped states")
                for variant, fields in b["variants"].items():
                    name = "raw" if variant == "raw" else f"same_{variant}"
                    for field, value in fields.items():
                        maximum = max(maximum, abs(a["variants"][name][field]-value))
                        records += 1
    if maximum > 1e-12: raise ValueError("Stage243 frozen control changed")
    return {"records": records, "max_abs_difference": maximum}
