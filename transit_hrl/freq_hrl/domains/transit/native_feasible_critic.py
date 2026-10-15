"""Decision objectives with fixed independent-future service/risk calibration."""

import copy

import numpy as np

from freq_hrl.domains.transit import native_cost_objectives as optimizer
from freq_hrl.domains.transit import native_disjoint_critic as previous

base, joint, strength = previous.base, previous.joint, previous.strength
OBJECTIVES = ("all_mse", "feasible_mse", "rank_regret")
FILTERS = base.FILTERS
VARIANTS = ("raw", *(f"{o}_{f}" for o in OBJECTIVES for f in FILTERS))


def component_mse(z, target, precision, difference, eligible, anchor):
    """Optimize the head sum, holding the cached head difference fixed."""
    mask = eligible & (np.arange(6)[None, :] != anchor[:, None])
    weights = (mask/np.maximum(mask.sum(axis=1, keepdims=True), 1))[..., None]*precision
    fixed = (z@difference)[..., None]*np.array([.5, -.5])
    n = len(z)
    def loss(theta):
        error = (z@theta)[..., None]/2+fixed-target
        return float(np.sum(weights*error**2)/n), np.einsum("nah,nak->k", weights*error, z)/n
    matrix = .5*np.einsum("na,nak,nal->kl", weights.sum(axis=-1), z, z)/n
    return loss, lambda theta: matrix


def fit_paths(rows, data, draws, cached):
    source, budget = cached["disjoint"]["variance_group"], cached["same"]["combined"]["scalar_path"]
    models = {o: copy.deepcopy(source) for o in OBJECTIVES}
    scale = source["scale"]
    residual, precision, reliability, _ = base.noise_targets(rows, data, draws, scale)
    target = residual*reliability
    anchor = data["originals"]
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)
    _, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    numerical = int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum())
    offset = joint.relative(data["heads"], anchor)[..., 0]/scale
    truth = offset+target.sum(axis=-1)
    eligible = data["raw_logits"] <= 0
    ledger, maximum = [], 0.
    for level in strength.lowrank.LEVELS[1:]:
        k = min(int(level), numerical)
        basis = vt[:k].T*np.sqrt(mask.sum())/s[:k]
        z = x@basis
        old = np.asarray(source["weights"][level])
        heads = (vt[:k]@old.T).T*s[:k]/np.sqrt(mask.sum())
        initial, difference = heads.sum(axis=0), heads[0]-heads[1]
        error = float(np.max(np.abs((x@(basis@heads.T)-x@old.T)*scale)))
        maximum = max(maximum, error)
        if error > 1e-12: raise ValueError("Cached split critic left the fixed subspace")
        old_budget = np.asarray(budget["weights"][level][0])
        radius = float(np.linalg.norm((vt[:k]@old_budget)*s[:k]/np.sqrt(mask.sum())))
        # Source GLS is optimal in this smaller class too; verify the loss/head mapping.
        all_loss, _ = component_mse(z, target, precision, difference, np.ones_like(eligible), anchor)
        _, control = optimizer.finish_fit(all_loss, initial, initial, 0, radius=radius)
        objectives = {"feasible_mse": component_mse(z, target, precision, difference, eligible, anchor),
            "rank_regret": optimizer.losses(z, offset, truth, eligible, anchor)["rank_regret"]}
        for name, (loss, hessian) in objectives.items():
            solve = optimizer.quadratic_fit if name == "feasible_mse" else optimizer.constrained_fit
            theta, fit = solve(loss, hessian, initial, radius=radius)
            models[name]["weights"][level] = (basis@np.stack(((theta+difference)/2, (theta-difference)/2), axis=-1)).T.tolist()
            ledger.append({"objective": name, "level": level, "retained_rank": k,
                "total_cost_function_df": k, "new_parameters": k, "cached_initial_norm": float(np.linalg.norm(initial)),
                "all_mse_control_optimality": control["optimality"], **fit})
    return models, {"fits": ledger, "subspace_max_abs_difference": maximum}


def candidates(data, models):
    return {o: base.candidates(data, m) for o, m in models.items()}


def select(blocks):
    roots = sorted({r for b in blocks for r in b["curves"]["all_mse"]["zero"]["by_root"]})
    scales = {r: np.mean([b["scales"] for b in blocks if str(b["held_group"][0]) == r], axis=0) for r in roots}
    summaries, selections = {}, {}
    for objective in OBJECTIVES:
        summaries[objective] = {c: {"pooled": base.previous.curve_summary([b["curves"][objective][c] for b in blocks], scales),
            "halves": [base.previous.curve_summary([b["curves"][objective][c] for b in blocks if b["half"] == h], scales) for h in (0, 1)]}
            for c in strength.CANDIDATES}
        selections[objective] = {}
        for mode in FILTERS:
            eligible = [c for c in strength.CANDIDATES if c != "zero" and summaries[objective][c]["pooled"]["score"] < -base.previous.TOL
                and base.previous.nonharm(summaries[objective][c]["pooled"])
                and (mode != "half_safe" or all(base.previous.nonharm(h) for h in summaries[objective][c]["halves"]))]
            selections[objective][mode] = min(eligible, key=lambda c: summaries[objective][c]["pooled"]["score"]) if eligible else None
    return selections, summaries


def predict(paths, selection):
    return {"raw": paths["all_mse"]["zero"],
        **{f"{o}_{f}": paths[o][selection[o][f] or "zero"] for o in OBJECTIVES for f in FILTERS}}


def aggregate(pieces):
    def average(xs):
        return {"comparisons": len(xs), "mean_cases": float(np.mean([p["cases"] for p in xs])),
            "variants": {v: {f: float(np.mean([p["variants"][v][f] for p in xs])) for f in xs[0]["variants"][v]} for v in VARIANTS}}
    roots = sorted({r for p in pieces for r in p["by_root"]})
    return {"all": average([p["all"] for p in pieces]),
        "by_root": {r: average([p["by_root"][r] for p in pieces if r in p["by_root"]]) for r in roots}}


def compare_cached(matrix, expected):
    maximum, records = 0., 0
    for scope, labels in expected.items():
        for label, stats in labels.items():
            actual = matrix[scope][label]
            for a, b in [(actual["all"], stats["all"]), *((actual["by_root"][r], p) for r, p in stats["by_root"].items())]:
                if a["mean_cases"] != b["mean_cases"] or a["comparisons"] != b["comparisons"]: raise ValueError("Dropped cached states")
                for name in ("raw", *(f"all_mse_{f}" for f in FILTERS)):
                    old = "raw" if name == "raw" else name.replace("all_mse", "disjoint_variance_group")
                    for field, value in b["variants"][old].items():
                        maximum = max(maximum, abs(a["variants"][name][field]-value))
                        records += 1
    if maximum > 1e-12: raise ValueError("Frozen independent-future control changed")
    return {"records": records, "max_abs_difference": maximum}
