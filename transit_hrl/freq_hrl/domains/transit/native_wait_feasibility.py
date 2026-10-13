"""Magnitude-aware, draw-level wait feasibility over causal plan features."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_conditional_wait as wait

RIDGE, MAX_ITERATIONS = .1, 200
FALSE_SAFE_COSTS = {"symmetric_feasibility": 1., "asymmetric_feasibility": 2.}


def wait_masses(values):
    """Retain harm magnitude: a positive mean need not have a majority of positive draws."""
    values = np.asarray(values)
    return np.maximum(values, 0).mean(axis=0), np.maximum(-values, 0).mean(axis=0)


def sample_weights(rows, masses):
    """Keep Stage212 target precision independent of optimization weighting."""
    targets = [masses[r["root"], r["scene_seed"], r["probe_index"]] for r in rows]
    return np.asarray([m.get("sample_weight", m["draw_count"]) for m in targets])


def design(raw, originals, center, scale):
    x = (raw-center)/scale
    x -= x[np.arange(len(x)), originals][:, None].copy()
    return np.concatenate([x, np.ones((*x.shape[:2], 1))], axis=-1)


def objective(beta, x, positive, negative, weights, false_safe_cost):
    logits = x@beta
    harm, benefit = false_safe_cost*positive, negative
    loss = np.average(harm*np.logaddexp(0, -logits)+benefit*np.logaddexp(0, logits), weights=weights)
    residual = (harm+benefit)*expit(logits)-harm
    gradient = x.T@(weights*residual)/weights.sum()+RIDGE*beta
    return float(loss+.5*RIDGE*(beta@beta)), gradient


def fit_wait(rows, masses, variant):
    false_safe_cost = FALSE_SAFE_COSTS[variant]
    raw = np.stack([wait.features(r, "conditional_wait") for r in rows])
    center, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.
    originals = np.asarray([r["original_choice"] for r in rows])
    mask = np.arange(6)[None, :] != originals[:, None]
    x = design(raw, originals, center, scale)[mask]
    keys = [(r["root"], r["scene_seed"], r["probe_index"]) for r in rows]
    positive = np.asarray([masses[k]["positive_wait_min"] for k in keys])[mask]
    negative = np.asarray([masses[k]["negative_wait_min"] for k in keys])[mask]
    weights = np.repeat(sample_weights(rows, masses), 5)
    target_scale = max(float(np.average(positive+negative, weights=weights)), 1e-6)
    result = minimize(objective, np.zeros(x.shape[1]), args=(x, positive/target_scale,
        negative/target_scale, weights, false_safe_cost), method="L-BFGS-B", jac=True,
        options={"maxiter": MAX_ITERATIONS, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success:
        raise RuntimeError(f"Wait feasibility fit did not converge: {result.message}")
    return {"method": "draw_mass_logistic", "variant": variant, "representation": "conditional_wait",
        "false_safe_cost": false_safe_cost, "ridge": RIDGE, "max_iterations": MAX_ITERATIONS,
        "parameter_count": x.shape[1], "coefficients": result.x.tolist(),
        "center": center.tolist(), "scale": scale.tolist(), "target_scale_min": target_scale,
        "training_states": len(rows), "training_groups": sorted(map(list, {paired.scene_group(r) for r in rows})),
        "training_mean_states": sum(masses[k]["draw_count"] == 8 for k in keys),
        "sample_weight_sum": float(weights.sum()/5), "optimizer_iterations": int(result.nit),
        "solver_success": bool(result.success), "final_loss": float(result.fun),
        "gradient_linf": float(abs(result.jac).max())}


def predict(row, model):
    raw = wait.features(row, model["representation"])[None]
    x = design(raw, [row["original_choice"]], model["center"], model["scale"])[0]
    share = expit(x@np.asarray(model["coefficients"]))
    share[row["original_choice"]] = 0.
    score = share-.5
    score[row["original_choice"]] = 0.
    return {"predicted_harm_mass_share": share.tolist(), "predicted_feasibility_score": score.tolist()}
