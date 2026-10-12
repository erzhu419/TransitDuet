"""Regularized linear action differences on the same causal critic features."""

import numpy as np

from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_floor_value as value


def fit_value(rows, reference=None, *, sample_weights=None):
    raw = np.stack([value.features(r, "tail_summary") for r in rows])
    labels = [{l["candidate"]: l for l in r["labels"]} for r in rows]
    targets = np.asarray([[[l[i][h] for h in value.HEADS] for i in range(6)] for l in labels])
    a, b = np.triu_indices(6, 1)
    if reference is None:
        scale = raw.std(axis=(0, 1))
        scale[scale < 1e-8] = 1.
        reference = {"center": raw.mean(axis=(0, 1)).tolist(), "scale": scale.tolist(),
            "target_scale": np.maximum(np.sqrt(np.mean((targets[:, a]-targets[:, b])**2, axis=(0, 1))), 1e-6).tolist()}
    x = (raw-reference["center"])/reference["scale"]
    y = targets/reference["target_scale"]
    z, gaps = (x[:, a]-x[:, b]).reshape(-1, x.shape[-1]), (y[:, a]-y[:, b]).reshape(-1, 3)
    state_weights = np.ones(len(rows)) if sample_weights is None else np.asarray(sample_weights)
    weights = np.repeat(state_weights, len(a))
    gram = z.T@(weights[:, None]*z)/weights.sum()+paired.RIDGE*np.eye(z.shape[-1])
    coefficients = np.linalg.solve(gram, z.T@(weights[:, None]*gaps)/weights.sum())
    pair_loss = float(np.average(((z@coefficients-gaps)**2).mean(axis=1), weights=weights))
    return {"method": "pair_ridge", "coefficients": coefficients.tolist(), "heads": list(value.HEADS),
        "center": reference["center"], "scale": reference["scale"], "target_scale": reference["target_scale"],
        "ridge": paired.RIDGE, "representation": "tail_summary", "parameter_count": coefficients.size,
        "steps": 0, "solver": "weighted_ridge_normal_equations", "objective": "paired_mse_plus_ridge",
        "final_pair_loss": pair_loss, "ridge_loss": float(paired.RIDGE*np.sum(coefficients**2)/3),
        "sample_weight_sum": float(state_weights.sum()), "training_states": len(rows),
        "training_groups": sorted(map(list, {paired.scene_group(r) for r in rows})),
        "absolute_fleet_forecast": False}


def predict(row, model):
    x = (value.features(row, "tail_summary")-model["center"])/model["scale"]
    x = x-x[row["original_choice"]].copy()
    heads = (x@np.asarray(model["coefficients"]))*model["target_scale"]
    return {"predicted_heads": heads[:, :2].tolist(), "predicted_wait_delta_min": heads[:, 2].tolist()}


def choose_action(row, model, policy):
    prediction = predict(row, model)
    cost = np.asarray(prediction["predicted_heads"]).sum(axis=1)
    if policy == "wait_feasible":
        cost = np.where(np.asarray(prediction["predicted_wait_delta_min"]) <= 0, cost, np.inf)
    elif policy != "cost_only":
        raise ValueError("Unknown paired-ridge decision policy")
    return paired.choice_from_cost(cost, row["original_choice"])


def evaluate(rows, model, policy):
    result = paired.evaluate_action_value(rows, model, predictor=lambda r, m: np.asarray(predict(r, m)["predicted_heads"]),
        selector=lambda r, m: choose_action(r, m, policy))
    for row, observed in zip(rows, result): observed.update(predict(row, model))
    return result
