"""Fixed, observable partition gates at equal column count and label-fit DF."""

import numpy as np

from .compute_budget import count_work
from .plan_validity_boundary_parent_scale import ridge_solution, scaled_basis


METHODS = ("tracking_distance", "state_rms")
TRACKING_DISTANCE_COLUMN = 1


def gate_coordinate(arms, model):
    if model["gate_kind"] == "tracking_distance":
        return arms[:, TRACKING_DISTANCE_COLUMN]
    if model["gate_kind"] == "state_rms":
        z = (arms[:, :39] - model["parent_mean"]) / model["parent_scale"]
        return np.sqrt(np.mean(z ** 2, axis=1))
    raise ValueError("unknown frozen conditioning method")


def conditioned_basis(arms, model):
    phi = scaled_basis(arms, model["parent_mean"], model["parent_scale"])
    g = .5 * (1 + np.tanh((gate_coordinate(arms, model) - model["gate_center"]) / model["gate_scale"]))
    return np.column_stack((np.sqrt(1 - g)[:, None] * phi, np.sqrt(g)[:, None] * phi))


def predict_conditioned(model, now, wait):
    return (conditioned_basis(wait, model) - conditioned_basis(now, model)) @ np.asarray(model["weights"])


def fit_conditioned_targets(now, wait, training_mean, train, *, total_df, gate_kind):
    count_work("ridge_fit_calls")
    arms = np.r_[now[train], wait[train]]
    mean, scale = arms[:, :39].mean(axis=0), arms[:, :39].std(axis=0)
    model = {"parent_mean": mean.tolist(), "parent_scale": np.where(scale > 1e-8, scale, 1.).tolist(), "gate_kind": gate_kind}
    coordinate = gate_coordinate(arms, model)
    spread = float(coordinate.std())
    model.update(gate_center=float(np.median(coordinate)), gate_scale=spread if spread > 1e-8 else 1.)
    x = conditioned_basis(wait, model) - conditioned_basis(now, model)
    weights, alpha, df = ridge_solution(x[train], training_mean[train], total_df=total_df)
    model.update(weights=weights.tolist(), alpha=alpha, total_effective_df=df, effective_df_slopes=df,
        intercept=0., feature_columns=240, geometry="training_parent_scale_unit_norm_partition_of_unity_no_contrast_scaling")
    return x @ weights, model


def paired_readout(candidate, reference, target, mask):
    p, q = np.asarray(candidate["prefix_held_predictions"])[mask], np.asarray(reference["prefix_held_predictions"])[mask]
    cm, rm = np.asarray(candidate["matched_prediction_mean"])[mask], np.asarray(reference["matched_prediction_mean"])[mask]
    cs, rs = np.asarray(candidate["matched_prediction_square_mean"])[mask], np.asarray(reference["matched_prediction_square_mean"])[mask]
    y = target[mask]
    return {"held_MSE_delta": float(np.mean(p ** 2 - q ** 2 - 2 * (p - q) * y)),
        "matched_future_MSE_delta": float(np.mean(cs - rs - 2 * (cm - rm) * y))}


def compare_views(candidate, reference, data):
    keys, labels = np.asarray(data["window_keys"]), np.asarray(data["training_replicas"])
    result = []
    for r, (c, ref) in enumerate(zip(candidate, reference)):
        score = lambda mask: paired_readout(c, ref, labels[:, 1 - r], mask)
        result.append({"fitting_A_column": r, "overall": score(np.ones(len(keys), dtype=bool)),
            "window_halves": [score(keys[:, 2] < 2), score(keys[:, 2] >= 2)],
            "prefix_reports": [{"seed": int(g), **score(keys[:, 0] == g)} for g in np.unique(keys[:, 0])]})
    return result


def followup_rule(public, contrasts):
    for r in (0, 1):
        for reference in ("uniform", "state_rms"):
            view = contrasts["tracking_distance_minus_" + reference][r]
            if not all(p["held_MSE_delta"] < 0 for p in (view["overall"], *view["window_halves"])):
                return False
        if contrasts["tracking_distance_minus_uniform"][r]["overall"]["matched_future_MSE_delta"] > 0:
            return False
        if public["tracking_distance"]["views"][r]["overall"]["matched_minus_zero"] >= 0:
            return False
    return True
