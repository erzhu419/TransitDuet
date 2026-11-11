"""Scale parent states before interactions; never standardize arm differences."""

from copy import deepcopy
import numpy as np

from .compute_budget import count_work
from .plan_validity_boundary_tail import METHODS, basis, dataset, readout, summarize_probe


def parent_dataset(rows, *, horizon, period):
    data = dataset(rows, horizon=horizon, period=period)
    for arm in ("now", "wait_one_check"):
        data[arm] = basis([w["boundary_context"][arm] for r in rows for w in r["windows"]])
    return data


def scaled_basis(arms, mean, scale):
    f = (arms[:, :39] - mean) / scale
    clock = arms[:, 39:42]
    return np.column_stack((f, clock, f * clock[:, 1:2], f * clock[:, 2:3]))


def predict_parent(model, now, wait):
    x = scaled_basis(wait, model["parent_mean"], model["parent_scale"]) - scaled_basis(now, model["parent_mean"], model["parent_scale"])
    return x @ np.asarray(model["weights"])


def fit_parent(data, train, *, total_df):
    count_work("ridge_fit_calls")
    arms = np.r_[data["now"][train, :39], data["wait_one_check"][train, :39]]
    mean, scale = arms.mean(axis=0), arms.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    x = scaled_basis(data["wait_one_check"][train], mean, scale) - scaled_basis(data["now"][train], mean, scale)
    y = data["tail"][train, :2].mean(axis=1)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    eigenvalues = s ** 2
    rank = int(np.sum(eigenvalues > max(1., eigenvalues[0]) * 1e-12))
    if not 0 < total_df < rank:
        raise ValueError("fixed potential DF exceeds the training contrast rank")
    lower, upper = 0., float(eigenvalues.sum() / total_df)
    for _ in range(70):
        alpha = (lower + upper) / 2
        if np.sum(eigenvalues / (eigenvalues + alpha)) > total_df:
            lower = alpha
        else:
            upper = alpha
    # Direct SVD ridge: reusing a column-standardizing solver would undo the repair.
    weights = vt.T @ ((s / (eigenvalues + upper)) * (u.T @ y))
    df = float(np.sum(eigenvalues / (eigenvalues + upper)))
    model = {"parent_mean": mean.tolist(), "parent_scale": scale.tolist(), "weights": weights.tolist(),
        "alpha": upper, "intercept": 0., "total_effective_df": df, "effective_df_slopes": df,
        "geometry": "training_arm_parent_scale_before_products_and_difference_no_contrast_scaling"}
    return predict_parent(model, data["now"], data["wait_one_check"]), model


def verify_prior(data, previous, public):
    if (not np.array_equal(previous["prefix_by_window"], data["groups"])
        or not np.array_equal(previous["window_indices"], data["windows"])):
        raise ValueError("saved predictions do not match the resident window roster")
    groups = np.unique(data["groups"])
    if ([f["held_prefix"] for f in previous["fold_models"]] != groups.tolist()
        or any(f["training_prefixes"] != groups[groups != f["held_prefix"]].tolist() for f in previous["fold_models"])):
        raise ValueError("saved whole-prefix folds differ")
    all_rows = np.ones(len(data["tail"]), dtype=bool)
    for panel, key in (("training", "training_predictions"), ("prefix_oof", "prefix_oof_predictions")):
        scores = {k: np.asarray(previous[key][k]) for k in METHODS}
        reproduced = readout(scores, data, all_rows, independent=panel == "prefix_oof")
        for method in METHODS:
            for block in ("A", "B"):
                if not np.isclose(reproduced[method][block]["mse"], public["panels"][panel][method][block]["mse"], rtol=1e-10, atol=1e-10):
                    raise ValueError("resident labels/predictions do not reproduce the frozen prior losses")


def fit_parent_probe(data, previous, *, total_df, publish=lambda _: None):
    groups = np.unique(data["groups"])
    full = {k: np.asarray(previous["training_predictions"][k]).copy() for k in METHODS}
    oof = {k: np.asarray(previous["prefix_oof_predictions"][k]).copy() for k in METHODS}
    models = deepcopy(previous["full_models"])
    folds = deepcopy(previous["fold_models"])
    full["boundary_delta"], models["boundary_delta"] = fit_parent(data, np.ones(len(data["tail"]), dtype=bool), total_df=total_df)
    for i, group in enumerate(groups):
        held = data["groups"] == group
        scores, model = fit_parent(data, ~held, total_df=total_df)
        oof["boundary_delta"][held] = scores[held]
        folds[i]["models"]["boundary_delta"] = model
        publish({"stage": "parent_scaled_prefix_folds", "completed_prefix_folds": i + 1})
    public, private = summarize_probe(data, full, oof, models, folds, total_df=total_df)
    public.update(geometry=models["boundary_delta"]["geometry"], baseline_predictions_reused=True)
    return public, private
