"""Development-only conditional tail prediction on frozen nested boundary banks."""

import numpy as np

from .compute_budget import count_work
from .pointmaze_compact_plan_validity import _base_matrix, _causal_validity_interactions
from .pointmaze_timing_baselines import _ridge_at_df


METHODS = ("zero", "constant", "causal_t", "boundary_delta")


def basis(contexts):
    f, names = _causal_validity_interactions(*_base_matrix(contexts))
    remaining = np.asarray([c["remaining_fraction"] for c in contexts])[:, None]
    spent = np.asarray([c["call_spent"] for c in contexts], dtype=float)[:, None]
    clock = np.column_stack(([c["offset_fraction"] for c in contexts], remaining, spent))
    return np.column_stack((f, clock, f * remaining, f * spent))


def dataset(rows, *, horizon, period):
    origins, now, wait, tails, groups, windows = [], [], [], [], [], []
    for row in rows:
        step = row["opportunity_step"]
        origin = {**row, "offset_fraction": (step % period) / 25,
            "remaining_fraction": (horizon - step) / horizon, "call_spent": False}
        for j, window in enumerate(row["windows"]):
            origins.append(origin)
            now.append(window["boundary_context"]["now"])
            wait.append(window["boundary_context"]["wait_one_check"])
            tails.append([d["tail"] for d in window["suffix_draws"]])
            groups.append(row["seed"])
            windows.append(j)
    return {"causal_t": basis(origins), "boundary_delta": basis(wait) - basis(now),
        "tail": np.asarray(tails), "groups": np.asarray(groups), "windows": np.asarray(windows)}


def predict(model, x):
    z = (x - model["feature_mean"]) / model["feature_scale"]
    return model["weights"][0] + z @ np.asarray(model["weights"][1:])


def fit_views(data, train, *, total_df):
    y = data["tail"][train, :2].mean(axis=1)
    count_work("constant_fit_calls")
    constant = float(y.mean())
    predictions = {"zero": np.zeros(len(data["tail"])), "constant": np.full(len(data["tail"]), constant)}
    models = {"constant": {"value": constant, "total_effective_df": 1}}
    for method in ("causal_t", "boundary_delta"):
        x, target = data[method][train], y
        symmetric = method == "boundary_delta"
        if symmetric:
            # A value difference has no free offset; reversal is the same sample.
            x, target = np.r_[x, -x], np.r_[target, -target]
        _, model = _ridge_at_df(x, target, data[method], total_df if symmetric else total_df - 1)
        if symmetric:
            model["feature_mean"] = [0.] * x.shape[1]
            model["weights"][0] = 0.
        model["total_effective_df"] = model["effective_df_slopes"] + (0 if symmetric else 1)
        predictions[method] = predict(model, data[method])
        models[method] = model
    return predictions, models


def errors(prediction, tail, *, independent):
    result = {}
    for block, selected in (("A", tail[:, :2]), ("B", tail[:, 2:])):
        mean = selected.mean(axis=1)
        mse = float(np.mean((prediction - mean) ** 2))
        noise = float(np.mean(selected.var(axis=1, ddof=1) / selected.shape[1]))
        result[block] = {"mse": mse, "mse_minus_zero": float(np.mean(prediction ** 2 - 2 * prediction * mean)),
            "label_mean_variance": noise, "bias": float(np.mean(prediction - mean)),
            "noise_corrected_mse": mse - noise if independent or block == "B" else None}
    result["conditional_squared_error_cross_AB"] = float(np.mean(
        (prediction - tail[:, :2].mean(axis=1)) * (prediction - tail[:, 2:].mean(axis=1)))) if independent else None
    return result


def readout(predictions, data, mask, *, independent):
    return {method: errors(predictions[method][mask], data["tail"][mask], independent=independent)
        for method in METHODS}


def fit_probe(data, *, total_df, publish=lambda _: None):
    groups = np.unique(data["groups"])
    all_rows = np.ones(len(data["tail"]), dtype=bool)
    full, full_models = fit_views(data, all_rows, total_df=total_df)
    oof = {name: np.empty(len(all_rows)) for name in METHODS}
    fold_models = []
    for group in groups:
        held = data["groups"] == group
        scores, models = fit_views(data, ~held, total_df=total_df)
        for method in METHODS:
            oof[method][held] = scores[method][held]
        fold_models.append({"held_prefix": int(group), "training_prefixes": groups[groups != group].tolist(), "models": models})
        publish({"stage": "prefix_folds", "completed_prefix_folds": len(fold_models)})
    return summarize_probe(data, full, oof, full_models, fold_models, total_df=total_df)


def summarize_probe(data, full, oof, full_models, fold_models, *, total_df):
    groups = np.unique(data["groups"])
    all_rows = np.ones(len(data["tail"]), dtype=bool)
    panels = {"training": readout(full, data, all_rows, independent=False),
        "prefix_oof": readout(oof, data, all_rows, independent=True)}
    halves = [readout(oof, data, data["windows"] < 2, independent=True),
        readout(oof, data, data["windows"] >= 2, independent=True)]
    comparisons = [panels["prefix_oof"], *halves]
    qualifies = all(p["boundary_delta"]["B"]["mse"] < p[baseline]["B"]["mse"]
        for p in comparisons for baseline in ("zero", "constant", "causal_t"))
    public = {"prefixes": len(groups), "states": len(all_rows) // 4, "paired_windows": len(all_rows),
        "feature_columns": data["causal_t"].shape[1], "matched_total_effective_df": total_df,
        "actual_effective_df": {m: [full_models[m]["total_effective_df"],
            *[f["models"][m]["total_effective_df"] for f in fold_models]] for m in ("causal_t", "boundary_delta")},
        "panels": panels, "prefix_oof_window_halves": halves,
        "prefix_reports": [{"seed": int(g), "readout": readout(oof, data, data["groups"] == g, independent=True)} for g in groups],
        "boundary_probe_qualifies_on_this_root": qualifies,
        "scope": "exposed_development_prefix_LOPO_not_new_confirmation_or_policy_performance",
        "boundary_scope": "label_side_information_at_b_not_available_at_earlier_t",
        "statistics": "conditional_suffix_noise_correction_retains_negatives_no_pseudoseed_CI"}
    return public, {"full_models": full_models, "fold_models": fold_models, "training_predictions": full,
        "prefix_oof_predictions": oof, "prefix_by_window": data["groups"], "window_indices": data["windows"]}
