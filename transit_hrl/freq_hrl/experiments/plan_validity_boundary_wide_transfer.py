"""Match 31-prefix exposure while separating A-future and prefix transfer."""

import numpy as np

from .compute_budget import count_work
from .plan_validity_boundary_parent_scale import fit_parent_targets


def errors(included_mean, included_square_mean, excluded, fit_target, future_target, mask):
    p, q, q2, y = excluded[mask], included_mean[mask], included_square_mean[mask], future_target[mask]
    train_y = fit_target[mask]
    return {"in_fit_MSE": float(np.mean(q2 - 2 * q * train_y + train_y ** 2)),
        "matched_future_MSE": float(np.mean(q2 - 2 * q * y + y ** 2)),
        "held_prefix_MSE": float(np.mean((p - y) ** 2)), "zero_future_MSE": float(np.mean(y ** 2)),
        "matched_minus_zero": float(np.mean(q2 - 2 * q * y)),
        "held_minus_zero": float(np.mean(p ** 2 - 2 * p * y)),
        "held_minus_matched": float(np.mean(p ** 2 - q2 - 2 * (p - q) * y))}


def diagnose(data, old_prefixes, *, total_df, fit_targets=None):
    fitter = fit_parent_targets if fit_targets is None else fit_targets
    keys = np.asarray(data["window_keys"])
    groups, windows = keys[:, 0], keys[:, 2]
    labels = np.asarray(data["training_replicas"])
    prefixes = np.unique(groups)
    if labels.shape != (384, 2) or len(prefixes) != 32 or any(np.sum(groups == g) != 12 for g in prefixes):
        raise ValueError("need the frozen 32-prefix, two-A-label Wide bank")
    public, private = [], []
    all_rows = np.ones(len(groups), dtype=bool)
    for replica in (0, 1):
        target, future = labels[:, replica], labels[:, 1 - replica]
        included_sum, included_square_sum = np.zeros(len(groups)), np.zeros(len(groups))
        excluded = np.empty(len(groups))
        folds = []
        for held in prefixes:
            train = groups != held
            scores, model = fitter(data["now"], data["wait_one_check"], target, train, total_df=total_df)
            count_work("diagnostic_model_window_predictions", len(scores))
            excluded[~train] = scores[~train]
            # Average losses of individual DF-3 models, not an ensemble prediction.
            included_sum[train] += scores[train]
            included_square_sum[train] += scores[train] ** 2
            folds.append({"excluded_prefix": int(held),
                "training_prefixes": prefixes[prefixes != held].tolist(), "model": model})
        included = included_sum / 31
        included_square = included_square_sum / 31
        score = lambda mask: errors(included, included_square, excluded, target, future, mask)
        public.append({"fitting_A_column": replica, "evaluation_A_column": 1 - replica,
            "training_prefixes_per_fit": 31, "training_windows_per_fit": 372, "matched_model_losses_per_window": 31,
            "overall": score(all_rows),
            "prefix_reports": [{"seed": int(g), "origin": "cached8" if g in old_prefixes else "new24",
                **score(groups == g)} for g in prefixes],
            "origin_reports": {name: score(mask) for name, mask in (
                ("cached8", np.isin(groups, old_prefixes)), ("new24", ~np.isin(groups, old_prefixes)))},
            "window_halves": [score(windows < 2), score(windows >= 2)]})
        private.append({"fitting_A_column": replica, "folds": folds,
            "matched_prediction_mean": included, "matched_prediction_square_mean": included_square,
            "prefix_held_predictions": excluded})
    return {"prefixes": 32, "windows": 384, "A_labels": 768,
        "matched_included_rule": "mean_individual_losses_of_all_31_folds_including_probe_prefix_not_loss_of_ensemble",
        "views": public, "mean_across_views_descriptive": {
            k: float(np.mean([v["overall"][k] for v in public])) for k in public[0]["overall"]},
        "statistics": "A_only_development_crossfit_two_correlated_views_no_CI_or_model_selection",
        "scope": "same_31_prefix_sample_size_DF3_replica_transfer_vs_whole_prefix_transfer_not_E_or_policy_validation"}, private
