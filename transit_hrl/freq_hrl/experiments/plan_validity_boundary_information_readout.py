"""Frozen-model E comparisons: prefix coverage and conditional suffix MC separately."""

import numpy as np

from .plan_validity_boundary_information_fit import checked_window, row_map
from .plan_validity_boundary_tail import basis


COMPARISONS = (("base", "zero"), ("dense", "zero"), ("wide", "zero"),
    ("dense", "base"), ("wide", "base"), ("wide", "dense"))


def evaluation_data(rows, cell, design, prefixes):
    keyed = row_map(rows, prefixes)
    origins, now, wait, tails, groups, windows, keys = [], [], [], [], [], [], []
    for (seed, step), row in sorted(keyed.items()):
        if row["boundary_step"] != step + design["period"] or len(row["windows"]) != design["windows_per_state"]:
            raise ValueError("E state/window roster differs from the frozen design")
        for j, window in enumerate(row["windows"]):
            tails.append(checked_window(window, root=cell["optimizer_seed"], seed=seed, step=step,
                j=j, indices=design["readout"]["suffix_indices"]))
            context = window["boundary_context"]
            now.append(context["now"])
            wait.append(context["wait_one_check"])
            origins.append({**row, "offset_fraction": (step % design["period"]) / 25,
                "remaining_fraction": (design["horizon"] - step) / design["horizon"], "call_spent": False})
            groups.append(seed)
            windows.append(j)
            keys.append([seed, step, j])
    return {"now": basis(now), "wait_one_check": basis(wait), "causal_t": basis(origins),
        "tail": np.asarray(tails), "groups": np.asarray(groups), "windows": np.asarray(windows), "window_keys": keys}


def paired_loss(p, q, tail):
    difference = p - q
    square = float(np.mean(p ** 2 - q ** 2))
    alignment = -2 * float(np.mean(difference * tail.mean(axis=1)))
    variance = 4 * difference ** 2 * tail.var(axis=1, ddof=1) / tail.shape[1]
    return {"MSE_delta": square + alignment, "squared_prediction_term": square,
        "alignment_term": alignment, "conditional_suffix_MC_SE": float(np.sqrt(variance.sum()) / len(p))}


def summarize(data, predictions, *, comparisons=COMPARISONS):
    tail, groups, windows = data["tail"], data["groups"], data["windows"]
    reports = {}
    for candidate, reference in comparisons:
        p, q = predictions[candidate], predictions[reference]
        prefixes = [{"seed": int(seed), **paired_loss(p[groups == seed], q[groups == seed], tail[groups == seed])}
            for seed in np.unique(groups)]
        contrasts = np.asarray([r["MSE_delta"] for r in prefixes])
        reports[candidate + "_minus_" + reference] = {**paired_loss(p, q, tail),
            "prefix_reports": prefixes, "improving_prefixes": int(np.sum(contrasts < 0)),
            "prefix_delta_SD_descriptive": float(np.std(contrasts, ddof=1)),
            "suffix_halves": [paired_loss(p, q, t) for t in (tail[:, :4], tail[:, 4:])],
            "window_halves": [paired_loss(p[m], q[m], tail[m]) for m in (windows < 2, windows >= 2)]}
    return {"prefixes": len(np.unique(groups)), "states": len(tail) // 4, "windows": len(tail),
        "E_pairs": int(tail.size), "MSE_to_E_mean": {k: float(np.mean((p - tail.mean(axis=1)) ** 2)) for k, p in predictions.items()},
        "empirical_E_mean_noise_variance": float(np.mean(tail.var(axis=1, ddof=1) / tail.shape[1])),
        "comparisons": reports,
        "statistics": "negative_delta_is_lower_MSE_shared_label_square_cancels_prefix_dispersion_not_conditional_MC_SE",
        "scope": "new_prefix_boundary_prediction_given_frozen_roots_not_policy_or_earlier_t_deployable_credit"}
