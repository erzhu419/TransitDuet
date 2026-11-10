"""Training-A decision calibration on fixed ridge scores and cached effects."""

from __future__ import annotations

import numpy as np

from .plan_validity_phase_support import feature_matrix, predict
from .plan_validity_reliability import paired_mean_interval
from .plan_validity_window_coverage import values
from .pointmaze_timing_baselines import _ridge_at_df


def reconstruct_oof(banks, rules, *, effective_df, quantile):
    x = feature_matrix(banks["candidate"], clock=True)[0]
    groups = np.asarray([r["seed"] for r in banks["candidate"]])
    output = {}
    for continuation in ("reference", "candidate"):
        name = "label_" + continuation
        y = values(banks[continuation], "W", "A").mean(axis=1)
        scores = np.empty(len(groups))
        for group in np.unique(groups):
            train = groups != group
            scores[~train], _ = _ridge_at_df(x[train], y[train], x[~train], effective_df)
        if not np.isclose(np.quantile(scores, quantile), rules[name].threshold, rtol=1e-12, atol=1e-12):
            raise RuntimeError("reconstructed OOF quantile differs from the frozen cut")
        output[name] = scores
    return output


def calibrate_cut(scores, window_A, full_A, original_cut):
    """Maximize apparent G while retaining the original cut's apparent W."""
    scores, window_A, full_A = map(np.asarray, (scores, window_A, full_A))
    base = scores >= original_cut
    ordered = np.unique(scores)
    candidates = [float(original_cut), float(np.nextafter(ordered[0], -np.inf)),
        *[float(a + (b - a) / 2) for a, b in zip(ordered[:-1], ordered[1:])],
        float(np.nextafter(ordered[-1], np.inf))]
    feasible = []
    for cut in candidates:
        delta = (scores >= cut).astype(int) - base.astype(int)
        w, g = float(np.mean(delta * window_A)), float(np.mean(delta * full_A))
        if w >= 0:
            changed = int(np.count_nonzero(delta))
            feasible.append({"cut": cut, "delta_W_A": w, "delta_G_A": g,
                "changed_A_actions": changed, "choose_now_A": int(np.count_nonzero(scores >= cut))})
    selected = max(feasible, key=lambda r: (r["delta_G_A"], -r["changed_A_actions"], -abs(r["cut"] - original_cut)))
    return {**selected, "original_cut": float(original_cut), "candidate_cut_count": len(candidates),
        "A_constraints_are_empirical_not_a_safety_certificate": True}


def _B_gain(change, samples, groups):
    paired = change[:, None] * samples
    means = paired.mean(axis=1)
    path_means = [float(means[groups == g].mean()) for g in np.unique(groups)]
    return {"mean": float(means.mean()),
        "conditional_MC_standard_error": float(np.sqrt(np.sum(paired.var(axis=1, ddof=1) / paired.shape[1])) / len(means)),
        "training_prefix_interval": paired_mean_interval(path_means)}


def decision_readout(rows, oof, rules, selected):
    groups = np.asarray([r["seed"] for r in rows])
    factual = np.asarray([r["factual_stop"] for r in rows], dtype=int)
    effects = {k: values(rows, k, "B") for k in ("W", "G")}
    effects["tail"] = effects["G"] - effects["W"]
    a_effects = {k: values(rows, k, "A").mean(axis=1) for k in ("W", "G")}
    a_effects["tail"] = a_effects["G"] - a_effects["W"]
    result = {}
    for head, rule in rules.items():
        result[head] = {}
        for view, scores in (("oof_scores", oof[head]), ("frozen_full_scores", predict(rule, rows))):
            original = (scores >= rule.threshold).astype(int)
            utility = (scores >= selected[head]["cut"]).astype(int)
            changes = utility - original
            panels = {}
            for block, subset in (("B1", slice(0, 2)), ("B2", slice(2, 4)), ("B_all", slice(0, 4))):
                sample = {k: v[:, subset] for k, v in effects.items()}
                panels[block] = {"utility_minus_quantile": {k: _B_gain(changes, v, groups) for k, v in sample.items()},
                    "against_C0": {name: {k: _B_gain(action - factual, v, groups) for k, v in sample.items()}
                        for name, action in (("quantile", original), ("utility", utility))}}
            result[head][view] = {"apparent_A_utility_minus_quantile": {
                k: float(np.mean(changes * v)) for k, v in a_effects.items()},
                "changed_actions": int(np.count_nonzero(changes)),
                "changed_prefixes": len(set(groups[changes != 0].tolist())),
                "choose_now": {"quantile": int(original.sum()), "utility": int(utility.sum()), "C0": int(factual.sum())},
                "B_panels": panels}
    return result


def score_source_readout(rows, oof, rules, selected):
    """Compare both A-selected cuts on the same scoring view and paired futures."""
    panels = {source: decision_readout(rows, oof, rules, cuts) for source, cuts in selected.items()}
    groups = np.asarray([r["seed"] for r in rows])
    b = {k: values(rows, k, "B") for k in ("W", "G")}
    a = {k: values(rows, k, "A").mean(axis=1) for k in ("W", "G")}
    contrast = {}
    for head, rule in rules.items():
        contrast[head] = {}
        for view, scores in (("oof_scores", oof[head]), ("frozen_full_scores", predict(rule, rows))):
            old = scores >= selected["oof_scores"][head]["cut"]
            new = scores >= selected["frozen_full_scores"][head]["cut"]
            change = new.astype(int) - old.astype(int)
            contrast[head][view] = {
                "changed_actions": int(np.count_nonzero(change)),
                "changed_prefixes": len(set(groups[change != 0].tolist())),
                "apparent_A_full_selected_minus_oof_selected": {k: float(np.mean(change * v)) for k, v in a.items()},
                "B_full_selected_minus_oof_selected": {
                    block: {k: _B_gain(change, v[:, subset], groups) for k, v in b.items()}
                    for block, subset in (("B1", slice(0, 2)), ("B2", slice(2, 4)), ("B_all", slice(0, 4)))}}
    return {"by_calibration_source": panels, "same_score_source_contrast": contrast}
