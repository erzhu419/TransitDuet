"""Signed fleet-margin support and bank stability, using cached draws only."""

import numpy as np

SIGNS = ("added_fleet", "saved_fleet")
GAP_PARTS = ("probability", "amplitude", "one_sided_support", "near_zero")
SCATTER_PARTS = ("occurrence", "within_event", "near_zero")
ZERO = 1e-12


def audit(margins):
    """Each row is one original-relative action with two disjoint future banks."""
    y = np.asarray(margins, float).reshape(-1, 2, 16)
    indicator = np.stack((y > ZERO, y < -ZERO), axis=-1)
    magnitude = np.abs(y)[..., None]*indicator
    counts = indicator.sum(axis=2)
    probability = counts/16
    amplitude = np.divide(magnitude.sum(axis=2), counts,
        out=np.full(counts.shape, np.nan), where=counts > 0)
    # Empty empirical contributions are zero; conditional amplitudes stay unknown.
    safe = np.nan_to_num(amplitude)
    products = probability*safe
    near = np.where(np.abs(y) <= ZERO, y, 0.)
    direct = y.mean(axis=2)
    reconstructed = products[..., 0]-products[..., 1]+near.mean(axis=2)
    both = (counts[:, 0] > 0)&(counts[:, 1] > 0)
    one = (counts[:, 0] > 0)^(counts[:, 1] > 0)
    dp = probability[:, 1]-probability[:, 0]
    da = safe[:, 1]-safe[:, 0]
    probability_gap = np.where(both, dp*safe.mean(axis=1), 0.)
    amplitude_gap = np.where(both, da*probability.mean(axis=1), 0.)
    support_gap = np.where(one, products[:, 1]-products[:, 0], 0.)
    signed = np.array([1., -1.])
    gap_parts = np.stack((probability_gap@signed, amplitude_gap@signed,
        support_gap@signed, near[:, 1].mean(axis=1)-near[:, 0].mean(axis=1)), axis=-1)
    gap = direct[:, 1]-direct[:, 0]
    occurrence = np.sum((indicator-probability[:, :, None])*safe[:, :, None]*signed, axis=-1)
    within = np.sum((magnitude-indicator*safe[:, :, None])*signed, axis=-1)
    near_centered = near-near.mean(axis=2, keepdims=True)
    centered = y-y.mean(axis=2, keepdims=True)
    scatter = np.stack((occurrence, within, near_centered), axis=-1)
    covariance = np.einsum("nbdi,nbdj->nij", scatter, scatter)/(16*15)
    variance = np.sum(centered**2, axis=(1, 2))/(16*15)
    signed_draws = magnitude*signed
    signed_centered = signed_draws-signed_draws.mean(axis=2, keepdims=True)
    sign_covariance = np.einsum("nbdi,nbdj->nij", signed_centered, signed_centered)/(16*15)
    errors = {"mean_product": float(np.max(np.abs(direct-reconstructed), initial=0.)),
        "gap_decomposition": float(np.max(np.abs(gap-gap_parts.sum(axis=1)), initial=0.)),
        "centered_decomposition": float(np.max(np.abs(centered-scatter.sum(axis=-1)), initial=0.)),
        "variance_decomposition": float(np.max(np.abs(variance-covariance.sum(axis=(1, 2))), initial=0.))}
    if max(errors.values()) > 1e-12: raise ValueError("Signed risk identities failed")
    return {"counts": counts, "probability": probability, "amplitude": amplitude,
        "products": products, "direct": direct, "both": both, "one": one,
        "gap_parts": gap_parts, "gap": gap, "variance": variance,
        "scatter_covariance": covariance, "sign_covariance": sign_covariance, "errors": errors}


def summarize(a, mask):
    ids = np.flatnonzero(mask)
    counts, p, mu = (a[k][ids] for k in ("counts", "probability", "amplitude"))
    gap, parts, variance = (a[k][ids] for k in ("gap", "gap_parts", "variance"))
    cov = a["scatter_covariance"][ids].sum(axis=0)
    gram = parts.T@parts
    total_v, total_g = float(variance.sum()), float(gap@gap)
    ratio = lambda x, y: float(x/y) if y else None
    signs = {}
    for i, name in enumerate(SIGNS):
        both, one = a["both"][ids, i], a["one"][ids, i]
        means = mu[both, :, i]
        signs[name] = {"event_draws": int(counts[..., i].sum()),
            "zero_event_action_banks": int(np.count_nonzero(counts[..., i] == 0)),
            "one_event_action_banks": int(np.count_nonzero(counts[..., i] == 1)),
            "both_banks_supported_actions": int(both.sum()), "one_bank_supported_actions": int(one.sum()),
            "neither_bank_supported_actions": int(np.count_nonzero(~both&~one)),
            "probability_gap_RMS": float(np.sqrt(np.mean(np.diff(p[..., i], axis=1)**2))) if len(ids) else None,
            "both_supported_amplitude_gap_RMS": float(np.sqrt(np.mean(np.diff(means, axis=1)**2))) if len(means) else None,
            "both_supported_amplitude_gap_over_midpoint_RMS": np.sqrt(ratio(float(np.sum(np.diff(means, axis=1)**2)), float(np.sum(means.mean(axis=1)**2)))) if len(means) else None,
            "both_supported_equal_amplitude_actions": int(np.count_nonzero(np.abs(np.diff(means, axis=1)[:, 0]) <= ZERO)),
            "unsigned_mean_sum_by_bank": a["products"][ids, :, i].sum(axis=0).tolist()}
    observed_cross = float(gram.sum()-np.trace(gram))
    signs_cov = a["sign_covariance"][ids].sum(axis=0)
    direct = a["direct"][ids]
    return {"unique_state_actions": len(ids), "action_future_keys": len(ids)*32,
        "zero_any_event_action_banks": int(np.count_nonzero(counts.sum(axis=-1) == 0)),
        "actions_with_no_event_in_either_bank": int(np.count_nonzero(counts.sum(axis=(1, 2)) == 0)),
        "actions_with_both_signs_observed": int(np.count_nonzero(np.all(counts.sum(axis=1) > 0, axis=1))),
        "bank_mean_sign_reversals": int(np.count_nonzero((direct[:, 0]*direct[:, 1] < 0)&np.all(np.abs(direct) > ZERO, axis=1))),
        "signs": signs, "direct_mean_sum_by_bank": direct.sum(axis=0).tolist(),
        "observed_gap_squared_sum": total_g, "plugin_two_bank_mean_variance_sum": total_v,
        "observed_gap_RMS": float(np.sqrt(total_g/len(ids))) if len(ids) else None,
        "plugin_noise_RMS": float(np.sqrt(total_v/len(ids))) if len(ids) else None,
        "observed_over_plugin_noise_RMS": np.sqrt(ratio(total_g, total_v)) if total_v else None,
        "gap_parts_squared_sum": dict(zip(GAP_PARTS, np.diag(gram).tolist())),
        "gap_signed_cross_sum": observed_cross,
        "gap_gram": gram.tolist(),
        "scatter_variance": dict(zip(SCATTER_PARTS, np.diag(cov).tolist())),
        "scatter_signed_cross_sum": float(cov.sum()-np.trace(cov)),
        "scatter_fraction": {k: ratio(cov[i, i], total_v) for i, k in enumerate(SCATTER_PARTS)},
        "saved_added_signed_noise_cross_sum": float(2*signs_cov[0, 1])}
