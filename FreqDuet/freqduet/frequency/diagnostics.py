"""Frequency-separation diagnostics for FreqDuet.

These helpers are intentionally policy-agnostic. The runner supplies compact
episode traces, and this module computes attribution and response metrics that
can be reused by ablation scripts or plotting code.
"""

import math
from collections import defaultdict

import numpy as np


def _float_array(values):
    return np.asarray(values, dtype=np.float64)


def _quantile_bins(values, bins):
    values = _float_array(values)
    if values.size == 0 or float(np.std(values)) < 1e-12:
        return None
    bins = max(2, int(bins))
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        lo = float(values.min())
        hi = float(values.max())
        if abs(hi - lo) < 1e-12:
            return None
        edges = np.linspace(lo, hi, bins + 1)
    return edges


def binned_mutual_information(xs, ys, bins=8, min_n=10, normalized=True):
    """Return binned mutual information between two scalar traces.

    Quantile bins make the score robust to the very different scales used by
    low-frequency demand, high-frequency residuals, and holding actions.
    Normalized MI is in [0, 1] when both marginal entropies are non-zero.
    """
    x = _float_array(xs)
    y = _float_array(ys)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < int(min_n):
        return 0.0
    x_edges = _quantile_bins(x, bins)
    y_edges = _quantile_bins(y, bins)
    if x_edges is None or y_edges is None:
        return 0.0

    xb = np.digitize(x, x_edges[1:-1], right=False)
    yb = np.digitize(y, y_edges[1:-1], right=False)
    joint = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64)
    for i, j in zip(xb, yb):
        joint[int(i), int(j)] += 1.0
    total = float(joint.sum())
    if total <= 0.0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    nz = pxy > 0.0
    for i, j in zip(*np.nonzero(nz)):
        mi += pxy[i, j] * math.log(pxy[i, j] / max(px[i] * py[j], 1e-12))
    if not normalized:
        return float(max(mi, 0.0))

    hx = -float(np.sum(px[px > 0.0] * np.log(px[px > 0.0])))
    hy = -float(np.sum(py[py > 0.0] * np.log(py[py > 0.0])))
    denom = math.sqrt(max(hx * hy, 1e-12))
    return float(np.clip(mi / denom, 0.0, 1.0))


def demand_attribution_mi(upper_samples, lower_samples, bins=8):
    """Compute the MI version of the FreqDuet focus score.

    Score =
      I(a_U; lambda_low) - I(a_U; lambda_high)
      + I(a_L; lambda_high) - I(a_L; lambda_low)
    """
    metrics = {
        "demand_attr_mi_score": 0.0,
        "demand_attr_mi_upper_low": 0.0,
        "demand_attr_mi_upper_high": 0.0,
        "demand_attr_mi_lower_high": 0.0,
        "demand_attr_mi_lower_low": 0.0,
    }

    if upper_samples:
        arr = np.asarray(upper_samples, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            metrics["demand_attr_mi_upper_low"] = binned_mutual_information(
                arr[:, 0], arr[:, 2], bins=bins)
            metrics["demand_attr_mi_upper_high"] = binned_mutual_information(
                arr[:, 1], arr[:, 2], bins=bins)

    if lower_samples:
        arr = np.asarray(lower_samples, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            metrics["demand_attr_mi_lower_high"] = binned_mutual_information(
                arr[:, 1], arr[:, 2], bins=bins)
            metrics["demand_attr_mi_lower_low"] = binned_mutual_information(
                arr[:, 0], arr[:, 2], bins=bins)

    metrics["demand_attr_mi_score"] = (
        metrics["demand_attr_mi_upper_low"]
        - metrics["demand_attr_mi_upper_high"]
        + metrics["demand_attr_mi_lower_high"]
        - metrics["demand_attr_mi_lower_low"]
    )
    return metrics


def shock_response_metrics(
        events,
        shock_threshold=0.10,
        action_threshold_s=10.0,
        response_window_s=900.0,
        same_station=False):
    """Measure holding response lag after local high-frequency shocks.

    A shock is a rising-edge event where |local high residual| crosses
    ``shock_threshold`` for a station-direction pair. A hit is the first later
    lower action above ``action_threshold_s`` in the same direction, optionally
    at the same station, within ``response_window_s``.
    """
    base = {
        "shock_response_time_mean_s": 0.0,
        "shock_response_time_std_s": 0.0,
        "shock_response_hit_rate": 0.0,
        "shock_events": 0.0,
        "shock_action_mean_s": 0.0,
    }
    if not events:
        return base

    clean = []
    for event in events:
        try:
            clean.append({
                "time_s": float(event.get("time_s", 0.0)),
                "station_id": int(event.get("station_id", -1)),
                "direction": bool(event.get("direction", True)),
                "high": float(event.get("high", 0.0)),
                "action_s": float(event.get("action_s", 0.0)),
            })
        except (TypeError, ValueError):
            continue
    if not clean:
        return base

    clean.sort(key=lambda x: x["time_s"])
    threshold = abs(float(shock_threshold))
    action_threshold_s = max(float(action_threshold_s), 0.0)
    response_window_s = max(float(response_window_s), 0.0)

    shock_indices = []
    prev_abs = defaultdict(float)
    for idx, event in enumerate(clean):
        shock_key = (event["station_id"], event["direction"])
        high_abs = abs(event["high"])
        if high_abs >= threshold and prev_abs[shock_key] < threshold:
            shock_indices.append(idx)
        prev_abs[shock_key] = high_abs

    if not shock_indices:
        return base

    lags = []
    shock_actions = []
    for idx in shock_indices:
        shock = clean[idx]
        shock_actions.append(shock["action_s"])
        t0 = shock["time_s"]
        for event in clean[idx:]:
            if event["time_s"] - t0 > response_window_s:
                break
            if event["direction"] != shock["direction"]:
                continue
            if same_station and event["station_id"] != shock["station_id"]:
                continue
            if event["action_s"] >= action_threshold_s:
                lags.append(event["time_s"] - t0)
                break

    base["shock_events"] = float(len(shock_indices))
    base["shock_action_mean_s"] = float(np.mean(shock_actions)) if shock_actions else 0.0
    base["shock_response_hit_rate"] = (
        float(len(lags) / len(shock_indices)) if shock_indices else 0.0)
    if lags:
        arr = np.asarray(lags, dtype=np.float64)
        base["shock_response_time_mean_s"] = float(arr.mean())
        base["shock_response_time_std_s"] = float(arr.std())
    return base
