"""Small statistical checks for experiment claim gates."""

from __future__ import annotations

from math import comb
from typing import Any, Iterable

import numpy as np


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def finite_array(values: Iterable[Any]) -> np.ndarray:
    out = [value for value in (finite_float(v) for v in values) if value is not None]
    return np.asarray(out, dtype=np.float64)


def bootstrap_mean_ci(
    values: Iterable[Any],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    arr = finite_array(values).reshape(-1)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, arr.size, size=(max(1, int(n_boot)), arr.size))
    means = arr[draws].mean(axis=1)
    low_q = 100.0 * float(alpha) / 2.0
    high_q = 100.0 * (1.0 - float(alpha) / 2.0)
    return float(np.percentile(means, low_q)), float(np.percentile(means, high_q))


def sign_test_p_value(improvements: Iterable[Any]) -> float:
    vals = finite_array(improvements)
    vals = vals[np.abs(vals) > 1e-12]
    n = int(vals.size)
    if n == 0:
        return 1.0
    wins = int(np.sum(vals > 0.0))
    tail = min(wins, n - wins)
    prob = sum(comb(n, k) for k in range(0, tail + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * prob))


def _row_key(row: dict[str, Any], key_fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in key_fields)


def paired_delta_stats(
    rows: list[dict[str, Any]],
    variant_key: str,
    pair_keys: tuple[str, ...],
    metric: str,
    treatment: str,
    control: str,
    lower_is_better: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return paired treatment-control deltas for a metric.

    Raw delta is `treatment - control`. Improvement is `-delta` when lower is
    better, otherwise `delta`.
    """

    indexed: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    for row in rows:
        variant = row.get(variant_key)
        value = finite_float(row.get(metric))
        if variant is None or value is None:
            continue
        indexed[(str(variant), _row_key(row, pair_keys))] = row

    treatment_keys = {
        pair for (variant, pair) in indexed
        if variant == str(treatment)
    }
    control_keys = {
        pair for (variant, pair) in indexed
        if variant == str(control)
    }
    common = sorted(treatment_keys & control_keys, key=repr)
    deltas = []
    for pair in common:
        t_val = finite_float(indexed[(str(treatment), pair)].get(metric))
        c_val = finite_float(indexed[(str(control), pair)].get(metric))
        if t_val is None or c_val is None:
            continue
        deltas.append(t_val - c_val)
    delta_arr = finite_array(deltas)
    improvements = -delta_arr if lower_is_better else delta_arr
    ci_low, ci_high = bootstrap_mean_ci(delta_arr, n_boot=n_boot, seed=seed)
    imp_low, imp_high = bootstrap_mean_ci(improvements, n_boot=n_boot, seed=seed + 7919)
    return {
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "direction": "decrease" if lower_is_better else "increase",
        "n_common": int(delta_arr.size),
        "delta_mean": float(delta_arr.mean()) if delta_arr.size else float("nan"),
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "improvement_mean": float(improvements.mean()) if improvements.size else float("nan"),
        "improvement_ci95_low": imp_low,
        "improvement_ci95_high": imp_high,
        "win_rate": float(np.mean(improvements > 0.0)) if improvements.size else float("nan"),
        "sign_p_value": sign_test_p_value(improvements),
    }


def claim_status(
    stats: dict[str, Any],
    *,
    min_pairs: int = 3,
    require_ci: bool = False,
) -> str:
    n_common = int(stats.get("n_common", 0) or 0)
    if n_common < int(min_pairs):
        return "underpowered"
    low = finite_float(stats.get("improvement_ci95_low"))
    mean = finite_float(stats.get("improvement_mean"))
    win_rate = finite_float(stats.get("win_rate"))
    if low is not None and low > 0.0:
        return "supported"
    if mean is not None and mean > 0.0 and (win_rate or 0.0) >= 0.50:
        return "positive_mixed" if not require_ci else "inconclusive"
    if mean is not None and mean <= 0.0:
        return "not_supported"
    return "inconclusive"


def format_ci(stats: dict[str, Any], digits: int = 4) -> str:
    mean = finite_float(stats.get("delta_mean"))
    low = finite_float(stats.get("delta_ci95_low"))
    high = finite_float(stats.get("delta_ci95_high"))
    if mean is None or low is None or high is None:
        return "NA"
    return f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"
