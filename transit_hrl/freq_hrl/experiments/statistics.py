"""Statistical primitives for paired, independently clustered claim gates."""

from __future__ import annotations

from math import exp, lgamma, log
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
    log_half_n = -float(n) * log(2.0)
    log_terms = [
        lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0) + log_half_n
        for k in range(0, tail + 1)
    ]
    max_log = max(log_terms)
    prob = exp(max_log) * sum(exp(term - max_log) for term in log_terms)
    return float(min(1.0, 2.0 * prob))


def _row_key(row: dict[str, Any], key_fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in key_fields)


def holm_adjusted_p_values(values: Iterable[Any]) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values in the original order."""

    arr = finite_array(values).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, 1.0)
    order = np.argsort(arr, kind="stable")
    adjusted = np.empty(arr.size, dtype=np.float64)
    running = 0.0
    m = int(arr.size)
    for rank, index in enumerate(order):
        running = max(running, float((m - rank) * arr[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def apply_holm_correction(
    rows: list[dict[str, Any]],
    *,
    family_key: str = "multiplicity_family",
    p_key: str = "sign_p_value",
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Annotate checks with family-wise Holm-Bonferroni decisions."""

    out = [dict(row) for row in rows]
    families: dict[str, list[int]] = {}
    for index, row in enumerate(out):
        family = str(row.get(family_key) or row.get("claim") or "global")
        row[family_key] = family
        if finite_float(row.get(p_key)) is not None:
            families.setdefault(family, []).append(index)
    for family, indices in families.items():
        adjusted = holm_adjusted_p_values(out[index].get(p_key) for index in indices)
        family_size = len(indices)
        for index, value in zip(indices, adjusted):
            out[index]["holm_adjusted_p_value"] = float(value)
            out[index]["multiplicity_family_size"] = int(family_size)
            out[index]["holm_reject"] = bool(float(value) <= float(alpha))
    for row in out:
        row.setdefault("holm_adjusted_p_value", float("nan"))
        row.setdefault("multiplicity_family_size", 0)
        row.setdefault("holm_reject", False)
    return out


def paired_delta_stats(
    rows: list[dict[str, Any]],
    variant_key: str,
    pair_keys: tuple[str, ...],
    metric: str,
    treatment: str,
    control: str,
    lower_is_better: bool = False,
    cluster_keys: tuple[str, ...] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return paired treatment-control deltas for a metric.

    Raw delta is `treatment - control`. Improvement is `-delta` when lower is
    better, otherwise `delta`.
    """

    indexed: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    duplicate_keys: list[tuple[str, tuple[Any, ...]]] = []
    for row in rows:
        variant = row.get(variant_key)
        value = finite_float(row.get(metric))
        if variant is None or value is None:
            continue
        index_key = (str(variant), _row_key(row, pair_keys))
        if index_key in indexed:
            duplicate_keys.append(index_key)
            continue
        indexed[index_key] = row
    if duplicate_keys:
        preview = ", ".join(repr(key) for key in duplicate_keys[:3])
        raise ValueError(
            "paired_delta_stats requires one row per variant/pair key; "
            f"found {len(duplicate_keys)} duplicate rows, including {preview}"
        )

    treatment_keys = {
        pair for (variant, pair) in indexed
        if variant == str(treatment)
    }
    control_keys = {
        pair for (variant, pair) in indexed
        if variant == str(control)
    }
    common = sorted(treatment_keys & control_keys, key=repr)
    # Reusing the same RNG seed across sources/scenarios creates repeated
    # measures, not new independent replications. Callers can override this
    # convention for studies with a different registered sampling unit.
    if cluster_keys is None and "seed" in pair_keys:
        cluster_fields = ("seed",)
    else:
        cluster_fields = tuple(pair_keys if cluster_keys is None else cluster_keys)
    deltas = []
    cluster_deltas: dict[tuple[Any, ...], list[float]] = {}
    for pair in common:
        treatment_row = indexed[(str(treatment), pair)]
        control_row = indexed[(str(control), pair)]
        t_val = finite_float(treatment_row.get(metric))
        c_val = finite_float(control_row.get(metric))
        if t_val is None or c_val is None:
            continue
        treatment_cluster = _row_key(treatment_row, cluster_fields)
        control_cluster = _row_key(control_row, cluster_fields)
        if treatment_cluster != control_cluster:
            raise ValueError(
                "treatment and control rows disagree on independent cluster: "
                f"{treatment_cluster!r} != {control_cluster!r}"
            )
        delta = float(t_val - c_val)
        deltas.append(delta)
        cluster_deltas.setdefault(treatment_cluster, []).append(delta)
    pair_delta_arr = finite_array(deltas)
    cluster_delta_arr = finite_array(
        np.mean(values) for values in cluster_deltas.values()
    )
    improvements = -cluster_delta_arr if lower_is_better else cluster_delta_arr
    ci_low, ci_high = bootstrap_mean_ci(cluster_delta_arr, n_boot=n_boot, seed=seed)
    if lower_is_better:
        imp_low, imp_high = -ci_high, -ci_low
    else:
        imp_low, imp_high = ci_low, ci_high
    delta_mean = (
        float(cluster_delta_arr.mean()) if cluster_delta_arr.size else float("nan")
    )
    improvement_mean = -delta_mean if lower_is_better else delta_mean
    sample_std = (
        float(cluster_delta_arr.std(ddof=1)) if cluster_delta_arr.size > 1 else float("nan")
    )
    standard_error = (
        sample_std / float(np.sqrt(cluster_delta_arr.size))
        if cluster_delta_arr.size > 1 else float("nan")
    )
    effect_size = (
        improvement_mean / sample_std
        if np.isfinite(sample_std) and sample_std > 1e-12 else float("nan")
    )
    cluster_sizes = [len(values) for values in cluster_deltas.values()]
    return {
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "direction": "decrease" if lower_is_better else "increase",
        "pair_keys": list(pair_keys),
        "cluster_keys": list(cluster_fields),
        "estimand": "equal_weight_mean_of_independent_cluster_deltas",
        "n_common": int(pair_delta_arr.size),
        "n_independent": int(cluster_delta_arr.size),
        "n_clusters": int(cluster_delta_arr.size),
        "cluster_size_min": int(min(cluster_sizes)) if cluster_sizes else 0,
        "cluster_size_max": int(max(cluster_sizes)) if cluster_sizes else 0,
        "delta_mean": delta_mean,
        "pair_weighted_delta_mean": (
            float(pair_delta_arr.mean()) if pair_delta_arr.size else float("nan")
        ),
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "delta_standard_error": standard_error,
        "improvement_mean": improvement_mean,
        "improvement_ci95_low": imp_low,
        "improvement_ci95_high": imp_high,
        "paired_effect_size_dz": effect_size,
        "win_rate": float(np.mean(improvements > 0.0)) if improvements.size else float("nan"),
        "sign_p_value": sign_test_p_value(improvements),
        "statistical_contract": "paired_cluster_bootstrap_v2",
    }


def _independent_count(stats: dict[str, Any]) -> int:
    return int(stats.get("n_independent", stats.get("n_clusters", stats.get("n_common", 0))) or 0)


def claim_status(
    stats: dict[str, Any],
    *,
    min_pairs: int = 3,
    require_ci: bool = False,
) -> str:
    if _independent_count(stats) < int(min_pairs):
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


def noninferiority_status(
    stats: dict[str, Any],
    *,
    max_loss: float = 0.0,
    min_pairs: int = 3,
) -> str:
    if _independent_count(stats) < int(min_pairs):
        return "underpowered"
    low = finite_float(stats.get("improvement_ci95_low"))
    high = finite_float(stats.get("improvement_ci95_high"))
    mean = finite_float(stats.get("improvement_mean"))
    margin = abs(float(max_loss))
    if low is not None and low >= -margin:
        return "supported"
    if high is not None and high < -margin:
        return "not_supported"
    if mean is not None and mean >= -margin:
        return "positive_mixed"
    return "inconclusive"


def format_ci(stats: dict[str, Any], digits: int = 4) -> str:
    mean = finite_float(stats.get("delta_mean"))
    low = finite_float(stats.get("delta_ci95_low"))
    high = finite_float(stats.get("delta_ci95_high"))
    if mean is None or low is None or high is None:
        return "NA"
    return f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"
