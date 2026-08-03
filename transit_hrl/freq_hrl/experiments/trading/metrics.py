"""Auditable financial metrics shared by every trading experiment."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


TRADING_DAYS_PER_YEAR = 252.0
TRADING_HOURS_PER_DAY = 6.5
METRIC_CONTRACT_VERSION = "trading_metrics_v2"
SELECTION_OBJECTIVE_VERSION = "log_growth_drawdown_utility_v3"
DEFAULT_SELECTION_DRAWDOWN_WEIGHT = 0.25
DEFAULT_TRAINING_REWARD_SCALE = 100.0


def validation_utility(
    row: dict[str, Any],
    *,
    drawdown_weight: float = DEFAULT_SELECTION_DRAWDOWN_WEIGHT,
) -> float:
    """Return the preregistered checkpoint/HPO selection utility.

    Net return already includes transaction costs, so turnover is not charged a
    second time. Episode information ratios remain reportable endpoints, but
    are excluded from selection because they are noisy on short episodes.
    """

    total_return = float(row["total_return"])
    max_drawdown_value = float(row["max_drawdown"])
    weight = float(drawdown_weight)
    if not np.isfinite(total_return) or total_return <= -1.0:
        raise ValueError("total_return must be finite and greater than -1")
    if not np.isfinite(max_drawdown_value) or max_drawdown_value < 0.0:
        raise ValueError("max_drawdown must be finite and non-negative")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("drawdown_weight must be finite and non-negative")
    return float(np.log1p(total_return) - weight * max_drawdown_value)


def periods_per_year_from_bar_seconds(
    bar_seconds: float,
    *,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
    trading_hours_per_day: float = TRADING_HOURS_PER_DAY,
) -> float:
    bar_seconds = float(bar_seconds)
    if not np.isfinite(bar_seconds) or bar_seconds <= 0.0:
        raise ValueError("bar_seconds must be positive and finite")
    if bar_seconds >= 12.0 * 3600.0:
        return float(trading_days_per_year)
    return float(trading_days_per_year * trading_hours_per_day * 3600.0 / bar_seconds)


def max_drawdown(equity: Iterable[float], *, initial_equity: float = 1.0) -> float:
    values = np.asarray(list(equity), dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    if not np.all(np.isfinite(values)):
        raise ValueError("equity contains non-finite values")
    path = np.concatenate([[float(initial_equity)], values])
    peaks = np.maximum.accumulate(path)
    drawdown = 1.0 - path / np.maximum(peaks, 1e-12)
    return float(np.max(drawdown))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 1e-12:
        return float(numerator / denominator)
    if abs(numerator) <= 1e-12:
        return 0.0
    return float("nan")


def summarize_pnl_series(
    values: Iterable[float],
    equity: Iterable[float],
    *,
    periods_per_year: float,
    initial_equity: float = 1.0,
    risk_free_rate_annual: float = 0.0,
    compounding: str = "multiplicative",
    min_annualization_years: float = 0.25,
) -> dict[str, Any]:
    """Summarize equally spaced net returns or normalized PnL increments.

    ``compounding='multiplicative'`` is for simple net returns. ``additive`` is
    for PnL increments normalized by a fixed initial notional, as used by the
    order-book replay adapters.
    """

    series = np.asarray(list(values), dtype=np.float64).reshape(-1)
    equity_arr = np.asarray(list(equity), dtype=np.float64).reshape(-1)
    if series.size != equity_arr.size:
        raise ValueError("PnL series and equity path must have the same length")
    if not np.all(np.isfinite(series)) or not np.all(np.isfinite(equity_arr)):
        raise ValueError("PnL series and equity path must be finite")
    periods = float(periods_per_year)
    if not np.isfinite(periods) or periods <= 0.0:
        raise ValueError("periods_per_year must be positive and finite")
    if compounding not in {"multiplicative", "additive"}:
        raise ValueError("compounding must be 'multiplicative' or 'additive'")
    if compounding == "multiplicative" and np.any(series < -1.0):
        raise ValueError("simple returns below -100% are invalid")

    n = int(series.size)
    mean = float(series.mean()) if n else 0.0
    sample_std = float(series.std(ddof=1)) if n > 1 else float("nan")
    risk_free_per_period = float(
        (1.0 + float(risk_free_rate_annual)) ** (1.0 / periods) - 1.0
    )
    excess_mean = mean - risk_free_per_period
    mean_over_std = _ratio(excess_mean, sample_std)
    annualized_sharpe = (
        float(np.sqrt(periods) * mean_over_std)
        if np.isfinite(mean_over_std) else float("nan")
    )
    episode_information_ratio = (
        float(np.sqrt(n) * mean_over_std)
        if n and np.isfinite(mean_over_std) else float("nan")
    )
    excess = series - risk_free_per_period
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(np.minimum(excess, 0.0)))))
        if n else float("nan")
    )
    annualized_sortino = _ratio(
        float(np.sqrt(periods) * excess_mean),
        downside_deviation,
    )

    if n:
        total_return = float(equity_arr[-1] / float(initial_equity) - 1.0)
    else:
        total_return = 0.0
    horizon_years = float(n / periods) if n else 0.0
    annualization_reliable = bool(
        horizon_years >= max(float(min_annualization_years), 0.0)
    )
    arithmetic_annualized_return = float(periods * mean) if n else 0.0
    if compounding == "multiplicative":
        reconstructed = float(initial_equity) * np.cumprod(1.0 + series)
        annualized_return_raw = (
            float((equity_arr[-1] / float(initial_equity)) ** (periods / n) - 1.0)
            if n and equity_arr[-1] > 0.0 else float("nan")
        )
    else:
        reconstructed = float(initial_equity) + np.cumsum(series)
        annualized_return_raw = arithmetic_annualized_return
    annualized_return = (
        annualized_return_raw if annualization_reliable else float("nan")
    )
    reconstruction_error = (
        float(np.max(np.abs(reconstructed - equity_arr))) if n else 0.0
    )
    drawdown = max_drawdown(equity_arr, initial_equity=float(initial_equity))
    calmar = _ratio(annualized_return, drawdown)
    episode_return_to_drawdown = _ratio(total_return, drawdown)
    sharpe_defined = bool(np.isfinite(annualized_sharpe))
    return {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "return_series_kind": (
            "net_simple_return" if compounding == "multiplicative"
            else "fixed_notional_normalized_pnl_increment"
        ),
        "return_observations": n,
        "periods_per_year": periods,
        "observation_horizon_years": horizon_years,
        "min_annualization_years": float(min_annualization_years),
        "annualization_reliable": annualization_reliable,
        "risk_free_rate_annual": float(risk_free_rate_annual),
        "return_mean": mean,
        "return_sample_std": sample_std,
        "episode_information_ratio": episode_information_ratio,
        "annualized_sharpe": annualized_sharpe,
        "annualized_sortino": annualized_sortino,
        "annualized_return": annualized_return,
        "arithmetic_annualized_return": arithmetic_annualized_return,
        "total_return": total_return,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "episode_return_to_drawdown": episode_return_to_drawdown,
        "equity_reconstruction_max_abs_error": reconstruction_error,
        "sharpe_defined": sharpe_defined,
        # Backward-compatible names now have explicit v2 definitions.
        "sharpe": annualized_sharpe,
        "sortino": annualized_sortino,
    }
