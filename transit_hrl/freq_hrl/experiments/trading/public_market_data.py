"""Public/CSV market-data evaluation path for Freq-HRL trading policies."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.domains.trading import (
    PortfolioExecutionConfig,
    PortfolioExecutionEnv,
    TradingFrequencyTracker,
)
from freq_hrl.policies import FrequencyTradingController, FrequencyTradingPlanner

from .performance_validation import max_drawdown


def _read_price_csv(path: Path, close_col: str = "Close") -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"empty price CSV: {path}")
    if close_col not in rows[0]:
        lower = {key.lower(): key for key in rows[0]}
        if close_col.lower() in lower:
            close_col = lower[close_col.lower()]
        else:
            raise ValueError(f"CSV {path} missing close column {close_col!r}")
    out = []
    for row in rows:
        date = row.get("Date") or row.get("date") or str(len(out))
        try:
            close = float(row[close_col])
        except (TypeError, ValueError):
            continue
        if math.isfinite(close) and close > 0.0:
            out.append({"date": date, "close": close})
    if len(out) < 3:
        raise ValueError(f"not enough valid closes in {path}")
    return out


def _download_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _symbol_to_yahoo(symbol: str) -> str:
    # Stooq's US ETF symbols are commonly written as spy.us; Yahoo uses SPY.
    if symbol.lower().endswith(".us"):
        return symbol[:-3].upper()
    return symbol.upper()


def _fetch_yahoo(symbol: str) -> list[dict[str, Any]]:
    ticker = _symbol_to_yahoo(symbol)
    encoded = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        "?range=10y&interval=1d&events=history&includeAdjustedClose=true"
    )
    payload = _download_text(url)
    data = json.loads(payload)
    result = ((data.get("chart") or {}).get("result") or [None])[0]
    if not result:
        raise ValueError(f"unexpected Yahoo response for {symbol}")
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or [{}]
    closes = (quotes[0] or {}).get("close") or []
    adjcloses = ((indicators.get("adjclose") or [{}])[0] or {}).get("adjclose") or []
    prices = adjcloses if len(adjcloses) == len(timestamps) else closes
    out = []
    for ts, close in zip(timestamps, prices):
        try:
            close_f = float(close)
        except (TypeError, ValueError):
            continue
        if math.isfinite(close_f) and close_f > 0.0:
            date = datetime.fromtimestamp(float(ts), tz=timezone.utc).date().isoformat()
            out.append({"date": date, "close": close_f})
    if len(out) < 3:
        raise ValueError(f"not enough Yahoo rows for {symbol}")
    return out


def _fetch_stooq(
    symbol: str,
    apikey: str | None = None,
    *,
    yahoo_fallback: bool = True,
) -> list[dict[str, Any]]:
    query = {"s": symbol.lower(), "i": "d"}
    if apikey:
        query["apikey"] = apikey
    url = f"https://stooq.com/q/d/l/?{urllib.parse.urlencode(query)}"
    payload = _download_text(url)
    rows = [dict(row) for row in csv.DictReader(io.StringIO(payload))]
    if not rows or "Close" not in rows[0]:
        if yahoo_fallback and "apikey" in payload.lower():
            print(f"Stooq requires an API key for {symbol}; falling back to Yahoo chart data.")
            return _fetch_yahoo(symbol)
        first_line = payload.splitlines()[0] if payload.splitlines() else "<empty>"
        raise ValueError(f"unexpected Stooq response for {symbol}: {first_line[:120]}")
    out = []
    for row in rows:
        try:
            close = float(row["Close"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(close) and close > 0.0:
            out.append({"date": row.get("Date", str(len(out))), "close": close})
    if len(out) < 3:
        raise ValueError(f"not enough Stooq rows for {symbol}")
    return out


def align_prices(series: list[tuple[str, list[dict[str, Any]]]]) -> tuple[list[str], np.ndarray]:
    maps = [{row["date"]: float(row["close"]) for row in rows} for _, rows in series]
    dates = sorted(set.intersection(*[set(m) for m in maps]))
    if len(dates) < 3:
        raise ValueError("not enough overlapping dates across market series")
    prices = np.asarray([[m[date] for m in maps] for date in dates], dtype=np.float64)
    return dates, prices


def price_returns(prices: np.ndarray) -> np.ndarray:
    logp = np.log(np.maximum(prices, 1e-12))
    return np.diff(logp, axis=0)


def run_dataset_eval(
    returns: np.ndarray,
    steps: int | None = None,
    freq_method: str = "ema",
) -> dict[str, Any]:
    if steps is not None:
        returns = returns[-int(steps):]
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    assets = returns.shape[1]
    predictor = np.zeros_like(returns)
    predictor[1:] = returns[:-1]
    volume_proxy = 1.0 + 100.0 * np.abs(returns)
    env = PortfolioExecutionEnv(
        returns,
        volumes=volume_proxy,
        config=PortfolioExecutionConfig(
            transaction_cost_bps=5.0,
            slippage_bps=1.0,
            max_leverage=1.0,
            inventory_drift_penalty=0.002,
        ),
    )
    tracker = TradingFrequencyTracker(
        bar_sec=24 * 3600.0,
        method=freq_method,
        low_period_s=90 * 24 * 3600.0,
        fast_period_s=5 * 24 * 3600.0,
        mid_period_s=30 * 24 * 3600.0,
        energy_period_s=10 * 24 * 3600.0,
        persistence_period_s=30 * 24 * 3600.0,
        persistence_threshold=0.010,
        feature_norm=np.ones(assets) * 0.015,
        promotion_enable=True,
        promotion_window_s=30 * 24 * 3600.0,
        promotion_residual_threshold=0.006,
        promotion_persistence_ratio=0.40,
        promotion_cooldown_s=60 * 24 * 3600.0,
        promotion_adapt_low=True,
        promotion_adapt_gain=0.25,
    )
    planner = FrequencyTradingPlanner(promotion_mid_gain=0.5)
    controller = FrequencyTradingController()
    env.reset()
    pnl_returns = []
    equity = []
    turnover = []
    promotions = 0
    for t in range(returns.shape[0]):
        freq = tracker.update_bar(predictor[t], t=float(t * 24 * 3600.0))
        promotions += 1 if dict(freq.get("promotion", {}) or {}).get("promote", False) else 0
        obs = {"raw_signal": predictor[t], "position": env.position.copy(), "t": t}
        upper = planner.plan(obs, tracker.upper_features(), context={"frequency": freq, "n_assets": assets})
        env.set_target(upper.action)
        lower = controller.act(obs, tracker.lower_features(upper.action, env.position), upper, context={"frequency": freq})
        _, _, done, info = env.lower_step(lower.action)
        pnl_returns.append(float(info["portfolio_return"] - info["transaction_cost"]))
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        if done:
            break
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    return {
        "bars": int(returns.shape[0]),
        "assets": int(assets),
        "freq_method": str(freq_method),
        "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "sharpe": float(np.sqrt(252.0) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0,
        "max_drawdown": max_drawdown(eq),
        "turnover": float(np.sum(turnover)),
        "promotion_count": int(promotions),
    }


def write_report(path: Path, symbols: list[str], summary: dict[str, Any], source: str) -> None:
    lines = [
        "# Public Market Data Evaluation",
        "",
        f"- source: `{source}`",
        f"- symbols: {symbols}",
        f"- frequency encoder: `{summary.get('freq_method', 'ema')}`",
        "- predictor: previous-bar log return only, so current/future returns are not used as policy input",
        f"- bars: {summary['bars']}",
        f"- total return: {summary['total_return']:.4f}",
        f"- annualized Sharpe: {summary['sharpe']:.3f}",
        f"- max drawdown: {summary['max_drawdown']:.4f}",
        f"- turnover: {summary['turnover']:.2f}",
        f"- promotions: {summary['promotion_count']}",
        "",
        "This is a public-data validation path for the Freq-HRL protocol. It is not investment advice and is not a production trading simulator.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["csv", "stooq", "yahoo"], default="csv")
    parser.add_argument("--csv-files", type=Path, nargs="*", default=[])
    parser.add_argument("--symbols", nargs="*", default=["spy.us", "qqq.us", "iwm.us"])
    parser.add_argument("--close-col", default="Close")
    parser.add_argument("--stooq-apikey", default=os.environ.get("STOOQ_APIKEY", ""))
    parser.add_argument("--no-stooq-yahoo-fallback", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--freq-method", choices=["ema", "state_space", "haar_wavelet"], default="ema")
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_public_market"))
    args = parser.parse_args()

    if args.source == "csv":
        if not args.csv_files:
            raise ValueError("--csv-files is required when --source csv")
        series = [
            (path.stem, _read_price_csv(path, close_col=args.close_col))
            for path in args.csv_files
        ]
    elif args.source == "yahoo":
        series = [(symbol, _fetch_yahoo(symbol)) for symbol in args.symbols]
    else:
        series = [
            (
                symbol,
                _fetch_stooq(
                    symbol,
                    apikey=args.stooq_apikey or None,
                    yahoo_fallback=not args.no_stooq_yahoo_fallback,
                ),
            )
            for symbol in args.symbols
        ]
    symbols = [name for name, _ in series]
    dates, prices = align_prices(series)
    returns = price_returns(prices)
    summary = run_dataset_eval(returns, steps=args.steps, freq_method=args.freq_method)
    summary.update({
        "source": args.source,
        "symbols": symbols,
        "first_date": dates[1],
        "last_date": dates[-1],
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(args.output_dir / "report.md", symbols, summary, args.source)
    print(f"wrote {args.output_dir}")
    print(f"public_market sharpe={summary['sharpe']:.3f} return={summary['total_return']:.4f}")


if __name__ == "__main__":
    main()
