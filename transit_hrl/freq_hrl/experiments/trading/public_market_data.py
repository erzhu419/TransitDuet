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
        date = (
            row.get("Datetime")
            or row.get("datetime")
            or row.get("Timestamp")
            or row.get("timestamp")
            or row.get("Time")
            or row.get("time")
            or row.get("Date")
            or row.get("date")
            or str(len(out))
        )
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


def _fetch_yahoo(symbol: str, range_: str = "10y", interval: str = "1d") -> list[dict[str, Any]]:
    ticker = _symbol_to_yahoo(symbol)
    encoded = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?range={urllib.parse.quote(str(range_), safe='')}"
        f"&interval={urllib.parse.quote(str(interval), safe='')}"
        "&events=history&includeAdjustedClose=true"
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
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            date = dt.date().isoformat() if interval == "1d" else dt.isoformat()
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
    bar_sec: float = 24 * 3600.0,
) -> dict[str, Any]:
    if steps is not None:
        returns = returns[-int(steps):]
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    assets = returns.shape[1]
    predictor = np.zeros_like(returns)
    predictor[1:] = returns[:-1]
    volume_proxy = 1.0 + 100.0 * np.abs(returns)
    bar_sec = max(float(bar_sec), 1.0)
    periods_per_year = 252.0 if bar_sec >= 12 * 3600.0 else 252.0 * 6.5 * 3600.0 / bar_sec
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
        bar_sec=bar_sec,
        method=freq_method,
        low_period_s=90 * bar_sec,
        fast_period_s=5 * bar_sec,
        mid_period_s=30 * bar_sec,
        energy_period_s=10 * bar_sec,
        persistence_period_s=30 * bar_sec,
        persistence_threshold=0.010,
        feature_norm=np.ones(assets) * 0.015,
        promotion_enable=True,
        promotion_window_s=30 * bar_sec,
        promotion_residual_threshold=0.006,
        promotion_persistence_ratio=0.40,
        promotion_cooldown_s=60 * bar_sec,
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
        freq = tracker.update_bar(predictor[t], t=float(t * bar_sec))
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
        "bar_sec": float(bar_sec),
        "freq_method": str(freq_method),
        "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "sharpe": float(np.sqrt(periods_per_year) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0,
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
        f"- bar seconds: {summary.get('bar_sec', 24 * 3600.0)}",
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
    parser.add_argument("--source", choices=["csv", "stooq", "yahoo", "yahoo_intraday"], default="csv")
    parser.add_argument("--csv-files", type=Path, nargs="*", default=[])
    parser.add_argument("--symbols", nargs="*", default=["spy.us", "qqq.us", "iwm.us"])
    parser.add_argument("--close-col", default="Close")
    parser.add_argument("--stooq-apikey", default=os.environ.get("STOOQ_APIKEY", ""))
    parser.add_argument("--no-stooq-yahoo-fallback", action="store_true")
    parser.add_argument("--yahoo-range", default="10y")
    parser.add_argument("--yahoo-interval", default="1d")
    parser.add_argument("--bar-sec", type=float, default=24 * 3600.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--freq-method",
        choices=["ema", "state_space", "haar_wavelet", "adaptive_wavelet", "neural_state_space"],
        default="ema",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_public_market"))
    args = parser.parse_args()

    if args.source == "csv":
        if not args.csv_files:
            raise ValueError("--csv-files is required when --source csv")
        series = [
            (path.stem, _read_price_csv(path, close_col=args.close_col))
            for path in args.csv_files
        ]
    elif args.source in {"yahoo", "yahoo_intraday"}:
        range_ = args.yahoo_range if args.source == "yahoo_intraday" else "10y"
        interval = args.yahoo_interval if args.source == "yahoo_intraday" else "1d"
        series = [(symbol, _fetch_yahoo(symbol, range_=range_, interval=interval)) for symbol in args.symbols]
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
    summary = run_dataset_eval(returns, steps=args.steps, freq_method=args.freq_method, bar_sec=args.bar_sec)
    summary.update({
        "source": args.source,
        "symbols": symbols,
        "first_date": dates[1],
        "last_date": dates[-1],
        "yahoo_range": args.yahoo_range if args.source == "yahoo_intraday" else "",
        "yahoo_interval": args.yahoo_interval if args.source == "yahoo_intraday" else "",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(args.output_dir / "report.md", symbols, summary, args.source)
    print(f"wrote {args.output_dir}")
    print(f"public_market sharpe={summary['sharpe']:.3f} return={summary['total_return']:.4f}")


if __name__ == "__main__":
    main()
