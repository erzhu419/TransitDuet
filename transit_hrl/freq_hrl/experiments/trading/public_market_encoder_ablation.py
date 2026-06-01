"""Run causal encoder ablations on public/CSV market data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .public_market_data import (
    _fetch_stooq,
    _fetch_yahoo,
    _read_price_csv,
    align_prices,
    price_returns,
    run_dataset_eval,
)


PUBLIC_ENCODERS = ("ema", "state_space", "haar_wavelet", "adaptive_wavelet", "neural_state_space")


def load_series(args: argparse.Namespace) -> tuple[list[str], list[str], np.ndarray]:
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
    return symbols, dates, price_returns(prices)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    source: str,
    symbols: list[str],
    first_date: str,
    last_date: str,
) -> None:
    best_sharpe = max(rows, key=lambda row: float(row["sharpe"]))
    best_return = max(rows, key=lambda row: float(row["total_return"]))
    lines = [
        "# Public Market Encoder Ablation",
        "",
        f"- source: `{source}`",
        f"- symbols: {symbols}",
        f"- date range: {first_date} through {last_date}",
        f"- bar seconds: {rows[0].get('bar_sec', 24 * 3600.0)}",
        "- predictor: previous-bar log return only",
        f"- best Sharpe encoder: `{best_sharpe['freq_method']}` ({best_sharpe['sharpe']:.3f})",
        f"- best return encoder: `{best_return['freq_method']}` ({best_return['total_return']:.4f})",
        "",
        "| encoder | bars | return | Sharpe | max DD | turnover | promotions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['freq_method']} "
            f"| {int(row['bars'])} "
            f"| {row['total_return']:.4f} "
            f"| {row['sharpe']:.3f} "
            f"| {row['max_drawdown']:.4f} "
            f"| {row['turnover']:.2f} "
            f"| {int(row['promotion_count'])} |"
        )
    lines.extend([
        "",
        "This evaluates causal encoders on public market data. It is not investment advice and is not a production trading simulator.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["csv", "stooq", "yahoo", "yahoo_intraday"], default="csv")
    parser.add_argument("--csv-files", type=Path, nargs="*", default=[])
    parser.add_argument("--symbols", nargs="*", default=["spy.us", "qqq.us", "iwm.us"])
    parser.add_argument("--close-col", default="Close")
    parser.add_argument("--stooq-apikey", default="")
    parser.add_argument("--no-stooq-yahoo-fallback", action="store_true")
    parser.add_argument("--yahoo-range", default="10y")
    parser.add_argument("--yahoo-interval", default="1d")
    parser.add_argument("--bar-sec", type=float, default=24 * 3600.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--methods", nargs="+", choices=PUBLIC_ENCODERS, default=list(PUBLIC_ENCODERS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/trading_public_market_encoder_ablation"),
    )
    args = parser.parse_args()

    symbols, dates, returns = load_series(args)
    rows = []
    for method in args.methods:
        row = run_dataset_eval(returns, steps=args.steps, freq_method=method, bar_sec=args.bar_sec)
        row.update({
            "source": args.source,
            "symbols": symbols,
            "first_date": dates[1],
            "last_date": dates[-1],
            "yahoo_range": args.yahoo_range if args.source == "yahoo_intraday" else "",
            "yahoo_interval": args.yahoo_interval if args.source == "yahoo_intraday" else "",
        })
        rows.append(row)
    rows.sort(key=lambda row: list(PUBLIC_ENCODERS).index(row["freq_method"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": rows}, f, indent=2)
    write_report(
        args.output_dir / "report.md",
        rows,
        source=args.source,
        symbols=symbols,
        first_date=dates[1],
        last_date=dates[-1],
    )
    best = max(rows, key=lambda row: float(row["sharpe"]))
    print(f"wrote {args.output_dir}")
    print(f"public_encoder best={best['freq_method']} sharpe={best['sharpe']:.3f} return={best['total_return']:.4f}")


if __name__ == "__main__":
    main()
