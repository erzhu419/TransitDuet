"""L2 order-book matching validation for Freq-HRL trading encoders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.domains.trading import TradingFrequencyTracker
from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.experiments.trading.order_book_data import ORDER_BOOK_ENCODERS
from freq_hrl.experiments.trading.performance_validation import max_drawdown


def _float(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    lower = {key.lower(): key for key in row}
    for name in names:
        key = lower.get(name.lower())
        if key is None:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return float(default)


def read_l2_order_book_csv(path: Path, *, levels: int = 5) -> list[dict[str, Any]]:
    """Read a multi-level order-book CSV into the matching simulator format."""
    with path.open("r", newline="", encoding="utf-8") as f:
        raw_rows = [dict(row) for row in csv.DictReader(f)]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        bid_prices = []
        ask_prices = []
        bid_sizes = []
        ask_sizes = []
        for level in range(1, max(1, int(levels)) + 1):
            bid = _float(
                row,
                f"bid_price_{level}",
                f"bid_px_{level}",
                f"bid{level}",
                f"bid_price{level}",
                default=float("nan"),
            )
            ask = _float(
                row,
                f"ask_price_{level}",
                f"ask_px_{level}",
                f"ask{level}",
                f"ask_price{level}",
                default=float("nan"),
            )
            bid_size = _float(
                row,
                f"bid_size_{level}",
                f"bid_qty_{level}",
                f"bid_volume_{level}",
                f"bid_size{level}",
                default=float("nan"),
            )
            ask_size = _float(
                row,
                f"ask_size_{level}",
                f"ask_qty_{level}",
                f"ask_volume_{level}",
                f"ask_size{level}",
                default=float("nan"),
            )
            if not all(np.isfinite([bid, ask, bid_size, ask_size])):
                continue
            if bid <= 0.0 or ask <= 0.0 or ask < bid:
                continue
            bid_prices.append(float(bid))
            ask_prices.append(float(ask))
            bid_sizes.append(float(max(bid_size, 1e-9)))
            ask_sizes.append(float(max(ask_size, 1e-9)))
        if not bid_prices:
            bid = _float(row, "bid", "best_bid", "bid_price", "bid_price_1", default=float("nan"))
            ask = _float(row, "ask", "best_ask", "ask_price", "ask_price_1", default=float("nan"))
            bid_size = _float(row, "bid_size", "best_bid_size", "bid_size_1", default=1.0)
            ask_size = _float(row, "ask_size", "best_ask_size", "ask_size_1", default=1.0)
            if bid > 0.0 and ask >= bid:
                bid_prices = [float(bid)]
                ask_prices = [float(ask)]
                bid_sizes = [float(max(bid_size, 1e-9))]
                ask_sizes = [float(max(ask_size, 1e-9))]
        if bid_prices:
            rows.append({
                "timestamp": _float(row, "timestamp", "time", "ts", default=float(idx)),
                "bid_prices": bid_prices,
                "ask_prices": ask_prices,
                "bid_sizes": bid_sizes,
                "ask_sizes": ask_sizes,
            })
    if len(rows) < 4:
        raise ValueError(f"not enough valid L2 order-book rows in {path}")
    return rows


def make_synthetic_l2_order_book(seed: int = 7, steps: int = 720, levels: int = 5) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    price = 100.0
    rows: list[dict[str, Any]] = []
    prev_imbalance = 0.0
    for t in range(max(4, int(steps))):
        imbalance = float(np.clip(0.50 * np.sin(2.0 * np.pi * t / 80.0) + rng.normal(0.0, 0.16), -0.85, 0.85))
        drift = 0.000015 * np.sin(2.0 * np.pi * t / 240.0) + 0.00012 * prev_imbalance
        price *= float(np.exp(drift + rng.normal(0.0, 0.00032)))
        spread = 0.006 + 0.004 * (1.0 + np.sin(2.0 * np.pi * t / 50.0))
        tick = max(0.002, spread * 0.50)
        base_depth = 650.0 + 200.0 * np.cos(2.0 * np.pi * t / 130.0)
        bid_prices = []
        ask_prices = []
        bid_sizes = []
        ask_sizes = []
        for level in range(max(1, int(levels))):
            depth_decay = 1.0 / (1.0 + 0.28 * level)
            bid_prices.append(float(price - 0.5 * spread - level * tick))
            ask_prices.append(float(price + 0.5 * spread + level * tick))
            bid_sizes.append(float(max(base_depth * depth_decay * (1.0 + imbalance) + rng.normal(0.0, 12.0), 1.0)))
            ask_sizes.append(float(max(base_depth * depth_decay * (1.0 - imbalance) + rng.normal(0.0, 12.0), 1.0)))
        rows.append({
            "timestamp": float(t),
            "bid_prices": bid_prices,
            "ask_prices": ask_prices,
            "bid_sizes": bid_sizes,
            "ask_sizes": ask_sizes,
        })
        prev_imbalance = imbalance
    return rows


def _book_mid(row: dict[str, Any]) -> float:
    return 0.5 * (float(row["bid_prices"][0]) + float(row["ask_prices"][0]))


def _book_imbalance(row: dict[str, Any]) -> float:
    bid = float(row["bid_sizes"][0])
    ask = float(row["ask_sizes"][0])
    return (bid - ask) / max(bid + ask, 1e-12)


def fill_market_order(row: dict[str, Any], signed_qty: float) -> dict[str, float]:
    qty = abs(float(signed_qty))
    if qty <= 1e-12:
        return {"filled": 0.0, "avg_price": _book_mid(row), "slippage_bps": 0.0, "levels_used": 0.0}
    side_buy = signed_qty > 0.0
    prices = row["ask_prices"] if side_buy else row["bid_prices"]
    sizes = row["ask_sizes"] if side_buy else row["bid_sizes"]
    remaining = qty
    notional = 0.0
    filled = 0.0
    levels_used = 0
    for price, size in zip(prices, sizes):
        take = min(remaining, max(float(size), 0.0))
        if take <= 0.0:
            continue
        notional += take * float(price)
        filled += take
        remaining -= take
        levels_used += 1
        if remaining <= 1e-12:
            break
    if filled <= 1e-12:
        return {"filled": 0.0, "avg_price": _book_mid(row), "slippage_bps": 0.0, "levels_used": 0.0}
    avg_price = notional / filled
    mid = _book_mid(row)
    signed_slip = (avg_price - mid) / max(mid, 1e-12)
    if not side_buy:
        signed_slip = (mid - avg_price) / max(mid, 1e-12)
    return {
        "filled": float(filled if side_buy else -filled),
        "avg_price": float(avg_price),
        "slippage_bps": float(signed_slip * 1e4),
        "levels_used": float(levels_used),
    }


def run_matching_eval(
    rows: list[dict[str, Any]],
    *,
    freq_method: str,
    latency_bins: int = 1,
    max_position: float = 6.0,
    max_order_qty: float = 1.5,
    participation: float = 0.0025,
    steps: int | None = None,
) -> dict[str, Any]:
    books = rows[-int(steps):] if steps is not None else list(rows)
    mids = np.asarray([_book_mid(row) for row in books], dtype=np.float64)
    returns = np.diff(np.log(np.maximum(mids, 1e-12)))
    if returns.size < 3:
        raise ValueError("not enough order-book rows")
    tracker = TradingFrequencyTracker(
        bar_sec=1.0,
        method=str(freq_method),
        low_period_s=300.0,
        fast_period_s=20.0,
        mid_period_s=90.0,
        energy_period_s=60.0,
        persistence_period_s=120.0,
        persistence_threshold=0.0006,
        feature_norm=[0.001],
        promotion_enable=True,
        promotion_window_s=120.0,
        promotion_residual_threshold=0.0005,
        promotion_persistence_ratio=0.35,
        promotion_cooldown_s=180.0,
        promotion_adapt_low=True,
        promotion_adapt_gain=0.20,
    )
    position = 0.0
    cash = 0.0
    equity_curve = []
    pnl_returns = []
    slippages = []
    fill_abs = []
    partials = 0
    levels = []
    promotions = 0
    delay = max(0, int(latency_bins))
    prev_mid = mids[0]
    for t in range(1, len(books) - 1):
        signal_book = books[max(0, t - delay)]
        signal = returns[t - 1] + 0.00035 * _book_imbalance(signal_book)
        freq = tracker.update_bar(np.asarray([signal], dtype=np.float64), t=float(t))
        promotions += 1 if dict(freq.get("promotion", {}) or {}).get("promote", False) else 0
        upper = tracker.upper_features()
        lower = tracker.lower_features(np.asarray([position / max(max_position, 1e-9)]), np.asarray([position]))
        low = float(upper[0]) if upper.size else 0.0
        high = float(lower[0]) if lower.size else 0.0
        target = np.tanh(7.0 * low + 1.5 * high + 2.0 * signal) * float(max_position)
        order = float(np.clip(target - position, -float(max_order_qty), float(max_order_qty)))
        visible_depth = float(signal_book["bid_sizes"][0] + signal_book["ask_sizes"][0])
        order = float(np.clip(order, -participation * visible_depth, participation * visible_depth))
        fill_book = books[min(t + delay, len(books) - 1)]
        fill = fill_market_order(fill_book, order)
        filled = float(fill["filled"])
        if abs(filled) + 1e-12 < abs(order):
            partials += 1
        if abs(filled) > 0.0:
            cash -= filled * float(fill["avg_price"])
            position += filled
        mid = mids[t]
        equity = cash + position * mid
        prev_equity = cash + position * prev_mid
        pnl_returns.append((equity - prev_equity) / max(abs(prev_mid) * max_position, 1e-9))
        equity_curve.append(1.0 + equity / max(abs(mids[0]) * max_position, 1e-9))
        slippages.append(float(fill["slippage_bps"]))
        fill_abs.append(abs(filled))
        levels.append(float(fill["levels_used"]))
        prev_mid = mid
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity_curve, dtype=np.float64)
    return {
        "freq_method": str(freq_method),
        "bars": int(len(books)),
        "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "sharpe": float(np.sqrt(252.0 * 6.5 * 3600.0) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0,
        "max_drawdown": max_drawdown(eq),
        "avg_slippage_bps": float(np.mean(slippages)) if slippages else 0.0,
        "avg_abs_fill": float(np.mean(fill_abs)) if fill_abs else 0.0,
        "partial_fill_rate": float(partials / max(len(fill_abs), 1)),
        "avg_levels_used": float(np.mean(levels)) if levels else 0.0,
        "promotion_count": int(promotions),
        "final_position": float(position),
    }


def paired_checks(rows: list[dict[str, Any]], *, baseline: str, min_pairs: int) -> list[dict[str, Any]]:
    checks = []
    treatments = sorted({
        str(row["freq_method"])
        for row in rows
        if str(row["freq_method"]) != str(baseline)
    })
    for treatment in treatments:
        for metric, lower_is_better in [
            ("sharpe", False),
            ("total_return", False),
            ("max_drawdown", True),
            ("avg_slippage_bps", True),
            ("partial_fill_rate", True),
        ]:
            stats = paired_delta_stats(
                rows,
                variant_key="freq_method",
                pair_keys=("source", "seed", "latency_bins"),
                metric=metric,
                treatment=treatment,
                control=baseline,
                lower_is_better=lower_is_better,
            )
            checks.append({
                "check": f"{treatment}_vs_{baseline}_{metric}",
                **stats,
                "status": claim_status(stats, min_pairs=int(min_pairs)),
            })
    return checks


def run_validation(
    output_dir: Path,
    *,
    seeds: list[int],
    latency_bins: list[int],
    methods: list[str],
    steps: int,
    levels: int,
    min_pairs: int,
    csv_files: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    books: list[tuple[str, int, list[dict[str, Any]]]] = []
    if csv_files:
        for idx, path in enumerate(csv_files):
            books.append((str(path), idx, read_l2_order_book_csv(Path(path), levels=int(levels))))
    else:
        for seed in seeds:
            books.append((
                f"synthetic_l2_seed{int(seed)}",
                int(seed),
                make_synthetic_l2_order_book(
                    seed=int(seed),
                    steps=max(int(steps) + 8, 64),
                    levels=int(levels),
                ),
            ))
    for source, seed, book in books:
        for latency in latency_bins:
            for method in methods:
                row = run_matching_eval(
                    book,
                    freq_method=str(method),
                    latency_bins=int(latency),
                    steps=int(steps),
                )
                row.update({
                    "source": str(source),
                    "seed": int(seed),
                    "latency_bins": int(latency),
                    "levels": int(levels),
                })
                rows.append(row)
    checks = paired_checks(rows, baseline="ema", min_pairs=int(min_pairs))
    write_outputs(output_dir, rows, checks, sources=[source for source, _, _ in books])
    return {"summary": rows, "paired_checks": checks}


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    sources: list[str],
) -> None:
    with (output_dir / "per_eval.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "paired_checks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checks)
    best = max(rows, key=lambda row: float(row["sharpe"]))
    payload = {
        "summary": rows,
        "paired_checks": checks,
        "best": best,
        "sources": sources,
        "boundary": "L2 market-order matching with latency, partial fills, and slippage; real CSV input is supported, but exchange queue priority is not modeled.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# L2 Order-Book Matching Validation",
        "",
        f"- best Sharpe: `{best['freq_method']}` latency={best['latency_bins']} ({best['sharpe']:.3f})",
        f"- sources: `{len(sources)}`",
        "- boundary: L2 market-order matching with latency and partial fills; no exchange queue priority",
        "",
        "| check | status | metric | n | delta | CI95 low | CI95 high | win rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in checks:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['metric']} "
            f"| {row['n_common']} | {row['delta_mean']:+.4f} "
            f"| {row['delta_ci95_low']:+.4f} | {row['delta_ci95_high']:+.4f} "
            f"| {row['win_rate']:.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--latency-bins", type=int, nargs="+", default=[0, 2, 5])
    parser.add_argument("--methods", nargs="+", choices=ORDER_BOOK_ENCODERS, default=list(ORDER_BOOK_ENCODERS))
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--csv-files", type=Path, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_order_book_matching_validation"))
    args = parser.parse_args()
    payload = run_validation(
        args.output_dir,
        seeds=list(args.seeds),
        latency_bins=list(args.latency_bins),
        methods=list(args.methods),
        steps=int(args.steps),
        levels=int(args.levels),
        min_pairs=int(args.min_pairs),
        csv_files=list(args.csv_files or []),
    )
    best = max(payload["summary"], key=lambda row: float(row["sharpe"]))
    print(
        "order_book_matching "
        f"best={best['freq_method']} latency={best['latency_bins']} sharpe={best['sharpe']:.3f}"
    )


if __name__ == "__main__":
    main()
