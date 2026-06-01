"""Order-book CSV validation path for Freq-HRL trading."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.domains.trading import (
    PortfolioExecutionConfig,
    PortfolioExecutionEnv,
    TradingFrequencyTracker,
)
from freq_hrl.experiments.trading.performance_validation import max_drawdown
from freq_hrl.policies import FrequencyTradingController, FrequencyTradingPlanner


ORDER_BOOK_ENCODERS = ("ema", "state_space", "adaptive_wavelet", "neural_state_space")


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


def read_order_book_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    out = []
    for idx, row in enumerate(rows):
        bid = _float(row, "bid_price_1", "bid_price", "best_bid")
        ask = _float(row, "ask_price_1", "ask_price", "best_ask")
        bid_size = _float(row, "bid_size_1", "bid_size", "best_bid_size", default=1.0)
        ask_size = _float(row, "ask_size_1", "ask_size", "best_ask_size", default=1.0)
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            continue
        out.append({
            "timestamp": _float(row, "timestamp", "time", "ts", default=float(idx)),
            "bid": bid,
            "ask": ask,
            "bid_size": max(bid_size, 1e-9),
            "ask_size": max(ask_size, 1e-9),
        })
    if len(out) < 4:
        raise ValueError(f"not enough valid order-book rows in {path}")
    return out


def make_synthetic_order_book(seed: int = 7, steps: int = 720) -> list[dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    price = 100.0
    rows = []
    prev_imbalance = 0.0
    for t in range(max(4, int(steps))):
        imbalance = float(np.clip(0.55 * np.sin(2.0 * np.pi * t / 60.0) + rng.normal(0.0, 0.12), -0.9, 0.9))
        drift = 0.00002 * np.sin(2.0 * np.pi * t / 180.0) + 0.00016 * prev_imbalance
        shock = rng.normal(0.0, 0.00035)
        price *= float(np.exp(drift + shock))
        spread = 0.006 + 0.004 * (1.0 + np.sin(2.0 * np.pi * t / 45.0))
        depth = 1000.0 + 250.0 * np.cos(2.0 * np.pi * t / 120.0)
        bid_size = depth * (1.0 + imbalance)
        ask_size = depth * (1.0 - imbalance)
        rows.append({
            "timestamp": float(t),
            "bid": price - 0.5 * spread,
            "ask": price + 0.5 * spread,
            "bid_size": max(bid_size, 1.0),
            "ask_size": max(ask_size, 1.0),
        })
        prev_imbalance = imbalance
    return rows


def write_order_book_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "timestamp": row["timestamp"],
                "bid_price_1": row["bid"],
                "ask_price_1": row["ask"],
                "bid_size_1": row["bid_size"],
                "ask_size_1": row["ask_size"],
            })


def order_book_arrays(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    bid = np.asarray([row["bid"] for row in rows], dtype=np.float64)
    ask = np.asarray([row["ask"] for row in rows], dtype=np.float64)
    bid_size = np.asarray([row["bid_size"] for row in rows], dtype=np.float64)
    ask_size = np.asarray([row["ask_size"] for row in rows], dtype=np.float64)
    mid = 0.5 * (bid + ask)
    micro = (ask * bid_size + bid * ask_size) / np.maximum(bid_size + ask_size, 1e-12)
    spread = (ask - bid) / np.maximum(mid, 1e-12)
    imbalance = (bid_size - ask_size) / np.maximum(bid_size + ask_size, 1e-12)
    depth = bid_size + ask_size
    returns = np.diff(np.log(np.maximum(micro, 1e-12)))
    signal = np.zeros_like(returns)
    if returns.size > 1:
        signal[1:] = returns[:-1]
    signal += 0.0004 * imbalance[:-1] - 0.10 * spread[:-1] * np.sign(signal)
    return {
        "returns": returns.reshape(-1, 1),
        "signal": signal.reshape(-1, 1),
        "spread": spread[:-1],
        "imbalance": imbalance[:-1],
        "depth": depth[:-1],
    }


def run_order_book_eval(rows: list[dict[str, float]], freq_method: str, steps: int | None = None) -> dict[str, Any]:
    arrays = order_book_arrays(rows)
    returns = arrays["returns"] if steps is None else arrays["returns"][-int(steps):]
    signal = arrays["signal"] if steps is None else arrays["signal"][-int(steps):]
    spread = arrays["spread"] if steps is None else arrays["spread"][-int(steps):]
    imbalance = arrays["imbalance"] if steps is None else arrays["imbalance"][-int(steps):]
    depth = arrays["depth"] if steps is None else arrays["depth"][-int(steps):]
    avg_half_spread_bps = 0.5 * float(np.mean(spread)) * 1e4
    env = PortfolioExecutionEnv(
        returns,
        volumes=1.0 + np.abs(imbalance).reshape(-1, 1) + depth.reshape(-1, 1) / max(float(np.mean(depth)), 1e-9),
        config=PortfolioExecutionConfig(
            transaction_cost_bps=max(0.5, avg_half_spread_bps),
            slippage_bps=max(0.1, 0.25 * avg_half_spread_bps),
            max_leverage=1.0,
            inventory_drift_penalty=0.001,
        ),
    )
    tracker = TradingFrequencyTracker(
        bar_sec=1.0,
        method=freq_method,
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
    planner = FrequencyTradingPlanner(promotion_mid_gain=0.5)
    controller = FrequencyTradingController()
    env.reset()
    pnl_returns = []
    equity = []
    turnover = []
    promotions = 0
    for t in range(returns.shape[0]):
        freq = tracker.update_bar(signal[t], t=float(t))
        promotions += 1 if dict(freq.get("promotion", {}) or {}).get("promote", False) else 0
        obs = {"raw_signal": signal[t], "position": env.position.copy(), "t": t}
        upper = planner.plan(obs, tracker.upper_features(), context={"frequency": freq, "n_assets": 1})
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
        "freq_method": str(freq_method),
        "bars": int(returns.shape[0]),
        "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "sharpe": float(np.sqrt(252.0 * 6.5 * 3600.0) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0,
        "max_drawdown": max_drawdown(eq),
        "turnover": float(np.sum(turnover)),
        "promotion_count": int(promotions),
        "avg_spread_bps": float(np.mean(spread) * 1e4),
        "avg_abs_imbalance": float(np.mean(np.abs(imbalance))),
        "avg_depth": float(np.mean(depth)),
    }


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], source: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"source": source, "summary": rows}, f, indent=2)
    best = max(rows, key=lambda row: float(row["sharpe"]))
    lines = [
        "# Order-Book Freq-HRL Encoder Validation",
        "",
        f"- source: `{source}`",
        f"- best Sharpe encoder: `{best['freq_method']}` ({best['sharpe']:.3f})",
        "",
        "| encoder | bars | return | Sharpe | max DD | turnover | promotions | spread bps | abs imbalance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['freq_method']} "
            f"| {row['bars']} "
            f"| {row['total_return']:.4f} "
            f"| {row['sharpe']:.3f} "
            f"| {row['max_drawdown']:.4f} "
            f"| {row['turnover']:.2f} "
            f"| {row['promotion_count']} "
            f"| {row['avg_spread_bps']:.3f} "
            f"| {row['avg_abs_imbalance']:.3f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", type=Path, default=None)
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--synthetic-seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--methods", nargs="+", choices=ORDER_BOOK_ENCODERS, default=list(ORDER_BOOK_ENCODERS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/trading_order_book_encoder_ablation"),
    )
    args = parser.parse_args()
    if args.csv_file is not None:
        rows_raw = read_order_book_csv(args.csv_file)
        source = str(args.csv_file)
    else:
        rows_raw = make_synthetic_order_book(seed=args.synthetic_seed, steps=max(args.steps, 4))
        source = f"synthetic_order_book_seed{args.synthetic_seed}"
        if args.generate_synthetic:
            write_order_book_csv(args.output_dir / "synthetic_order_book.csv", rows_raw)
    rows = [run_order_book_eval(rows_raw, method, steps=args.steps) for method in args.methods]
    write_outputs(args.output_dir, rows, source=source)
    best = max(rows, key=lambda row: float(row["sharpe"]))
    print(f"wrote {args.output_dir}")
    print(f"order_book best={best['freq_method']} sharpe={best['sharpe']:.3f} return={best['total_return']:.4f}")


if __name__ == "__main__":
    main()
