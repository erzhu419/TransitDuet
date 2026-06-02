"""L3 order-event replay validation for Freq-HRL trading encoders."""

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


def read_l3_events_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        raw_rows = [dict(row) for row in csv.DictReader(f)]
    events: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        event_type = str(row.get("event_type", row.get("type", ""))).lower()
        side = str(row.get("side", "")).lower()
        if event_type not in {"add", "cancel", "trade"} or side not in {"bid", "ask"}:
            continue
        size = _float(row, "size", "qty", "quantity", default=0.0)
        price = _float(row, "price", "px", default=0.0)
        if size <= 0.0 or price <= 0.0:
            continue
        events.append({
            "timestamp": _float(row, "timestamp", "time", "ts", default=float(idx)),
            "event_type": event_type,
            "side": side,
            "price": float(price),
            "size": float(size),
            "order_id": str(row.get("order_id", row.get("id", f"csv_{idx}"))),
        })
    if len(events) < 8:
        raise ValueError(f"not enough valid L3 events in {path}")
    events.sort(key=lambda item: (float(item["timestamp"]), str(item["event_type"])))
    return events


class L3QueueBook:
    def __init__(self) -> None:
        self.queues: dict[tuple[str, float], list[dict[str, Any]]] = {}
        self.order_index: dict[str, tuple[str, float]] = {}
        self.agent_fills: list[dict[str, float]] = []

    def add_order(
        self,
        order_id: str,
        side: str,
        price: float,
        size: float,
        *,
        owner: str = "market",
    ) -> None:
        if float(size) <= 0.0:
            return
        key = (str(side), round(float(price), 6))
        item = {
            "order_id": str(order_id),
            "size": float(size),
            "owner": str(owner),
        }
        self.queues.setdefault(key, []).append(item)
        self.order_index[str(order_id)] = key

    def cancel_order(self, order_id: str, size: float | None = None) -> float:
        key = self.order_index.get(str(order_id))
        if key is None:
            return 0.0
        queue = self.queues.get(key, [])
        for idx, order in enumerate(queue):
            if str(order["order_id"]) != str(order_id):
                continue
            cancel_size = float(order["size"] if size is None else min(float(size), float(order["size"])))
            order["size"] = float(order["size"]) - cancel_size
            if order["size"] <= 1e-12:
                queue.pop(idx)
                self.order_index.pop(str(order_id), None)
            if not queue:
                self.queues.pop(key, None)
            return float(cancel_size)
        return 0.0

    def cancel_owner(self, owner: str) -> None:
        owned = [
            order_id
            for order_id, key in list(self.order_index.items())
            for order in self.queues.get(key, [])
            if str(order["order_id"]) == str(order_id) and str(order["owner"]) == str(owner)
        ]
        for order_id in owned:
            self.cancel_order(order_id)

    def trade(self, side: str, price: float, size: float, *, timestamp: float = 0.0) -> float:
        key = (str(side), round(float(price), 6))
        queue = self.queues.get(key, [])
        remaining = max(float(size), 0.0)
        executed = 0.0
        while remaining > 1e-12 and queue:
            order = queue[0]
            take = min(remaining, float(order["size"]))
            order["size"] = float(order["size"]) - take
            remaining -= take
            executed += take
            if str(order["owner"]) == "agent":
                signed = take if str(side) == "bid" else -take
                self.agent_fills.append({
                    "timestamp": float(timestamp),
                    "signed_qty": float(signed),
                    "price": float(price),
                })
            if order["size"] <= 1e-12:
                self.order_index.pop(str(order["order_id"]), None)
                queue.pop(0)
        if not queue:
            self.queues.pop(key, None)
        return float(executed)

    def process(self, event: dict[str, Any]) -> None:
        event_type = str(event["event_type"])
        if event_type == "add":
            self.add_order(
                str(event["order_id"]),
                str(event["side"]),
                float(event["price"]),
                float(event["size"]),
            )
        elif event_type == "cancel":
            self.cancel_order(str(event["order_id"]), float(event["size"]))
        elif event_type == "trade":
            self.trade(
                str(event["side"]),
                float(event["price"]),
                float(event["size"]),
                timestamp=float(event.get("timestamp", 0.0)),
            )

    def best_price(self, side: str) -> float:
        prices = [price for (book_side, price), queue in self.queues.items() if book_side == side and queue]
        if not prices:
            return 0.0
        return float(max(prices) if side == "bid" else min(prices))

    def depth_at(self, side: str, price: float) -> float:
        return float(sum(float(order["size"]) for order in self.queues.get((str(side), round(float(price), 6)), [])))

    def snapshot(self) -> dict[str, float]:
        bid = self.best_price("bid")
        ask = self.best_price("ask")
        if bid <= 0.0 or ask <= 0.0:
            return {
                "bid": 0.0,
                "ask": 0.0,
                "mid": 0.0,
                "bid_size": 0.0,
                "ask_size": 0.0,
                "imbalance": 0.0,
                "spread_bps": 0.0,
            }
        bid_size = self.depth_at("bid", bid)
        ask_size = self.depth_at("ask", ask)
        mid = 0.5 * (bid + ask)
        return {
            "bid": float(bid),
            "ask": float(ask),
            "mid": float(mid),
            "bid_size": float(bid_size),
            "ask_size": float(ask_size),
            "imbalance": float((bid_size - ask_size) / max(bid_size + ask_size, 1e-12)),
            "spread_bps": float((ask - bid) / max(mid, 1e-12) * 1e4),
        }


def _book_order_ids(book: L3QueueBook, side: str) -> list[str]:
    out = []
    for (book_side, _price), queue in book.queues.items():
        if book_side == side:
            out.extend(str(order["order_id"]) for order in queue if str(order["owner"]) == "market")
    return out


def make_synthetic_l3_events(seed: int = 7, steps: int = 360, levels: int = 3) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    book = L3QueueBook()
    events: list[dict[str, Any]] = []
    mid = 100.0
    tick = 0.01
    oid = 0

    def add_event(timestamp: float, event_type: str, side: str, price: float, size: float, order_id: str) -> None:
        event = {
            "timestamp": float(timestamp),
            "event_type": str(event_type),
            "side": str(side),
            "price": float(round(price, 6)),
            "size": float(max(size, 0.0)),
            "order_id": str(order_id),
        }
        events.append(event)
        book.process(event)

    for level in range(1, max(1, int(levels)) + 1):
        for side, price in [("bid", mid - level * tick), ("ask", mid + level * tick)]:
            for _ in range(3):
                oid += 1
                add_event(0.0, "add", side, price, rng.uniform(20.0, 80.0), f"m{oid}")

    for t in range(1, max(4, int(steps)) + 1):
        snap = book.snapshot()
        if snap["mid"] > 0.0:
            mid = float(snap["mid"])
        pressure = float(np.clip(0.55 * np.sin(2.0 * np.pi * t / 75.0) + rng.normal(0.0, 0.18), -0.9, 0.9))
        drift = 0.00020 * pressure + 0.00004 * np.sin(2.0 * np.pi * t / 180.0)
        mid = float(mid * np.exp(drift))
        best_bid = book.best_price("bid")
        best_ask = book.best_price("ask")
        if best_bid > 0.0 and rng.random() < 0.50:
            ids = _book_order_ids(book, "bid")
            if ids:
                event = {
                    "timestamp": float(t),
                    "event_type": "cancel",
                    "side": "bid",
                    "price": best_bid,
                    "size": float(rng.uniform(5.0, 35.0)),
                    "order_id": str(rng.choice(ids)),
                }
                events.append(event)
                book.process(event)
        if best_ask > 0.0 and rng.random() < 0.50:
            ids = _book_order_ids(book, "ask")
            if ids:
                event = {
                    "timestamp": float(t),
                    "event_type": "cancel",
                    "side": "ask",
                    "price": best_ask,
                    "size": float(rng.uniform(5.0, 35.0)),
                    "order_id": str(rng.choice(ids)),
                }
                events.append(event)
                book.process(event)
        # Aggressive flow is correlated with pressure, so the encoder has a real signal.
        if best_ask > 0.0 and rng.random() < 0.45 + 0.30 * max(pressure, 0.0):
            event = {
                "timestamp": float(t),
                "event_type": "trade",
                "side": "ask",
                "price": best_ask,
                "size": float(rng.uniform(20.0, 120.0) * (1.0 + max(pressure, 0.0))),
                "order_id": f"tb{t}",
            }
            events.append(event)
            book.process(event)
        if best_bid > 0.0 and rng.random() < 0.45 + 0.30 * max(-pressure, 0.0):
            event = {
                "timestamp": float(t),
                "event_type": "trade",
                "side": "bid",
                "price": best_bid,
                "size": float(rng.uniform(20.0, 120.0) * (1.0 + max(-pressure, 0.0))),
                "order_id": f"ts{t}",
            }
            events.append(event)
            book.process(event)
        quote_mid = mid + rng.normal(0.0, 0.003)
        for level in range(1, max(1, int(levels)) + 1):
            for side, price in [("bid", quote_mid - level * tick), ("ask", quote_mid + level * tick)]:
                if rng.random() < 0.70 / level:
                    oid += 1
                    size = rng.uniform(15.0, 70.0) * (1.0 + (pressure if side == "bid" else -pressure) * 0.25)
                    add_event(float(t), "add", side, price, max(size, 1.0), f"m{oid}")
    return events


def _group_events(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for event in sorted(events, key=lambda item: (float(item["timestamp"]), str(item["event_type"]))):
        grouped.setdefault(int(float(event["timestamp"])), []).append(event)
    return grouped


def run_l3_replay_eval(
    events: list[dict[str, Any]],
    *,
    freq_method: str,
    steps: int | None = None,
    max_position: float = 6.0,
    max_order_qty: float = 2.0,
    participation: float = 0.015,
) -> dict[str, Any]:
    grouped = _group_events(events)
    if not grouped:
        raise ValueError("empty L3 event stream")
    max_t = max(grouped)
    horizon = min(max_t, int(steps)) if steps is not None else max_t
    book = L3QueueBook()
    for event in grouped.get(0, []):
        book.process(event)
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
    last_mid = book.snapshot()["mid"] or 100.0
    last_fill_idx = 0
    pnl_returns = []
    equity_curve = []
    fill_abs = []
    slippages = []
    spread_capture = []
    promotions = 0
    agent_seq = 0
    for t in range(1, max(2, horizon)):
        snap = book.snapshot()
        if snap["mid"] <= 0.0:
            for event in grouped.get(t, []):
                book.process(event)
            continue
        signal = np.log(max(snap["mid"], 1e-12) / max(last_mid, 1e-12)) + 0.00030 * snap["imbalance"]
        freq = tracker.update_bar(np.asarray([signal], dtype=np.float64), t=float(t))
        promotions += 1 if dict(freq.get("promotion", {}) or {}).get("promote", False) else 0
        upper = tracker.upper_features()
        lower = tracker.lower_features(np.asarray([position / max(max_position, 1e-9)]), np.asarray([position]))
        low = float(upper[0]) if upper.size else 0.0
        high = float(lower[0]) if lower.size else 0.0
        target = np.tanh(8.0 * low + 1.5 * high + 2.0 * signal) * float(max_position)
        desired = float(np.clip(target - position, -float(max_order_qty), float(max_order_qty)))
        visible_depth = float(snap["bid_size"] + snap["ask_size"])
        desired = float(np.clip(desired, -participation * visible_depth, participation * visible_depth))
        book.cancel_owner("agent")
        if abs(desired) > 1e-12:
            agent_seq += 1
            side = "bid" if desired > 0.0 else "ask"
            price = snap["bid"] if desired > 0.0 else snap["ask"]
            book.add_order(f"agent_{t}_{agent_seq}", side, price, abs(desired), owner="agent")
        for event in grouped.get(t, []):
            book.process(event)
        new_fills = book.agent_fills[last_fill_idx:]
        last_fill_idx = len(book.agent_fills)
        for fill in new_fills:
            signed = float(fill["signed_qty"])
            price = float(fill["price"])
            cash -= signed * price
            position += signed
            fill_abs.append(abs(signed))
            fill_mid = max(snap["mid"], 1e-12)
            signed_slip = (price - fill_mid) / fill_mid
            if signed < 0.0:
                signed_slip = (fill_mid - price) / fill_mid
            slip_bps = signed_slip * 1e4
            slippages.append(float(slip_bps))
            spread_capture.append(float(-slip_bps))
        end_snap = book.snapshot()
        mid = end_snap["mid"] or snap["mid"]
        equity = cash + position * mid
        prev_equity = cash + position * last_mid
        pnl_returns.append((equity - prev_equity) / max(abs(last_mid) * max_position, 1e-9))
        equity_curve.append(1.0 + equity / max(abs(last_mid) * max_position, 1e-9))
        last_mid = mid
    book.cancel_owner("agent")
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity_curve, dtype=np.float64)
    return {
        "freq_method": str(freq_method),
        "events": int(len(events)),
        "bars": int(horizon),
        "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "sharpe": float(np.sqrt(252.0 * 6.5 * 3600.0) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0,
        "max_drawdown": max_drawdown(eq),
        "fill_rate": float(len(fill_abs) / max(horizon - 1, 1)),
        "avg_abs_fill": float(np.mean(fill_abs)) if fill_abs else 0.0,
        "avg_slippage_bps": float(np.mean(slippages)) if slippages else 0.0,
        "avg_spread_capture_bps": float(np.mean(spread_capture)) if spread_capture else 0.0,
        "promotion_count": int(promotions),
        "final_position": float(position),
    }


def paired_checks(rows: list[dict[str, Any]], *, baseline: str, min_pairs: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
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
            ("fill_rate", False),
            ("avg_spread_capture_bps", False),
        ]:
            stats = paired_delta_stats(
                rows,
                variant_key="freq_method",
                pair_keys=("source", "seed"),
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
    methods: list[str],
    steps: int,
    levels: int,
    min_pairs: int,
    csv_files: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tapes: list[tuple[str, int, list[dict[str, Any]]]] = []
    if csv_files:
        for idx, path in enumerate(csv_files):
            tapes.append((str(path), idx, read_l3_events_csv(Path(path))))
    else:
        for seed in seeds:
            tapes.append((
                f"synthetic_l3_seed{int(seed)}",
                int(seed),
                make_synthetic_l3_events(seed=int(seed), steps=max(int(steps), 16), levels=int(levels)),
            ))
    rows: list[dict[str, Any]] = []
    for source, seed, events in tapes:
        for method in methods:
            row = run_l3_replay_eval(events, freq_method=str(method), steps=int(steps))
            row.update({
                "source": str(source),
                "seed": int(seed),
                "levels": int(levels),
            })
            rows.append(row)
    checks = paired_checks(rows, baseline="ema", min_pairs=int(min_pairs))
    write_outputs(output_dir, rows, checks, sources=[source for source, _, _ in tapes])
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
        "sources": list(sources),
        "boundary": "Synthetic or CSV L3 add/cancel/trade event replay with FIFO queue priority for agent passive orders; committed validation uses synthetic L3 tapes unless csv-files are provided.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# L3 Order-Event Replay Validation",
        "",
        f"- best Sharpe: `{best['freq_method']}` ({best['sharpe']:.3f})",
        f"- sources: `{len(sources)}`",
        "- boundary: L3 FIFO queue replay with add/cancel/trade events; committed run uses synthetic tapes unless CSVs are supplied",
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
    parser.add_argument("--methods", nargs="+", choices=ORDER_BOOK_ENCODERS, default=list(ORDER_BOOK_ENCODERS))
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--csv-files", type=Path, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_order_book_l3_replay_validation"))
    args = parser.parse_args()
    payload = run_validation(
        args.output_dir,
        seeds=list(args.seeds),
        methods=list(args.methods),
        steps=int(args.steps),
        levels=int(args.levels),
        min_pairs=int(args.min_pairs),
        csv_files=list(args.csv_files or []),
    )
    best = max(payload["summary"], key=lambda row: float(row["sharpe"]))
    print(f"order_book_l3 best={best['freq_method']} sharpe={best['sharpe']:.3f}")


if __name__ == "__main__":
    main()
