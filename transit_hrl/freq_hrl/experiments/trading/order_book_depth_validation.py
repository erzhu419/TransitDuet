"""Order-book depth, spread, and latency stress validation for Freq-HRL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.experiments.trading.order_book_data import (
    ORDER_BOOK_ENCODERS,
    make_synthetic_order_book,
    read_order_book_csv,
    run_order_book_eval,
)


STRESS_REGIMES: dict[str, dict[str, float]] = {
    "baseline": {"spread_mult": 1.0, "depth_mult": 1.0, "latency_bins": 0.0},
    "wide_spread": {"spread_mult": 2.0, "depth_mult": 1.0, "latency_bins": 0.0},
    "shallow_depth": {"spread_mult": 1.0, "depth_mult": 0.35, "latency_bins": 0.0},
    "stale_book": {"spread_mult": 1.0, "depth_mult": 1.0, "latency_bins": 3.0},
    "combined_stress": {"spread_mult": 2.5, "depth_mult": 0.30, "latency_bins": 5.0},
}


def apply_order_book_stress(
    rows: list[dict[str, float]],
    *,
    spread_mult: float,
    depth_mult: float,
    latency_bins: int,
) -> list[dict[str, float]]:
    """Apply deterministic execution stress while preserving mid-price path."""

    if not rows:
        return []
    stressed: list[dict[str, float]] = []
    delay = max(int(latency_bins), 0)
    for idx, row in enumerate(rows):
        book_row = rows[max(0, idx - delay)]
        mid = 0.5 * (float(row["bid"]) + float(row["ask"]))
        spread = max(float(book_row["ask"]) - float(book_row["bid"]), 1e-9) * max(float(spread_mult), 0.0)
        stressed.append({
            "timestamp": float(row.get("timestamp", idx)),
            "bid": mid - 0.5 * spread,
            "ask": mid + 0.5 * spread,
            "bid_size": max(float(book_row["bid_size"]) * max(float(depth_mult), 1e-6), 1e-9),
            "ask_size": max(float(book_row["ask_size"]) * max(float(depth_mult), 1e-6), 1e-9),
        })
    return stressed


def eval_rows(
    *,
    seeds: list[int],
    steps: int,
    methods: list[str],
    stress_regimes: dict[str, dict[str, float]],
    csv_files: list[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if csv_files:
        datasets = [(str(path), 0, read_order_book_csv(path)) for path in csv_files]
    else:
        datasets = [
            (f"synthetic_order_book_seed{int(seed)}", int(seed), make_synthetic_order_book(seed=int(seed), steps=max(int(steps) + 8, 64)))
            for seed in seeds
        ]
    for dataset, seed, raw_rows in datasets:
        for stress, spec in stress_regimes.items():
            stressed = apply_order_book_stress(
                raw_rows,
                spread_mult=float(spec["spread_mult"]),
                depth_mult=float(spec["depth_mult"]),
                latency_bins=int(spec["latency_bins"]),
            )
            for method in methods:
                item = run_order_book_eval(stressed, str(method), steps=int(steps))
                item.update({
                    "dataset": dataset,
                    "seed": int(seed),
                    "stress": stress,
                    "spread_mult": float(spec["spread_mult"]),
                    "depth_mult": float(spec["depth_mult"]),
                    "latency_bins": int(spec["latency_bins"]),
                })
                rows.append(item)
    return rows


def paired_checks(rows: list[dict[str, Any]], *, baseline: str, min_pairs: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    candidates = sorted({
        str(row.get("freq_method"))
        for row in rows
        if str(row.get("freq_method")) != str(baseline)
    })
    for treatment in candidates:
        for metric, lower_is_better in [
            ("sharpe", False),
            ("total_return", False),
            ("max_drawdown", True),
            ("turnover", True),
        ]:
            stats = paired_delta_stats(
                rows,
                variant_key="freq_method",
                pair_keys=("dataset", "stress"),
                cluster_keys=("dataset",),
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


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
        "boundary": "Synthetic or CSV L1/L2 order-book stress validation; not a full exchange matching engine.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Order-Book Depth Stress Validation",
        "",
        f"- best Sharpe: `{best['freq_method']}` under `{best['stress']}` ({best['sharpe']:.3f})",
        "- boundary: synthetic or CSV L1/L2 stress validation, not a full exchange matching engine",
        "",
        "## Paired Checks",
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


def run_validation(
    output_dir: Path,
    *,
    seeds: list[int],
    steps: int,
    methods: list[str],
    csv_files: list[Path],
    min_pairs: int,
) -> dict[str, Any]:
    rows = eval_rows(
        seeds=seeds,
        steps=int(steps),
        methods=methods,
        stress_regimes=STRESS_REGIMES,
        csv_files=csv_files,
    )
    checks = paired_checks(rows, baseline="ema", min_pairs=int(min_pairs))
    write_outputs(output_dir, rows, checks)
    return {"summary": rows, "paired_checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--methods", nargs="+", choices=ORDER_BOOK_ENCODERS, default=list(ORDER_BOOK_ENCODERS))
    parser.add_argument("--csv-files", type=Path, nargs="*", default=[])
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/trading_order_book_depth_validation"),
    )
    args = parser.parse_args()
    payload = run_validation(
        args.output_dir,
        seeds=list(args.seeds),
        steps=int(args.steps),
        methods=list(args.methods),
        csv_files=list(args.csv_files),
        min_pairs=int(args.min_pairs),
    )
    best = max(payload["summary"], key=lambda row: float(row["sharpe"]))
    print(
        "order_book_depth "
        f"best={best['freq_method']} stress={best['stress']} sharpe={best['sharpe']:.3f}"
    )


if __name__ == "__main__":
    main()
