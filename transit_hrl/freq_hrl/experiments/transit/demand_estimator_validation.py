"""Validate causal Transit demand-intensity estimators on count data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.domains.transit import TransitFrequencyTracker


def make_count_series(seed: int, steps: int, dispersion: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    t = np.arange(int(steps), dtype=np.float64)
    rate = (
        18.0
        + 5.0 * np.sin(2.0 * np.pi * t / 96.0)
        + 2.0 * np.cos(2.0 * np.pi * t / 48.0)
    )
    shift = int(0.58 * steps)
    rate[shift:] += 4.0
    rate = np.maximum(rate, 1.0)
    k = max(float(dispersion), 1e-6)
    p = k / (k + rate)
    counts = rng.negative_binomial(k, p).astype(np.float64)
    return rate, counts


def make_tracker(method: str) -> TransitFrequencyTracker:
    return TransitFrequencyTracker(
        update_interval_s=60.0,
        bin_sec=60.0,
        method=method,
        low_period_s=30 * 60.0,
        fast_period_s=5 * 60.0,
        mid_period_s=15 * 60.0,
        energy_period_s=10 * 60.0,
        harmonic_period_s=96 * 60.0,
        fourier_k=2,
        harmonic_learning_rate=0.4,
        harmonic_ridge=0.1,
        harmonic_nb_dispersion=20.0,
        global_demand_norm=50.0,
        local_demand_norm=10.0,
        forecast_horizon_s=60.0,
    )


def evaluate_method(method: str, seed: int, steps: int, warmup: int, dispersion: float) -> dict[str, Any]:
    true_rate, counts = make_count_series(seed=seed, steps=steps, dispersion=dispersion)
    tracker = make_tracker(method)
    preds: list[float] = []
    targets: list[float] = []
    nlls: list[float] = []
    for idx, count in enumerate(counts):
        if tracker.total_updates >= warmup:
            pred = max(float(tracker.summary()["freq_low_forecast"]) * 50.0, 1e-6)
            target = float(true_rate[idx])
            preds.append(pred)
            targets.append(target)
            nlls.append(pred - target * np.log(pred))
        tracker.update({(0, True): float(count)})
    pred_arr = np.asarray(preds, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    err = pred_arr - target_arr
    return {
        "method": method,
        "seed": int(seed),
        "mse": float(np.mean(err * err)),
        "mae": float(np.mean(np.abs(err))),
        "poisson_nll_no_const": float(np.mean(nlls)),
        "n": int(pred_arr.size),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({row["method"] for row in rows})
    out = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        out.append({
            "method": method,
            "seeds": len(subset),
            "mse_mean": float(np.mean([row["mse"] for row in subset])),
            "mae_mean": float(np.mean([row["mae"] for row in subset])),
            "poisson_nll_no_const_mean": float(np.mean([
                row["poisson_nll_no_const"] for row in subset
            ])),
        })
    best = min(out, key=lambda row: row["mse_mean"])
    for row in out:
        row["delta_mse_vs_best"] = row["mse_mean"] - best["mse_mean"]
    return out


def write_report(path: Path, summary: list[dict[str, Any]]) -> None:
    best = min(summary, key=lambda row: row["mse_mean"])
    lines = [
        "# Transit Demand Estimator Validation",
        "",
        f"- best by MSE: `{best['method']}`",
        "",
        "| method | seeds | MSE | MAE | Poisson NLL | delta MSE vs best |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: item["mse_mean"]):
        lines.append(
            f"| {row['method']} "
            f"| {row['seeds']} "
            f"| {row['mse_mean']:.4f} "
            f"| {row['mae_mean']:.4f} "
            f"| {row['poisson_nll_no_const_mean']:.4f} "
            f"| {row['delta_mse_vs_best']:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dispersion", type=float, default=20.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ema", "fourier", "dynamic_harmonic_nb"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_demand_estimator_validation"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        evaluate_method(method, seed, args.steps, args.warmup, args.dispersion)
        for method in args.methods
        for seed in args.seeds
    ]
    summary = summarize(rows)
    with (args.output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)
    write_report(args.output_dir / "report.md", summary)
    best = min(summary, key=lambda row: row["mse_mean"])
    print(f"wrote {args.output_dir}")
    print(f"demand_estimator best={best['method']} mse={best['mse_mean']:.4f}")


if __name__ == "__main__":
    main()
