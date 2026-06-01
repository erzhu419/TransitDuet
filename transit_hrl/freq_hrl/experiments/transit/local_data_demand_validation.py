"""Validate Transit demand estimators on copied TransitDuet OD spreadsheets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freq_hrl.domains.transit import TransitFrequencyTracker
from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.experiments.transit.demand_estimator_validation import COUNT_CALIBRATION_ID


def _time_label(value: Any) -> str:
    text = str(value)
    if " " in text:
        text = text.split()[-1]
    return text


def load_hourly_od_counts(path: Path, max_series: int | None = None) -> dict[str, np.ndarray]:
    frame = pd.read_excel(path)
    if frame.empty or frame.shape[1] < 3:
        raise ValueError(f"OD spreadsheet has no usable count columns: {path}")
    time_col = frame.columns[0]
    origin_col = frame.columns[1]
    count_cols = list(frame.columns[2:])
    out: dict[str, list[tuple[str, float]]] = {}
    for _, row in frame.iterrows():
        origin = str(row[origin_col])
        if not origin or origin.lower() == "nan":
            continue
        counts = pd.to_numeric(row[count_cols], errors="coerce").fillna(0.0)
        total = float(counts.sum())
        out.setdefault(origin, []).append((_time_label(row[time_col]), total))
    series: dict[str, np.ndarray] = {}
    for origin, values in sorted(out.items()):
        values.sort(key=lambda item: item[0])
        arr = np.asarray([count for _, count in values], dtype=np.float64)
        if np.count_nonzero(arr) >= 3:
            series[origin] = arr
    if max_series is not None:
        return dict(list(series.items())[:max(1, int(max_series))])
    return series


def expand_hourly_counts(hourly: np.ndarray, bins_per_hour: int = 12) -> np.ndarray:
    hourly = np.asarray(hourly, dtype=np.float64).reshape(-1)
    bins = np.arange(max(1, int(bins_per_hour)), dtype=np.float64)
    phase = (bins + 0.5) / max(float(bins.size), 1.0)
    weights = 1.0 + 0.25 * np.sin(2.0 * np.pi * phase) + 0.10 * np.cos(4.0 * np.pi * phase)
    weights = np.maximum(weights, 0.20)
    weights = weights / float(np.sum(weights))
    return np.concatenate([float(max(count, 0.0)) * weights for count in hourly])


def make_tracker(method: str, bin_sec: float, norm: float) -> TransitFrequencyTracker:
    return TransitFrequencyTracker(
        update_interval_s=float(bin_sec),
        bin_sec=float(bin_sec),
        method=method,
        low_period_s=3.0 * 3600.0,
        fast_period_s=20.0 * 60.0,
        mid_period_s=75.0 * 60.0,
        energy_period_s=45.0 * 60.0,
        harmonic_period_s=14.0 * 3600.0,
        fourier_k=3,
        harmonic_learning_rate=0.8,
        harmonic_ridge=0.10,
        harmonic_nb_dispersion=120.0,
        global_demand_norm=float(norm),
        local_demand_norm=max(float(norm) / 4.0, 1.0),
        forecast_horizon_s=float(bin_sec),
    )


def evaluate_series(
    method: str,
    series_id: str,
    series_index: int,
    counts: np.ndarray,
    warmup: int,
    bin_sec: float,
) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    norm = max(float(np.percentile(counts, 95)), 1.0)
    tracker = make_tracker(method, bin_sec=bin_sec, norm=norm)
    preds: list[float] = []
    targets: list[float] = []
    for idx, count in enumerate(counts):
        if tracker.total_updates >= int(warmup):
            pred = max(float(tracker.summary()["freq_low_forecast"]) * norm, 1e-6)
            preds.append(pred)
            targets.append(float(count))
        tracker.update({(series_index, True): float(count)})
    pred_arr = np.asarray(preds, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    err = pred_arr - target_arr
    nll = pred_arr - target_arr * np.log(np.maximum(pred_arr, 1e-6))
    return {
        "method": str(method),
        "seed": int(series_index),
        "series_id": str(series_id),
        "mse": float(np.mean(err * err)),
        "mae": float(np.mean(np.abs(err))),
        "poisson_nll_no_const": float(np.mean(nll)),
        "n": int(pred_arr.size),
        "mean_count": float(np.mean(counts)),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    out = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        out.append({
            "method": method,
            "series": len(subset),
            "mse_mean": float(np.mean([float(row["mse"]) for row in subset])),
            "mae_mean": float(np.mean([float(row["mae"]) for row in subset])),
            "poisson_nll_no_const_mean": float(np.mean([
                float(row["poisson_nll_no_const"]) for row in subset
            ])),
        })
    best = min(out, key=lambda row: row["mse_mean"])
    for row in out:
        row["delta_mse_vs_best"] = float(row["mse_mean"] - best["mse_mean"])
    return out


def paired_method_stats(rows: list[dict[str, Any]], reference: str = "fourier") -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    out = []
    for idx, method in enumerate(methods):
        if method == reference:
            continue
        for metric in ("mse", "mae", "poisson_nll_no_const"):
            stats = paired_delta_stats(
                rows,
                variant_key="method",
                pair_keys=("seed",),
                metric=metric,
                treatment=method,
                control=reference,
                lower_is_better=True,
                seed=3100 + 17 * idx,
            )
            out.append({
                "comparison": f"{method}_vs_{reference}",
                **stats,
                "status": claim_status(stats, min_pairs=5),
            })
    return out


def write_report(
    path: Path,
    source_path: Path,
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> None:
    best = min(summary, key=lambda row: row["mse_mean"])
    lines = [
        "# Local Transit OD Demand Estimator Validation",
        "",
        f"- source: `{source_path}`",
        "- data path: copied TransitDuet OD spreadsheet, expanded from hourly OD counts into causal 5-minute AFC/APC-style bins",
        f"- best by MSE: `{best['method']}`",
        "",
        "| method | series | MSE | MAE | Poisson NLL | delta MSE vs best |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: item["mse_mean"]):
        lines.append(
            f"| {row['method']} "
            f"| {row['series']} "
            f"| {row['mse_mean']:.4f} "
            f"| {row['mae_mean']:.4f} "
            f"| {row['poisson_nll_no_const_mean']:.4f} "
            f"| {row['delta_mse_vs_best']:+.4f} |"
        )
    if paired:
        lines.extend([
            "",
            "## Paired Method Deltas",
            "",
            "Deltas are `method - fourier`; lower is better for all listed metrics.",
            "",
            "| comparison | metric | n | delta | CI95 low | CI95 high | win rate | status |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in paired:
            lines.append(
                f"| {row['comparison']} "
                f"| {row['metric']} "
                f"| {row['n_common']} "
                f"| {row['delta_mean']:+.4f} "
                f"| {row['delta_ci95_low']:+.4f} "
                f"| {row['delta_ci95_high']:+.4f} "
                f"| {row['win_rate']:.2f} "
                f"| {row['status']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    output_dir: Path,
    od_path: Path,
    methods: list[str],
    max_series: int,
    bins_per_hour: int,
    warmup: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly = load_hourly_od_counts(od_path, max_series=max_series)
    bin_sec = 3600.0 / max(int(bins_per_hour), 1)
    rows = []
    for series_index, (series_id, counts) in enumerate(hourly.items()):
        expanded = expand_hourly_counts(counts, bins_per_hour=bins_per_hour)
        for method in methods:
            rows.append(evaluate_series(
                method=method,
                series_id=series_id,
                series_index=series_index,
                counts=expanded,
                warmup=warmup,
                bin_sec=bin_sec,
            ))
    summary = summarize(rows)
    paired = paired_method_stats(rows, reference="fourier") if "fourier" in methods else []
    with (output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if paired:
        with (output_dir / "paired_deltas.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(paired)
    payload = {
        "metadata": {
            "estimator_calibration": COUNT_CALIBRATION_ID,
            "source": "local_transitduet_od_xlsx",
            "source_path": str(od_path),
            "bins_per_hour": int(bins_per_hour),
            "warmup": int(warmup),
        },
        "rows": rows,
        "summary": summary,
        "paired_deltas": paired,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_report(output_dir / "report.md", od_path, summary, paired)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--od-path",
        type=Path,
        default=Path("transit_hrl/freq_transitduet/env/data/passenger_OD.xlsx"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ema", "fourier", "dynamic_harmonic_nb", "adaptive_wavelet"],
    )
    parser.add_argument("--max-series", type=int, default=12)
    parser.add_argument("--bins-per-hour", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_local_demand_estimator_validation"),
    )
    args = parser.parse_args()
    payload = run_validation(
        output_dir=args.output_dir,
        od_path=args.od_path,
        methods=list(args.methods),
        max_series=int(args.max_series),
        bins_per_hour=int(args.bins_per_hour),
        warmup=int(args.warmup),
    )
    best = min(payload["summary"], key=lambda row: row["mse_mean"])
    print(f"wrote {args.output_dir}")
    print(f"local_demand best={best['method']} mse={best['mse_mean']:.4f}")


if __name__ == "__main__":
    main()
