"""Validate demand estimators on public GTFS schedule-event traces.

GTFS static feeds do not contain passenger AFC/APC boardings.  This module
uses real scheduled stop events as a causal activity proxy, which is useful for
testing the Transit data path without claiming passenger-demand ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.experiments.transit.demand_estimator_validation import COUNT_CALIBRATION_ID
from freq_hrl.experiments.transit.local_data_demand_validation import evaluate_series


DEFAULT_GTFS_URL = "https://www.bart.gov/dev/schedules/google_transit.zip"


def _safe_name(url: str) -> str:
    name = url.rstrip("/").split("/")[-1] or "gtfs.zip"
    return name if name.endswith(".zip") else f"{name}.zip"


def download_gtfs(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / _safe_name(url)
    if out.exists() and out.stat().st_size > 0:
        return out
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Freq-HRL-GTFS-Validator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data:
        raise ValueError(f"empty GTFS response from {url}")
    out.write_bytes(data)
    return out


def _seconds_from_gtfs_time(value: Any) -> float | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hour, minute, second = (int(float(part)) for part in parts)
    except ValueError:
        return None
    return float(hour * 3600 + minute * 60 + second)


def load_gtfs_stop_event_counts(
    gtfs_zip: Path,
    *,
    bin_sec: float = 300.0,
    max_series: int = 24,
    min_events: int = 20,
) -> dict[str, np.ndarray]:
    """Return stop-level scheduled event counts binned into a service day."""
    bins_per_day = int(np.ceil(24.0 * 3600.0 / max(float(bin_sec), 1.0)))
    with zipfile.ZipFile(gtfs_zip) as zf:
        names = set(zf.namelist())
        if "stop_times.txt" not in names:
            raise ValueError(f"GTFS feed has no stop_times.txt: {gtfs_zip}")
        with zf.open("stop_times.txt") as f:
            frame = pd.read_csv(
                f,
                usecols=lambda col: col in {"stop_id", "departure_time", "arrival_time"},
                dtype=str,
            )
    if frame.empty or "stop_id" not in frame.columns:
        raise ValueError(f"GTFS stop_times has no usable stop rows: {gtfs_zip}")
    time_col = "departure_time" if "departure_time" in frame.columns else "arrival_time"
    counts_by_stop: dict[str, np.ndarray] = {}
    totals: dict[str, int] = {}
    for stop_id, time_value in zip(frame["stop_id"], frame[time_col]):
        seconds = _seconds_from_gtfs_time(time_value)
        if seconds is None:
            continue
        bin_idx = int((seconds % (24.0 * 3600.0)) // float(bin_sec))
        stop = str(stop_id)
        arr = counts_by_stop.get(stop)
        if arr is None:
            arr = np.zeros(bins_per_day, dtype=np.float64)
            counts_by_stop[stop] = arr
        arr[min(max(bin_idx, 0), bins_per_day - 1)] += 1.0
        totals[stop] = totals.get(stop, 0) + 1
    ranked = [
        (stop, counts_by_stop[stop])
        for stop, total in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
        if total >= int(min_events)
    ]
    return dict(ranked[:max(1, int(max_series))])


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in sorted({str(row["method"]) for row in rows}):
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
    out = []
    methods = sorted({str(row["method"]) for row in rows})
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
                seed=4900 + 23 * idx,
            )
            out.append({
                "comparison": f"{method}_vs_{reference}",
                **stats,
                "status": claim_status(stats, min_pairs=5),
            })
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    paired = payload["paired_deltas"]
    best = min(summary, key=lambda row: row["mse_mean"])
    lines = [
        "# Public GTFS Demand-Proxy Validation",
        "",
        f"- source URL: `{payload['metadata'].get('source_url', '')}`",
        f"- cached feed: `{payload['metadata'].get('gtfs_zip', '')}`",
        f"- bin size: `{payload['metadata'].get('bin_sec', 0.0):.0f}s`",
        "- data path: public GTFS `stop_times.txt`, converted into causal stop-level scheduled event bins",
        "- boundary: scheduled events are a real Transit activity proxy, not AFC/APC passenger counts",
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
    *,
    gtfs_zip: Path | None = None,
    url: str = DEFAULT_GTFS_URL,
    cache_dir: Path = Path("transit_hrl/data/public_gtfs"),
    methods: list[str] | None = None,
    max_series: int = 24,
    bin_sec: float = 300.0,
    warmup: int = 12,
    min_events: int = 20,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = methods or ["ema", "fourier", "dynamic_harmonic_nb"]
    source_zip = gtfs_zip if gtfs_zip is not None else download_gtfs(url, cache_dir)
    counts = load_gtfs_stop_event_counts(
        source_zip,
        bin_sec=float(bin_sec),
        max_series=int(max_series),
        min_events=int(min_events),
    )
    rows: list[dict[str, Any]] = []
    for series_index, (series_id, series_counts) in enumerate(counts.items()):
        for method in methods:
            rows.append(evaluate_series(
                method=method,
                series_id=series_id,
                series_index=series_index,
                counts=series_counts,
                warmup=int(warmup),
                bin_sec=float(bin_sec),
            ))
    if not rows:
        raise ValueError(f"no usable GTFS stop-event series in {source_zip}")
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
            "source": "public_gtfs_schedule_proxy",
            "source_url": str(url),
            "gtfs_zip": str(source_zip),
            "bin_sec": float(bin_sec),
            "warmup": int(warmup),
            "max_series": int(max_series),
            "min_events": int(min_events),
            "real_transit_feed": True,
            "passenger_demand_ground_truth": False,
        },
        "rows": rows,
        "summary": summary,
        "paired_deltas": paired,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_report(output_dir / "report.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_GTFS_URL)
    parser.add_argument("--gtfs-zip", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("transit_hrl/data/public_gtfs_bart"))
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/transit_public_gtfs_demand_estimator"))
    parser.add_argument("--methods", nargs="+", default=["ema", "fourier", "dynamic_harmonic_nb"])
    parser.add_argument("--max-series", type=int, default=24)
    parser.add_argument("--bin-sec", type=float, default=300.0)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--min-events", type=int, default=20)
    args = parser.parse_args()
    payload = run_validation(
        output_dir=args.output_dir,
        gtfs_zip=args.gtfs_zip,
        url=str(args.url),
        cache_dir=args.cache_dir,
        methods=list(args.methods),
        max_series=int(args.max_series),
        bin_sec=float(args.bin_sec),
        warmup=int(args.warmup),
        min_events=int(args.min_events),
    )
    best = min(payload["summary"], key=lambda row: row["mse_mean"])
    print(
        f"public_gtfs best={best['method']} mse={best['mse_mean']:.4f} "
        f"series={best['series']} source={payload['metadata']['source_url']}"
    )


if __name__ == "__main__":
    main()
