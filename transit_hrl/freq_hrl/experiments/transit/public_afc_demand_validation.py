"""Validate demand estimators on public AFC-style subway ridership data.

The MTA hourly ridership feed aggregates OMNY/MetroCard passenger entries by
station complex and hour.  This is closer to AFC passenger-demand evidence than
GTFS schedule-event proxies because the target is observed ridership, not
scheduled service activity.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.experiments.transit.demand_estimator_validation import COUNT_CALIBRATION_ID
from freq_hrl.experiments.transit.local_data_demand_validation import evaluate_series


DEFAULT_AFC_ENDPOINT = "https://data.ny.gov/resource/wujg-7c2s.json"
DEFAULT_START = "2024-10-01T00:00:00"
DEFAULT_END = "2024-10-15T00:00:00"
AFC_PROFILE_METHODS = {
    "afc_daily_profile",
    "afc_daily_profile_nb",
    "afc_calibrated_profile",
}


def _timestamp_key(value: Any) -> pd.Timestamp:
    return pd.Timestamp(str(value)).tz_localize(None)


def fetch_mta_hourly_ridership(
    *,
    endpoint: str = DEFAULT_AFC_ENDPOINT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_csv: Path | None = None,
    limit: int = 50000,
) -> list[dict[str, Any]]:
    """Fetch station-hour subway ridership aggregates from NY Open Data."""
    if cache_csv is not None and cache_csv.exists():
        with cache_csv.open("r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    params = {
        "$select": (
            "transit_timestamp, station_complex_id, station_complex, "
            "sum(ridership) as ridership"
        ),
        "$where": (
            "transit_mode='subway' "
            f"AND transit_timestamp between '{start}' and '{end}'"
        ),
        "$group": "transit_timestamp, station_complex_id, station_complex",
        "$order": "station_complex_id, transit_timestamp",
        "$limit": str(int(limit)),
    }
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Freq-HRL-AFC-Validator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        rows = json.loads(response.read().decode("utf-8"))
    if cache_csv is not None:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["transit_timestamp", "station_complex_id", "station_complex", "ridership"]
        with cache_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return rows


def rows_to_station_hour_series(
    rows: list[dict[str, Any]],
    *,
    max_series: int = 24,
    min_hours: int = 72,
) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("no AFC ridership rows")
    frame = pd.DataFrame(rows)
    required = {"transit_timestamp", "station_complex_id", "station_complex", "ridership"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"AFC rows missing columns: {missing}")
    frame["timestamp"] = frame["transit_timestamp"].map(_timestamp_key)
    frame["ridership_value"] = pd.to_numeric(frame["ridership"], errors="coerce").fillna(0.0)
    grouped = (
        frame.groupby(["station_complex_id", "station_complex", "timestamp"], as_index=False)["ridership_value"]
        .sum()
        .sort_values(["station_complex_id", "timestamp"])
    )
    start = grouped["timestamp"].min().floor("h")
    end = grouped["timestamp"].max().ceil("h")
    hourly_index = pd.date_range(start=start, end=end, freq="h")
    ranked: list[tuple[float, str, np.ndarray]] = []
    for (station_id, station_name), subset in grouped.groupby(["station_complex_id", "station_complex"]):
        series = (
            subset.set_index("timestamp")["ridership_value"]
            .reindex(hourly_index, fill_value=0.0)
            .astype(float)
        )
        nonzero_hours = int(np.count_nonzero(series.to_numpy() > 0.0))
        if nonzero_hours < int(min_hours):
            continue
        series_id = f"{station_id}:{station_name}"
        ranked.append((float(series.sum()), series_id, series.to_numpy(dtype=np.float64)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return {
        series_id: values
        for _, series_id, values in ranked[:max(1, int(max_series))]
    }


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


def evaluate_afc_profile_series(
    method: str,
    series_id: str,
    series_index: int,
    counts: np.ndarray,
    *,
    warmup: int,
    period_hours: int = 24,
    alpha: float = 0.80,
    global_shrink: float = 0.35,
) -> dict[str, Any]:
    """Causal station-hour AFC profile estimator.

    The model keeps an online seasonal profile for each hour-of-day slot and
    shrinks it toward a global EMA.  It is intentionally simple but important:
    real AFC station entries have strong daily periodicity, and the default
    harmonic NB path underfits this structure on the current MTA feed.
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    period = max(1, int(period_hours))
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    shrink = float(np.clip(global_shrink, 0.0, 1.0))
    profile = np.zeros(period, dtype=np.float64)
    seen = np.zeros(period, dtype=np.float64)
    global_mean = 0.0
    preds: list[float] = []
    targets: list[float] = []
    for idx, count in enumerate(counts):
        count = float(max(count, 0.0))
        slot = int(idx % period)
        if idx >= int(warmup):
            slot_mean = profile[slot] if seen[slot] > 0.0 else global_mean
            pred = (1.0 - shrink) * slot_mean + shrink * global_mean
            preds.append(max(float(pred), 1e-6))
            targets.append(count)
        if idx == 0:
            global_mean = count
            profile[slot] = count
            seen[slot] = 1.0
            continue
        global_mean = (1.0 - alpha) * global_mean + alpha * count
        if seen[slot] <= 0.0:
            profile[slot] = count
        else:
            profile[slot] = (1.0 - alpha) * profile[slot] + alpha * count
        seen[slot] += 1.0
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


def paired_method_stats(rows: list[dict[str, Any]], reference: str = "fourier") -> list[dict[str, Any]]:
    out = []
    for idx, method in enumerate(sorted({str(row["method"]) for row in rows})):
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
                seed=6100 + 29 * idx,
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
    metadata = payload["metadata"]
    best = min(summary, key=lambda row: row["mse_mean"])
    lines = [
        "# Public AFC Demand Validation",
        "",
        f"- source: `{metadata.get('source_endpoint', '')}`",
        f"- window: `{metadata.get('start', '')}` to `{metadata.get('end', '')}`",
        f"- rows fetched: {metadata.get('rows_fetched', 0)}",
        "- data path: NY Open Data/MTA hourly subway ridership, aggregated by station complex and hour",
        "- boundary: subway station entries are AFC-style passenger demand; they are not APC onboard loads or OD flows",
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
    endpoint: str = DEFAULT_AFC_ENDPOINT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_csv: Path | None = None,
    methods: list[str] | None = None,
    max_series: int = 24,
    min_hours: int = 72,
    warmup: int = 24,
    limit: int = 50000,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Validate timestamps early so malformed Socrata predicates are easier to diagnose.
    datetime.fromisoformat(start)
    datetime.fromisoformat(end)
    methods = methods or ["ema", "fourier", "dynamic_harmonic_nb", "afc_daily_profile"]
    raw_rows = fetch_mta_hourly_ridership(
        endpoint=str(endpoint),
        start=str(start),
        end=str(end),
        cache_csv=cache_csv,
        limit=int(limit),
    )
    series = rows_to_station_hour_series(
        raw_rows,
        max_series=int(max_series),
        min_hours=int(min_hours),
    )
    rows: list[dict[str, Any]] = []
    for series_index, (series_id, counts) in enumerate(series.items()):
        for method in methods:
            if str(method) in AFC_PROFILE_METHODS:
                rows.append(evaluate_afc_profile_series(
                    method=method,
                    series_id=series_id,
                    series_index=series_index,
                    counts=counts,
                    warmup=int(warmup),
                ))
            else:
                rows.append(evaluate_series(
                    method=method,
                    series_id=series_id,
                    series_index=series_index,
                    counts=counts,
                    warmup=int(warmup),
                    bin_sec=3600.0,
                ))
    if not rows:
        raise ValueError("no usable AFC station-hour series")
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
            "source": "public_mta_hourly_ridership_afc",
            "source_endpoint": str(endpoint),
            "start": str(start),
            "end": str(end),
            "rows_fetched": len(raw_rows),
            "max_series": int(max_series),
            "min_hours": int(min_hours),
            "warmup": int(warmup),
            "real_passenger_demand": True,
            "afc_style_entries": True,
            "apc_onboard_loads": False,
            "afc_calibrated_profile": bool({str(method) for method in methods} & AFC_PROFILE_METHODS),
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
    parser.add_argument("--endpoint", default=DEFAULT_AFC_ENDPOINT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cache-csv", type=Path, default=Path("transit_hrl/data/public_afc_mta/hourly_ridership.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/transit_public_afc_demand_estimator"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ema", "fourier", "dynamic_harmonic_nb", "afc_daily_profile"],
    )
    parser.add_argument("--max-series", type=int, default=24)
    parser.add_argument("--min-hours", type=int, default=72)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--limit", type=int, default=50000)
    args = parser.parse_args()
    payload = run_validation(
        output_dir=args.output_dir,
        endpoint=str(args.endpoint),
        start=str(args.start),
        end=str(args.end),
        cache_csv=args.cache_csv,
        methods=list(args.methods),
        max_series=int(args.max_series),
        min_hours=int(args.min_hours),
        warmup=int(args.warmup),
        limit=int(args.limit),
    )
    best = min(payload["summary"], key=lambda row: row["mse_mean"])
    print(
        f"public_afc best={best['method']} mse={best['mse_mean']:.4f} "
        f"series={best['series']} rows={payload['metadata']['rows_fetched']}"
    )


if __name__ == "__main__":
    main()
