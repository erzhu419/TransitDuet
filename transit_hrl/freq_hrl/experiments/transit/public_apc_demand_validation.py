"""Validate demand estimators on public APC route boarding data.

Halifax Transit publishes half-hourly route boardings collected from bus
Automatic Passenger Counters.  This is real APC passenger-demand evidence, but
the public table reports route-level boardings rather than onboard occupancy,
alightings, or OD flows.
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
from freq_hrl.experiments.transit.public_afc_demand_validation import evaluate_afc_profile_series


DEFAULT_APC_ENDPOINT = (
    "https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/ArcGIS/rest/services/"
    "Transit_Automated_Passenger_Counts/FeatureServer/0/query"
)
DEFAULT_START = "2026-01-01"
DEFAULT_END = "2026-02-01"
APC_PROFILE_METHODS = {"apc_route_profile", "apc_daily_profile"}


def _read_cache(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "OBJECTID",
        "Route_Number",
        "Route_Name",
        "Ridership_Total",
        "Route_Hour",
        "Route_Hour_Description",
        "Route_Date",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fetch_halifax_apc_boardings(
    *,
    endpoint: str = DEFAULT_APC_ENDPOINT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_csv: Path | None = None,
    limit: int = 50000,
    page_size: int = 2000,
) -> list[dict[str, Any]]:
    """Fetch half-hourly route boardings from Halifax's ArcGIS REST table."""
    if cache_csv is not None and cache_csv.exists():
        return _read_cache(cache_csv)
    datetime.fromisoformat(start)
    datetime.fromisoformat(end)
    where = f"Route_Date >= DATE '{start}' AND Route_Date < DATE '{end}'"
    rows: list[dict[str, Any]] = []
    offset = 0
    max_rows = max(1, int(limit))
    page = max(1, min(int(page_size), 2000))
    while len(rows) < max_rows:
        count = min(page, max_rows - len(rows))
        params = {
            "f": "json",
            "where": where,
            "outFields": "OBJECTID,Route_Number,Route_Name,Ridership_Total,Route_Hour,Route_Hour_Description,Route_Date",
            "orderByFields": "Route_Number,Route_Date,Route_Hour,OBJECTID",
            "resultRecordCount": str(count),
            "resultOffset": str(offset),
        }
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Freq-HRL-APC-Validator/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
        features = payload.get("features", [])
        if not features:
            break
        rows.extend(dict(feature.get("attributes", {})) for feature in features)
        if len(features) < count:
            break
        offset += len(features)
    if cache_csv is not None:
        _write_cache(cache_csv, rows)
    return rows


def _route_date(value: Any) -> pd.Timestamp:
    if isinstance(value, (int, float)) or str(value).replace(".", "", 1).isdigit():
        return pd.to_datetime(float(value), unit="ms", utc=True).tz_localize(None).floor("D")
    return pd.Timestamp(str(value)).tz_localize(None).floor("D")


def _route_hour_minutes(value: Any) -> int:
    return int(round(float(value) * 60.0))


def rows_to_route_halfhour_series(
    rows: list[dict[str, Any]],
    *,
    max_series: int = 24,
    min_bins: int = 72,
) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("no APC boarding rows")
    frame = pd.DataFrame(rows)
    required = {"Route_Number", "Route_Name", "Ridership_Total", "Route_Hour", "Route_Date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"APC rows missing columns: {missing}")
    frame["date"] = frame["Route_Date"].map(_route_date)
    frame["route_minutes"] = frame["Route_Hour"].map(_route_hour_minutes)
    frame["timestamp"] = frame["date"] + pd.to_timedelta(frame["route_minutes"], unit="m")
    frame["ridership_value"] = pd.to_numeric(frame["Ridership_Total"], errors="coerce").fillna(0.0)
    grouped = (
        frame.groupby(["Route_Number", "Route_Name", "timestamp"], as_index=False)["ridership_value"]
        .sum()
        .sort_values(["Route_Number", "timestamp"])
    )
    start = grouped["timestamp"].min().floor("30min")
    end = grouped["timestamp"].max().ceil("30min")
    halfhour_index = pd.date_range(start=start, end=end, freq="30min")
    ranked: list[tuple[float, str, np.ndarray]] = []
    for (route_number, route_name), subset in grouped.groupby(["Route_Number", "Route_Name"]):
        series = (
            subset.set_index("timestamp")["ridership_value"]
            .reindex(halfhour_index, fill_value=0.0)
            .astype(float)
        )
        values = series.to_numpy(dtype=np.float64)
        if int(np.count_nonzero(values > 0.0)) < int(min_bins):
            continue
        series_id = f"{route_number}:{route_name}"
        ranked.append((float(values.sum()), series_id, values))
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
                seed=7100 + 31 * idx,
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
        "# Public APC Demand Validation",
        "",
        f"- source: `{metadata.get('source_endpoint', '')}`",
        f"- window: `{metadata.get('start', '')}` to `{metadata.get('end', '')}`",
        f"- rows fetched: {metadata.get('rows_fetched', 0)}",
        "- data path: Halifax Transit half-hourly route boardings collected by bus Automatic Passenger Counters",
        "- boundary: route boardings are real APC passenger demand; they are not onboard occupancy, alightings, or OD flows",
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
    endpoint: str = DEFAULT_APC_ENDPOINT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_csv: Path | None = None,
    methods: list[str] | None = None,
    max_series: int = 24,
    min_bins: int = 72,
    warmup: int = 48,
    limit: int = 50000,
    page_size: int = 2000,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = methods or ["ema", "fourier", "dynamic_harmonic_nb", "apc_route_profile"]
    raw_rows = fetch_halifax_apc_boardings(
        endpoint=str(endpoint),
        start=str(start),
        end=str(end),
        cache_csv=cache_csv,
        limit=int(limit),
        page_size=int(page_size),
    )
    series = rows_to_route_halfhour_series(
        raw_rows,
        max_series=int(max_series),
        min_bins=int(min_bins),
    )
    rows: list[dict[str, Any]] = []
    bin_sec = 30.0 * 60.0
    period_bins = int(round(24.0 * 3600.0 / bin_sec))
    for series_index, (series_id, counts) in enumerate(series.items()):
        for method in methods:
            if str(method) in APC_PROFILE_METHODS:
                rows.append(evaluate_afc_profile_series(
                    method=str(method),
                    series_id=series_id,
                    series_index=series_index,
                    counts=counts,
                    warmup=int(warmup),
                    period_hours=period_bins,
                ))
            else:
                rows.append(evaluate_series(
                    method=str(method),
                    series_id=series_id,
                    series_index=series_index,
                    counts=counts,
                    warmup=int(warmup),
                    bin_sec=bin_sec,
                ))
    if not rows:
        raise ValueError("no usable APC route-halfhour series")
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
            "source": "public_halifax_apc_route_boardings",
            "source_endpoint": str(endpoint),
            "start": str(start),
            "end": str(end),
            "rows_fetched": len(raw_rows),
            "max_series": int(max_series),
            "min_bins": int(min_bins),
            "warmup": int(warmup),
            "real_passenger_demand": True,
            "apc_style_boardings": True,
            "apc_onboard_loads": False,
            "apc_calibrated_profile": bool({str(method) for method in methods} & APC_PROFILE_METHODS),
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
    parser.add_argument("--endpoint", default=DEFAULT_APC_ENDPOINT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cache-csv", type=Path, default=Path("transit_hrl/data/public_apc_halifax/route_boardings.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/transit_public_apc_demand_estimator"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ema", "fourier", "dynamic_harmonic_nb", "apc_route_profile"],
    )
    parser.add_argument("--max-series", type=int, default=24)
    parser.add_argument("--min-bins", type=int, default=72)
    parser.add_argument("--warmup", type=int, default=48)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--page-size", type=int, default=2000)
    args = parser.parse_args()
    payload = run_validation(
        output_dir=args.output_dir,
        endpoint=str(args.endpoint),
        start=str(args.start),
        end=str(args.end),
        cache_csv=args.cache_csv,
        methods=list(args.methods),
        max_series=int(args.max_series),
        min_bins=int(args.min_bins),
        warmup=int(args.warmup),
        limit=int(args.limit),
        page_size=int(args.page_size),
    )
    best = min(payload["summary"], key=lambda row: row["mse_mean"])
    print(
        f"public_apc best={best['method']} mse={best['mse_mean']:.4f} "
        f"series={best['series']} rows={payload['metadata']['rows_fetched']}"
    )


if __name__ == "__main__":
    main()
