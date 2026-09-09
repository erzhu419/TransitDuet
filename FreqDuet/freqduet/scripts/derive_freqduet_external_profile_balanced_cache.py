#!/usr/bin/env python3
"""Derive balanced AFC/APC profile caches from the tracked public samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "external_afc_apc"
DEFAULT_AFC = RAW_ROOT / "public_afc_mta" / "hourly_ridership.csv"
DEFAULT_APC = RAW_ROOT / "public_apc_halifax" / "route_boardings.csv"
DEFAULT_OUT = RAW_ROOT / "balanced_profile_cache_v1"
MTA_SERVICE_DAY = "2024-10-01"
HALIFAX_FIRST_DAY = "2026-01-01"
HALIFAX_LAST_DAY = "2026-01-07"
HALIFAX_DAILY_ROUTES = ("1", "10B", "10C")
HALIFAX_WEEKDAY_ROUTES = ("10A", "123", "127", "135")
HALIFAX_WEEKDAY_SERVICE_DAYS = (
    "2026-01-02",
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def derive_mta(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = pd.read_csv(path)
    required = {
        "transit_timestamp",
        "station_complex_id",
        "station_complex",
        "ridership",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing MTA columns: {sorted(missing)}")
    data["timestamp"] = pd.to_datetime(
        data["transit_timestamp"], errors="coerce"
    )
    data["ridership"] = pd.to_numeric(data["ridership"], errors="coerce")
    day = data[data["timestamp"].dt.strftime("%Y-%m-%d").eq(MTA_SERVICE_DAY)]
    complete_ids = []
    for station_id, group in day.groupby("station_complex_id"):
        hours = set(group["timestamp"].dt.hour.dropna().astype(int))
        if len(group) == 24 and hours == set(range(24)):
            complete_ids.append(station_id)
    selected = day[day["station_complex_id"].isin(complete_ids)].copy()
    if len(complete_ids) != 39 or len(selected) != 39 * 24:
        raise ValueError(
            "tracked MTA cache no longer has the frozen 39 x 24 complete subset"
        )
    hourly = (
        selected.assign(hour_bin=selected["timestamp"].dt.hour)
        .groupby("hour_bin", as_index=False)
        .agg(demand=("ridership", "sum"), source_rows=("ridership", "size"))
    )
    rows = [{
        "hour_bin": int(row.hour_bin),
        "demand": float(row.demand),
        "series_count": len(complete_ids),
        "profile_units": len(complete_ids),
        "source_rows": int(row.source_rows),
        "first_time": f"{MTA_SERVICE_DAY}T00:00:00",
        "last_time": f"{MTA_SERVICE_DAY}T23:00:00",
        "aggregation_scope": "39 complete cached station-complex days",
    } for row in hourly.itertuples(index=False)]
    return rows, {
        "source_file": str(path.relative_to(ROOT)),
        "raw_rows": int(len(data)),
        "service_day": MTA_SERVICE_DAY,
        "selected_station_complexes": len(complete_ids),
        "selected_source_rows": int(len(selected)),
        "selection": "station complex has exactly one observation for each hour 0-23",
        "excluded_rows": int(len(data) - len(selected)),
    }


def derive_halifax(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = pd.read_csv(path, dtype={"Route_Number": str})
    required = {
        "OBJECTID",
        "Route_Number",
        "Ridership_Total",
        "Route_Hour",
        "Route_Date",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing Halifax columns: {sorted(missing)}")
    data["service_date"] = pd.to_datetime(
        data["Route_Date"], unit="ms", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["Route_Hour"] = pd.to_numeric(data["Route_Hour"], errors="coerce")
    data["Ridership_Total"] = pd.to_numeric(
        data["Ridership_Total"], errors="coerce"
    )

    all_days = pd.date_range(
        HALIFAX_FIRST_DAY, HALIFAX_LAST_DAY, freq="D"
    ).strftime("%Y-%m-%d").tolist()
    expected_pairs = {
        *((route, day) for route in HALIFAX_DAILY_ROUTES for day in all_days),
        *((route, day) for route in HALIFAX_WEEKDAY_ROUTES
          for day in HALIFAX_WEEKDAY_SERVICE_DAYS),
    }
    selected_routes = set(HALIFAX_DAILY_ROUTES + HALIFAX_WEEKDAY_ROUTES)
    selected = data[
        data["Route_Number"].isin(selected_routes)
        & data["service_date"].between(HALIFAX_FIRST_DAY, HALIFAX_LAST_DAY)
    ].copy()
    observed_pairs = set(zip(selected["Route_Number"], selected["service_date"]))
    if observed_pairs != expected_pairs:
        raise ValueError("tracked Halifax cache no longer has the frozen route-day set")
    if selected.duplicated(["Route_Number", "service_date", "Route_Hour"]).any():
        raise ValueError("tracked Halifax cache has duplicate route-day time bins")

    selected["hour_bin"] = selected["Route_Hour"] % 24.0
    hourly = (
        selected.groupby("hour_bin", as_index=False)
        .agg(
            demand=("Ridership_Total", "sum"),
            source_rows=("Ridership_Total", "size"),
        )
        .sort_values("hour_bin")
    )
    rows = [{
        "hour_bin": float(row.hour_bin),
        "demand": float(row.demand),
        "series_count": len(selected_routes),
        "profile_units": len(observed_pairs),
        "source_rows": int(row.source_rows),
        "first_time": HALIFAX_FIRST_DAY,
        "last_time": HALIFAX_LAST_DAY,
        "aggregation_scope": "7 complete cached routes across 37 route-days",
    } for row in hourly.itertuples(index=False)]
    excluded_routes = sorted(set(data["Route_Number"]) - selected_routes)
    return rows, {
        "source_file": str(path.relative_to(ROOT)),
        "raw_rows": int(len(data)),
        "service_days": [HALIFAX_FIRST_DAY, HALIFAX_LAST_DAY],
        "selected_routes": sorted(selected_routes),
        "selected_route_days": len(observed_pairs),
        "selected_source_rows": int(len(selected)),
        "selection": (
            "all observed days for routes 1, 10B, and 10C; all weekdays for "
            "routes 10A, 123, 127, and 135"
        ),
        "excluded_incomplete_routes": excluded_routes,
        "excluded_rows": int(len(data) - len(selected)),
    }


def derive(afc_path: Path, apc_path: Path, out_dir: Path) -> dict[str, Any]:
    mta_rows, mta_meta = derive_mta(afc_path)
    halifax_rows, halifax_meta = derive_halifax(apc_path)
    mta_name = "mta_complete_station_day_2024-10-01.csv"
    halifax_name = "halifax_complete_route_days_2026-01-01_2026-01-07.csv"
    write_csv(out_dir / mta_name, mta_rows)
    write_csv(out_dir / halifax_name, halifax_rows)
    manifest = {
        "manifest_version": "freqduet-balanced-external-profile-cache-v1",
        "claim_boundary": (
            "Complete subsets of bounded public cache files for descriptive "
            "demand-shape realism only; not population estimates, matched field "
            "calibration, or policy-effect evidence."
        ),
        "sources": {
            "public_afc_mta": {"output_file": mta_name, **mta_meta},
            "public_apc_halifax": {
                "output_file": halifax_name,
                **halifax_meta,
            },
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "derivation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--afc-csv", type=Path, default=DEFAULT_AFC)
    parser.add_argument("--apc-csv", type=Path, default=DEFAULT_APC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = derive(
        args.afc_csv.resolve(), args.apc_csv.resolve(), args.out_dir.resolve()
    )
    print(json.dumps({
        "status": "balanced_external_profile_cache_complete",
        "out_dir": str(args.out_dir.resolve()),
        "mta_station_complexes": manifest["sources"]["public_afc_mta"][
            "selected_station_complexes"
        ],
        "halifax_route_days": manifest["sources"]["public_apc_halifax"][
            "selected_route_days"
        ],
    }, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
