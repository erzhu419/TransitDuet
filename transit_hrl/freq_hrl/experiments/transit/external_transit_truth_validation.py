"""Validate public external Transit board/alight/load and OD truth sources.

This module keeps external data-source evidence separate from native simulator
performance evidence.  MBTA bus ridership supplies observed stop/trip
boardings, alightings, and load.  MTA subway OD supplies an agency-published OD
estimate, not direct bus OD or onboard load.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("transit_hrl/results/external_transit_truth_validation_latest")
DEFAULT_MBTA_ZIP = Path(
    "transit_hrl/data/public_mbta_bus_ridership_raw/"
    "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop.zip"
)
DEFAULT_MTA_OD_JSON = Path("transit_hrl/data/public_mta_od_raw/mta_subway_od_2024_sample.json")
MBTA_ITEM_URL = "https://mbta-massdot.opendata.arcgis.com/datasets/8daf4a33925a4df59183f860826d29ee"
MBTA_DATA_URL = "https://www.arcgis.com/sharing/rest/content/items/8daf4a33925a4df59183f860826d29ee/data"
MTA_OD_URL = "https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj"
MTA_OD_ENDPOINT = "https://data.ny.gov/resource/jsu2-fbtj.json?$limit=5000"

MBTA_ROUTE_COLUMNS = ("route_id", "gtfs_route_id", "route")
MBTA_DIRECTION_COLUMNS = ("direction_id", "gtfs_direction_id")
MBTA_TRIP_TIME_COLUMNS = ("trip_start_time", "trip start time", "start_time")
MBTA_DAY_TYPE_COLUMNS = ("day_type_name", "day_type", "day type")
MBTA_STOP_COLUMNS = ("stop_id", "gtfs_stop_id", "gtfs stop_id")
MBTA_STOP_SEQUENCE_COLUMNS = ("stop_sequence", "stop sequence")
BOARDING_COLUMNS = ("boardings", "boarding", "ons", "passenger_ons", "boarding_count")
ALIGHTING_COLUMNS = ("alightings", "alighting", "offs", "passenger_offs", "alighting_count")
LOAD_COLUMNS = ("load", "load_", "load_count", "current_load", "onboard_load", "occupancy")
SAMPLE_COLUMNS = ("sample_size", "# of trip samples", "# of trip samples ")

MTA_ORIGIN_COLUMNS = ("origin_station_complex_id", "origin_station_complex_name", "origin")
MTA_DESTINATION_COLUMNS = (
    "destination_station_complex_id",
    "destination_station_complex_name",
    "destination",
)
MTA_RIDERSHIP_COLUMNS = ("estimated_average_ridership", "estimated_ridership", "ridership")
MTA_TIME_COLUMNS = ("timestamp", "hour_of_day", "month", "day_of_week")


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _column_lookup(fieldnames: list[str]) -> dict[str, str]:
    return {_normalize_name(name): str(name) for name in fieldnames}


def _first_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    lookup = _column_lookup(fieldnames)
    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in lookup:
            return lookup[key]
    return ""


def _float_value(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def _season_key(name: str) -> tuple[int, int, str]:
    match = re.search(r"(Spring|Fall)_(\d{4})", name, flags=re.IGNORECASE)
    if not match:
        return (0, 0, name)
    season = match.group(1).lower()
    year = int(match.group(2))
    season_rank = {"spring": 1, "fall": 2}.get(season, 0)
    return (year, season_rank, name)


def _select_mbta_member(members: list[str], requested: str = "") -> str:
    csv_members = [name for name in members if name.lower().endswith(".csv")]
    if requested:
        for name in csv_members:
            if name == requested or name.endswith(requested):
                return name
        raise FileNotFoundError(f"MBTA member not found in zip: {requested}")
    if not csv_members:
        raise FileNotFoundError("No CSV files found in MBTA zip")
    return max(csv_members, key=_season_key)


def _count_unique(rows_seen: set[str], value: Any) -> None:
    text = str(value or "").strip()
    if text:
        rows_seen.add(text)


def summarize_mbta_bus_zip(
    zip_path: Path,
    *,
    member: str = "",
    min_rows: int = 1000,
    min_routes: int = 10,
    min_stops: int = 100,
) -> dict[str, Any]:
    if not zip_path.exists():
        return {
            "source": "mbta_bus_stop_trip_ridership",
            "source_kind": "public_agency_observed_bus_apc",
            "path_status": "missing",
            "claim_status": "external_missing",
            "source_url": MBTA_ITEM_URL,
            "boundary": "MBTA bus ridership zip was not available locally",
        }

    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        selected = _select_mbta_member(members, requested=member)
        routes: set[str] = set()
        stops: set[str] = set()
        trip_bins: set[tuple[str, str, str, str]] = set()
        day_types: set[str] = set()
        row_count = 0
        total_boardings = 0.0
        total_alightings = 0.0
        total_load = 0.0
        max_load = 0.0
        total_samples = 0.0
        fieldnames: list[str] = []
        with zf.open(selected) as raw:
            text_rows = (line.decode("utf-8-sig", errors="replace") for line in raw)
            reader = csv.DictReader(text_rows)
            fieldnames = [str(name) for name in reader.fieldnames or []]
            route_col = _first_column(fieldnames, MBTA_ROUTE_COLUMNS)
            direction_col = _first_column(fieldnames, MBTA_DIRECTION_COLUMNS)
            trip_time_col = _first_column(fieldnames, MBTA_TRIP_TIME_COLUMNS)
            day_type_col = _first_column(fieldnames, MBTA_DAY_TYPE_COLUMNS)
            stop_col = _first_column(fieldnames, MBTA_STOP_COLUMNS)
            boarding_col = _first_column(fieldnames, BOARDING_COLUMNS)
            alighting_col = _first_column(fieldnames, ALIGHTING_COLUMNS)
            load_col = _first_column(fieldnames, LOAD_COLUMNS)
            sample_col = _first_column(fieldnames, SAMPLE_COLUMNS)
            for row in reader:
                row_count += 1
                _count_unique(routes, row.get(route_col))
                _count_unique(stops, row.get(stop_col))
                _count_unique(day_types, row.get(day_type_col))
                if route_col and direction_col and trip_time_col and day_type_col:
                    trip_bins.add((
                        str(row.get(route_col, "")),
                        str(row.get(direction_col, "")),
                        str(row.get(trip_time_col, "")),
                        str(row.get(day_type_col, "")),
                    ))
                boardings = _float_value(row.get(boarding_col))
                alightings = _float_value(row.get(alighting_col))
                load = _float_value(row.get(load_col))
                total_boardings += boardings
                total_alightings += alightings
                total_load += load
                max_load = max(max_load, load)
                total_samples += _float_value(row.get(sample_col))

    has_board_alight = total_boardings > 0.0 and total_alightings > 0.0
    has_load = total_load > 0.0 and max_load > 0.0
    supported = (
        row_count >= int(min_rows)
        and len(routes) >= int(min_routes)
        and len(stops) >= int(min_stops)
        and has_board_alight
        and has_load
    )
    return {
        "source": "mbta_bus_stop_trip_ridership",
        "source_kind": "public_agency_observed_bus_apc",
        "path_status": "present",
        "claim_status": "supported" if supported else "not_supported",
        "source_url": MBTA_ITEM_URL,
        "data_url": MBTA_DATA_URL,
        "agency": "Massachusetts Bay Transportation Authority",
        "license": "CC0 per ArcGIS item metadata",
        "zip_path": str(zip_path),
        "selected_member": selected,
        "csv_members": len(members),
        "rows": row_count,
        "unique_routes": len(routes),
        "unique_stops": len(stops),
        "unique_trip_bins": len(trip_bins),
        "day_types": sorted(day_types),
        "columns": fieldnames,
        "has_boardings": total_boardings > 0.0,
        "has_alightings": total_alightings > 0.0,
        "has_onboard_load": has_load,
        "total_boardings": total_boardings,
        "total_alightings": total_alightings,
        "mean_load": total_load / row_count if row_count else 0.0,
        "max_load": max_load,
        "total_trip_samples": total_samples,
        "boundary": "observed MBTA stop/trip bus averages with boardings, alightings, and onboard load; not OD",
    }


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [dict(row) for row in data["data"] if isinstance(row, dict)]
    return []


def summarize_mta_od_json(
    json_path: Path,
    *,
    min_rows: int = 1000,
    min_origins: int = 20,
    min_destinations: int = 20,
    full_table_rows: int = 0,
) -> dict[str, Any]:
    rows = _read_json_rows(json_path)
    if not rows:
        return {
            "source": "mta_subway_od_estimate_2024",
            "source_kind": "public_agency_estimated_subway_od",
            "path_status": "missing",
            "claim_status": "external_missing",
            "source_url": MTA_OD_URL,
            "boundary": "MTA OD sample was not available locally",
        }

    fieldnames = sorted({str(key) for row in rows for key in row})
    origin_id_col = _first_column(fieldnames, ("origin_station_complex_id",))
    origin_name_col = _first_column(fieldnames, ("origin_station_complex_name",))
    dest_id_col = _first_column(fieldnames, ("destination_station_complex_id",))
    dest_name_col = _first_column(fieldnames, ("destination_station_complex_name",))
    ridership_col = _first_column(fieldnames, MTA_RIDERSHIP_COLUMNS)
    month_col = _first_column(fieldnames, ("month",))
    day_col = _first_column(fieldnames, ("day_of_week",))
    hour_col = _first_column(fieldnames, ("hour_of_day",))
    timestamp_col = _first_column(fieldnames, ("timestamp",))
    origins: set[str] = set()
    destinations: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    months: set[str] = set()
    day_hours: set[tuple[str, str]] = set()
    total_ridership = 0.0
    for row in rows:
        origin = str(row.get(origin_id_col) or row.get(origin_name_col) or "").strip()
        dest = str(row.get(dest_id_col) or row.get(dest_name_col) or "").strip()
        if origin:
            origins.add(origin)
        if dest:
            destinations.add(dest)
        if origin and dest:
            pairs.add((origin, dest))
        _count_unique(months, row.get(month_col))
        if day_col and hour_col:
            day_hours.add((str(row.get(day_col, "")), str(row.get(hour_col, ""))))
        total_ridership += _float_value(row.get(ridership_col))

    has_od = bool(origin_id_col and dest_id_col and pairs)
    has_time = bool(timestamp_col or (month_col and day_col and hour_col))
    supported = (
        len(rows) >= int(min_rows)
        and len(origins) >= int(min_origins)
        and len(destinations) >= int(min_destinations)
        and has_od
        and has_time
        and total_ridership > 0.0
    )
    return {
        "source": "mta_subway_od_estimate_2024",
        "source_kind": "public_agency_estimated_subway_od",
        "path_status": "present",
        "claim_status": "supported" if supported else "not_supported",
        "source_url": MTA_OD_URL,
        "api_endpoint": MTA_OD_ENDPOINT,
        "agency": "Metropolitan Transportation Authority",
        "json_path": str(json_path),
        "sample_rows": len(rows),
        "full_table_rows": int(full_table_rows),
        "unique_origins": len(origins),
        "unique_destinations": len(destinations),
        "unique_od_pairs": len(pairs),
        "unique_months": len(months),
        "unique_day_hour_bins": len(day_hours),
        "columns": fieldnames,
        "has_od_fields": has_od,
        "has_ridership": total_ridership > 0.0,
        "total_estimated_average_ridership_sample": total_ridership,
        "boundary": "agency-published subway OD estimate from fare data; not observed bus OD or onboard load",
    }


def build_claim_boundaries(mbta: dict[str, Any], mta_od: dict[str, Any]) -> list[dict[str, Any]]:
    mbta_supported = str(mbta.get("claim_status", "")) == "supported"
    mta_supported = str(mta_od.get("claim_status", "")) == "supported"
    return [
        {
            "id": "E1",
            "evidence_item": "real_public_bus_stop_board_alight",
            "status": "supported" if mbta_supported and mbta.get("has_alightings") else mbta.get("claim_status", "missing"),
            "allowed_wording": "real public bus stop/trip boardings and alightings from MBTA",
            "forbidden_wording": "GTFS-ride-native board_alight feed unless supplied separately",
            "evidence": (
                f"rows={mbta.get('rows', 0)} routes={mbta.get('unique_routes', 0)} "
                f"stops={mbta.get('unique_stops', 0)} "
                f"total_boardings={mbta.get('total_boardings', 0.0):.1f} "
                f"total_alightings={mbta.get('total_alightings', 0.0):.1f}"
            ),
        },
        {
            "id": "E2",
            "evidence_item": "real_public_bus_stop_onboard_load",
            "status": "supported" if mbta_supported and mbta.get("has_onboard_load") else mbta.get("claim_status", "missing"),
            "allowed_wording": "real public bus stop/trip onboard load averages from MBTA",
            "forbidden_wording": "onboard-load improvement under Freq-HRL unless linked to a control validation",
            "evidence": (
                f"rows={mbta.get('rows', 0)} mean_load={mbta.get('mean_load', 0.0):.4f} "
                f"max_load={mbta.get('max_load', 0.0):.4f}"
            ),
        },
        {
            "id": "E3",
            "evidence_item": "real_public_subway_od_estimate",
            "status": "supported" if mta_supported and mta_od.get("has_od_fields") else mta_od.get("claim_status", "missing"),
            "allowed_wording": "real public agency subway OD estimates from MTA",
            "forbidden_wording": "observed individual OD truth or bus OD/onboard load",
            "evidence": (
                f"sample_rows={mta_od.get('sample_rows', 0)} "
                f"full_table_rows={mta_od.get('full_table_rows', 0)} "
                f"origins={mta_od.get('unique_origins', 0)} "
                f"destinations={mta_od.get('unique_destinations', 0)} "
                f"od_pairs={mta_od.get('unique_od_pairs', 0)}"
            ),
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# External Transit Truth Source Validation",
        "",
        payload["boundary"],
        "",
        f"- evidence scope: `{payload['summary']['evidence_scope']}`",
        f"- supported boundaries: `{payload['summary']['supported_boundaries']}`",
        "",
        "## Sources",
        "",
        "| source | status | rows | coverage | boundary |",
        "|---|---|---:|---|---|",
    ]
    for row in payload["source_coverage"]:
        rows = row.get("rows", row.get("sample_rows", 0))
        coverage = (
            f"routes={row.get('unique_routes', '')} stops={row.get('unique_stops', '')} "
            f"origins={row.get('unique_origins', '')} destinations={row.get('unique_destinations', '')}"
        )
        lines.append(
            f"| {row.get('source', '')} | {row.get('claim_status', '')} | {rows} "
            f"| {coverage} | {row.get('boundary', '')} |"
        )
    lines.extend([
        "",
        "## Claim Boundaries",
        "",
        "| id | evidence item | status | allowed wording | forbidden wording | evidence |",
        "|---|---|---|---|---|---|",
    ])
    for row in payload["claim_boundaries"]:
        lines.append(
            f"| {row['id']} | {row['evidence_item']} | {row['status']} "
            f"| {row['allowed_wording']} | {row['forbidden_wording']} | {row['evidence']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_external_truth_validation(
    output_dir: Path,
    *,
    mbta_zip: Path = DEFAULT_MBTA_ZIP,
    mbta_member: str = "",
    mta_od_json: Path = DEFAULT_MTA_OD_JSON,
    mta_od_total_rows: int = 0,
    download_missing: bool = False,
    min_mbta_rows: int = 1000,
    min_mbta_routes: int = 10,
    min_mbta_stops: int = 100,
    min_mta_od_rows: int = 1000,
    min_mta_od_origins: int = 20,
    min_mta_od_destinations: int = 20,
) -> dict[str, Any]:
    if download_missing and not mbta_zip.exists():
        _download_file(MBTA_DATA_URL, mbta_zip)
    if download_missing and not mta_od_json.exists():
        _download_file(MTA_OD_ENDPOINT, mta_od_json)
    mbta = summarize_mbta_bus_zip(
        mbta_zip,
        member=mbta_member,
        min_rows=min_mbta_rows,
        min_routes=min_mbta_routes,
        min_stops=min_mbta_stops,
    )
    mta_od = summarize_mta_od_json(
        mta_od_json,
        min_rows=min_mta_od_rows,
        min_origins=min_mta_od_origins,
        min_destinations=min_mta_od_destinations,
        full_table_rows=mta_od_total_rows,
    )
    boundaries = build_claim_boundaries(mbta, mta_od)
    supported = sum(1 for row in boundaries if row["status"] == "supported")
    evidence_scope = (
        "real_public_board_alight_load_and_estimated_od"
        if supported == len(boundaries)
        else "partial_external_transit_truth"
    )
    payload = {
        "summary": {
            "evidence_scope": evidence_scope,
            "supported_boundaries": supported,
            "source_count": 2,
        },
        "source_coverage": [mbta, mta_od],
        "claim_boundaries": boundaries,
        "inputs": {
            "mbta_zip": str(mbta_zip),
            "mbta_member": mbta_member,
            "mta_od_json": str(mta_od_json),
            "mta_od_total_rows": int(mta_od_total_rows),
            "download_missing": bool(download_missing),
        },
        "boundary": (
            "This ledger validates public external truth sources for Transit data "
            "coverage. It does not by itself prove a native Freq-HRL control "
            "improvement on those exact files."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "source_coverage.csv", payload["source_coverage"])
    _write_csv(output_dir / "claim_boundaries.csv", payload["claim_boundaries"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_report(output_dir / "report.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mbta-zip", type=Path, default=DEFAULT_MBTA_ZIP)
    parser.add_argument("--mbta-member", default="")
    parser.add_argument("--mta-od-json", type=Path, default=DEFAULT_MTA_OD_JSON)
    parser.add_argument("--mta-od-total-rows", type=int, default=0)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--min-mbta-rows", type=int, default=1000)
    parser.add_argument("--min-mbta-routes", type=int, default=10)
    parser.add_argument("--min-mbta-stops", type=int, default=100)
    parser.add_argument("--min-mta-od-rows", type=int, default=1000)
    parser.add_argument("--min-mta-od-origins", type=int, default=20)
    parser.add_argument("--min-mta-od-destinations", type=int, default=20)
    args = parser.parse_args()
    payload = run_external_truth_validation(
        output_dir=args.output_dir,
        mbta_zip=args.mbta_zip,
        mbta_member=str(args.mbta_member),
        mta_od_json=args.mta_od_json,
        mta_od_total_rows=int(args.mta_od_total_rows),
        download_missing=bool(args.download_missing),
        min_mbta_rows=int(args.min_mbta_rows),
        min_mbta_routes=int(args.min_mbta_routes),
        min_mbta_stops=int(args.min_mbta_stops),
        min_mta_od_rows=int(args.min_mta_od_rows),
        min_mta_od_origins=int(args.min_mta_od_origins),
        min_mta_od_destinations=int(args.min_mta_od_destinations),
    )
    print(
        "external_transit_truth_validation "
        f"scope={payload['summary']['evidence_scope']} "
        f"supported={payload['summary']['supported_boundaries']}"
    )


if __name__ == "__main__":
    main()
