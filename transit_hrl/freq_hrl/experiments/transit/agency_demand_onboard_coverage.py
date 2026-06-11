"""Audit real-agency demand, OD, onboard-load, and native service evidence.

This module is deliberately conservative.  Public AFC/APC feeds and native
simulation metrics answer different questions: AFC/APC files can establish
observed demand coverage, while the copied/native Transit loop establishes
onboard-load and alighting behavior only inside the simulator unless an
external GTFS-ride/APC feed supplies those fields directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_AFC_CSV = Path("transit_hrl/data/public_afc_mta/hourly_ridership.csv")
DEFAULT_APC_CSV = Path("transit_hrl/data/public_apc_halifax/route_boardings.csv")
DEFAULT_NATIVE_SUMMARY = Path(
    "transit_hrl/results/transit_native_real_demand_service_response_v7_48pair_merged/summary.json"
)
DEFAULT_OUTPUT_DIR = Path("transit_hrl/results/agency_demand_onboard_coverage_latest")

AFC_TIMESTAMP_COLUMNS = ("transit_timestamp", "timestamp", "datetime", "date_time")
AFC_STATION_COLUMNS = ("station_complex_id", "station_id", "stop_id", "station")
AFC_COUNT_COLUMNS = ("ridership", "entries", "entry_count", "count")

APC_ROUTE_COLUMNS = ("Route_Number", "route_id", "route", "route_number")
APC_DATE_COLUMNS = ("Route_Date", "service_date", "date")
APC_TIME_COLUMNS = ("Route_Hour", "hour", "time_bin", "timestamp")
BOARDING_COLUMNS = (
    "Ridership_Total",
    "boardings",
    "boarding",
    "boarding_count",
    "ons",
    "passenger_ons",
)
ALIGHTING_COLUMNS = (
    "alightings",
    "alighting",
    "alighting_count",
    "offs",
    "passenger_offs",
)
LOAD_COLUMNS = (
    "current_load",
    "load_count",
    "onboard_load",
    "onboard",
    "passenger_load",
    "occupancy",
)
ORIGIN_COLUMNS = ("origin_stop_id", "boarding_stop_id", "from_stop_id", "origin", "origin_id")
DESTINATION_COLUMNS = (
    "destination_stop_id",
    "alighting_stop_id",
    "to_stop_id",
    "destination",
    "destination_id",
)
NATIVE_METRICS = (
    "control_score",
    "ep_reward",
    "avg_wait_min",
    "native_avg_board_wait_min",
    "native_boarded_pax",
    "native_alighted_pax",
    "native_completed_throughput_pax",
    "native_unalighted_pax",
    "native_avg_onboard_load",
    "native_peak_onboard_load",
    "LowerLFDrift",
)


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path or not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _read_json(path: Path) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _column_lookup(fieldnames: list[str]) -> dict[str, str]:
    return {str(name).lower(): str(name) for name in fieldnames}


def _first_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    lookup = _column_lookup(fieldnames)
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return ""


def _has_column(fieldnames: list[str], candidates: tuple[str, ...]) -> bool:
    return bool(_first_column(fieldnames, candidates))


def _float_value(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sum_column(rows: list[dict[str, str]], column: str) -> float:
    if not column:
        return 0.0
    return float(sum(_float_value(row.get(column)) for row in rows))


def _unique_count(rows: list[dict[str, str]], column: str) -> int:
    if not column:
        return 0
    return len({str(row.get(column, "")) for row in rows if str(row.get(column, ""))})


def _range_strings(rows: list[dict[str, str]], column: str) -> tuple[str, str]:
    if not column:
        return "", ""
    values = sorted({str(row.get(column, "")) for row in rows if str(row.get(column, ""))})
    return (values[0], values[-1]) if values else ("", "")


def _status_from(condition: bool, *, missing: bool = False) -> str:
    if missing:
        return "external_missing"
    return "supported" if condition else "not_supported"


def summarize_afc_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    *,
    min_rows: int,
    min_station_count: int,
    min_time_bins: int,
) -> dict[str, Any]:
    timestamp_col = _first_column(fieldnames, AFC_TIMESTAMP_COLUMNS)
    station_col = _first_column(fieldnames, AFC_STATION_COLUMNS)
    count_col = _first_column(fieldnames, AFC_COUNT_COLUMNS)
    start, end = _range_strings(rows, timestamp_col)
    unique_stations = _unique_count(rows, station_col)
    unique_timestamps = _unique_count(rows, timestamp_col)
    total_ridership = _sum_column(rows, count_col)
    supported = (
        len(rows) >= int(min_rows)
        and unique_stations >= int(min_station_count)
        and unique_timestamps >= int(min_time_bins)
        and total_ridership > 0.0
    )
    return {
        "source": "public_afc_station_hour",
        "path_status": "present" if rows else "missing",
        "rows": len(rows),
        "unique_station_complexes": unique_stations,
        "unique_time_bins": unique_timestamps,
        "time_start": start,
        "time_end": end,
        "total_ridership": total_ridership,
        "has_station_entries": bool(count_col),
        "has_onboard_load": _has_column(fieldnames, LOAD_COLUMNS),
        "has_alightings": _has_column(fieldnames, ALIGHTING_COLUMNS),
        "has_od_fields": _has_column(fieldnames, ORIGIN_COLUMNS) and _has_column(fieldnames, DESTINATION_COLUMNS),
        "claim_status": _status_from(supported, missing=not rows),
        "boundary": "observed station-hour entries; not onboard occupancy, alightings, or OD unless those fields are present",
    }


def summarize_apc_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    *,
    min_rows: int,
    min_route_count: int,
    min_time_bins: int,
) -> dict[str, Any]:
    route_col = _first_column(fieldnames, APC_ROUTE_COLUMNS)
    date_col = _first_column(fieldnames, APC_DATE_COLUMNS)
    time_col = _first_column(fieldnames, APC_TIME_COLUMNS)
    boarding_col = _first_column(fieldnames, BOARDING_COLUMNS)
    route_time_bins = {
        (
            str(row.get(route_col, "")),
            str(row.get(date_col, "")),
            str(row.get(time_col, "")),
        )
        for row in rows
        if route_col and date_col and time_col
    }
    start, end = _range_strings(rows, date_col)
    unique_routes = _unique_count(rows, route_col)
    total_boardings = _sum_column(rows, boarding_col)
    supported = (
        len(rows) >= int(min_rows)
        and unique_routes >= int(min_route_count)
        and len(route_time_bins) >= int(min_time_bins)
        and total_boardings > 0.0
    )
    return {
        "source": "public_apc_route_boarding",
        "path_status": "present" if rows else "missing",
        "rows": len(rows),
        "unique_routes": unique_routes,
        "unique_route_time_bins": len(route_time_bins),
        "date_start": start,
        "date_end": end,
        "total_boardings": total_boardings,
        "has_boardings": bool(boarding_col),
        "has_onboard_load": _has_column(fieldnames, LOAD_COLUMNS),
        "has_alightings": _has_column(fieldnames, ALIGHTING_COLUMNS),
        "has_od_fields": _has_column(fieldnames, ORIGIN_COLUMNS) and _has_column(fieldnames, DESTINATION_COLUMNS),
        "claim_status": _status_from(supported, missing=not rows),
        "boundary": "observed route boardings; not onboard occupancy, alightings, or OD unless those fields are present",
    }


def _real_agency_source(source_kind: str) -> bool:
    return str(source_kind).lower() in {
        "real_agency",
        "public_agency",
        "agency_export",
        "gtfs_ride_public",
    }


def summarize_gtfs_ride(
    gtfs_ride_dir: Path | None,
    *,
    source_kind: str = "unknown",
    source_url: str = "",
    agency: str = "",
) -> dict[str, Any]:
    if gtfs_ride_dir is None:
        return {
            "source": "gtfs_ride_external",
            "path_status": "not_configured",
            "claim_status": "external_missing",
            "source_kind": str(source_kind),
            "source_verified": False,
            "source_url": str(source_url),
            "agency": str(agency),
            "boundary": "optional external GTFS-ride directory was not supplied",
        }
    board_rows, board_fields = _read_csv_rows(gtfs_ride_dir / "board_alight.txt")
    rider_rows, rider_fields = _read_csv_rows(gtfs_ride_dir / "rider_trip.txt")
    capacity_rows, capacity_fields = _read_csv_rows(gtfs_ride_dir / "trip_capacity.txt")
    trip_col = _first_column(board_fields, ("trip_id",))
    stop_col = _first_column(board_fields, ("stop_id",))
    board_col = _first_column(board_fields, BOARDING_COLUMNS)
    alight_col = _first_column(board_fields, ALIGHTING_COLUMNS)
    load_col = _first_column(board_fields, LOAD_COLUMNS)
    origin_col = _first_column(rider_fields or board_fields, ORIGIN_COLUMNS)
    dest_col = _first_column(rider_fields or board_fields, DESTINATION_COLUMNS)
    od_rows = rider_rows if rider_rows else board_rows
    has_od = bool(origin_col and dest_col and od_rows)
    source_verified = _real_agency_source(source_kind)
    rows_present = bool(board_rows or rider_rows)
    claim_status = "external_missing"
    if rows_present and source_verified:
        claim_status = "supported"
    elif rows_present:
        claim_status = "schema_supported_unverified_source"
    return {
        "source": "gtfs_ride_external",
        "path_status": "present" if gtfs_ride_dir.exists() else "missing",
        "source_kind": str(source_kind),
        "source_verified": bool(source_verified),
        "source_url": str(source_url),
        "agency": str(agency),
        "board_alight_rows": len(board_rows),
        "rider_trip_rows": len(rider_rows),
        "trip_capacity_rows": len(capacity_rows),
        "unique_trips": _unique_count(board_rows, trip_col),
        "unique_stops": _unique_count(board_rows, stop_col),
        "total_boardings": _sum_column(board_rows, board_col),
        "total_alightings": _sum_column(board_rows, alight_col),
        "mean_load_count": (
            float(np.mean([_float_value(row.get(load_col)) for row in board_rows]))
            if board_rows and load_col
            else 0.0
        ),
        "has_boardings": bool(board_col),
        "has_alightings": bool(alight_col),
        "has_onboard_load": bool(load_col),
        "has_od_fields": has_od,
        "has_capacity": bool(capacity_rows and capacity_fields),
        "claim_status": claim_status,
        "boundary": (
            "external stop-level board/alight, load, and OD support only closes "
            "paper claims when source_kind is a verified real agency feed"
        ),
    }


def _native_metric_summaries(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    subset = [row for row in rows if str(row.get("variant", "")) == str(variant)]
    out: list[dict[str, Any]] = []
    for metric in NATIVE_METRICS:
        values = [
            _float_value(row.get(metric))
            for row in subset
            if row.get(metric) is not None and str(row.get(metric)) != ""
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        out.append({
            "variant": variant,
            "metric": metric,
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
    return out


def _find_check(checks: list[dict[str, Any]], metric: str, contains: str = "") -> dict[str, Any]:
    for row in checks:
        if str(row.get("metric", "")) != metric:
            continue
        if contains and contains not in str(row.get("check", "")):
            continue
        return dict(row)
    return {}


def summarize_native_summary(data: dict[str, Any], *, variant: str) -> dict[str, Any]:
    rows = [dict(row) for row in data.get("rows", []) if isinstance(row, dict)]
    checks = [dict(row) for row in data.get("paired_checks", []) if isinstance(row, dict)]
    variant_rows = [row for row in rows if str(row.get("variant", "")) == str(variant)]
    metrics = _native_metric_summaries(rows, variant)
    key_checks = {
        "score": _find_check(checks, "control_score"),
        "reward": _find_check(checks, "ep_reward"),
        "board_wait": _find_check(checks, "native_avg_board_wait_min"),
        "alighted": _find_check(checks, "native_alighted_pax"),
        "throughput": _find_check(checks, "native_completed_throughput_pax"),
        "onboard_load": _find_check(checks, "native_avg_onboard_load"),
        "lower_lf_drift": _find_check(checks, "LowerLFDrift"),
    }
    supported_service = all(
        str(key_checks[name].get("status", "")) == "supported"
        for name in ("score", "reward", "board_wait", "alighted", "throughput")
    )
    return {
        "source": "native_real_demand_service_response",
        "path_status": "present" if data else "missing",
        "variant": variant,
        "rows": len(variant_rows),
        "seeds": len({int(row.get("seed", 0)) for row in variant_rows if str(row.get("seed", "")).strip()}),
        "sources": sorted({str(row.get("source", "")) for row in variant_rows if row.get("source")}),
        "metrics": metrics,
        "key_checks": key_checks,
        "native_service_response_status": "supported" if supported_service else "not_supported",
        "native_onboard_loop_status": "supported" if any(row["metric"] == "native_avg_onboard_load" for row in metrics) else "not_supported",
        "native_onboard_improvement_status": str(key_checks["onboard_load"].get("status", "missing")),
        "boundary": "native simulator service loop; onboard/alighting metrics are not external ground truth unless supplied by agency data",
    }


def build_claim_boundaries(
    afc: dict[str, Any],
    apc: dict[str, Any],
    gtfs_ride: dict[str, Any],
    native: dict[str, Any],
) -> list[dict[str, Any]]:
    gtfs_has_alight = bool(gtfs_ride.get("has_alightings"))
    gtfs_has_load = bool(gtfs_ride.get("has_onboard_load"))
    gtfs_has_od = bool(gtfs_ride.get("has_od_fields"))
    gtfs_source_verified = bool(gtfs_ride.get("source_verified", False))

    def _external_truth_status(has_fields: bool) -> str:
        if has_fields and gtfs_source_verified:
            return "supported"
        if has_fields:
            return "schema_supported_unverified_source"
        return "external_missing"

    gtfs_evidence_suffix = (
        f" source_kind={gtfs_ride.get('source_kind', 'unknown')} "
        f"source_verified={gtfs_source_verified}"
    )
    return [
        {
            "id": "A1",
            "evidence_item": "real_afc_station_hour_demand",
            "status": afc.get("claim_status", "missing"),
            "allowed_wording": "real AFC-style station-hour entry demand",
            "forbidden_wording": "real OD or onboard-load ground truth",
            "evidence": f"rows={afc.get('rows', 0)} stations={afc.get('unique_station_complexes', 0)} time_bins={afc.get('unique_time_bins', 0)}",
        },
        {
            "id": "A2",
            "evidence_item": "real_apc_route_boarding_demand",
            "status": apc.get("claim_status", "missing"),
            "allowed_wording": "real APC-style route boarding demand",
            "forbidden_wording": "real onboard occupancy, alighting, or OD ground truth unless columns exist",
            "evidence": f"rows={apc.get('rows', 0)} routes={apc.get('unique_routes', 0)} route_time_bins={apc.get('unique_route_time_bins', 0)}",
        },
        {
            "id": "A3",
            "evidence_item": "native_service_response_wait_alighting_throughput",
            "status": native.get("native_service_response_status", "missing"),
            "allowed_wording": "native public-demand service-response loop improves wait/alighting/throughput",
            "forbidden_wording": "external agency alighting or onboard-load ground-truth improvement",
            "evidence": (
                f"rows={native.get('rows', 0)} seeds={native.get('seeds', 0)} "
                f"board_wait={native.get('key_checks', {}).get('board_wait', {}).get('status', 'missing')} "
                f"alighted={native.get('key_checks', {}).get('alighted', {}).get('status', 'missing')} "
                f"throughput={native.get('key_checks', {}).get('throughput', {}).get('status', 'missing')}"
            ),
        },
        {
            "id": "A4",
            "evidence_item": "native_onboard_load_loop",
            "status": native.get("native_onboard_loop_status", "missing"),
            "allowed_wording": "native onboard-load metric is recorded and audited",
            "forbidden_wording": "native onboard-load improvement is supported if CI is inconclusive",
            "evidence": f"onboard_improvement={native.get('native_onboard_improvement_status', 'missing')}",
        },
        {
            "id": "A5",
            "evidence_item": "real_gtfs_ride_board_alight",
            "status": _external_truth_status(gtfs_has_alight),
            "allowed_wording": "real stop-level board/alight validation when GTFS-ride board_alight is supplied",
            "forbidden_wording": "real alighting ground truth for the current AFC/APC-only cache",
            "evidence": (
                f"board_alight_rows={gtfs_ride.get('board_alight_rows', 0)} "
                f"has_alightings={gtfs_has_alight}{gtfs_evidence_suffix}"
            ),
        },
        {
            "id": "A6",
            "evidence_item": "real_gtfs_ride_onboard_load",
            "status": _external_truth_status(gtfs_has_load),
            "allowed_wording": "real onboard-load validation when GTFS-ride load_count/current_load is supplied",
            "forbidden_wording": "real onboard-load ground truth for the current AFC/APC-only cache",
            "evidence": (
                f"board_alight_rows={gtfs_ride.get('board_alight_rows', 0)} "
                f"has_onboard_load={gtfs_has_load}{gtfs_evidence_suffix}"
            ),
        },
        {
            "id": "A7",
            "evidence_item": "real_gtfs_ride_od",
            "status": _external_truth_status(gtfs_has_od),
            "allowed_wording": "real OD validation when rider_trip or origin/destination fields are supplied",
            "forbidden_wording": "real OD ground truth for the current AFC/APC-only cache",
            "evidence": (
                f"rider_trip_rows={gtfs_ride.get('rider_trip_rows', 0)} "
                f"has_od_fields={gtfs_has_od}{gtfs_evidence_suffix}"
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
    source_rows = payload["source_coverage"]
    boundaries = payload["claim_boundaries"]
    native = payload["native_service"]
    lines = [
        "# Agency Demand and Onboard-Load Coverage Ledger",
        "",
        payload["boundary"],
        "",
        f"- overall scope: `{payload['summary']['evidence_scope']}`",
        f"- supported boundary rows: `{payload['summary']['supported_boundaries']}`",
        f"- external-missing boundary rows: `{payload['summary']['external_missing_boundaries']}`",
        "",
        "## Source Coverage",
        "",
        "| source | status | rows | coverage | boundary |",
        "|---|---|---:|---|---|",
    ]
    for row in source_rows:
        coverage = (
            f"stations={row.get('unique_station_complexes', '')} "
            f"routes={row.get('unique_routes', '')} "
            f"time_bins={row.get('unique_time_bins', row.get('unique_route_time_bins', ''))}"
        )
        rows = row.get("rows", row.get("board_alight_rows", 0))
        lines.append(
            f"| {row.get('source', '')} | {row.get('claim_status', row.get('path_status', ''))} "
            f"| {rows} | {coverage} | {row.get('boundary', '')} |"
        )
    lines.extend([
        "",
        "## Claim Boundaries",
        "",
        "| id | evidence item | status | allowed wording | forbidden wording | evidence |",
        "|---|---|---|---|---|---|",
    ])
    for row in boundaries:
        lines.append(
            f"| {row['id']} | {row['evidence_item']} | {row['status']} "
            f"| {row['allowed_wording']} | {row['forbidden_wording']} | {row['evidence']} |"
        )
    lines.extend([
        "",
        "## Native Service Metrics",
        "",
        f"- variant: `{native.get('variant', '')}`",
        f"- rows: `{native.get('rows', 0)}`",
        f"- seeds: `{native.get('seeds', 0)}`",
        f"- service-response status: `{native.get('native_service_response_status', '')}`",
        f"- onboard improvement status: `{native.get('native_onboard_improvement_status', '')}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_coverage(
    output_dir: Path,
    *,
    afc_csv: Path = DEFAULT_AFC_CSV,
    apc_csv: Path = DEFAULT_APC_CSV,
    native_summary: Path = DEFAULT_NATIVE_SUMMARY,
    gtfs_ride_dir: Path | None = None,
    gtfs_ride_source_kind: str = "unknown",
    gtfs_ride_source_url: str = "",
    gtfs_ride_agency: str = "",
    native_variant: str = "native_real_freqhrl",
    min_afc_rows: int = 100,
    min_afc_stations: int = 4,
    min_afc_time_bins: int = 24,
    min_apc_rows: int = 100,
    min_apc_routes: int = 4,
    min_apc_time_bins: int = 40,
) -> dict[str, Any]:
    afc_rows, afc_fields = _read_csv_rows(afc_csv)
    apc_rows, apc_fields = _read_csv_rows(apc_csv)
    native_data = _read_json(native_summary)
    afc = summarize_afc_rows(
        afc_rows,
        afc_fields,
        min_rows=int(min_afc_rows),
        min_station_count=int(min_afc_stations),
        min_time_bins=int(min_afc_time_bins),
    )
    apc = summarize_apc_rows(
        apc_rows,
        apc_fields,
        min_rows=int(min_apc_rows),
        min_route_count=int(min_apc_routes),
        min_time_bins=int(min_apc_time_bins),
    )
    gtfs_ride = summarize_gtfs_ride(
        gtfs_ride_dir,
        source_kind=str(gtfs_ride_source_kind),
        source_url=str(gtfs_ride_source_url),
        agency=str(gtfs_ride_agency),
    )
    native = summarize_native_summary(native_data, variant=str(native_variant))
    boundaries = build_claim_boundaries(afc, apc, gtfs_ride, native)
    supported_boundaries = sum(1 for row in boundaries if row["status"] == "supported")
    external_missing = sum(1 for row in boundaries if row["status"] == "external_missing")
    evidence_scope = (
        "real_afc_apc_demand_plus_native_service_response"
        if afc["claim_status"] == "supported"
        and apc["claim_status"] == "supported"
        and native["native_service_response_status"] == "supported"
        else "partial"
    )
    if (
        gtfs_ride.get("has_od_fields")
        and gtfs_ride.get("has_onboard_load")
        and gtfs_ride.get("source_verified")
    ):
        evidence_scope = "real_gtfs_ride_od_onboard_plus_native_service_response"
    payload = {
        "summary": {
            "evidence_scope": evidence_scope,
            "supported_boundaries": supported_boundaries,
            "external_missing_boundaries": external_missing,
            "source_count": 3,
        },
        "source_coverage": [afc, apc, gtfs_ride],
        "native_service": native,
        "claim_boundaries": boundaries,
        "inputs": {
            "afc_csv": str(afc_csv),
            "apc_csv": str(apc_csv),
            "native_summary": str(native_summary),
            "gtfs_ride_dir": str(gtfs_ride_dir) if gtfs_ride_dir else "",
            "gtfs_ride_source_kind": str(gtfs_ride_source_kind),
            "gtfs_ride_source_url": str(gtfs_ride_source_url),
            "gtfs_ride_agency": str(gtfs_ride_agency),
        },
        "boundary": (
            "This ledger separates observed agency demand from native simulator "
            "service metrics.  Claims about real OD/onboard-load/alighting ground "
            "truth require external files that expose those fields."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "source_coverage.csv", payload["source_coverage"])
    _write_csv(output_dir / "claim_boundaries.csv", payload["claim_boundaries"])
    _write_csv(output_dir / "native_service_metrics.csv", native.get("metrics", []))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_report(output_dir / "report.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--afc-csv", type=Path, default=DEFAULT_AFC_CSV)
    parser.add_argument("--apc-csv", type=Path, default=DEFAULT_APC_CSV)
    parser.add_argument("--native-summary", type=Path, default=DEFAULT_NATIVE_SUMMARY)
    parser.add_argument("--gtfs-ride-dir", type=Path, default=None)
    parser.add_argument("--gtfs-ride-source-kind", default="unknown")
    parser.add_argument("--gtfs-ride-source-url", default="")
    parser.add_argument("--gtfs-ride-agency", default="")
    parser.add_argument("--native-variant", default="native_real_freqhrl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-afc-rows", type=int, default=100)
    parser.add_argument("--min-afc-stations", type=int, default=4)
    parser.add_argument("--min-afc-time-bins", type=int, default=24)
    parser.add_argument("--min-apc-rows", type=int, default=100)
    parser.add_argument("--min-apc-routes", type=int, default=4)
    parser.add_argument("--min-apc-time-bins", type=int, default=40)
    args = parser.parse_args()
    payload = run_coverage(
        output_dir=args.output_dir,
        afc_csv=args.afc_csv,
        apc_csv=args.apc_csv,
        native_summary=args.native_summary,
        gtfs_ride_dir=args.gtfs_ride_dir,
        gtfs_ride_source_kind=str(args.gtfs_ride_source_kind),
        gtfs_ride_source_url=str(args.gtfs_ride_source_url),
        gtfs_ride_agency=str(args.gtfs_ride_agency),
        native_variant=str(args.native_variant),
        min_afc_rows=int(args.min_afc_rows),
        min_afc_stations=int(args.min_afc_stations),
        min_afc_time_bins=int(args.min_afc_time_bins),
        min_apc_rows=int(args.min_apc_rows),
        min_apc_routes=int(args.min_apc_routes),
        min_apc_time_bins=int(args.min_apc_time_bins),
    )
    print(
        "agency_demand_onboard_coverage "
        f"scope={payload['summary']['evidence_scope']} "
        f"supported={payload['summary']['supported_boundaries']} "
        f"external_missing={payload['summary']['external_missing_boundaries']}"
    )


if __name__ == "__main__":
    main()
