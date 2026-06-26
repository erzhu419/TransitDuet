#!/usr/bin/env python3
"""Download an offline MTA Bus Time API cache for FreqDuet.

The API key is read from ``MTA_BUS_TIME_API_KEY`` or ``--api-key`` and is never
written to disk. Raw responses, flattened CSVs, and a small claim-boundary audit
are saved under FreqDuet's external truth-source tree for offline use.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = (
    ROOT / "data" / "external_truth_sources" / "mta_bus_time_api" / "offline_cache"
)
OBA_BASE = "https://bustime-classic.mta.info/api/where"
SIRI_VM = "https://bustime-classic.mta.info/api/siri/vehicle-monitoring.json"
DEFAULT_AGENCIES = ["MTA NYCT", "MTABC"]
DEFAULT_VEHICLE_ROUTES = [
    "MTA NYCT_M15+",
    "MTA NYCT_B63",
    "MTA NYCT_M1",
    "MTABC_Q60",
    "MTA NYCT_B46+",
    "MTA NYCT_Q44+",
    "MTA NYCT_S79+",
    "MTA NYCT_BX12+",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("_")


def _redacted_query(params: dict[str, Any]) -> dict[str, Any]:
    return {k: ("<redacted>" if k == "key" else v) for k, v in params.items()}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _fetch_json(url: str, params: dict[str, Any], *, timeout: int = 60) -> Any:
    encoded = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{url}?{encoded}",
        headers={"User-Agent": "FreqDuet-MTA-BusTime-offline-cache/0.1"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _oba(path: str, params: dict[str, Any]) -> Any:
    return _fetch_json(f"{OBA_BASE}/{path}", params)


def _data_list(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("list"), list):
        return [x for x in data["list"] if isinstance(x, dict)]
    return []


def _vehicle_activities(payload: Any) -> list[dict[str, Any]]:
    delivery = (
        payload.get("Siri", {})
        .get("ServiceDelivery", {})
        .get("VehicleMonitoringDelivery", [])
    )
    if not delivery:
        return []
    activities = delivery[0].get("VehicleActivity", [])
    if isinstance(activities, dict):
        return [activities]
    if isinstance(activities, list):
        return [x for x in activities if isinstance(x, dict)]
    return []


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _flatten_vehicle_activity(source: str, activity: dict[str, Any]) -> dict[str, Any]:
    journey = activity.get("MonitoredVehicleJourney", {})
    location = journey.get("VehicleLocation", {}) or {}
    framed = journey.get("FramedVehicleJourneyRef", {}) or {}
    call = journey.get("MonitoredCall", {}) or {}
    return {
        "snapshot_source": source,
        "recorded_at_time": activity.get("RecordedAtTime"),
        "line_ref": journey.get("LineRef"),
        "direction_ref": journey.get("DirectionRef"),
        "journey_pattern_ref": journey.get("JourneyPatternRef"),
        "published_line_name": _first(journey.get("PublishedLineName")),
        "operator_ref": journey.get("OperatorRef"),
        "origin_ref": journey.get("OriginRef"),
        "destination_ref": journey.get("DestinationRef"),
        "destination_name": _first(journey.get("DestinationName")),
        "data_frame_ref": framed.get("DataFrameRef"),
        "dated_vehicle_journey_ref": framed.get("DatedVehicleJourneyRef"),
        "vehicle_ref": journey.get("VehicleRef"),
        "latitude": location.get("Latitude"),
        "longitude": location.get("Longitude"),
        "bearing": journey.get("Bearing"),
        "progress_rate": journey.get("ProgressRate"),
        "progress_status": "|".join(journey.get("ProgressStatus", []) or []),
        "block_ref": journey.get("BlockRef"),
        "vehicle_status": journey.get("VehicleStatus"),
        "monitored_call_stop_point_ref": call.get("StopPointRef"),
        "monitored_call_visit_number": call.get("VisitNumber"),
        "monitored_call_stop_name": _first(call.get("StopPointName")),
        "monitored_call_extensions": json.dumps(call.get("Extensions", {}), ensure_ascii=False),
    }


def flatten_route_stops(route_id: str, payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    entry = data.get("entry", {}) if isinstance(data, dict) else {}
    refs = data.get("references", {}) if isinstance(data, dict) else {}
    stops = []
    for stop in refs.get("stops", []) or []:
        stops.append({
            "stop_id": stop.get("id"),
            "code": stop.get("code"),
            "name": stop.get("name"),
            "lat": stop.get("lat"),
            "lon": stop.get("lon"),
            "direction": stop.get("direction"),
            "wheelchair_boarding": stop.get("wheelchairBoarding"),
        })
    sequences = []
    for grouping in entry.get("stopGroupings", []) or []:
        grouping_type = grouping.get("type")
        for stop_group in grouping.get("stopGroups", []) or []:
            direction_id = stop_group.get("id")
            direction_name = (stop_group.get("name") or {}).get("name")
            for seq, stop_id in enumerate(stop_group.get("stopIds", []) or []):
                sequences.append({
                    "route_id": route_id,
                    "grouping_type": grouping_type,
                    "direction_id": direction_id,
                    "direction_name": direction_name,
                    "stop_sequence": seq,
                    "stop_id": stop_id,
                })
    return stops, sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("MTA_BUS_TIME_API_KEY"))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--snapshot-id", default=_stamp())
    parser.add_argument("--agencies", nargs="*", default=DEFAULT_AGENCIES)
    parser.add_argument("--max-routes", type=int, default=0, help="0 downloads all routes.")
    parser.add_argument("--delay-sec", type=float, default=0.08)
    parser.add_argument("--skip-route-stops", action="store_true")
    parser.add_argument("--vehicle-routes", nargs="*", default=DEFAULT_VEHICLE_ROUTES)
    parser.add_argument("--vehicle-max", type=int, default=40)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("MTA Bus Time API key required via --api-key or MTA_BUS_TIME_API_KEY.")

    out_dir = Path(args.out_root) / args.snapshot_id
    raw_dir = out_dir / "raw"
    parsed_dir = out_dir / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    request_log: list[dict[str, Any]] = []

    def fetch_and_store(label: str, url: str, params: dict[str, Any], path: Path) -> Any:
        payload = _fetch_json(url, params)
        _write_json(path, payload)
        request_log.append({
            "label": label,
            "endpoint": url,
            "params": _redacted_query(params),
            "raw_path": str(path.relative_to(out_dir)),
            "status_code": payload.get("code") if isinstance(payload, dict) else None,
            "status_text": payload.get("text") if isinstance(payload, dict) else None,
        })
        time.sleep(args.delay_sec)
        return payload

    agencies_payload = fetch_and_store(
        "agencies_with_coverage",
        f"{OBA_BASE}/agencies-with-coverage.json",
        {"key": args.api_key},
        raw_dir / "agencies_with_coverage.json",
    )
    agencies_rows = []
    for item in _data_list(agencies_payload):
        agency = item.get("agency", {})
        agencies_rows.append({
            "agency_id": agency.get("id"),
            "agency_name": agency.get("name"),
            "timezone": agency.get("timezone"),
            "lat": item.get("lat"),
            "lon": item.get("lon"),
            "lat_span": item.get("latSpan"),
            "lon_span": item.get("lonSpan"),
        })
    pd.DataFrame(agencies_rows).to_csv(parsed_dir / "mta_bus_time_agencies.csv", index=False)

    routes: list[dict[str, Any]] = []
    for agency in args.agencies:
        payload = fetch_and_store(
            f"routes_for_agency_{agency}",
            f"{OBA_BASE}/routes-for-agency/{urllib.parse.quote(agency, safe='')}.json",
            {"key": args.api_key},
            raw_dir / f"routes_for_agency_{_slug(agency)}.json",
        )
        for route in _data_list(payload):
            routes.append({
                "agency_id": route.get("agencyId"),
                "route_id": route.get("id"),
                "short_name": route.get("shortName"),
                "long_name": route.get("longName"),
                "description": route.get("description"),
                "route_type": route.get("type"),
                "color": route.get("color"),
                "text_color": route.get("textColor"),
            })
    routes_df = pd.DataFrame(routes).drop_duplicates("route_id")
    routes_df.to_csv(parsed_dir / "mta_bus_time_routes.csv", index=False)

    route_ids = routes_df["route_id"].dropna().astype(str).tolist()
    if args.max_routes > 0:
        route_ids = route_ids[:args.max_routes]

    all_stops: list[dict[str, Any]] = []
    route_stop_rows: list[dict[str, Any]] = []
    failed_routes: list[dict[str, Any]] = []
    if not args.skip_route_stops:
        for i, route_id in enumerate(route_ids, 1):
            raw_path = raw_dir / "stops_for_route" / f"{_slug(route_id)}.json"
            params = {
                "key": args.api_key,
                "includePolylines": "false",
                "version": "2",
            }
            try:
                payload = fetch_and_store(
                    f"stops_for_route_{route_id}",
                    f"{OBA_BASE}/stops-for-route/{urllib.parse.quote(route_id, safe='')}.json",
                    params,
                    raw_path,
                )
                stops, seqs = flatten_route_stops(route_id, payload)
                all_stops.extend(stops)
                route_stop_rows.extend(seqs)
            except Exception as exc:  # pragma: no cover - network resilience
                failed_routes.append({"route_id": route_id, "error": str(exc)})
            if i % 50 == 0:
                print(f"downloaded stops-for-route {i}/{len(route_ids)}")

    stops_df = pd.DataFrame(all_stops)
    if not stops_df.empty:
        stops_df = stops_df.drop_duplicates("stop_id")
    stops_df.to_csv(parsed_dir / "mta_bus_time_stops.csv", index=False)
    pd.DataFrame(route_stop_rows).to_csv(parsed_dir / "mta_bus_time_route_stop_sequences.csv", index=False)
    pd.DataFrame(failed_routes).to_csv(parsed_dir / "mta_bus_time_failed_routes.csv", index=False)

    valid_route_ids = set(routes_df["route_id"].dropna().astype(str))
    vehicle_route_ids = [r for r in args.vehicle_routes if r in valid_route_ids]
    vehicle_rows: list[dict[str, Any]] = []
    vehicle_snapshot_meta: list[dict[str, Any]] = []
    for route_id in vehicle_route_ids:
        params = {
            "key": args.api_key,
            "version": "2",
            "LineRef": route_id,
            "VehicleMonitoringDetailLevel": "normal",
            "MaximumStopVisits": str(args.vehicle_max),
        }
        payload = fetch_and_store(
            f"vehicle_monitoring_{route_id}",
            SIRI_VM,
            params,
            raw_dir / "vehicle_monitoring" / f"{_slug(route_id)}.json",
        )
        activities = _vehicle_activities(payload)
        vehicle_rows.extend(_flatten_vehicle_activity(route_id, activity) for activity in activities)
        delivery = payload.get("Siri", {}).get("ServiceDelivery", {}).get("VehicleMonitoringDelivery", [{}])[0]
        vehicle_snapshot_meta.append({
            "line_ref": route_id,
            "response_timestamp": delivery.get("ResponseTimestamp"),
            "valid_until": delivery.get("ValidUntil"),
            "vehicle_activity_count": len(activities),
        })
    pd.DataFrame(vehicle_rows).to_csv(parsed_dir / "mta_bus_time_vehicle_snapshots.csv", index=False)
    pd.DataFrame(vehicle_snapshot_meta).to_csv(parsed_dir / "mta_bus_time_vehicle_snapshot_meta.csv", index=False)

    source_coverage = pd.DataFrame([
        {
            "source": "mta_bus_time_oba_discovery_static",
            "source_kind": "mta_api_static_route_stop_geometry",
            "claim_status": "supported_offline_cache",
            "agencies": ",".join(args.agencies),
            "routes": int(len(routes_df)),
            "routes_with_stop_sequences": int(pd.DataFrame(route_stop_rows)["route_id"].nunique()) if route_stop_rows else 0,
            "unique_stops": int(stops_df["stop_id"].nunique()) if not stops_df.empty else 0,
            "boundary": "Static MTA Bus Time OBA discovery route/stop/sequence cache for FreqDuet external data only; no FreqHRL result imported.",
        },
        {
            "source": "mta_bus_time_siri_vehicle_monitoring",
            "source_kind": "mta_api_realtime_vehicle_snapshot",
            "claim_status": "supported_route_filtered_snapshot" if vehicle_rows else "empty_or_not_available_at_download_time",
            "route_filtered_snapshots": int(len(vehicle_snapshot_meta)),
            "vehicle_rows": int(len(vehicle_rows)),
            "boundary": "Route-filtered SIRI VehicleMonitoring snapshots for offline replay/realism evidence; not a full-day historical AVL archive.",
        },
    ])
    source_coverage.to_csv(parsed_dir / "mta_bus_time_source_coverage.csv", index=False)

    claim_boundaries = pd.DataFrame([
        {
            "id": "MTA1",
            "evidence_item": "mta_bus_time_static_route_stop_cache",
            "status": "supported",
            "allowed_wording": "offline MTA Bus Time route, stop, and route-stop sequence cache",
            "forbidden_wording": "MTA APC/onboard-load or FreqDuet field outcome",
            "evidence": f"routes={len(routes_df)} stops={stops_df['stop_id'].nunique() if not stops_df.empty else 0} route_sequences={len(route_stop_rows)}",
        },
        {
            "id": "MTA2",
            "evidence_item": "mta_bus_time_vehicle_snapshot_cache",
            "status": "supported" if vehicle_rows else "empty_or_not_available_at_download_time",
            "allowed_wording": "route-filtered SIRI VehicleMonitoring snapshots are cached for offline AVL realism checks",
            "forbidden_wording": "full-day historical AVL archive or control-loop field validation",
            "evidence": f"route_snapshots={len(vehicle_snapshot_meta)} vehicle_rows={len(vehicle_rows)}",
        },
        {
            "id": "MTA3",
            "evidence_item": "separation_from_freqhrl_results",
            "status": "supported",
            "allowed_wording": "raw MTA API data downloaded for FreqDuet external data audit",
            "forbidden_wording": "reuse of FreqHRL paper results or checkpoints",
            "evidence": "cache path is under FreqDuet/freqduet/data/external_truth_sources/mta_bus_time_api",
        },
    ])
    claim_boundaries.to_csv(parsed_dir / "mta_bus_time_claim_boundaries.csv", index=False)

    manifest = {
        "snapshot_id": args.snapshot_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "api_key_written_to_disk": False,
        "agencies": args.agencies,
        "route_count": int(len(routes_df)),
        "stop_count": int(stops_df["stop_id"].nunique()) if not stops_df.empty else 0,
        "route_stop_sequence_rows": int(len(route_stop_rows)),
        "failed_route_count": int(len(failed_routes)),
        "vehicle_route_snapshots": int(len(vehicle_snapshot_meta)),
        "vehicle_rows": int(len(vehicle_rows)),
        "raw_dir": "raw",
        "parsed_dir": "parsed",
        "requests": request_log,
        "boundary": (
            "MTA Bus Time API cache for FreqDuet external data only. It is not "
            "FreqHRL paper result data, and it is not a field deployment result."
        ),
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(Path(args.out_root) / "latest_manifest.json", manifest)
    print(json.dumps({
        "out_dir": str(out_dir),
        "routes": manifest["route_count"],
        "stops": manifest["stop_count"],
        "route_stop_sequence_rows": manifest["route_stop_sequence_rows"],
        "failed_route_count": manifest["failed_route_count"],
        "vehicle_rows": manifest["vehicle_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
