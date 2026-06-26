#!/usr/bin/env python3
"""Audit MBTA same-network APC/GTFS calibration readiness for FreqDuet.

This is a data-boundary audit, not a deployment claim. It checks whether the
public MBTA APC board/alight/load data can be matched to MBTA static GTFS route
and stop geometry, then records what is still missing for exact AFC/APC/AVL
field calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results_freqduet" / "mbta_same_network_calibration_audit" / "v1"
DEFAULT_MBTA_APC_CSV = Path(os.environ.get(
    "FREQDUET_MBTA_BUS_RIDERSHIP_CSV",
    "/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/mbta/"
    "ridership/MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/"
    "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/"
    "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop_Fall_2025.csv",
))
DEFAULT_MBTA_GTFS_DIR = Path(os.environ.get(
    "FREQDUET_MBTA_GTFS_DIR",
    "/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/mbta/gtfs",
))
MBTA_APC_SOURCE_URL = "https://gis.data.mass.gov/datasets/8daf4a33925a4df59183f860826d29ee"
MBTA_GTFS_DOC_URL = "https://github.com/mbta/gtfs-documentation"
MBTA_TRANSIT_PERFORMANCE_URL = "https://github.com/mbta/transit-performance"
GTFS_RT_REFERENCE_URL = "https://gtfs.org/documentation/realtime/reference/"
MTA_BUS_TIME_KEY_URL = "https://register.developer.obanyc.com/"


APC_USECOLS = [
    "GTFS route_id",
    "GTFS direction_id",
    "trip start time",
    "Stop Name",
    "GTFS stop_id",
    "stop sequence",
    "Year",
    "Day Type",
    "Boardings",
    "Alightings",
    "Load",
    "Route/Variant",
    "# of Trip Samples ",
]


def _safe_num(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _hour_from_trip_start(value: Any) -> int:
    text = str(value or "")
    if ":" not in text:
        return -1
    try:
        return int(text.split(":", 1)[0])
    except ValueError:
        return -1


def _as_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_gtfs(gtfs_dir: Path, apc_routes: set[str], focus_route: str) -> dict[str, Any]:
    required = ["routes.txt", "stops.txt", "trips.txt", "stop_times.txt", "feed_info.txt"]
    missing = [name for name in required if not (gtfs_dir / name).exists()]
    if missing:
        return {
            "missing": missing,
            "routes": pd.DataFrame(),
            "stops": pd.DataFrame(),
            "feed_info": pd.DataFrame(),
            "route_stop_pairs": pd.DataFrame(),
            "focus_stop_times": pd.DataFrame(),
        }

    routes = pd.read_csv(gtfs_dir / "routes.txt", dtype=str)
    stops = pd.read_csv(gtfs_dir / "stops.txt", dtype=str)
    feed_info = pd.read_csv(gtfs_dir / "feed_info.txt", dtype=str)
    trips = pd.read_csv(
        gtfs_dir / "trips.txt",
        dtype=str,
        usecols=["trip_id", "route_id", "direction_id"],
    )
    trips = trips[trips["route_id"].isin(apc_routes)].copy()
    trip_lookup = trips.set_index("trip_id")[["route_id", "direction_id"]]
    trip_ids = set(trip_lookup.index)

    route_stop_parts: list[pd.DataFrame] = []
    focus_parts: list[pd.DataFrame] = []
    stop_cols = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    for chunk in pd.read_csv(gtfs_dir / "stop_times.txt", dtype=str, usecols=stop_cols, chunksize=500_000):
        chunk = chunk[chunk["trip_id"].isin(trip_ids)]
        if chunk.empty:
            continue
        merged = chunk.merge(trip_lookup, left_on="trip_id", right_index=True, how="inner")
        route_stop_parts.append(
            merged[["route_id", "direction_id", "stop_id"]].drop_duplicates()
        )
        focus = merged[merged["route_id"].eq(focus_route)].copy()
        if not focus.empty:
            focus_parts.append(focus)

    route_stop_pairs = (
        pd.concat(route_stop_parts, ignore_index=True).drop_duplicates()
        if route_stop_parts else pd.DataFrame(columns=["route_id", "direction_id", "stop_id"])
    )
    focus_stop_times = (
        pd.concat(focus_parts, ignore_index=True)
        if focus_parts else pd.DataFrame(columns=["route_id", "direction_id", "stop_id", "stop_sequence"])
    )
    if not focus_stop_times.empty:
        focus_stop_times["stop_sequence_num"] = pd.to_numeric(
            focus_stop_times["stop_sequence"], errors="coerce"
        )
        focus_stop_times = focus_stop_times.groupby(
            ["route_id", "direction_id", "stop_id"], as_index=False
        ).agg(
            gtfs_stop_sequence_min=("stop_sequence_num", "min"),
            gtfs_stop_sequence_max=("stop_sequence_num", "max"),
            gtfs_trip_rows=("trip_id", "size"),
            gtfs_unique_trips=("trip_id", "nunique"),
        )

    return {
        "missing": [],
        "routes": routes,
        "stops": stops,
        "feed_info": feed_info,
        "route_stop_pairs": route_stop_pairs,
        "focus_stop_times": focus_stop_times,
    }


def load_apc(apc_csv: Path, focus_route: str, chunksize: int) -> dict[str, Any]:
    if not apc_csv.exists():
        return {
            "exists": False,
            "routes": set(),
            "stops": set(),
            "route_stop_pairs": pd.DataFrame(columns=["route_id", "direction_id", "stop_id"]),
            "route_summary": pd.DataFrame(),
            "focus_profile": pd.DataFrame(),
            "top_routes": pd.DataFrame(),
            "coverage": {},
        }

    route_parts: list[pd.DataFrame] = []
    focus_parts: list[pd.DataFrame] = []
    top_parts: list[pd.DataFrame] = []
    route_stop_parts: list[pd.DataFrame] = []
    routes: set[str] = set()
    stops: set[str] = set()
    years: set[str] = set()
    day_types: set[str] = set()
    row_count = 0
    total_boardings = 0.0
    total_alightings = 0.0
    load_sum = 0.0
    max_load = 0.0

    for chunk in pd.read_csv(apc_csv, usecols=APC_USECOLS, chunksize=chunksize):
        for col in ["GTFS route_id", "GTFS direction_id", "GTFS stop_id", "Year", "Day Type"]:
            chunk[col] = chunk[col].astype(str)
        for col in ["Boardings", "Alightings", "Load", "# of Trip Samples "]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
        chunk["hour"] = chunk["trip start time"].map(_hour_from_trip_start)
        row_count += len(chunk)
        routes.update(chunk["GTFS route_id"].dropna().unique())
        stops.update(chunk["GTFS stop_id"].dropna().unique())
        years.update(chunk["Year"].dropna().unique())
        day_types.update(chunk["Day Type"].dropna().unique())
        total_boardings += float(chunk["Boardings"].sum())
        total_alightings += float(chunk["Alightings"].sum())
        load_sum += float(chunk["Load"].sum())
        max_load = max(max_load, float(chunk["Load"].max()))

        route_stop_parts.append(
            chunk[["GTFS route_id", "GTFS direction_id", "GTFS stop_id"]]
            .rename(columns={
                "GTFS route_id": "route_id",
                "GTFS direction_id": "direction_id",
                "GTFS stop_id": "stop_id",
            })
            .drop_duplicates()
        )
        route_parts.append(chunk.groupby(
            ["GTFS route_id", "GTFS direction_id", "Day Type"], as_index=False
        ).agg(
            rows=("Load", "size"),
            boardings=("Boardings", "sum"),
            alightings=("Alightings", "sum"),
            mean_load=("Load", "mean"),
            p95_load=("Load", lambda x: float(np.percentile(x, 95))),
            max_load=("Load", "max"),
            trip_samples=("# of Trip Samples ", "sum"),
        ))
        top_parts.append(chunk.groupby(["GTFS route_id"], as_index=False).agg(
            rows=("Load", "size"),
            boardings=("Boardings", "sum"),
            max_load=("Load", "max"),
        ))

        focus = chunk[chunk["GTFS route_id"].eq(focus_route)].copy()
        if not focus.empty:
            focus_parts.append(focus.groupby(
                [
                    "GTFS route_id",
                    "GTFS direction_id",
                    "Day Type",
                    "stop sequence",
                    "GTFS stop_id",
                    "Stop Name",
                ],
                as_index=False,
            ).agg(
                rows=("Load", "size"),
                boardings=("Boardings", "sum"),
                alightings=("Alightings", "sum"),
                mean_load=("Load", "mean"),
                p95_load=("Load", lambda x: float(np.percentile(x, 95))),
                max_load=("Load", "max"),
                trip_samples=("# of Trip Samples ", "sum"),
                first_hour=("hour", "min"),
                last_hour=("hour", "max"),
            ))

    route_summary = pd.concat(route_parts, ignore_index=True)
    route_summary = route_summary.groupby(
        ["GTFS route_id", "GTFS direction_id", "Day Type"], as_index=False
    ).agg(
        rows=("rows", "sum"),
        boardings=("boardings", "sum"),
        alightings=("alightings", "sum"),
        mean_load=("mean_load", "mean"),
        p95_load=("p95_load", "mean"),
        max_load=("max_load", "max"),
        trip_samples=("trip_samples", "sum"),
    ).sort_values(["boardings", "max_load"], ascending=False)

    top_routes = pd.concat(top_parts, ignore_index=True).groupby("GTFS route_id", as_index=False).agg(
        rows=("rows", "sum"),
        boardings=("boardings", "sum"),
        max_load=("max_load", "max"),
    ).sort_values(["boardings", "max_load"], ascending=False)

    focus_profile = (
        pd.concat(focus_parts, ignore_index=True)
        if focus_parts else pd.DataFrame()
    )
    if not focus_profile.empty:
        focus_profile = focus_profile.groupby(
            [
                "GTFS route_id",
                "GTFS direction_id",
                "Day Type",
                "stop sequence",
                "GTFS stop_id",
                "Stop Name",
            ],
            as_index=False,
        ).agg(
            rows=("rows", "sum"),
            boardings=("boardings", "sum"),
            alightings=("alightings", "sum"),
            mean_load=("mean_load", "mean"),
            p95_load=("p95_load", "mean"),
            max_load=("max_load", "max"),
            trip_samples=("trip_samples", "sum"),
            first_hour=("first_hour", "min"),
            last_hour=("last_hour", "max"),
        )
        focus_profile["stop_sequence_num"] = pd.to_numeric(
            focus_profile["stop sequence"], errors="coerce"
        )
        focus_profile = focus_profile.sort_values(
            ["Day Type", "GTFS direction_id", "stop_sequence_num", "GTFS stop_id"]
        )

    coverage = {
        "source": "mbta_bus_ridership_fall2025_apc",
        "source_kind": "public_agency_observed_bus_apc",
        "claim_status": "supported",
        "agency": "Massachusetts Bay Transportation Authority",
        "source_url": MBTA_APC_SOURCE_URL,
        "local_file": str(apc_csv),
        "rows": int(row_count),
        "unique_routes": int(len(routes)),
        "unique_stops": int(len(stops)),
        "years": sorted(years),
        "day_types": sorted(day_types),
        "total_boardings": total_boardings,
        "total_alightings": total_alightings,
        "mean_load": load_sum / row_count if row_count else math.nan,
        "max_load": max_load,
        "boundary": "Observed MBTA Fall 2025 bus stop/trip boardings, alightings, and load. No OD and no AVL.",
    }
    route_stop_pairs = pd.concat(route_stop_parts, ignore_index=True).drop_duplicates()
    return {
        "exists": True,
        "routes": routes,
        "stops": stops,
        "route_stop_pairs": route_stop_pairs,
        "route_summary": route_summary,
        "focus_profile": focus_profile,
        "top_routes": top_routes,
        "coverage": coverage,
    }


def build_overlap_tables(apc: dict[str, Any], gtfs: dict[str, Any], focus_route: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    routes_df = gtfs["routes"]
    stops_df = gtfs["stops"]
    gtfs_route_ids = set(routes_df["route_id"].astype(str)) if not routes_df.empty else set()
    gtfs_stop_ids = set(stops_df["stop_id"].astype(str)) if not stops_df.empty else set()

    apc_routes = set(apc["routes"])
    apc_stops = set(apc["stops"])
    apc_route_stop = apc["route_stop_pairs"].copy()
    gtfs_route_stop = gtfs["route_stop_pairs"].copy()
    for df in [apc_route_stop, gtfs_route_stop]:
        for col in ["route_id", "direction_id", "stop_id"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
    gtfs_keys = set(map(tuple, gtfs_route_stop[["route_id", "direction_id", "stop_id"]].to_numpy())) if not gtfs_route_stop.empty else set()
    apc_route_stop["in_current_gtfs_route_stop"] = [
        (route_id, direction_id, stop_id) in gtfs_keys
        for route_id, direction_id, stop_id in apc_route_stop[["route_id", "direction_id", "stop_id"]].to_numpy()
    ] if not apc_route_stop.empty else []

    route_stop_overlap = (
        float(apc_route_stop["in_current_gtfs_route_stop"].mean())
        if not apc_route_stop.empty else math.nan
    )
    summary = pd.DataFrame([{
        "apc_routes": len(apc_routes),
        "gtfs_routes": len(gtfs_route_ids),
        "apc_routes_in_current_gtfs": len(apc_routes & gtfs_route_ids),
        "apc_route_overlap_rate": len(apc_routes & gtfs_route_ids) / len(apc_routes) if apc_routes else math.nan,
        "apc_stops": len(apc_stops),
        "gtfs_stops": len(gtfs_stop_ids),
        "apc_stops_in_current_gtfs": len(apc_stops & gtfs_stop_ids),
        "apc_stop_overlap_rate": len(apc_stops & gtfs_stop_ids) / len(apc_stops) if apc_stops else math.nan,
        "apc_route_stop_pairs": len(apc_route_stop),
        "gtfs_route_stop_pairs_for_apc_routes": len(gtfs_route_stop),
        "apc_route_stop_pairs_in_current_gtfs": int(apc_route_stop["in_current_gtfs_route_stop"].sum()) if not apc_route_stop.empty else 0,
        "apc_route_stop_overlap_rate": route_stop_overlap,
        "focus_route": focus_route,
    }])
    return summary, apc_route_stop


def enrich_focus_profile(apc_focus: pd.DataFrame, gtfs: dict[str, Any]) -> pd.DataFrame:
    if apc_focus.empty:
        return apc_focus
    profile = apc_focus.copy()
    profile = profile.rename(columns={
        "GTFS route_id": "route_id",
        "GTFS direction_id": "direction_id",
        "GTFS stop_id": "stop_id",
        "Stop Name": "stop_name",
        "Day Type": "day_type",
        "stop sequence": "apc_stop_sequence",
    })
    stops = gtfs["stops"].copy()
    if not stops.empty:
        keep = [c for c in ["stop_id", "stop_name", "stop_lat", "stop_lon"] if c in stops.columns]
        stops = stops[keep].drop_duplicates("stop_id")
        stops = stops.rename(columns={"stop_name": "gtfs_stop_name"})
        profile = profile.merge(stops, on="stop_id", how="left")
    gtfs_focus = gtfs["focus_stop_times"].copy()
    if not gtfs_focus.empty:
        for col in ["route_id", "direction_id", "stop_id"]:
            gtfs_focus[col] = gtfs_focus[col].astype(str)
            profile[col] = profile[col].astype(str)
        profile = profile.merge(
            gtfs_focus,
            on=["route_id", "direction_id", "stop_id"],
            how="left",
        )
    profile["in_current_gtfs_focus_route"] = profile["gtfs_trip_rows"].notna()
    return profile.sort_values(["day_type", "direction_id", "stop_sequence_num", "stop_id"])


def build_source_coverage(apc: dict[str, Any], gtfs: dict[str, Any], gtfs_dir: Path, apc_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(apc["coverage"] if apc.get("coverage") else {
        "source": "mbta_bus_ridership_fall2025_apc",
        "source_kind": "public_agency_observed_bus_apc",
        "claim_status": "external_missing",
        "source_url": MBTA_APC_SOURCE_URL,
        "local_file": str(apc_csv),
        "boundary": "MBTA APC CSV missing locally.",
    })
    feed_info = gtfs["feed_info"]
    if not feed_info.empty:
        info = feed_info.iloc[0].to_dict()
        rows.append({
            "source": "mbta_static_gtfs_current_cache",
            "source_kind": "public_agency_static_schedule_geometry",
            "claim_status": "supported_current_schedule",
            "agency": "Massachusetts Bay Transportation Authority",
            "source_url": MBTA_GTFS_DOC_URL,
            "local_file": str(gtfs_dir),
            "feed_start_date": info.get("feed_start_date"),
            "feed_end_date": info.get("feed_end_date"),
            "feed_version": info.get("feed_version"),
            "boundary": "Current cached static GTFS gives route/stop/schedule geometry; it is not the Fall 2025 historical schedule archive.",
        })
    else:
        rows.append({
            "source": "mbta_static_gtfs_current_cache",
            "source_kind": "public_agency_static_schedule_geometry",
            "claim_status": "external_missing",
            "source_url": MBTA_GTFS_DOC_URL,
            "local_file": str(gtfs_dir),
            "boundary": f"Missing GTFS files: {gtfs.get('missing', [])}",
        })
    rows.extend([
        {
            "source": "mbta_gtfs_realtime_vehicle_positions",
            "source_kind": "public_or_developer_realtime_avl_feed",
            "claim_status": "not_collected_locally",
            "source_url": MBTA_TRANSIT_PERFORMANCE_URL,
            "standard_url": GTFS_RT_REFERENCE_URL,
            "boundary": "Can support future AVL collection/replay, but no local same-day historical VehiclePositions archive is present.",
        },
        {
            "source": "mta_bus_time_realtime_api",
            "source_kind": "developer_realtime_bus_avl_api",
            "claim_status": "api_key_needed_for_mta_bus",
            "source_url": MTA_BUS_TIME_KEY_URL,
            "boundary": "Useful only if switching to MTA bus AVL; it does not solve MBTA APC+AVL same-day matching.",
        },
    ])
    return pd.DataFrame(rows)


def build_claim_boundaries(overlap: pd.DataFrame, focus_profile: pd.DataFrame) -> pd.DataFrame:
    row = overlap.iloc[0]
    route_overlap = _safe_num(row.get("apc_route_overlap_rate"))
    stop_overlap = _safe_num(row.get("apc_stop_overlap_rate"))
    route_stop_overlap = _safe_num(row.get("apc_route_stop_overlap_rate"))
    focus_overlap = float(focus_profile["in_current_gtfs_focus_route"].mean()) if not focus_profile.empty else math.nan
    return pd.DataFrame([
        {
            "id": "S1",
            "evidence_item": "same_agency_apc_static_gtfs_route_stop_match",
            "status": "supported" if route_overlap >= 0.95 and stop_overlap >= 0.90 else "partial",
            "allowed_wording": "same-agency MBTA APC board/alight/load can be joined to MBTA GTFS route and stop identifiers",
            "forbidden_wording": "same-day AVL-calibrated field deployment or exact historical schedule replay",
            "evidence": (
                f"route_overlap={route_overlap:.4f}; stop_overlap={stop_overlap:.4f}; "
                f"route_stop_overlap={route_stop_overlap:.4f}"
            ),
            "api_or_data_needed": "historical same-period GTFS archive if exact Fall 2025 schedule replay is required",
        },
        {
            "id": "S2",
            "evidence_item": "route111_apc_load_calibration_targets",
            "status": "supported" if not focus_profile.empty and focus_overlap >= 0.80 else "partial",
            "allowed_wording": "Route 111 has same-network public APC load/boarding/alighting targets matched to current MBTA GTFS stops",
            "forbidden_wording": "Route 111 FreqDuet field outcome or observed wait-time improvement",
            "evidence": f"route111_profile_rows={len(focus_profile)}; current_gtfs_focus_stop_match={focus_overlap:.4f}",
            "api_or_data_needed": "same-day AVL/arrival archive and route-specific OD if claiming control-loop calibration",
        },
        {
            "id": "S3",
            "evidence_item": "same_period_static_schedule_alignment",
            "status": "not_supported",
            "allowed_wording": "current GTFS route/stop geometry used as a same-agency structural match",
            "forbidden_wording": "Fall 2025 APC matched to exact Fall 2025 scheduled trips",
            "evidence": "local APC is Fall 2025; local GTFS feed_info is current Spring 2026",
            "api_or_data_needed": "MBTA historical GTFS archive for the APC season",
        },
        {
            "id": "S4",
            "evidence_item": "same_period_avl_arrival_departure_events",
            "status": "not_supported",
            "allowed_wording": "GTFS-RT/AVL collection path is identified",
            "forbidden_wording": "same-day historical AVL arrival/departure calibration",
            "evidence": "no local historical GTFS-RT VehiclePositions/TripUpdates archive paired with Fall 2025 APC",
            "api_or_data_needed": "MBTA historical AVL archive or prospective GTFS-RT collection window",
        },
        {
            "id": "S5",
            "evidence_item": "same_network_od_apc_avl_control_loop",
            "status": "not_supported",
            "allowed_wording": "same-agency APC/load targets plus separate public OD evidence",
            "forbidden_wording": "exact AFC/APC/AVL OD-load control-loop field calibration",
            "evidence": "MBTA APC has board/alight/load but no OD; MTA OD is a separate subway network",
            "api_or_data_needed": "agency AFC/OD export or APC-derived OD estimates for the same MBTA bus route/day",
        },
    ])


def plot_focus_route(profile: pd.DataFrame, out_dir: Path, focus_route: str) -> None:
    if profile.empty:
        return
    subset = profile[profile["day_type"].eq("Wkdy")].copy()
    if subset.empty:
        subset = profile.copy()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for direction, part in subset.groupby("direction_id"):
        part = part.sort_values("stop_sequence_num")
        ax.plot(
            part["stop_sequence_num"],
            part["mean_load"],
            marker="o",
            linewidth=1.6,
            markersize=3.2,
            label=f"direction {direction}",
        )
    ax.set_title(f"MBTA Route {focus_route} APC onboard-load target")
    ax.set_xlabel("APC stop sequence")
    ax.set_ylabel("Mean onboard load")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"mbta_route{focus_route}_apc_load_profile.png", dpi=220)
    fig.savefig(out_dir / f"mbta_route{focus_route}_apc_load_profile.pdf")
    plt.close(fig)


def write_audit_note(
    out_dir: Path,
    overlap: pd.DataFrame,
    claim_boundaries: pd.DataFrame,
    focus_route: str,
) -> None:
    row = overlap.iloc[0]
    lines = [
        "# MBTA Same-Network Calibration Audit",
        "",
        "This audit checks public MBTA APC board/alight/load data against MBTA static GTFS.",
        "It is intentionally conservative: it records route/stop matching evidence and",
        "keeps same-day AVL, exact OD, and deployment claims out of scope until those",
        "data exist.",
        "",
        "## Match Summary",
        "",
        f"- APC routes: {int(row['apc_routes'])}; route overlap with current GTFS: {row['apc_route_overlap_rate']:.4f}",
        f"- APC stops: {int(row['apc_stops'])}; stop overlap with current GTFS: {row['apc_stop_overlap_rate']:.4f}",
        f"- APC route-direction-stop pairs: {int(row['apc_route_stop_pairs'])}; current GTFS overlap: {row['apc_route_stop_overlap_rate']:.4f}",
        f"- Focus route: `{focus_route}`",
        "",
        "## Claim Boundary",
        "",
        "Allowed: same-agency MBTA APC load/boarding/alighting targets can be matched",
        "to static MBTA route/stop geometry, with Route 111 as a concrete route-level",
        "calibration target.",
        "",
        "Not allowed yet: exact same-day AFC/APC/AVL field calibration, historical",
        "arrival/departure replay, route-level OD calibration, or observed field",
        "wait-time improvement.",
        "",
        "## Claim Table",
        "",
    ]
    lines.extend(
        f"- `{r.id}` {r.evidence_item}: {r.status}; {r.evidence}"
        for r in claim_boundaries.itertuples(index=False)
    )
    (out_dir / "mbta_same_network_calibration_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apc-csv", default=str(DEFAULT_MBTA_APC_CSV))
    parser.add_argument("--gtfs-dir", default=str(DEFAULT_MBTA_GTFS_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--focus-route", default="111")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apc_csv = Path(args.apc_csv)
    gtfs_dir = Path(args.gtfs_dir)
    focus_route = str(args.focus_route)

    apc = load_apc(apc_csv, focus_route, args.chunksize)
    gtfs = load_gtfs(gtfs_dir, set(apc["routes"]), focus_route)
    overlap, route_stop_pairs = build_overlap_tables(apc, gtfs, focus_route)
    focus_profile = enrich_focus_profile(apc["focus_profile"], gtfs)
    source_coverage = build_source_coverage(apc, gtfs, gtfs_dir, apc_csv)
    claim_boundaries = build_claim_boundaries(overlap, focus_profile)

    _write_csv(out_dir / "mbta_same_network_source_coverage.csv", source_coverage)
    _write_csv(out_dir / "mbta_same_network_overlap_summary.csv", overlap)
    _write_csv(out_dir / "mbta_apc_route_stop_gtfs_overlap.csv", route_stop_pairs)
    _write_csv(out_dir / f"mbta_route{focus_route}_apc_gtfs_profile.csv", focus_profile)
    _write_csv(out_dir / "mbta_same_network_claim_boundaries.csv", claim_boundaries)
    _write_csv(out_dir / "mbta_apc_top_routes.csv", apc["top_routes"].head(30))
    plot_focus_route(focus_profile, out_dir, focus_route)
    write_audit_note(out_dir, overlap, claim_boundaries, focus_route)

    summary = {
        "out_dir": str(out_dir),
        "apc_csv": str(apc_csv),
        "gtfs_dir": str(gtfs_dir),
        "focus_route": focus_route,
        "overlap": overlap.iloc[0].to_dict(),
        "source_coverage_csv": _as_rel(out_dir / "mbta_same_network_source_coverage.csv"),
        "claim_boundaries_csv": _as_rel(out_dir / "mbta_same_network_claim_boundaries.csv"),
        "focus_profile_csv": _as_rel(out_dir / f"mbta_route{focus_route}_apc_gtfs_profile.csv"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir}")
    print(json.dumps(summary["overlap"], indent=2))


if __name__ == "__main__":
    main()
