#!/usr/bin/env python3
"""Build external OD and onboard-load truth-source audits for FreqDuet.

The audit is deliberately data-facing. It validates public agency sources that
contain OD estimates and onboard-load/board-alight counts, then emits compact
calibration-target tables for the paper package. It does not import the
separate ``transit_hrl`` algorithm stack and it does not claim that FreqDuet has
been field-deployed on these exact agency feeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results_freqduet" / "external_od_onboard_truth_audit" / "v1"
DEFAULT_MTA_SAMPLE = (
    ROOT / "data" / "external_truth_sources" / "mta_subway_od"
    / "mta_subway_od_2024_sample_5000.json"
)
DEFAULT_MTA_COUNT = (
    ROOT / "data" / "external_truth_sources" / "mta_subway_od"
    / "mta_subway_od_2024_count.json"
)
DEFAULT_MTA_METADATA = (
    ROOT / "data" / "external_truth_sources" / "mta_subway_od"
    / "mta_subway_od_2024_metadata.json"
)
DEFAULT_MBTA_CSV = Path(os.environ.get(
    "FREQDUET_MBTA_BUS_RIDERSHIP_CSV",
    "/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/mbta/"
    "ridership/MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/"
    "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/"
    "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop_Fall_2025.csv",
))

MBTA_SOURCE_URL = "https://hub.arcgis.com/datasets/8daf4a33925a4df59183f860826d29ee"
MBTA_DATA_URL = "https://www.arcgis.com/sharing/rest/content/items/8daf4a33925a4df59183f860826d29ee/data"
MTA_OD_URL = "https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj"
MTA_OD_SAMPLE_URL = "https://data.ny.gov/resource/jsu2-fbtj.json?%24limit=5000"
MTA_OD_COUNT_URL = "https://data.ny.gov/resource/jsu2-fbtj.json?%24select=count(*)"
MTA_OD_METADATA_URL = "https://data.ny.gov/api/views/jsu2-fbtj"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as response:
        path.write_bytes(response.read())


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _json_load(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _hour_from_trip_start(value: Any) -> int:
    text = str(value or "")
    if ":" not in text:
        return -1
    return _safe_int(text.split(":", 1)[0], -1)


def _as_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_mbta_targets(csv_path: Path, *, chunksize: int = 200_000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = [
        "GTFS route_id",
        "GTFS direction_id",
        "trip start time",
        "GTFS stop_id",
        "stop sequence",
        "Day Type",
        "Boardings",
        "Alightings",
        "Load",
        "# of Trip Samples ",
    ]
    if not csv_path.exists():
        coverage = {
            "source": "mbta_bus_stop_trip_ridership",
            "source_kind": "public_agency_observed_bus_apc",
            "claim_status": "external_missing",
            "source_url": MBTA_SOURCE_URL,
            "data_url": MBTA_DATA_URL,
            "local_file": str(csv_path),
            "boundary": "MBTA bus ridership CSV was not available locally.",
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), coverage

    route_parts: list[pd.DataFrame] = []
    hour_parts: list[pd.DataFrame] = []
    stop_parts: list[pd.DataFrame] = []
    row_count = 0
    route_set: set[str] = set()
    stop_set: set[str] = set()
    day_types: set[str] = set()
    total_board = 0.0
    total_alight = 0.0
    load_sum = 0.0
    max_load = 0.0
    trip_samples = 0.0

    for chunk in pd.read_csv(csv_path, usecols=required, chunksize=chunksize):
        row_count += len(chunk)
        for col in ["Boardings", "Alightings", "Load", "# of Trip Samples "]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
        chunk["hour"] = chunk["trip start time"].map(_hour_from_trip_start)
        route_set.update(chunk["GTFS route_id"].astype(str).unique())
        stop_set.update(chunk["GTFS stop_id"].astype(str).unique())
        day_types.update(chunk["Day Type"].astype(str).unique())
        total_board += float(chunk["Boardings"].sum())
        total_alight += float(chunk["Alightings"].sum())
        load_sum += float(chunk["Load"].sum())
        max_load = max(max_load, float(chunk["Load"].max()))
        trip_samples += float(chunk["# of Trip Samples "].sum())

        route_parts.append(chunk.groupby(
            ["GTFS route_id", "GTFS direction_id", "Day Type"], as_index=False
        ).agg(
            rows=("Load", "size"),
            boardings=("Boardings", "sum"),
            alightings=("Alightings", "sum"),
            mean_load=("Load", "mean"),
            max_load=("Load", "max"),
            trip_samples=("# of Trip Samples ", "sum"),
        ))
        hour_parts.append(chunk.groupby(["Day Type", "hour"], as_index=False).agg(
            rows=("Load", "size"),
            boardings=("Boardings", "sum"),
            alightings=("Alightings", "sum"),
            mean_load=("Load", "mean"),
            p95_load=("Load", lambda x: float(np.percentile(x, 95))),
            max_load=("Load", "max"),
        ))
        stop_parts.append(chunk.groupby(
            ["GTFS route_id", "GTFS direction_id", "Day Type", "stop sequence"],
            as_index=False,
        ).agg(
            rows=("Load", "size"),
            boardings=("Boardings", "sum"),
            alightings=("Alightings", "sum"),
            mean_load=("Load", "mean"),
            max_load=("Load", "max"),
        ))

    route_targets = pd.concat(route_parts, ignore_index=True)
    route_targets = route_targets.groupby(
        ["GTFS route_id", "GTFS direction_id", "Day Type"], as_index=False
    ).agg(
        rows=("rows", "sum"),
        boardings=("boardings", "sum"),
        alightings=("alightings", "sum"),
        mean_load=("mean_load", "mean"),
        max_load=("max_load", "max"),
        trip_samples=("trip_samples", "sum"),
    )
    route_targets["load_per_boarding"] = route_targets["mean_load"] / route_targets["boardings"].replace(0, np.nan)
    route_targets = route_targets.sort_values(["boardings", "max_load"], ascending=False)

    hourly = pd.concat(hour_parts, ignore_index=True)
    hourly = hourly.groupby(["Day Type", "hour"], as_index=False).agg(
        rows=("rows", "sum"),
        boardings=("boardings", "sum"),
        alightings=("alightings", "sum"),
        mean_load=("mean_load", "mean"),
        p95_load=("p95_load", "mean"),
        max_load=("max_load", "max"),
    )
    hourly = hourly[hourly["hour"].ge(0)].sort_values(["Day Type", "hour"])

    stop_curve = pd.concat(stop_parts, ignore_index=True)
    stop_curve = stop_curve.groupby(
        ["GTFS route_id", "GTFS direction_id", "Day Type", "stop sequence"],
        as_index=False,
    ).agg(
        rows=("rows", "sum"),
        boardings=("boardings", "sum"),
        alightings=("alightings", "sum"),
        mean_load=("mean_load", "mean"),
        max_load=("max_load", "max"),
    )

    coverage = {
        "source": "mbta_bus_stop_trip_ridership",
        "source_kind": "public_agency_observed_bus_apc",
        "claim_status": "supported",
        "agency": "Massachusetts Bay Transportation Authority",
        "source_url": MBTA_SOURCE_URL,
        "data_url": MBTA_DATA_URL,
        "local_file": str(csv_path),
        "rows": int(row_count),
        "unique_routes": int(len(route_set)),
        "unique_stops": int(len(stop_set)),
        "day_types": sorted(day_types),
        "has_boardings": True,
        "has_alightings": True,
        "has_onboard_load": True,
        "total_boardings": total_board,
        "total_alightings": total_alight,
        "mean_load": load_sum / row_count if row_count else math.nan,
        "max_load": max_load,
        "total_trip_samples": trip_samples,
        "boundary": "Observed MBTA bus stop/trip averages with boardings, alightings, and onboard load; not OD.",
    }
    return route_targets, hourly, stop_curve, coverage


def load_mta_od(sample_path: Path, count_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = _json_load(sample_path)
    count_rows = _json_load(count_path)
    metadata = _json_load(metadata_path) or {}
    full_table_rows = 0
    if isinstance(count_rows, list) and count_rows:
        full_table_rows = _safe_int(count_rows[0].get("count"), 0)
    if not isinstance(rows, list) or not rows:
        coverage = {
            "source": "mta_subway_od_estimate_2024",
            "source_kind": "public_agency_estimated_subway_od",
            "claim_status": "external_missing",
            "agency": "Metropolitan Transportation Authority",
            "source_url": MTA_OD_URL,
            "api_endpoint": MTA_OD_SAMPLE_URL,
            "local_file": str(sample_path),
            "full_table_rows": int(full_table_rows),
            "boundary": "MTA OD sample is missing locally; only metadata can be reported.",
        }
        return pd.DataFrame(), pd.DataFrame(), coverage

    df = pd.DataFrame(rows)
    df["estimated_average_ridership"] = pd.to_numeric(
        df["estimated_average_ridership"], errors="coerce"
    ).fillna(0.0)
    df["hour_of_day"] = pd.to_numeric(df["hour_of_day"], errors="coerce").astype("Int64")
    df["origin_station_complex_id"] = df["origin_station_complex_id"].astype(str)
    df["destination_station_complex_id"] = df["destination_station_complex_id"].astype(str)

    top_pairs = df.groupby([
        "origin_station_complex_id",
        "origin_station_complex_name",
        "destination_station_complex_id",
        "destination_station_complex_name",
    ], as_index=False).agg(
        estimated_average_ridership=("estimated_average_ridership", "sum"),
        rows=("estimated_average_ridership", "size"),
    ).sort_values("estimated_average_ridership", ascending=False)

    hourly = df.groupby(["day_of_week", "hour_of_day"], as_index=False).agg(
        estimated_average_ridership=("estimated_average_ridership", "sum"),
        od_pairs=("estimated_average_ridership", "size"),
    ).sort_values(["day_of_week", "hour_of_day"])

    coverage = {
        "source": "mta_subway_od_estimate_2024",
        "source_kind": "public_agency_estimated_subway_od",
        "claim_status": "supported",
        "agency": metadata.get("attribution") or "Metropolitan Transportation Authority",
        "source_url": MTA_OD_URL,
        "api_endpoint": MTA_OD_SAMPLE_URL,
        "local_file": _as_rel(sample_path),
        "sample_rows": int(len(df)),
        "full_table_rows": int(full_table_rows),
        "unique_origins": int(df["origin_station_complex_id"].nunique()),
        "unique_destinations": int(df["destination_station_complex_id"].nunique()),
        "unique_od_pairs": int(top_pairs.shape[0]),
        "unique_day_hour_bins": int(hourly.shape[0]),
        "total_estimated_average_ridership_sample": float(df["estimated_average_ridership"].sum()),
        "has_od_fields": True,
        "has_ridership": True,
        "dataset_description": str(metadata.get("description", "")),
        "boundary": "Agency-published subway OD estimate from fare-derived inference; not observed individual OD truth, bus OD, or onboard load.",
    }
    return top_pairs, hourly, coverage


def build_claim_boundaries(mbta: dict[str, Any], mta: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "id": "E1",
            "evidence_item": "real_public_bus_stop_board_alight",
            "status": "supported" if mbta.get("has_boardings") and mbta.get("has_alightings") else mbta.get("claim_status", "missing"),
            "allowed_wording": "real public MBTA bus stop/trip boardings and alightings",
            "forbidden_wording": "complete OD calibration or field improvement unless linked to same-route control validation",
            "evidence": (
                f"rows={mbta.get('rows', 0)} routes={mbta.get('unique_routes', 0)} "
                f"stops={mbta.get('unique_stops', 0)} "
                f"boardings={mbta.get('total_boardings', 0.0):.1f} "
                f"alightings={mbta.get('total_alightings', 0.0):.1f}"
            ),
        },
        {
            "id": "E2",
            "evidence_item": "real_public_bus_stop_onboard_load",
            "status": "supported" if mbta.get("has_onboard_load") else mbta.get("claim_status", "missing"),
            "allowed_wording": "real public MBTA bus onboard-load calibration targets",
            "forbidden_wording": "onboard-load improvement under FreqDuet unless evaluated in a matched control loop",
            "evidence": (
                f"rows={mbta.get('rows', 0)} mean_load={mbta.get('mean_load', 0.0):.4f} "
                f"max_load={mbta.get('max_load', 0.0):.4f}"
            ),
        },
        {
            "id": "E3",
            "evidence_item": "real_public_subway_od_estimate",
            "status": "supported" if mta.get("has_od_fields") and mta.get("has_ridership") else mta.get("claim_status", "missing"),
            "allowed_wording": "real public agency-estimated subway OD matrices from MTA",
            "forbidden_wording": "observed individual AFC OD truth, bus OD, or onboard-load calibration",
            "evidence": (
                f"sample_rows={mta.get('sample_rows', 0)} full_rows={mta.get('full_table_rows', 0)} "
                f"origins={mta.get('unique_origins', 0)} destinations={mta.get('unique_destinations', 0)} "
                f"od_pairs={mta.get('unique_od_pairs', 0)}"
            ),
        },
        {
            "id": "E4",
            "evidence_item": "joint_same_network_od_onboard_control_loop",
            "status": "not_supported",
            "allowed_wording": "separate public OD and onboard-load truth-source coverage",
            "forbidden_wording": "exact same-network AFC/APC OD plus onboard-load calibration for FreqDuet deployment",
            "evidence": "MTA OD is subway agency-estimated OD; MBTA APC is bus stop/trip load. They are not one joint route/day/control loop.",
        },
    ]
    return pd.DataFrame(rows)


def write_plots(route_targets: pd.DataFrame, hourly_load: pd.DataFrame, mta_pairs: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if not route_targets.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        values = route_targets["max_load"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(values, bins=36, color="#2b6cb0", alpha=0.85, edgecolor="white")
        ax.set_xlabel("Route-direction-day maximum onboard load")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#cbd5e0", alpha=0.8)
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(out_dir / f"mbta_onboard_load_distribution.{fmt}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if not hourly_load.empty:
        fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
        wkdy = hourly_load[hourly_load["Day Type"].astype(str).eq("Wkdy")].sort_values("hour")
        ax1.plot(wkdy["hour"], wkdy["boardings"], color="#2b6cb0", marker="o", label="Boardings")
        ax1.plot(wkdy["hour"], wkdy["alightings"], color="#b83280", marker="o", label="Alightings")
        ax1.set_xlabel("Trip start hour")
        ax1.set_ylabel("Passenger counts")
        ax2 = ax1.twinx()
        ax2.plot(wkdy["hour"], wkdy["mean_load"], color="#2f855a", marker="s", label="Mean load")
        ax2.set_ylabel("Mean onboard load")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
        ax1.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax1.grid(axis="y", color="#cbd5e0", alpha=0.8)
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(out_dir / f"mbta_weekday_hourly_board_alight_load.{fmt}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if not mta_pairs.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        top = mta_pairs.head(12).copy()
        labels = [
            f"{o} -> {d}"
            for o, d in zip(top["origin_station_complex_name"], top["destination_station_complex_name"])
        ]
        y = np.arange(len(top))
        ax.barh(y, top["estimated_average_ridership"], color="#805ad5", alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Estimated average ridership")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#cbd5e0", alpha=0.8)
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(out_dir / f"mta_od_sample_top_pairs.{fmt}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_report(out_dir: Path, coverage: pd.DataFrame, boundaries: pd.DataFrame) -> None:
    lines = [
        "# External OD And Onboard-Load Truth Audit",
        "",
        "This FreqDuet audit adds stronger public agency truth-source coverage than the AFC/APC profile audit.",
        "It validates separate OD-estimate and onboard-load sources, but it does not create a same-network field calibration loop.",
        "",
        "## Supported Evidence",
        "",
    ]
    for _, row in boundaries.iterrows():
        lines.append(f"- `{row['id']}` {row['evidence_item']}: `{row['status']}`. {row['evidence']}")
    lines.extend([
        "",
        "## Source Boundary",
        "",
    ])
    for _, row in coverage.iterrows():
        lines.append(f"- `{row['source']}`: {row['boundary']}")
    (out_dir / "external_od_onboard_truth_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbta-csv", default=str(DEFAULT_MBTA_CSV))
    parser.add_argument("--mta-sample-json", default=str(DEFAULT_MTA_SAMPLE))
    parser.add_argument("--mta-count-json", default=str(DEFAULT_MTA_COUNT))
    parser.add_argument("--mta-metadata-json", default=str(DEFAULT_MTA_METADATA))
    parser.add_argument("--download-mta", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip().lower() for x in str(args.formats).split(",") if x.strip()]

    mta_sample = Path(args.mta_sample_json)
    mta_count = Path(args.mta_count_json)
    mta_metadata = Path(args.mta_metadata_json)
    if args.download_mta:
        if not mta_sample.exists():
            _download(MTA_OD_SAMPLE_URL, mta_sample)
        if not mta_count.exists():
            _download(MTA_OD_COUNT_URL, mta_count)
        if not mta_metadata.exists():
            _download(MTA_OD_METADATA_URL, mta_metadata)

    route_targets, hourly_load, stop_curve, mbta_coverage = load_mbta_targets(Path(args.mbta_csv))
    mta_pairs, mta_hourly, mta_coverage = load_mta_od(mta_sample, mta_count, mta_metadata)
    coverage = pd.DataFrame([mbta_coverage, mta_coverage])
    boundaries = build_claim_boundaries(mbta_coverage, mta_coverage)

    coverage.to_csv(out_dir / "external_truth_source_coverage.csv", index=False)
    boundaries.to_csv(out_dir / "external_truth_claim_boundaries.csv", index=False)
    if not route_targets.empty:
        route_targets.to_csv(out_dir / "mbta_onboard_route_targets.csv", index=False)
        hourly_load.to_csv(out_dir / "mbta_hourly_board_alight_load.csv", index=False)
        stop_curve.to_csv(out_dir / "mbta_stop_sequence_load_curve.csv", index=False)
    if not mta_pairs.empty:
        mta_pairs.to_csv(out_dir / "mta_od_sample_top_pairs.csv", index=False)
        mta_hourly.to_csv(out_dir / "mta_od_sample_hourly_profile.csv", index=False)
    write_plots(route_targets, hourly_load, mta_pairs, out_dir, formats)
    write_report(out_dir, coverage, boundaries)

    payload = {
        "status": "generated",
        "does_not_import_transit_hrl": True,
        "truth_scope": "separate_public_agency_od_estimate_and_onboard_load_sources",
        "not_supported": "same-network exact AFC/APC OD plus onboard-load field calibration for FreqDuet deployment",
        "inputs": {
            "mbta_csv": str(Path(args.mbta_csv)),
            "mta_sample_json": str(mta_sample),
            "mta_count_json": str(mta_count),
            "mta_metadata_json": str(mta_metadata),
        },
        "coverage": coverage.to_dict(orient="records"),
        "claim_boundaries": boundaries.to_dict(orient="records"),
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_dir}")
    print(f"claims_supported={int(boundaries['status'].eq('supported').sum())}/{len(boundaries)}")


if __name__ == "__main__":
    main()
