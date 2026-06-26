#!/usr/bin/env python3
"""Build route-family and service-day held-out readiness artifacts.

This audit converts the already downloaded MBTA/MTA external caches into
paper-facing tables. It is intentionally a readiness/protocol audit: it
documents which route-family and service-day splits are supported by local
data, and which claims still require a full FreqDuet route/day experiment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results_freqduet" / "route_day_heldout_readiness" / "v1"
MTA_CACHE = (
    ROOT
    / "data/external_truth_sources/mta_bus_time_api/offline_cache/20260626T144132Z/parsed"
)
MBTA_AUDIT = ROOT / "results_freqduet" / "mbta_same_network_calibration_audit" / "v1"


KEY_MBTA_ROUTES = {
    "1",
    "15",
    "22",
    "23",
    "28",
    "32",
    "39",
    "57",
    "66",
    "71",
    "73",
    "77",
    "111",
    "116",
    "117",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def route_prefix(short_name: object) -> str:
    text = str(short_name or "").upper().replace("-SBS", "").replace("+", "")
    match = re.match(r"([A-Z]+)", text)
    return match.group(1) if match else "NUMERIC"


def mta_service_class(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(name, ""))
        for name in ("route_id", "short_name", "long_name", "description")
    ).upper()
    if "SBS" in text or "SELECT BUS" in text or "+" in str(row.get("route_id", "")):
        return "select_bus"
    if int(row.get("route_type", 0) or 0) == 711:
        return "temporary_shuttle"
    if str(row.get("short_name", "")).upper().startswith("SIM"):
        return "express"
    return "local_bus"


def mbta_route_bucket(route_id: object) -> str:
    text = str(route_id)
    if text in KEY_MBTA_ROUTES:
        return "key_high_ridership"
    try:
        value = int(float(text))
    except ValueError:
        return "non_numeric"
    lower = (value // 100) * 100
    return f"numbered_{lower:03d}_{lower + 99:03d}"


def summarize_mta() -> tuple[pd.DataFrame, dict[str, object]]:
    routes = read_csv(MTA_CACHE / "mta_bus_time_routes.csv")
    seq = read_csv(MTA_CACHE / "mta_bus_time_route_stop_sequences.csv")
    vehicles = read_csv(MTA_CACHE / "mta_bus_time_vehicle_snapshots.csv")
    if routes.empty:
        return pd.DataFrame(), {"mta_routes": 0}

    routes = routes.copy()
    routes["route_prefix"] = routes["short_name"].map(route_prefix)
    routes["service_class"] = routes.apply(mta_service_class, axis=1)
    routes["family_id"] = (
        "mta_"
        + routes["agency_id"].astype(str).str.replace(r"\s+", "_", regex=True)
        + "_"
        + routes["route_prefix"].astype(str)
        + "_"
        + routes["service_class"].astype(str)
    )

    seq_by_route = pd.DataFrame()
    if not seq.empty:
        seq_by_route = (
            seq.groupby("route_id", as_index=False)
            .agg(
                route_stop_rows=("stop_id", "size"),
                route_stop_unique_stops=("stop_id", "nunique"),
                route_directions=("direction_id", "nunique"),
                route_stop_sequence_max=("stop_sequence", "max"),
            )
            .assign(route_stop_sequence_len=lambda x: x["route_stop_sequence_max"] + 1)
        )
        routes = routes.merge(seq_by_route, on="route_id", how="left")
    else:
        for col in (
            "route_stop_rows",
            "route_stop_unique_stops",
            "route_directions",
            "route_stop_sequence_len",
        ):
            routes[col] = np.nan

    vehicle_by_route = pd.DataFrame()
    if not vehicles.empty and "line_ref" in vehicles.columns:
        v = vehicles.copy()
        extensions = v.get("monitored_call_extensions", pd.Series(index=v.index, dtype=str))
        v["has_occupancy_payload"] = extensions.astype(str).str.contains(
            "EstimatedPassenger", na=False
        )
        vehicle_by_route = (
            v.groupby("line_ref", as_index=False)
            .agg(
                vehicle_snapshot_rows=("line_ref", "size"),
                vehicle_snapshot_vehicles=("vehicle_ref", "nunique"),
                vehicle_snapshot_occupancy_rows=("has_occupancy_payload", "sum"),
            )
            .rename(columns={"line_ref": "route_id"})
        )
        routes = routes.merge(vehicle_by_route, on="route_id", how="left")
    else:
        for col in (
            "vehicle_snapshot_rows",
            "vehicle_snapshot_vehicles",
            "vehicle_snapshot_occupancy_rows",
        ):
            routes[col] = 0

    for col in (
        "route_stop_rows",
        "route_stop_unique_stops",
        "route_directions",
        "route_stop_sequence_len",
        "vehicle_snapshot_rows",
        "vehicle_snapshot_vehicles",
        "vehicle_snapshot_occupancy_rows",
    ):
        routes[col] = pd.to_numeric(routes[col], errors="coerce").fillna(0)

    rows = []
    for family_id, group in routes.groupby("family_id", sort=True):
        rows.append(
            {
                "source": "mta_bus_time_api_offline_cache_v1",
                "agency": ",".join(sorted(group["agency_id"].astype(str).unique())),
                "family_id": family_id,
                "route_family": f"{group['route_prefix'].iloc[0]} / {group['service_class'].iloc[0]}",
                "n_routes": int(group["route_id"].nunique()),
                "example_routes": ",".join(group["short_name"].astype(str).head(12)),
                "route_stop_rows": int(group["route_stop_rows"].sum()),
                "unique_route_stops": int(group["route_stop_unique_stops"].sum()),
                "mean_stops_per_route": float(group["route_stop_sequence_len"].mean()),
                "directions_observed": int(group["route_directions"].max()),
                "vehicle_snapshot_rows": int(group["vehicle_snapshot_rows"].sum()),
                "vehicle_snapshot_vehicles": int(group["vehicle_snapshot_vehicles"].sum()),
                "vehicle_snapshot_occupancy_rows": int(
                    group["vehicle_snapshot_occupancy_rows"].sum()
                ),
                "day_types_available": "snapshot_date_only",
                "readiness_status": "ready_for_route_family_protocol",
                "paper_use": "held-out route-family split design and AVL/geometry realism audit",
                "boundary": (
                    "Route/stop geometry and route-filtered vehicle snapshots; "
                    "not APC/onboard-load, not multi-day historical AVL, and not a "
                    "completed FreqDuet policy evaluation."
                ),
            }
        )

    summary = {
        "mta_routes": int(routes["route_id"].nunique()),
        "mta_route_families": int(len(rows)),
        "mta_route_stop_rows": int(seq.shape[0]) if not seq.empty else 0,
        "mta_unique_stops_sum_by_route": int(routes["route_stop_unique_stops"].sum()),
        "mta_vehicle_snapshot_rows": int(vehicles.shape[0]) if not vehicles.empty else 0,
    }
    return pd.DataFrame(rows), summary


def summarize_mbta() -> tuple[pd.DataFrame, dict[str, object]]:
    source = read_csv(MBTA_AUDIT / "mbta_same_network_source_coverage.csv")
    top_routes = read_csv(MBTA_AUDIT / "mbta_apc_top_routes.csv")
    overlap = read_csv(MBTA_AUDIT / "mbta_apc_route_stop_gtfs_overlap.csv")
    route111 = read_csv(MBTA_AUDIT / "mbta_route111_apc_gtfs_profile.csv")
    if source.empty and top_routes.empty and overlap.empty:
        return pd.DataFrame(), {"mbta_routes": 0}

    rows = []
    day_types = "Wkdy,Sat,Sun"
    if not source.empty and "day_types" in source.columns:
        vals = source["day_types"].dropna().astype(str).tolist()
        if vals:
            day_types = vals[0].replace("[", "").replace("]", "").replace("'", "")

    if not top_routes.empty:
        top_routes = top_routes.copy()
        top_routes["route_id"] = top_routes["GTFS route_id"].astype(str)
        top_routes["family_id"] = "mbta_" + top_routes["route_id"].map(mbta_route_bucket)
        overlap_by_route = pd.DataFrame()
        if not overlap.empty:
            overlap = overlap.copy()
            overlap["route_id"] = overlap["route_id"].astype(str)
            overlap_by_route = (
                overlap.groupby("route_id", as_index=False)
                .agg(
                    overlap_route_stop_rows=("stop_id", "size"),
                    overlap_unique_stops=("stop_id", "nunique"),
                    overlap_gtfs_rate=("in_current_gtfs_route_stop", "mean"),
                )
            )
            top_routes = top_routes.merge(overlap_by_route, on="route_id", how="left")
        else:
            for col in (
                "overlap_route_stop_rows",
                "overlap_unique_stops",
                "overlap_gtfs_rate",
            ):
                top_routes[col] = np.nan

        for family_id, group in top_routes.groupby("family_id", sort=True):
            rows.append(
                {
                    "source": "mbta_same_network_calibration_audit_v1",
                    "agency": "MBTA",
                    "family_id": family_id,
                    "route_family": family_id.replace("mbta_", ""),
                    "n_routes": int(group["route_id"].nunique()),
                    "example_routes": ",".join(group["route_id"].astype(str).head(12)),
                    "route_stop_rows": int(
                        pd.to_numeric(group["overlap_route_stop_rows"], errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                    "unique_route_stops": int(
                        pd.to_numeric(group["overlap_unique_stops"], errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                    "mean_stops_per_route": float(
                        pd.to_numeric(group["overlap_unique_stops"], errors="coerce")
                        .dropna()
                        .mean()
                    )
                    if pd.to_numeric(group["overlap_unique_stops"], errors="coerce").notna().any()
                    else np.nan,
                    "directions_observed": 2,
                    "vehicle_snapshot_rows": 0,
                    "vehicle_snapshot_vehicles": 0,
                    "vehicle_snapshot_occupancy_rows": 0,
                    "day_types_available": day_types,
                    "readiness_status": "ready_for_day_type_protocol",
                    "paper_use": "APC day-type route/load split design and route-stop structural calibration",
                    "boundary": (
                        "Observed Fall 2025 APC board/alight/load with current static "
                        "GTFS overlap; not exact same-day historical AVL/control replay."
                    ),
                }
            )

    if not route111.empty:
        rows.append(
            {
                "source": "mbta_same_network_calibration_audit_v1",
                "agency": "MBTA",
                "family_id": "mbta_route111_focus",
                "route_family": "route111_focus_profile",
                "n_routes": 1,
                "example_routes": "111",
                "route_stop_rows": int(route111.shape[0]),
                "unique_route_stops": int(route111["stop_id"].nunique()),
                "mean_stops_per_route": float(route111.groupby("direction_id")["stop_id"].nunique().mean()),
                "directions_observed": int(route111["direction_id"].nunique()),
                "vehicle_snapshot_rows": 0,
                "vehicle_snapshot_vehicles": 0,
                "vehicle_snapshot_occupancy_rows": 0,
                "day_types_available": ",".join(sorted(route111["day_type"].astype(str).unique())),
                "readiness_status": "ready_for_route_day_case_study_protocol",
                "paper_use": "concrete MBTA route load-profile case study and day-type split seed",
                "boundary": "Route 111 APC/GTFS load profile only; no full FreqDuet route replay result.",
            }
        )

    summary = {
        "mbta_source_rows": int(source.shape[0]) if not source.empty else 0,
        "mbta_top_routes": int(top_routes["route_id"].nunique()) if not top_routes.empty else 0,
        "mbta_overlap_rows": int(overlap.shape[0]) if not overlap.empty else 0,
        "mbta_route111_rows": int(route111.shape[0]) if not route111.empty else 0,
        "mbta_route_families": int(len(rows)),
    }
    return pd.DataFrame(rows), summary


def build_protocol() -> pd.DataFrame:
    rows = [
        {
            "protocol_id": "mta_route_family_static_holdout",
            "fit_split": "fit harmonic prior and route geometry on selected MTA prefix/agency families",
            "heldout_split": "evaluate on withheld borough/prefix/service-class families",
            "available_inputs": "MTA Bus Time route, stop, route-stop sequence, and route-filtered vehicle snapshots",
            "minimum_next_artifact": "generate FreqDuet route-family configs and run paired seed matrix",
            "current_status": "protocol_ready_not_run",
            "safe_claim": "route-family held-out protocol is specified from offline MTA route/AVL cache",
            "unsafe_claim": "FreqDuet has already generalized across MTA route families",
        },
        {
            "protocol_id": "mbta_apc_day_type_holdout",
            "fit_split": "fit demand/load profile on Wkdy APC profiles",
            "heldout_split": "evaluate on Sat/Sun APC profiles or reverse day-type splits",
            "available_inputs": "MBTA Fall 2025 APC board/alight/load by route, stop, trip, and day type",
            "minimum_next_artifact": "convert day-type APC profiles into FreqDuet OD/load replay configs",
            "current_status": "protocol_ready_not_run",
            "safe_claim": "day-type held-out calibration protocol is specified from MBTA APC data",
            "unsafe_claim": "same-day MBTA APC/AVL control-loop validation is complete",
        },
        {
            "protocol_id": "mbta_route111_case_holdout",
            "fit_split": "fit on one Route 111 direction/day-type slice",
            "heldout_split": "evaluate on the opposite direction or held-out day type",
            "available_inputs": "Route 111 APC/GTFS load profile with stop-level load targets",
            "minimum_next_artifact": "build route111 replay environment and paired policy evaluation",
            "current_status": "case_protocol_ready_not_run",
            "safe_claim": "Route 111 is a concrete same-agency load-profile case-study candidate",
            "unsafe_claim": "Route 111 field operation improvement has been observed",
        },
        {
            "protocol_id": "freqduet_existing_synthetic_day_stress",
            "fit_split": "current historical OD table and harmonic prior",
            "heldout_split": "noise10/noise20/noise40, od20/od50, rush early/late/extreme",
            "available_inputs": "completed 60-seed broad generalization matrix",
            "minimum_next_artifact": "none for current perturbation claim",
            "current_status": "completed_for_perturbation_not_route_day",
            "safe_claim": "robustness to controlled demand-noise, OD-shift, and rush-shift perturbations",
            "unsafe_claim": "multi-route or multi-service-day field generalization",
        },
    ]
    return pd.DataFrame(rows)


def build_claim_boundaries() -> pd.DataFrame:
    rows = [
        {
            "claim_id": "route_family_split_design",
            "status": "supported_protocol",
            "evidence": "route_family_coverage.csv",
            "safe_language": "offline MTA route/stop data support route-family held-out split design",
            "boundary": "policy matrix not run yet",
        },
        {
            "claim_id": "mbta_day_type_split_design",
            "status": "supported_protocol",
            "evidence": "service_day_split_protocol.csv and MBTA APC source coverage",
            "safe_language": "MBTA APC day types support a service-day held-out calibration protocol",
            "boundary": "not same-day AFC/APC/AVL/OD calibration",
        },
        {
            "claim_id": "route111_case_profile",
            "status": "supported_data_profile",
            "evidence": "mbta_route111_apc_gtfs_profile.csv",
            "safe_language": "Route 111 provides a concrete load-profile case-study candidate",
            "boundary": "not a completed route111 control experiment",
        },
        {
            "claim_id": "completed_route_day_policy_generalization",
            "status": "not_supported_yet",
            "evidence": "none",
            "safe_language": "future route/day matrix",
            "boundary": "do not claim completed multi-route or multi-service-day FreqDuet evaluation",
        },
        {
            "claim_id": "same_day_field_calibration",
            "status": "not_supported_yet",
            "evidence": "none",
            "safe_language": "external realism and calibration-readiness audit",
            "boundary": "do not claim same-day AFC/APC/AVL/OD field calibration or deployment outcome",
        },
    ]
    return pd.DataFrame(rows)


def write_note(out_dir: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Route/Day Held-Out Readiness Audit",
        "",
        "This audit packages the non-text evidence needed to design route-family",
        "and service-day held-out experiments without claiming that those policy",
        "matrices have already been run.",
        "",
        "## Coverage",
        "",
        f"- MTA route families: {summary.get('mta_route_families', 0)}",
        f"- MTA routes: {summary.get('mta_routes', 0)}",
        f"- MTA route-stop sequence rows: {summary.get('mta_route_stop_rows', 0)}",
        f"- MTA vehicle snapshot rows: {summary.get('mta_vehicle_snapshot_rows', 0)}",
        f"- MBTA route/day families: {summary.get('mbta_route_families', 0)}",
        f"- MBTA top APC routes represented: {summary.get('mbta_top_routes', 0)}",
        f"- MBTA APC/GTFS overlap rows: {summary.get('mbta_overlap_rows', 0)}",
        "",
        "## Paper Use",
        "",
        "- Use `route_family_coverage.csv` to show that the package now contains",
        "  route-family split metadata for MTA and MBTA evidence sources.",
        "- Use `service_day_split_protocol.csv` to state the exact next experiment",
        "  required for route/day held-out validation.",
        "- Use `route_day_claim_boundaries.csv` to keep the claim conservative:",
        "  route/day protocols are ready; completed route/day policy generalization",
        "  and same-day field calibration are not yet supported.",
        "",
    ]
    (out_dir / "route_day_heldout_readiness.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mta, mta_summary = summarize_mta()
    mbta, mbta_summary = summarize_mbta()
    coverage = pd.concat([mta, mbta], ignore_index=True)
    protocol = build_protocol()
    boundaries = build_claim_boundaries()
    summary = {**mta_summary, **mbta_summary}
    summary["route_family_rows"] = int(len(coverage))
    summary["protocol_rows"] = int(len(protocol))
    summary["claim_boundary_rows"] = int(len(boundaries))

    coverage.to_csv(out_dir / "route_family_coverage.csv", index=False)
    protocol.to_csv(out_dir / "service_day_split_protocol.csv", index=False)
    boundaries.to_csv(out_dir / "route_day_claim_boundaries.csv", index=False)
    with (out_dir / "route_day_heldout_readiness_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_note(out_dir, summary)

    print(f"wrote {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
