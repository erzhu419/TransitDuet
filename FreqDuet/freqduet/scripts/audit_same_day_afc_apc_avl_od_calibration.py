#!/usr/bin/env python3
"""Audit whether exact same-day AFC/APC/AVL/OD calibration is supported locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results_freqduet/same_day_field_calibration_audit/v1"
DEFAULT_EXTERNAL_AUDIT = ROOT / "results_freqduet/external_od_onboard_truth_audit/v1"
DEFAULT_H2O_ROOT = Path(
    "/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/mbta/h2o_city_envs"
)


def exists(path: Path) -> bool:
    return path.exists()


def count_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_dir())


def add(rows: list[dict], source: str, evidence_type: str, path: Path, status: str, note: str) -> None:
    rows.append({
        "source": source,
        "evidence_type": evidence_type,
        "path": str(path),
        "exists": bool(path.exists()),
        "status": status,
        "note": note,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--external-audit-dir", default=str(DEFAULT_EXTERNAL_AUDIT))
    parser.add_argument("--h2o-root", default=str(DEFAULT_H2O_ROOT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    audit_dir = Path(args.external_audit_dir)
    h2o_root = Path(args.h2o_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    add(
        rows,
        "MBTA",
        "APC route/stop/trip/day-type aggregate",
        audit_dir / "mbta_onboard_route_targets.csv",
        "supported",
        "Route/direction/day-type APC aggregate is available and used for route-day demand scaling.",
    )
    add(
        rows,
        "MBTA",
        "APC hourly day-type profile",
        audit_dir / "mbta_hourly_board_alight_load.csv",
        "supported",
        "Hourly board/alight/load profile is available and used for service-day shape multipliers.",
    )
    add(
        rows,
        "MBTA",
        "Route 111 APC/GTFS profile",
        ROOT / "results_freqduet/mbta_same_network_calibration_audit/v1/mbta_route111_apc_gtfs_profile.csv",
        "supported",
        "Single-route detailed profile exists for case-study realism checks.",
    )
    mbta_env_root = h2o_root / "MBTA_weekday_all_routes/_line_envs"
    add(
        rows,
        "MBTA/H2Oplus",
        "GTFS/APC-derived line simulation environments",
        mbta_env_root,
        "supported",
        f"{count_dirs(mbta_env_root)} line-level env directories with config/OD/route/stop/timetable files.",
    )
    add(
        rows,
        "MBTA",
        "same-day historical AVL trajectory archive",
        audit_dir / "mbta_historical_avl_same_day.csv",
        "missing",
        "No matched same-day historical AVL archive was found in the current FreqDuet cache.",
    )
    add(
        rows,
        "MBTA",
        "same-day AFC tap OD matrix",
        audit_dir / "mbta_same_day_afc_od.csv",
        "missing",
        "No exact same-day AFC tap OD matrix was found. Existing MBTA evidence is APC/load + GTFS-derived OD simulation.",
    )
    add(
        rows,
        "MTA",
        "route/stop/vehicle snapshot cache",
        ROOT / "results_freqduet/route_day_heldout_readiness/v1/route_family_coverage.csv",
        "partial",
        "MTA cache supports static/real-time route realism but is not matched APC/AFC/AVL/OD calibration.",
    )

    coverage = pd.DataFrame(rows)
    coverage.to_csv(out_dir / "same_day_field_calibration_source_coverage.csv", index=False)

    supported = set(coverage[coverage["status"].eq("supported")]["evidence_type"])
    exact_supported = all([
        "APC route/stop/trip/day-type aggregate" in supported,
        "GTFS/APC-derived line simulation environments" in supported,
        coverage[
            coverage["evidence_type"].eq("same-day historical AVL trajectory archive")
            & coverage["exists"]
        ].shape[0] > 0,
        coverage[
            coverage["evidence_type"].eq("same-day AFC tap OD matrix")
            & coverage["exists"]
        ].shape[0] > 0,
    ])
    verdict = {
        "exact_same_day_afc_apc_avl_od_calibration_supported": bool(exact_supported),
        "supported_claim": (
            "route-family/service-day policy matrix and APC/GTFS realism calibration"
            if not exact_supported
            else "same-day AFC/APC/AVL/OD field calibration"
        ),
        "blocked_claim": (
            None if exact_supported
            else "Do not claim exact same-day AFC/APC/AVL/OD field calibration until matched AFC OD and historical AVL are added."
        ),
    }
    with (out_dir / "same_day_field_calibration_verdict.json").open("w") as f:
        json.dump(verdict, f, indent=2)

    md = [
        "# Same-day AFC/APC/AVL/OD calibration audit",
        "",
        f"Exact same-day calibration supported: **{verdict['exact_same_day_afc_apc_avl_od_calibration_supported']}**.",
        "",
        f"Supported claim now: {verdict['supported_claim']}.",
        "",
        f"Blocked claim: {verdict['blocked_claim'] or 'none'}.",
        "",
        "## Evidence",
        "",
    ]
    for row in rows:
        md.append(
            f"- {row['source']} / {row['evidence_type']}: {row['status']} "
            f"({row['note']})"
        )
    (out_dir / "same_day_field_calibration_audit.md").write_text("\n".join(md) + "\n")
    print(coverage.to_string(index=False))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
