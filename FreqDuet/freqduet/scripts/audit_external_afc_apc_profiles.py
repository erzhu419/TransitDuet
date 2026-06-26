#!/usr/bin/env python3
"""Audit public AFC/APC demand profiles for the FreqDuet paper package.

This script intentionally treats AFC/APC inputs as external demand-profile
evidence only. It does not import or reuse the separate ``transit_hrl``
algorithm code path. The outputs document coverage, profile shape, and the
boundary between public count profiles and full OD/onboard-load calibration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AFC = ROOT / "data" / "external_afc_apc" / "public_afc_mta" / "hourly_ridership.csv"
DEFAULT_APC = ROOT / "data" / "external_afc_apc" / "public_apc_halifax" / "route_boardings.csv"
DEFAULT_OD = ROOT / "env" / "data" / "passenger_OD.xlsx"
DEFAULT_OUT = ROOT / "results_freqduet" / "real_afc_apc_profile_audit" / "v1"

SOURCE_METADATA = {
    "public_afc_mta": {
        "agency": "MTA / New York State Open Data",
        "source_url": "https://data.ny.gov/resource/wujg-7c2s.json",
        "observation": "station-complex hourly entries",
        "source_kind": "public AFC station-entry count profile",
        "boundary": "AFC station entries only; not OD geometry, onboard load, alighting, or field deployment outcomes.",
    },
    "public_apc_halifax": {
        "agency": "Halifax Transit Open Data",
        "source_url": (
            "https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/ArcGIS/rest/services/"
            "Transit_Automated_Passenger_Counts/FeatureServer/0/query"
        ),
        "observation": "route half-hour boardings",
        "source_kind": "public APC route-boarding count profile",
        "boundary": "APC route boardings only; not full OD geometry, onboard occupancy, alighting, or field deployment outcomes.",
    },
    "freqduet_od": {
        "agency": "FreqDuet local corridor input table",
        "source_url": "FreqDuet/freqduet/env/data/passenger_OD.xlsx",
        "observation": "hourly OD demand intensities",
        "source_kind": "local historical OD-intensity simulator input",
        "boundary": "Simulator OD intensity table used for FreqDuet experiments; provenance and agency calibration remain separate manuscript boundaries.",
    },
}


def _hour_float(value: object) -> float:
    if pd.isna(value):
        return math.nan
    if hasattr(value, "hour"):
        return float(value.hour) + float(getattr(value, "minute", 0)) / 60.0
    text = str(value).strip()
    if not text:
        return math.nan
    if " " in text:
        text = text.split(" ")[-1]
    parts = text.split(":")
    try:
        if len(parts) >= 2:
            return float(int(parts[0])) + float(int(parts[1])) / 60.0
        return float(text)
    except ValueError:
        return math.nan


def _normalise_profile(profile: pd.DataFrame) -> pd.DataFrame:
    profile = profile.copy()
    total = float(profile["demand"].sum())
    profile["share"] = profile["demand"] / total if total > 0 else np.nan
    profile["hour_floor"] = np.floor(profile["hour_bin"]).astype(int)
    return profile


def _coverage_row(
    source: str,
    path: Path,
    df: pd.DataFrame,
    profile: pd.DataFrame,
    series_count: int,
    first_time: str,
    last_time: str,
) -> dict:
    meta = SOURCE_METADATA[source]
    peak = profile.sort_values("demand", ascending=False).head(1)
    peak_hour = float(peak["hour_bin"].iloc[0]) if not peak.empty else math.nan
    peak_value = float(peak["demand"].iloc[0]) if not peak.empty else math.nan
    return {
        "source": source,
        "agency": meta["agency"],
        "source_kind": meta["source_kind"],
        "source_url": meta["source_url"],
        "local_file": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "series_count": int(series_count),
        "time_bins": int(profile["hour_bin"].nunique()),
        "first_time": first_time,
        "last_time": last_time,
        "total_count": float(profile["demand"].sum()),
        "peak_hour_bin": peak_hour,
        "peak_count": peak_value,
        "observation": meta["observation"],
        "boundary": meta["boundary"],
    }


def load_afc(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    required = {"transit_timestamp", "station_complex_id", "station_complex", "ridership"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing AFC columns: {sorted(missing)}")
    df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"], errors="coerce")
    df["ridership"] = pd.to_numeric(df["ridership"], errors="coerce").fillna(0.0)
    df["hour_bin"] = df["transit_timestamp"].dt.hour + df["transit_timestamp"].dt.minute / 60.0
    profile = (
        df.groupby("hour_bin", as_index=False)["ridership"]
        .sum()
        .rename(columns={"ridership": "demand"})
    )
    profile = _normalise_profile(profile)
    profile["source"] = "public_afc_mta"
    coverage = _coverage_row(
        "public_afc_mta",
        path,
        df,
        profile,
        int(df["station_complex_id"].nunique()),
        str(df["transit_timestamp"].min()),
        str(df["transit_timestamp"].max()),
    )
    return profile, coverage


def load_apc(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    required = {
        "Route_Number",
        "Route_Name",
        "Ridership_Total",
        "Route_Hour",
        "Route_Date",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing APC columns: {sorted(missing)}")
    df["Ridership_Total"] = pd.to_numeric(df["Ridership_Total"], errors="coerce").fillna(0.0)
    df["Route_Hour"] = pd.to_numeric(df["Route_Hour"], errors="coerce")
    df["Route_Date_dt"] = pd.to_datetime(df["Route_Date"], unit="ms", errors="coerce")
    profile = (
        df.groupby("Route_Hour", as_index=False)["Ridership_Total"]
        .sum()
        .rename(columns={"Route_Hour": "hour_bin", "Ridership_Total": "demand"})
    )
    profile = _normalise_profile(profile)
    profile["source"] = "public_apc_halifax"
    coverage = _coverage_row(
        "public_apc_halifax",
        path,
        df,
        profile,
        int(df["Route_Number"].astype(str).nunique()),
        str(df["Route_Date_dt"].min()),
        str(df["Route_Date_dt"].max()),
    )
    return profile, coverage


def load_freqduet_od(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_excel(path)
    if len(df.columns) < 3:
        raise ValueError(f"{path} does not look like the FreqDuet OD table")
    time_col = df.columns[0]
    origin_col = df.columns[1]
    dest_cols = [c for c in df.columns if str(c).startswith("X")]
    if not dest_cols:
        raise ValueError(f"{path} has no destination columns named X*")
    work = df[[time_col, origin_col, *dest_cols]].copy()
    work["hour_bin"] = work[time_col].map(_hour_float)
    for col in dest_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    work["row_demand"] = work[dest_cols].sum(axis=1)
    profile = (
        work.groupby("hour_bin", as_index=False)["row_demand"]
        .sum()
        .rename(columns={"row_demand": "demand"})
    )
    profile = _normalise_profile(profile)
    profile["source"] = "freqduet_od"
    coverage = _coverage_row(
        "freqduet_od",
        path,
        work,
        profile,
        int(work[origin_col].astype(str).nunique()),
        str(work[time_col].min()),
        str(work[time_col].max()),
    )
    return profile, coverage


def _window_share(hourly: pd.DataFrame, lo: int, hi: int) -> float:
    window = hourly[(hourly["hour"] >= lo) & (hourly["hour"] < hi)]
    total = float(hourly["demand"].sum())
    return float(window["demand"].sum() / total) if total > 0 else math.nan


def build_alignment(profile: pd.DataFrame, reference_source: str = "freqduet_od") -> pd.DataFrame:
    hourly = (
        profile.groupby(["source", "hour_floor"], as_index=False)["demand"]
        .sum()
        .rename(columns={"hour_floor": "hour"})
    )
    rows: list[dict] = []
    ref = hourly[hourly["source"].eq(reference_source)][["hour", "demand"]]
    for source in sorted(profile["source"].unique()):
        src = hourly[hourly["source"].eq(source)][["hour", "demand"]].copy()
        total = float(src["demand"].sum())
        peak_hour = int(src.sort_values("demand", ascending=False)["hour"].iloc[0])
        row = {
            "source": source,
            "total_count": total,
            "peak_hour": peak_hour,
            "morning_share_06_10": _window_share(src, 6, 10),
            "midday_share_10_15": _window_share(src, 10, 15),
            "evening_share_15_20": _window_share(src, 15, 20),
        }
        if source == reference_source:
            row.update({
                "reference_source": reference_source,
                "common_hour_count": int(len(src)),
                "pearson_r_vs_freqduet_od": 1.0,
                "abs_peak_hour_gap_vs_freqduet_od": 0,
            })
        else:
            merged = src.merge(ref, on="hour", suffixes=("_source", "_reference"))
            if len(merged) >= 2 and merged["demand_source"].std() > 0 and merged["demand_reference"].std() > 0:
                corr = float(np.corrcoef(merged["demand_source"], merged["demand_reference"])[0, 1])
            else:
                corr = math.nan
            ref_peak = int(ref.sort_values("demand", ascending=False)["hour"].iloc[0])
            row.update({
                "reference_source": reference_source,
                "common_hour_count": int(len(merged)),
                "pearson_r_vs_freqduet_od": corr,
                "abs_peak_hour_gap_vs_freqduet_od": abs(peak_hour - ref_peak),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def plot_profiles(profile: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    colors = {
        "freqduet_od": "#2b6cb0",
        "public_afc_mta": "#b83280",
        "public_apc_halifax": "#2f855a",
    }
    labels = {
        "freqduet_od": "FreqDuet OD input",
        "public_afc_mta": "MTA AFC entries",
        "public_apc_halifax": "Halifax APC boardings",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for source, group in profile.sort_values("hour_bin").groupby("source"):
        ax.plot(
            group["hour_bin"],
            group["share"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            color=colors.get(source),
            label=labels.get(source, source),
        )
    ax.axvspan(6, 19, color="#edf2f7", alpha=0.75, zorder=0)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of observed daily profile")
    ax.set_xlim(0, 25)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", color="#cbd5e0", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"external_afc_apc_profile_overlay.{fmt}", dpi=220)
    plt.close(fig)


def write_note(out_dir: Path, coverage: pd.DataFrame, alignment: pd.DataFrame) -> None:
    afc = coverage[coverage["source"].eq("public_afc_mta")].iloc[0]
    apc = coverage[coverage["source"].eq("public_apc_halifax")].iloc[0]
    lines = [
        "# External AFC/APC Profile Audit",
        "",
        "This audit uses public AFC/APC count profiles as external demand-shape evidence for FreqDuet.",
        "It does not reuse the separate transit_hrl algorithm implementation, checkpoints, or result claims.",
        "",
        "## Coverage",
        "",
        f"- MTA AFC cache: {int(afc['rows'])} station-hour rows, {int(afc['series_count'])} station complexes.",
        f"- Halifax APC cache: {int(apc['rows'])} route half-hour rows, {int(apc['series_count'])} routes.",
        "- FreqDuet OD input is included only as the local simulator demand-shape reference.",
        "",
        "## Boundary",
        "",
        "These files support a claim that the paper package contains public AFC/APC demand-profile evidence.",
        "They do not support claims of exact AFC/APC OD geometry, onboard-load calibration, field deployment,",
        "or observed wait-time improvement on an agency system.",
        "",
        "## Outputs",
        "",
        "- `external_afc_apc_source_coverage.csv`",
        "- `external_afc_apc_aggregate_profile.csv`",
        "- `external_afc_apc_profile_alignment.csv`",
        "- `external_afc_apc_profile_overlay.png/.pdf`",
        "- `summary.json`",
        "",
        "## Alignment Summary",
        "",
    ]
    for _, row in alignment.sort_values("source").iterrows():
        corr = row["pearson_r_vs_freqduet_od"]
        corr_text = "NA" if pd.isna(corr) else f"{float(corr):.3f}"
        lines.append(
            f"- `{row['source']}`: peak hour {int(row['peak_hour'])}, "
            f"common hours {int(row['common_hour_count'])}, "
            f"corr vs FreqDuet OD {corr_text}."
        )
    (out_dir / "external_afc_apc_profile_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--afc-csv", default=str(DEFAULT_AFC))
    parser.add_argument("--apc-csv", default=str(DEFAULT_APC))
    parser.add_argument("--freqduet-od", default=str(DEFAULT_OD))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]

    afc_profile, afc_coverage = load_afc(Path(args.afc_csv))
    apc_profile, apc_coverage = load_apc(Path(args.apc_csv))
    od_profile, od_coverage = load_freqduet_od(Path(args.freqduet_od))

    profile = pd.concat([afc_profile, apc_profile, od_profile], ignore_index=True)
    coverage = pd.DataFrame([afc_coverage, apc_coverage, od_coverage])
    alignment = build_alignment(profile)

    coverage.to_csv(out_dir / "external_afc_apc_source_coverage.csv", index=False)
    profile[["source", "hour_bin", "hour_floor", "demand", "share"]].to_csv(
        out_dir / "external_afc_apc_aggregate_profile.csv", index=False
    )
    alignment.to_csv(out_dir / "external_afc_apc_profile_alignment.csv", index=False)
    plot_profiles(profile, out_dir, formats)
    write_note(out_dir, coverage, alignment)

    payload = {
        "status": "generated",
        "claim_boundary": (
            "Public AFC/APC demand-profile evidence only; not exact AFC/APC OD geometry, "
            "onboard-load calibration, agency deployment, or observed field wait-time improvement."
        ),
        "does_not_import_transit_hrl": True,
        "inputs": {
            "afc_csv": str(Path(args.afc_csv)),
            "apc_csv": str(Path(args.apc_csv)),
            "freqduet_od": str(Path(args.freqduet_od)),
        },
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "coverage": coverage.to_dict(orient="records"),
        "alignment": alignment.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_dir}")
    print(f"sources={len(coverage)} profile_rows={len(profile)}")


if __name__ == "__main__":
    main()
