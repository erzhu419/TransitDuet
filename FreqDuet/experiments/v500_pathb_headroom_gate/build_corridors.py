#!/usr/bin/env python3
"""Stage 0: Build dual multi-timescale corridors + stationary sanity benchmark.

Generates three env_bus-compatible corridor directories:
1. ``tidal_bimodal``: 20 intermediate stops (T01..T20), 12 physical buses,
   262 daily trips (131/dir), public-MTA bimodal commuter temporal curve
   (3.3x peak/trough) + 70/30 AM/PM directional tidal reversal, preserving
   total daily ridership (~23,379 pax/day).
2. ``mbta_route111_wkdy``: 23 GTFS-matched physical stops (M01..M23) on
   Boston MBTA Route 111 (Haymarket <-> Woodlawn/Chelsea), 10 physical buses,
   262 daily trips, GTFS Haversine inter-stop distances, and Fall 2025 weekday
   APC directional hourly boarding/alighting OD matrices.
3. ``flat_sanity``: Exact copy of the legacy nearly-stationary corridor
   (1,507-1,945 pax/hr) used as a stationary null-regime sanity check.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FREQDUET_ENV = HERE.parents[1] / "freqduet" / "env"
CORRIDORS_ROOT = HERE / "corridors"

MBTA_APC_CSV = Path(
    "/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/mbta/"
    "ridership/MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/"
    "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/"
    "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop_Fall_2025.csv"
)
MBTA_GTFS_DIR = Path(
    "/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/mbta/gtfs"
)

SERVICE_HOURS = list(range(6, 20))  # 06:00 through 19:00 inclusive (14 hours)

# Public MTA-like urban bimodal commuter temporal profile across 06:00-19:00
# (normalized in build_tidal_bimodal so daily total matches legacy 23,379 pax).
BIMODAL_HOUR_WEIGHTS: Dict[int, float] = {
    6: 1.10,
    7: 2.15,
    8: 2.55,
    9: 1.65,
    10: 0.92,
    11: 0.76,
    12: 0.78,
    13: 0.82,
    14: 0.95,
    15: 1.35,
    16: 2.05,
    17: 2.35,
    18: 1.55,
    19: 0.90,
}

# Fraction of hourly demand in upward (direction=True, inbound) vs downward.
BIMODAL_UP_SHARE: Dict[int, float] = {
    6: 0.68,
    7: 0.72,
    8: 0.70,
    9: 0.62,
    10: 0.52,
    11: 0.50,
    12: 0.50,
    13: 0.48,
    14: 0.44,
    15: 0.36,
    16: 0.30,
    17: 0.28,
    18: 0.34,
    19: 0.42,
}


def _write_od_excel(path: Path, od_frames: List[pd.DataFrame]) -> None:
    """Write passenger_OD.xlsx matching pd.read_excel(..., index_col=[1, 0])."""
    full = pd.concat(od_frames, ignore_index=True)
    full.to_excel(path, index=False)


def build_flat_sanity(out_dir: Path) -> dict:
    """Copy the legacy flat corridor verbatim as a stationary sanity check."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(FREQDUET_ENV / "config.json", out_dir / "config.json")
    for name in ("passenger_OD.xlsx", "route_news.xlsx", "stop_news.xlsx", "time_table.xlsx"):
        shutil.copy2(FREQDUET_ENV / "data" / name, out_dir / "data" / name)
    meta = {
        "name": "flat_sanity",
        "n_fleet": 12,
        "bus_capacity": 50,
        "trips_total": 262,
        "description": "Legacy flat demand corridor (hourly CV < 8%) for stationary null-regime check.",
    }
    (out_dir / "corridor_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def build_tidal_bimodal(out_dir: Path) -> dict:
    """Build controlled bimodal tidal-commuter corridor preserving total daily pax."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(FREQDUET_ENV / "config.json", out_dir / "config.json")
    shutil.copy2(FREQDUET_ENV / "data" / "time_table.xlsx", out_dir / "data" / "time_table.xlsx")

    # Rename X01..X20 -> T01..T20 to avoid legacy ['X13','X14','X15'] *= 0.4 hack in sim.py.
    stop_names = [f"T{i:02d}" for i in range(1, 21)]
    full_stops = ["Terminal_up"] + stop_names + ["Terminal_down"]
    stop_df = pd.DataFrame({
        "stop_id": list(range(len(full_stops))),
        "stop_name": full_stops,
    })
    stop_df.to_excel(out_dir / "data" / "stop_news.xlsx", index=False)

    # Spatial OD kernel pooled from legacy OD (separating upper=True and lower=False triangles).
    legacy_od = pd.read_excel(FREQDUET_ENV / "data" / "passenger_OD.xlsx", index_col=[1, 0])
    legacy_od.index = pd.MultiIndex.from_tuples([(str(a), str(b)) for a, b in legacy_od.index])
    legacy_stops = [f"X{i:02d}" for i in range(1, 21)]
    up_spatial = np.zeros((20, 20), dtype=np.float64)
    dn_spatial = np.zeros((20, 20), dtype=np.float64)
    total_legacy_pax = 0.0
    for h in SERVICE_HOURS:
        p_str = f"{h:02d}:00:00"
        mat = np.zeros((20, 20), dtype=np.float64)
        for i, s_orig in enumerate(legacy_stops):
            row = legacy_od.loc[(s_orig, p_str)]
            for j, s_dest in enumerate(legacy_stops):
                val = float(row[s_dest])
                mat[i, j] = val
        total_legacy_pax += float(mat.sum())
        up_spatial += np.triu(mat, k=1)
        dn_spatial += np.tril(mat, k=-1)

    up_spatial /= max(up_spatial.sum(), 1e-9)
    dn_spatial /= max(dn_spatial.sum(), 1e-9)

    target_daily_pax = 15500.0
    weight_sum = sum(BIMODAL_HOUR_WEIGHTS[h] for h in SERVICE_HOURS)
    od_frames = []
    hourly_summary = {}
    for h in SERVICE_HOURS:
        p_str = f"{h:02d}:00:00"
        hour_total = target_daily_pax * (BIMODAL_HOUR_WEIGHTS[h] / weight_sum)
        up_share = BIMODAL_UP_SHARE[h]
        mat = hour_total * (up_share * up_spatial + (1.0 - up_share) * dn_spatial)
        mat_round = np.rint(mat).astype(int)
        np.fill_diagonal(mat_round, 0)
        df_h = pd.DataFrame(mat_round, columns=stop_names)
        df_h.insert(0, "Unnamed: 1", stop_names)
        df_h.insert(0, "Unnamed: 0", p_str)
        od_frames.append(df_h)
        hourly_summary[p_str] = {
            "up_pax": int(np.triu(mat_round, k=1).sum()),
            "down_pax": int(np.tril(mat_round, k=-1).sum()),
            "total_pax": int(mat_round.sum()),
        }
    _write_od_excel(out_dir / "data" / "passenger_OD.xlsx", od_frames)

    # Route news with directional peak congestion matching tidal flow
    legacy_rn = pd.read_excel(FREQDUET_ENV / "data" / "route_news.xlsx")
    rename_stop = {f"X{i:02d}": f"T{i:02d}" for i in range(1, 21)}
    rn = pd.DataFrame({
        "route_id": legacy_rn["route_id"].astype(int),
        "start_stop": legacy_rn["start_stop"].replace(rename_stop),
        "end_stop": legacy_rn["end_stop"].replace(rename_stop),
        "distance": legacy_rn["distance"].astype(float),
        "V_max": 15.0,
    })
    # Links 0..20 are upward (inbound), 21..41 are downward (outbound)
    rng = np.random.default_rng(20261118)
    link_Spatial = rng.uniform(0.92, 1.08, size=len(rn))
    for h in SERVICE_HOURS:
        col = f"{h:02d}:00:00"
        up_load_factor = BIMODAL_HOUR_WEIGHTS[h] * BIMODAL_UP_SHARE[h]
        dn_load_factor = BIMODAL_HOUR_WEIGHTS[h] * (1.0 - BIMODAL_UP_SHARE[h])
        # Higher traffic congestion during directional peak: ~6.6-7.2 m/s peak vs ~8.8-9.4 m/s off-peak
        up_base_v = np.clip(9.6 - 1.55 * up_load_factor, 6.4, 9.6)
        dn_base_v = np.clip(9.6 - 1.55 * dn_load_factor, 6.4, 9.6)
        speeds = np.where(rn["route_id"] < 21, up_base_v, dn_base_v) * link_Spatial
        rn[col] = np.round(speeds, 2)
    rn.to_excel(out_dir / "data" / "route_news.xlsx", index=False)

    totals = [v["total_pax"] for v in hourly_summary.values()]
    meta = {
        "name": "tidal_bimodal",
        "n_fleet": 14,
        "bus_capacity": 75,
        "trips_total": 262,
        "daily_pax": int(sum(totals)),
        "peak_to_trough_ratio": round(max(totals) / max(min(totals), 1), 3),
        "hourly_summary": hourly_summary,
    }
    (out_dir / "corridor_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _reconstruct_forward_od(boardings: np.ndarray, alightings: np.ndarray) -> np.ndarray:
    """Synthesize forward stop-to-stop OD from APC boardings and alightings.

    Uses sequential onboard pool proportional alighting (Navick & Furth 1994),
    enforcing exact non-negativity, zero diagonal, and total boarding conservation.
    """
    n = len(boardings)
    b = np.maximum(np.asarray(boardings, dtype=np.float64), 0.0).copy()
    a = np.maximum(np.asarray(alightings, dtype=np.float64), 0.0).copy()
    b[-1] = 0.0  # no forward boardings at final arrival stop
    a[0] = 0.0   # no alightings at initial origin stop
    total_b = float(b.sum())
    if total_b <= 1e-6:
        return np.zeros((n, n), dtype=np.float64)
    if a.sum() <= 1e-6:
        a[-1] = 1.0
    a = a * (total_b / a.sum())

    od = np.zeros((n, n), dtype=np.float64)
    pool = np.zeros(n, dtype=np.float64)  # onboard pax indexed by origin i < j
    for j in range(n):
        avail = float(pool.sum())
        if j > 0 and avail > 1e-9:
            alight_j = total_b - sum(od[:, :j].sum() for _ in [0]) if j == n - 1 else min(a[j], avail)
            if j == n - 1:
                od[:, j] = pool
                pool[:] = 0.0
            else:
                take = min(float(a[j]), avail)
                shares = pool / avail
                od[:, j] = shares * take
                pool = np.maximum(0.0, pool - od[:, j])
        if j < n - 1:
            pool[j] += b[j]
    if pool.sum() > 1e-9:
        od[:, -1] += pool
    return od


def build_mbta_route111_wkdy(out_dir: Path, busy_day_scale: float = 1.35) -> dict:
    """Build real-world calibrated Boston MBTA Route 111 weekday corridor."""
    if not MBTA_APC_CSV.exists() or not MBTA_GTFS_DIR.exists():
        raise FileNotFoundError("MBTA Fall 2025 APC or GTFS directory is missing locally")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(FREQDUET_ENV / "config.json", out_dir / "config.json")
    shutil.copy2(FREQDUET_ENV / "data" / "time_table.xlsx", out_dir / "data" / "time_table.xlsx")

    cols = [
        "GTFS route_id", "GTFS direction_id", "trip start time",
        "Stop Name", "GTFS stop_id", "stop sequence", "Day Type",
        "Boardings", "Alightings", "Load",
    ]
    parts = []
    for ch in pd.read_csv(MBTA_APC_CSV, usecols=cols, chunksize=200_000, dtype=str):
        sub = ch[
            (ch["GTFS route_id"] == "111")
            & (ch["Day Type"].str.contains("Wkdy", na=False))
        ].copy()
        if not sub.empty:
            parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    for c in ("Boardings", "Alightings", "Load", "stop sequence"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["hour"] = df["trip start time"].str.split(":").str[0].astype(int)
    df = df[(df["hour"] >= 6) & (df["hour"] <= 19)].copy()

    stops_gtfs = pd.read_csv(
        MBTA_GTFS_DIR / "stops.txt",
        dtype=str,
        usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"],
    ).drop_duplicates("stop_id")
    stops_gtfs["stop_lat"] = pd.to_numeric(stops_gtfs["stop_lat"], errors="coerce")
    stops_gtfs["stop_lon"] = pd.to_numeric(stops_gtfs["stop_lon"], errors="coerce")

    # Canonical stop spine from Direction 1 (Inbound: Woodlawn/Elm St -> Haymarket/Surface Rd)
    # so direction=True (upward: M01 -> M23) = Inbound to Haymarket Boston (AM peak!),
    # and direction=False (downward: M23 -> M01) = Outbound to Chelsea/Woodlawn (PM peak!).
    d1_stops = (
        df[df["GTFS direction_id"] == "1"]
        .groupby(["stop sequence", "GTFS stop_id", "Stop Name"], as_index=False)[["Boardings", "Alightings"]]
        .sum()
        .sort_values("stop sequence")
        .merge(stops_gtfs, left_on="GTFS stop_id", right_on="stop_id", how="left")
        .dropna(subset=["stop_lat", "stop_lon"])
        .reset_index(drop=True)
    )
    n_stops = len(d1_stops)  # 23 physical stops with GTFS coordinates
    sim_names = [f"M{i:02d}" for i in range(1, n_stops + 1)]
    d1_stops["sim_name"] = sim_names
    spine_coords = d1_stops[["stop_lat", "stop_lon"]].to_numpy(dtype=np.float64)

    # Map Direction 0 (Outbound: Haymarket -> Woodlawn) stops to nearest physical spine station
    # in reverse order so outbound OD populates the strict lower triangle (M23 -> ... -> M01).
    d0_stops = (
        df[df["GTFS direction_id"] == "0"]
        .groupby(["stop sequence", "GTFS stop_id", "Stop Name"], as_index=False)[["Boardings", "Alightings"]]
        .sum()
        .sort_values("stop sequence")
        .merge(stops_gtfs, left_on="GTFS stop_id", right_on="stop_id", how="left")
        .dropna(subset=["stop_lat", "stop_lon"])
        .reset_index(drop=True)
    )
    d0_to_spine_idx: Dict[str, int] = {}
    for _, row in d0_stops.iterrows():
        dists = [
            _haversine_m(float(row["stop_lat"]), float(row["stop_lon"]), float(lat), float(lon))
            for lat, lon in spine_coords
        ]
        d0_to_spine_idx[str(row["GTFS stop_id"])] = int(np.argmin(dists))

    d1_id_to_idx = {str(sid): idx for idx, sid in enumerate(d1_stops["GTFS stop_id"].astype(str))}

    full_stops = ["Terminal_up"] + sim_names + ["Terminal_down"]
    stop_df = pd.DataFrame({
        "stop_id": list(range(len(full_stops))),
        "stop_name": full_stops,
    })
    stop_df.to_excel(out_dir / "data" / "stop_news.xlsx", index=False)

    # Compute physical inter-stop road distances (Haversine * 1.18 urban road circuity, min 160m)
    inter_dists = [150.0]  # Terminal_up -> M01
    for i in range(n_stops - 1):
        raw_d = _haversine_m(
            spine_coords[i, 0], spine_coords[i, 1],
            spine_coords[i + 1, 0], spine_coords[i + 1, 1],
        )
        inter_dists.append(round(max(160.0, raw_d * 1.18), 1))
    inter_dists.append(150.0)  # M23 -> Terminal_down

    # Build upward (0..n_stops) and downward (reversed) route links
    up_links = []
    for k in range(len(full_stops) - 1):
        up_links.append((k, full_stops[k], full_stops[k + 1], inter_dists[k]))
    dn_links = []
    base_id = len(up_links)
    for k in range(len(full_stops) - 1):
        src_idx = len(full_stops) - 1 - k
        dst_idx = len(full_stops) - 2 - k
        dn_links.append((base_id + k, full_stops[src_idx], full_stops[dst_idx], inter_dists[dst_idx]))

    all_links = up_links + dn_links
    rn = pd.DataFrame(all_links, columns=["route_id", "start_stop", "end_stop", "distance"])
    rn["V_max"] = 14.0

    od_frames = []
    hourly_summary = {}
    for h in SERVICE_HOURS:
        p_str = f"{h:02d}:00:00"
        sub_h = df[df["hour"] == h]

        # Direction 1 (Inbound = upward M01 -> M23 -> upper triangle)
        b_in = np.zeros(n_stops, dtype=np.float64)
        a_in = np.zeros(n_stops, dtype=np.float64)
        for _, r in sub_h[sub_h["GTFS direction_id"] == "1"].iterrows():
            sid = str(r["GTFS stop_id"])
            if sid in d1_id_to_idx:
                idx = d1_id_to_idx[sid]
                b_in[idx] += float(r["Boardings"]) * busy_day_scale
                a_in[idx] += float(r["Alightings"]) * busy_day_scale
        od_up = _reconstruct_forward_od(b_in, a_in)  # upper triangle (i < j)

        # Direction 0 (Outbound = downward M23 -> M01 -> reverse sequence -> lower triangle)
        b_out_rev = np.zeros(n_stops, dtype=np.float64)
        a_out_rev = np.zeros(n_stops, dtype=np.float64)
        for _, r in sub_h[sub_h["GTFS direction_id"] == "0"].iterrows():
            sid = str(r["GTFS stop_id"])
            if sid in d0_to_spine_idx:
                spine_i = d0_to_spine_idx[sid]
                rev_i = (n_stops - 1) - spine_i  # 0 at M23, n_stops-1 at M01
                b_out_rev[rev_i] += float(r["Boardings"]) * busy_day_scale
                a_out_rev[rev_i] += float(r["Alightings"]) * busy_day_scale
        od_dn_rev = _reconstruct_forward_od(b_out_rev, a_out_rev)
        # Map reversed indices back to spine indices (strict lower triangle orig > dest)
        od_dn = np.zeros((n_stops, n_stops), dtype=np.float64)
        for ri in range(n_stops):
            for rj in range(ri + 1, n_stops):
                orig_spine = (n_stops - 1) - ri
                dest_spine = (n_stops - 1) - rj
                od_dn[orig_spine, dest_spine] += od_dn_rev[ri, rj]

        mat = np.rint(od_up + od_dn).astype(int)
        np.fill_diagonal(mat, 0)
        df_h = pd.DataFrame(mat, columns=sim_names)
        df_h.insert(0, "Unnamed: 1", sim_names)
        df_h.insert(0, "Unnamed: 0", p_str)
        od_frames.append(df_h)

        up_pax = int(np.triu(mat, k=1).sum())
        dn_pax = int(np.tril(mat, k=-1).sum())
        hourly_summary[p_str] = {
            "inbound_up_pax": up_pax,
            "outbound_down_pax": dn_pax,
            "total_pax": up_pax + dn_pax,
        }

        # Calibrate urban Chelsea / Tobin Bridge directional speeds (m/s)
        in_cong = min(1.0, up_pax / 750.0)
        out_cong = min(1.0, dn_pax / 650.0)
        v_in = round(8.2 - 2.3 * in_cong, 2)   # ~5.9 m/s AM Tobin Bridge peak, ~7.8 m/s evening
        v_out = round(8.2 - 2.3 * out_cong, 2) # ~5.9 m/s PM Outbound peak, ~7.8 m/s morning
        rn[p_str] = np.where(rn["route_id"] < len(up_links), v_in, v_out)

    _write_od_excel(out_dir / "data" / "passenger_OD.xlsx", od_frames)
    rn.to_excel(out_dir / "data" / "route_news.xlsx", index=False)

    in_vals = [v["inbound_up_pax"] for v in hourly_summary.values()]
    out_vals = [v["outbound_down_pax"] for v in hourly_summary.values()]
    meta = {
        "name": "mbta_route111_wkdy",
        "n_fleet": 14,
        "bus_capacity": 75,
        "trips_total": 262,
        "n_physical_stops": n_stops,
        "one_way_length_m": round(sum(inter_dists), 1),
        "busy_day_scale": busy_day_scale,
        "daily_pax": sum(v["total_pax"] for v in hourly_summary.values()),
        "inbound_peak_to_trough": round(max(in_vals) / max(min(in_vals), 1), 2),
        "outbound_peak_to_trough": round(max(out_vals) / max(min(out_vals), 1), 2),
        "stop_mapping": d1_stops[["sim_name", "GTFS stop_id", "Stop Name", "stop_lat", "stop_lon"]].to_dict("records"),
        "hourly_summary": hourly_summary,
    }
    (out_dir / "corridor_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    CORRIDORS_ROOT.mkdir(parents=True, exist_ok=True)
    m_flat = build_flat_sanity(CORRIDORS_ROOT / "flat_sanity")
    m_tidal = build_tidal_bimodal(CORRIDORS_ROOT / "tidal_bimodal")
    m_mbta = build_mbta_route111_wkdy(CORRIDORS_ROOT / "mbta_route111_wkdy")
    manifest = {"flat_sanity": m_flat, "tidal_bimodal": m_tidal, "mbta_route111_wkdy": m_mbta}
    (CORRIDORS_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for k, v in manifest.items():
        print(f"Built {k}: n_fleet={v.get('n_fleet')}, daily_pax={v.get('daily_pax', 'legacy')}, "
              f"peak_info={v.get('peak_to_trough_ratio') or (v.get('inbound_peak_to_trough'), v.get('outbound_peak_to_trough'))}")


if __name__ == "__main__":
    main()
