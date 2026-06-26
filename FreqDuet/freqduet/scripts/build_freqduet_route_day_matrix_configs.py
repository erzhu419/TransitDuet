#!/usr/bin/env python3
"""Build MBTA route-family/service-day FreqDuet config matrix.

The generated configs point at copied H2O line environments under
``data/external_route_envs`` so the same relative path works locally and on
scheduleurm CPU nodes after rsync. Service-day variation is encoded from MBTA
APC hourly profiles as deterministic demand multipliers plus a route/day total
scale, rather than as a renamed synthetic scenario.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MBTA_ENV_ROOT = Path(
    "/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/"
    "mbta/h2o_city_envs/MBTA_weekday_all_routes/_line_envs"
)
DEFAULT_AUDIT_DIR = ROOT / "results_freqduet/external_od_onboard_truth_audit/v1"
DEFAULT_ENV_CACHE = ROOT / "data/external_route_envs/mbta_route_day_v1/_line_envs"
DEFAULT_CONFIG_DIR = ROOT / "configs_freqduet"
DEFAULT_SETUP_DIR = ROOT / "results_freqduet/route_day_policy_matrix_v1/config_setup"

PARENT_CONFIGS = {
    "main": "F_freqduet_terminal_paper_main_hiro.yaml",
    "nofreq": "F_freqduet_terminal_final_nofreq_hiro.yaml",
    "noleakage": "F_freqduet_terminal_final_noleakage_hiro.yaml",
}
DAY_TYPES = ("Wkdy", "Sat", "Sun")


def route_id_to_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return value or "na"


def route_family(route_id: str, rank: int) -> str:
    if rank <= 10:
        return "key_high_ridership"
    if route_id.isdigit():
        number = int(route_id)
        if number < 100:
            return "numbered_000_099"
        if number < 200:
            return "numbered_100_199"
        return "numbered_200_plus"
    return "non_numeric"


def parse_env_name(name: str) -> tuple[str, int | None]:
    match = re.match(r"(.+)_D([01])(?:_|$)", name)
    if not match:
        return name, None
    return match.group(1), int(match.group(2))


def env_sort_key(path: Path) -> tuple[int, int, str]:
    has_pattern = 1 if "_P" in path.name else 0
    return has_pattern, len(path.name), path.name


def available_envs(env_root: Path) -> dict[tuple[str, int], list[Path]]:
    envs: dict[tuple[str, int], list[Path]] = {}
    for path in sorted(env_root.iterdir()):
        if not path.is_dir():
            continue
        route_id, direction = parse_env_name(path.name)
        if direction is None:
            continue
        envs.setdefault((route_id, direction), []).append(path)
    for key, values in envs.items():
        envs[key] = sorted(values, key=env_sort_key)
    return envs


def select_routes(
    targets: pd.DataFrame,
    envs: dict[tuple[str, int], list[Path]],
    route_count: int,
) -> list[dict[str, Any]]:
    wkdy = targets[targets["Day Type"].eq("Wkdy")].copy()
    wkdy["route_id"] = wkdy["GTFS route_id"].map(route_id_to_str)
    wkdy["direction"] = wkdy["GTFS direction_id"].astype(int)
    wkdy = wkdy.sort_values("boardings", ascending=False)

    selected: list[dict[str, Any]] = []
    seen_routes: set[str] = set()
    for _, row in wkdy.iterrows():
        route_id = row["route_id"]
        direction = int(row["direction"])
        if not route_id or route_id in seen_routes:
            continue
        candidates = envs.get((route_id, direction), [])
        chosen_direction = direction
        if not candidates:
            alternate = 1 - direction
            candidates = envs.get((route_id, alternate), [])
            chosen_direction = alternate
        if not candidates:
            continue
        rank = len(selected) + 1
        env_path = candidates[0]
        selected.append({
            "scenario_rank": rank,
            "route_id": route_id,
            "target_direction": direction,
            "env_direction": chosen_direction,
            "env_name": env_path.name,
            "source_env_path": str(env_path),
            "route_family": route_family(route_id, rank),
            "wkdy_boardings": float(row["boardings"]),
            "wkdy_mean_load": float(row["mean_load"]),
            "wkdy_max_load": float(row["max_load"]),
            "wkdy_trip_samples": int(row["trip_samples"]),
        })
        seen_routes.add(route_id)
        if len(selected) >= route_count:
            break
    return selected


def build_hourly_shape_profiles(hourly: pd.DataFrame) -> dict[str, dict[int, float]]:
    subset = hourly[hourly["hour"].between(6, 19)].copy()
    profiles: dict[str, dict[int, float]] = {}
    wkdy = subset[subset["Day Type"].eq("Wkdy")].set_index("hour")["boardings"]
    wkdy_total = float(wkdy.sum())
    wkdy_share = wkdy / wkdy_total if wkdy_total > 0 else wkdy * 0 + 1.0 / len(wkdy)
    for day_type in DAY_TYPES:
        day = subset[subset["Day Type"].eq(day_type)].set_index("hour")["boardings"]
        if day.empty or float(day.sum()) <= 0 or day_type == "Wkdy":
            profiles[day_type] = {h: 1.0 for h in range(6, 20)}
            continue
        day_share = day / float(day.sum())
        ratios = (day_share / wkdy_share).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        profiles[day_type] = {
            int(h): round(float(np.clip(ratios.get(h, 1.0), 0.45, 1.85)), 4)
            for h in range(6, 20)
        }
    return profiles


def route_day_scale(targets: pd.DataFrame, route_id: str, direction: int, day_type: str) -> float:
    if day_type == "Wkdy":
        return 1.0
    df = targets.copy()
    df["route_id"] = df["GTFS route_id"].map(route_id_to_str)
    df["direction"] = df["GTFS direction_id"].astype(int)
    base = df[
        df["route_id"].eq(route_id)
        & df["direction"].eq(direction)
        & df["Day Type"].eq("Wkdy")
    ]
    day = df[
        df["route_id"].eq(route_id)
        & df["direction"].eq(direction)
        & df["Day Type"].eq(day_type)
    ]
    if base.empty or day.empty or float(base["boardings"].iloc[0]) <= 0:
        total_base = float(df[df["Day Type"].eq("Wkdy")]["boardings"].sum())
        total_day = float(df[df["Day Type"].eq(day_type)]["boardings"].sum())
        ratio = total_day / total_base if total_base > 0 else 1.0
    else:
        ratio = float(day["boardings"].iloc[0]) / float(base["boardings"].iloc[0])
    return round(float(np.clip(ratio, 0.35, 1.25)), 4)


def normalize_cached_timetable(dest: Path) -> tuple[int, int, int]:
    data_dir = dest / "data"
    timetable_path = data_dir / "time_table.xlsx"
    config_path = dest / "config.json"
    marker_path = dest / "freqduet_route_day_normalization.json"
    if marker_path.exists():
        meta = json.loads(marker_path.read_text())
        return (
            int(meta.get("service_start_hour", 6)),
            int(meta.get("service_end_hour", 19)),
            int(meta.get("launch_offset_s", 0)),
        )

    timetable = pd.read_excel(timetable_path)
    launch_offset = int(timetable["launch_time"].min()) if not timetable.empty else 0
    if launch_offset > 0:
        timetable["launch_time"] = (timetable["launch_time"] - launch_offset).clip(lower=0)
        timetable.to_excel(timetable_path, index=False)

    service_start_hour = int(launch_offset // 3600)
    service_end_hour = 23
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        service_end_hour = int(cfg.get("sim_end_hour", service_end_hour))
    marker = {
        "normalized": True,
        "launch_offset_s": launch_offset,
        "service_start_hour": service_start_hour,
        "service_end_hour": service_end_hour,
        "note": "time_table.launch_time was shifted so first copied dispatch starts at t=0 for FreqDuet.",
    }
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return service_start_hour, service_end_hour, launch_offset


def hour_columns(frame: pd.DataFrame) -> list[Any]:
    cols = []
    for col in frame.columns[5:]:
        text = str(col)
        if re.match(r"^\d{2}:\d{2}:\d{2}$", text):
            cols.append(col)
        elif hasattr(col, "hour"):
            cols.append(col)
    return cols


def make_bidirectional_terminal_env(dest: Path) -> None:
    marker_path = dest / "freqduet_route_day_topology.json"
    if marker_path.exists():
        return
    data_dir = dest / "data"
    stop_path = data_dir / "stop_news.xlsx"
    route_path = data_dir / "route_news.xlsx"
    timetable_path = data_dir / "time_table.xlsx"

    stops = pd.read_excel(stop_path)
    if stops["stop_name"].astype(str).isin(["Terminal_up", "Terminal_down"]).any():
        marker_path.write_text(json.dumps({
            "converted": False,
            "reason": "terminal stops already present",
        }, indent=2) + "\n")
        return

    stop_names = stops["stop_name"].astype(str).tolist()
    terminalized_stops = pd.DataFrame({
        "stop_id": list(range(len(stop_names) + 2)),
        "stop_name": ["Terminal_up"] + stop_names + ["Terminal_down"],
    })
    terminalized_stops.to_excel(stop_path, index=False)

    routes = pd.read_excel(route_path)
    hcols = hour_columns(routes)
    if not hcols:
        raise RuntimeError(f"No hourly speed columns in {route_path}")
    route_rows: list[dict[str, Any]] = []

    def from_template(start: str, end: str, template: pd.Series, route_id: int) -> dict[str, Any]:
        row = {
            "route_id": route_id,
            "start_stop": start,
            "end_stop": end,
            "distance": float(template.get("distance", 500.0)),
            "V_max": float(template.get("V_max", 15.0)),
        }
        for col in hcols:
            row[col] = float(template[col])
        return row

    first = routes.iloc[0]
    last = routes.iloc[-1]
    route_rows.append(from_template("Terminal_up", stop_names[0], first, len(route_rows)))
    for _, row in routes.iterrows():
        route_rows.append(from_template(
            str(row["start_stop"]), str(row["end_stop"]), row, len(route_rows)))
    route_rows.append(from_template(stop_names[-1], "Terminal_down", last, len(route_rows)))

    forward_rows = list(route_rows)
    for row in reversed(forward_rows):
        rev = dict(row)
        rev["route_id"] = len(route_rows)
        rev["start_stop"], rev["end_stop"] = row["end_stop"], row["start_stop"]
        route_rows.append(rev)
    pd.DataFrame(route_rows).to_excel(route_path, index=False)

    timetable = pd.read_excel(timetable_path)
    timetable = timetable[["launch_time", "direction"]].copy()
    forward = timetable.copy()
    forward["direction"] = 1
    reverse = timetable.copy()
    reverse["direction"] = 0
    timetable2 = pd.concat([forward, reverse], ignore_index=True)
    timetable2 = timetable2.sort_values(["launch_time", "direction"]).reset_index(drop=True)
    timetable2.to_excel(timetable_path, index=False)

    marker_path.write_text(json.dumps({
        "converted": True,
        "stop_count_original": int(len(stop_names)),
        "stop_count_terminalized": int(len(terminalized_stops)),
        "route_count_original": int(len(routes)),
        "route_count_terminalized": int(len(route_rows)),
        "timetable_rows_original": int(len(timetable)),
        "timetable_rows_terminalized": int(len(timetable2)),
        "note": "single-direction H2O line env converted into FreqDuet-compatible bidirectional terminal corridor",
    }, indent=2) + "\n")


def copy_env(source: Path, env_cache: Path) -> tuple[Path, int, int, int]:
    dest = env_cache / source.name
    if not dest.exists():
        shutil.copytree(source, dest)
    make_bidirectional_terminal_env(dest)
    service_start_hour, service_end_hour, launch_offset = normalize_cached_timetable(dest)
    return dest, service_start_hour, service_end_hour, launch_offset


def build_config(
    method: str,
    parent: str,
    scenario: dict[str, Any],
    day_type: str,
    env_rel_path: str,
    day_scale: float,
    hourly_profile: dict[int, float],
    service_start_hour: int,
    service_end_hour: int,
) -> dict[str, Any]:
    route_id = scenario["route_id"]
    family = scenario["route_family"]
    scenario_id = (
        f"mbta_r{slugify(route_id)}_d{scenario['env_direction']}_"
        f"{slugify(family)}"
    )
    cfg_name = (
        f"F_freqduet_routeday_{scenario_id}_{slugify(day_type)}_{method}_hiro"
    )
    return {
        "_extends": parent,
        "_name": cfg_name,
        "_paper_route_day": {
            "scenario_id": scenario_id,
            "agency": "MBTA",
            "route_id": route_id,
            "target_direction": int(scenario["target_direction"]),
            "env_direction": int(scenario["env_direction"]),
            "env_name": scenario["env_name"],
            "route_family": family,
            "day_type": day_type,
            "method": method,
            "source": "H2Oplus GTFS/APC line env + MBTA APC day-type profile",
            "wkdy_boardings": float(scenario["wkdy_boardings"]),
            "wkdy_mean_load": float(scenario["wkdy_mean_load"]),
            "wkdy_max_load": float(scenario["wkdy_max_load"]),
        },
        "env": {
            "path": env_rel_path,
            "route_sigma": 1.5,
            "demand_noise": 0.0,
            "demand_scale": float(day_scale),
            "demand_hourly_multipliers": {
                int(h): float(v) for h, v in hourly_profile.items()
            },
            "service_start_hour": int(service_start_hour),
            "service_end_hour": int(service_end_hour),
            "od_noise": 0.0,
            "peak_shift_choices": [0],
            "peak_shift_probs": [1.0],
        },
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mbta-env-root", default=str(DEFAULT_MBTA_ENV_ROOT))
    parser.add_argument("--route-targets-csv", default=str(DEFAULT_AUDIT_DIR / "mbta_onboard_route_targets.csv"))
    parser.add_argument("--hourly-profile-csv", default=str(DEFAULT_AUDIT_DIR / "mbta_hourly_board_alight_load.csv"))
    parser.add_argument("--env-cache", default=str(DEFAULT_ENV_CACHE))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--setup-dir", default=str(DEFAULT_SETUP_DIR))
    parser.add_argument("--route-count", type=int, default=20)
    parser.add_argument("--methods", default="main,nofreq,noleakage")
    parser.add_argument("--day-types", default="Wkdy,Sat,Sun")
    parser.add_argument("--no-copy-envs", action="store_true")
    args = parser.parse_args()

    env_root = Path(args.mbta_env_root)
    env_cache = Path(args.env_cache)
    config_dir = Path(args.config_dir)
    setup_dir = Path(args.setup_dir)
    setup_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    env_cache.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(args.route_targets_csv)
    hourly = pd.read_csv(args.hourly_profile_csv)
    envs = available_envs(env_root)
    selected = select_routes(targets, envs, route_count=int(args.route_count))
    if not selected:
        raise SystemExit(f"No route environments selected from {env_root}")

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = sorted(set(methods) - set(PARENT_CONFIGS))
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {unknown_methods}")
    day_types = [x.strip() for x in args.day_types.split(",") if x.strip()]
    unknown_days = sorted(set(day_types) - set(DAY_TYPES))
    if unknown_days:
        raise SystemExit(f"Unknown day types: {unknown_days}")

    hourly_profiles = build_hourly_shape_profiles(hourly)
    manifest_rows: list[dict[str, Any]] = []
    for scenario in selected:
        source_env = Path(scenario["source_env_path"])
        if args.no_copy_envs:
            cached_env = env_cache / source_env.name
            make_bidirectional_terminal_env(cached_env)
            service_start_hour, service_end_hour, launch_offset = normalize_cached_timetable(cached_env)
        else:
            cached_env, service_start_hour, service_end_hour, launch_offset = copy_env(source_env, env_cache)
        env_rel_path = str(cached_env.relative_to(ROOT))
        for day_type in day_types:
            day_scale = route_day_scale(
                targets, scenario["route_id"], int(scenario["target_direction"]), day_type)
            for method in methods:
                parent = PARENT_CONFIGS[method]
                cfg = build_config(
                    method=method,
                    parent=parent,
                    scenario=scenario,
                    day_type=day_type,
                    env_rel_path=env_rel_path,
                    day_scale=day_scale,
                    hourly_profile=hourly_profiles[day_type],
                    service_start_hour=service_start_hour,
                    service_end_hour=service_end_hour,
                )
                cfg_name = cfg["_name"]
                write_yaml(config_dir / f"{cfg_name}.yaml", cfg)
                manifest_rows.append({
                    "config": cfg_name,
                    "config_path": str((config_dir / f"{cfg_name}.yaml").relative_to(ROOT)),
                    "method": method,
                    "agency": "MBTA",
                    "scenario_id": cfg["_paper_route_day"]["scenario_id"],
                    "route_id": scenario["route_id"],
                    "target_direction": int(scenario["target_direction"]),
                    "env_direction": int(scenario["env_direction"]),
                    "env_name": scenario["env_name"],
                    "route_family": scenario["route_family"],
                    "day_type": day_type,
                    "env_path": env_rel_path,
                    "source_env_path": scenario["source_env_path"],
                    "demand_scale": day_scale,
                    "service_start_hour": int(service_start_hour),
                    "service_end_hour": int(service_end_hour),
                    "launch_offset_s": int(launch_offset),
                    "wkdy_boardings": float(scenario["wkdy_boardings"]),
                    "wkdy_mean_load": float(scenario["wkdy_mean_load"]),
                    "wkdy_max_load": float(scenario["wkdy_max_load"]),
                })

    selected_df = pd.DataFrame(selected)
    selected_df.to_csv(setup_dir / "selected_route_envs.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(setup_dir / "config_manifest.csv", index=False)
    profile_rows = []
    for day_type, profile in hourly_profiles.items():
        for hour, value in profile.items():
            profile_rows.append({"day_type": day_type, "hour": int(hour), "hourly_shape_multiplier": float(value)})
    pd.DataFrame(profile_rows).to_csv(setup_dir / "service_day_hourly_shape_multipliers.csv", index=False)
    with (setup_dir / "config_manifest.json").open("w") as f:
        json.dump({
            "route_count": len(selected),
            "config_count": len(manifest_rows),
            "methods": methods,
            "day_types": day_types,
            "env_cache": str(env_cache.relative_to(ROOT)),
        }, f, indent=2)

    main_configs = [
        row["config"] for row in manifest_rows if row["method"] == "main"
    ]
    print(f"Selected route envs: {len(selected)}")
    print(f"Wrote configs: {len(manifest_rows)}")
    print(f"Wrote manifest: {setup_dir / 'config_manifest.csv'}")
    print("Learned configs CSV:")
    print(",".join(row["config"] for row in manifest_rows))
    print("External baseline configs CSV:")
    print(",".join(main_configs))


if __name__ == "__main__":
    main()
