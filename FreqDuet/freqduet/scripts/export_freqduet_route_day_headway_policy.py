#!/usr/bin/env python3
"""Export executable route-day headway policies from fixed-headway value labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results_freqduet/route_day_policy_matrix_v1/config_setup/config_manifest.csv"


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def fixed_headway_target_s(method: object) -> float | None:
    text = str(method).strip()
    if text == "fixed_headway":
        return 360.0
    for prefix in ("fixed_headway_", "fixed_h"):
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix):]
        if suffix.startswith("h"):
            suffix = suffix[1:]
        try:
            value = float(suffix)
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def load_main_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" in df.columns:
        df = df[df["method"].astype(str).eq("main")].copy()
    required = {"config", "scenario_id", "route_id", "route_family", "day_type"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"manifest missing columns: {missing}")
    keep = [
        col for col in [
            "config",
            "agency",
            "scenario_id",
            "route_id",
            "target_direction",
            "env_direction",
            "env_name",
            "route_family",
            "day_type",
            "demand_scale",
            "service_start_hour",
            "service_end_hour",
            "launch_offset_s",
            "wkdy_boardings",
            "wkdy_mean_load",
            "wkdy_max_load",
        ]
        if col in df.columns
    ]
    return df[keep].drop_duplicates("config")


def method_cost_columns(wide: pd.DataFrame) -> list[str]:
    cols = [col for col in wide.columns if col.startswith("cost_fixed_headway")]
    if not cols:
        raise SystemExit("oracle wide file has no cost_fixed_headway* columns")
    return cols


def attach_targets(rows: pd.DataFrame, method_col: str) -> pd.DataFrame:
    out = rows.copy()
    out["target_headway_s"] = out[method_col].map(fixed_headway_target_s)
    bad = out["target_headway_s"].isna()
    if bad.any():
        methods = sorted(out.loc[bad, method_col].astype(str).unique().tolist())
        raise SystemExit(f"could not parse target from methods: {methods}")
    return out


def export_cv_policy(
    value_dir: Path,
    manifest: pd.DataFrame,
    protocols: set[str],
) -> list[pd.DataFrame]:
    path = value_dir / "route_day_fixed_headway_value_cv_rows.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    cv = pd.read_csv(path)
    if cv.empty:
        return []
    parts = []
    merge_cols = ["scenario_id", "day_type"]
    for protocol, variant in [
        ("seed_cv", "route_value_seed_cv"),
        ("route_id_cv", "route_value_route_cv"),
    ]:
        if protocol not in protocols:
            continue
        cur = cv[cv["cv_protocol"].astype(str).eq(protocol)].copy()
        if cur.empty:
            continue
        cur = cur.merge(
            manifest,
            on=merge_cols,
            how="inner",
            suffixes=("", "_manifest"),
        )
        if cur.empty:
            continue
        cur = attach_targets(cur, "selected_method")
        cur["policy_variant"] = variant
        cur["policy_mode"] = protocol
        cur["source"] = str(path)
        parts.append(cur)
    return parts


def export_oracle_mean_policy(
    value_dir: Path,
    manifest: pd.DataFrame,
    include_seed_oracle: bool,
) -> list[pd.DataFrame]:
    path = value_dir / "route_day_fixed_headway_oracle_wide.csv"
    if not path.exists():
        return []
    wide = pd.read_csv(path)
    if wide.empty:
        return []
    cost_cols = method_cost_columns(wide)
    parts = []

    grouped = wide.groupby(["scenario_id", "day_type"], as_index=False)[cost_cols].mean()
    best_col = grouped[cost_cols].idxmin(axis=1)
    grouped["selected_method"] = best_col.str.replace("cost_", "", regex=False)
    grouped["selected_cost_mean"] = grouped.lookup(grouped.index, best_col) if hasattr(grouped, "lookup") else [
        grouped.loc[idx, col] for idx, col in best_col.items()
    ]
    route_day = grouped.merge(
        manifest,
        on=["scenario_id", "day_type"],
        how="inner",
        suffixes=("", "_manifest"),
    )
    route_day = attach_targets(route_day, "selected_method")
    route_day["policy_variant"] = "route_oracle_mean"
    route_day["policy_mode"] = "oracle_route_day_mean"
    route_day["source"] = str(path)
    parts.append(route_day)

    global_means = wide[cost_cols].mean()
    global_method = str(global_means.idxmin()).replace("cost_", "")
    global_policy = manifest.copy()
    global_policy["selected_method"] = global_method
    global_policy["selected_cost_mean"] = float(global_means.min())
    global_policy = attach_targets(global_policy, "selected_method")
    global_policy["policy_variant"] = "route_value_global_best"
    global_policy["policy_mode"] = "global_best_mean"
    global_policy["source"] = str(path)
    parts.append(global_policy)

    if include_seed_oracle:
        seed_oracle = wide[
            ["scenario_id", "day_type", "seed", "oracle_best_method", "oracle_best_cost"]
        ].copy()
        seed_oracle = seed_oracle.rename(columns={
            "oracle_best_method": "selected_method",
            "oracle_best_cost": "selected_cost",
        })
        seed_oracle = seed_oracle.merge(
            manifest,
            on=["scenario_id", "day_type"],
            how="inner",
            suffixes=("", "_manifest"),
        )
        seed_oracle = attach_targets(seed_oracle, "selected_method")
        seed_oracle["policy_variant"] = "route_oracle_seed"
        seed_oracle["policy_mode"] = "oracle_route_day_seed"
        seed_oracle["source"] = str(path)
        parts.append(seed_oracle)
    return parts


def normalize_output(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    base_cols = [
        "policy_variant",
        "policy_mode",
        "config",
        "scenario_id",
        "route_id",
        "route_family",
        "day_type",
        "seed",
        "selected_method",
        "target_headway_s",
        "selected_cost",
        "selected_cost_mean",
        "delta_vs_baseline",
        "oracle_regret",
        "source",
        "agency",
        "target_direction",
        "env_direction",
        "env_name",
        "demand_scale",
        "service_start_hour",
        "service_end_hour",
        "launch_offset_s",
        "wkdy_boardings",
        "wkdy_mean_load",
        "wkdy_max_load",
    ]
    cols = [col for col in base_cols if col in out.columns]
    extra = [col for col in out.columns if col not in cols]
    out = out[cols + extra].copy()
    for col in ["target_headway_s", "selected_cost", "selected_cost_mean"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(
        [col for col in ["policy_variant", "config", "seed"] if col in out.columns]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--modes",
        default="seed_cv,route_id_cv,oracle_mean,global_best",
        help="comma-separated: seed_cv,route_id_cv,oracle_mean,global_best,oracle_seed",
    )
    args = parser.parse_args()

    value_dir = resolve(args.value_dir)
    manifest = load_main_manifest(resolve(args.manifest))
    modes = {item.strip() for item in str(args.modes).split(",") if item.strip()}
    protocols = modes & {"seed_cv", "route_id_cv"}

    parts = []
    parts.extend(export_cv_policy(value_dir, manifest, protocols))
    if modes & {"oracle_mean", "global_best", "oracle_seed"}:
        oracle_parts = export_oracle_mean_policy(
            value_dir,
            manifest,
            include_seed_oracle="oracle_seed" in modes,
        )
        if "oracle_mean" not in modes:
            oracle_parts = [
                part for part in oracle_parts
                if not part.get("policy_variant", pd.Series(dtype=str)).astype(str)
                .eq("route_oracle_mean").any()
            ]
        if "global_best" not in modes:
            oracle_parts = [
                part for part in oracle_parts
                if not part.get("policy_variant", pd.Series(dtype=str)).astype(str)
                .eq("route_value_global_best").any()
            ]
        parts.extend(oracle_parts)

    out = normalize_output(parts)
    if out.empty:
        raise SystemExit(f"no policies exported from {value_dir}")
    out_path = resolve(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    summary = (
        out.groupby(["policy_variant", "target_headway_s"], as_index=False)
        .agg(n_rows=("config", "size"), n_configs=("config", "nunique"))
        .sort_values(["policy_variant", "target_headway_s"])
    )
    print(f"Wrote {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
