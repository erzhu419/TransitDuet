#!/usr/bin/env python3
"""Summarize route-family/service-day matrix results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results_freqduet/route_day_policy_matrix_v1/config_setup/config_manifest.csv"


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 2026) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.RandomState(seed)
    draws = rng.choice(values, size=(int(n_boot), values.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_learned(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    lookup = manifest.set_index("config")
    rows = []
    for _, row in df.iterrows():
        cfg = str(row["config"])
        if cfg not in lookup.index:
            continue
        meta = lookup.loc[cfg]
        out = row.to_dict()
        out.update({
            "method": str(meta["method"]),
            "scenario_id": str(meta["scenario_id"]),
            "agency": str(meta["agency"]),
            "route_id": str(meta["route_id"]),
            "target_direction": int(meta["target_direction"]),
            "env_direction": int(meta["env_direction"]),
            "env_name": str(meta["env_name"]),
            "route_family": str(meta["route_family"]),
            "day_type": str(meta["day_type"]),
            "source_kind": "freqduet_learned",
        })
        rows.append(out)
    return pd.DataFrame(rows)


def load_external(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    main_manifest = manifest[manifest["method"].eq("main")].set_index("config")
    rows = []
    for _, row in df.iterrows():
        cfg = str(row["config"])
        if cfg not in main_manifest.index:
            continue
        meta = main_manifest.loc[cfg]
        out = row.to_dict()
        out.update({
            "method": str(row.get("method", row.get("variant", "fixed_headway"))),
            "scenario_id": str(meta["scenario_id"]),
            "agency": str(meta["agency"]),
            "route_id": str(meta["route_id"]),
            "target_direction": int(meta["target_direction"]),
            "env_direction": int(meta["env_direction"]),
            "env_name": str(meta["env_name"]),
            "route_family": str(meta["route_family"]),
            "day_type": str(meta["day_type"]),
            "source_kind": "external_baseline",
        })
        rows.append(out)
    return pd.DataFrame(rows)


def summary_table(per_seed: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    metrics = ["wait", "cv", "overshoot", "composite"]
    for keys, group in per_seed.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_rows"] = int(len(group))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_cols = ["scenario_id", "day_type", "seed"]
    metrics = ["wait", "cv", "overshoot", "composite"]
    for keys, group in per_seed.groupby(key_cols, sort=False):
        if "main" not in set(group["method"]):
            continue
        main = group[group["method"].eq("main")].iloc[0]
        for baseline in sorted(set(group["method"]) - {"main"}):
            base = group[group["method"].eq(baseline)].iloc[0]
            row = dict(zip(key_cols, keys))
            row.update({
                "route_id": str(main["route_id"]),
                "route_family": str(main["route_family"]),
                "baseline": baseline,
            })
            for metric in metrics:
                row[f"delta_{metric}"] = float(main[metric]) - float(base[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def delta_summary(deltas: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    rows = []
    metrics = ["wait", "cv", "overshoot", "composite"]
    for keys, group in deltas.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_pairs"] = int(len(group))
        for metric in metrics:
            col = f"delta_{metric}"
            vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=np.float64)
            lo, hi = bootstrap_ci(vals)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
            row[f"{col}_win_rate"] = float(np.nanmean(vals < 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--learned-per-seed", default=None)
    parser.add_argument("--external-per-seed", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(resolve(args.manifest))
    learned = load_learned(resolve(args.learned_per_seed), manifest) if args.learned_per_seed else pd.DataFrame()
    external = load_external(resolve(args.external_per_seed), manifest) if args.external_per_seed else pd.DataFrame()
    per_seed = pd.concat([learned, external], ignore_index=True, sort=False)
    if per_seed.empty:
        raise SystemExit("No route-day rows found from learned/external inputs")

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_dir / "route_day_policy_per_seed.csv", index=False)
    summary_table(per_seed, ["method"]).to_csv(out_dir / "route_day_method_summary.csv", index=False)
    summary_table(per_seed, ["route_family", "day_type", "method"]).to_csv(
        out_dir / "route_day_family_day_method_summary.csv", index=False)
    summary_table(per_seed, ["scenario_id", "day_type", "method"]).to_csv(
        out_dir / "route_day_scenario_method_summary.csv", index=False)

    deltas = paired_deltas(per_seed)
    deltas.to_csv(out_dir / "route_day_paired_deltas.csv", index=False)
    delta_summary(deltas, ["baseline"]).to_csv(out_dir / "route_day_overall_delta_summary.csv", index=False)
    delta_summary(deltas, ["route_family", "day_type", "baseline"]).to_csv(
        out_dir / "route_day_family_day_delta_summary.csv", index=False)

    with (out_dir / "route_day_summary_manifest.json").open("w") as f:
        json.dump({
            "manifest": str(resolve(args.manifest)),
            "learned_per_seed": str(resolve(args.learned_per_seed)) if args.learned_per_seed else None,
            "external_per_seed": str(resolve(args.external_per_seed)) if args.external_per_seed else None,
            "n_rows": int(len(per_seed)),
            "methods": sorted(per_seed["method"].dropna().astype(str).unique().tolist()),
            "n_scenarios": int(per_seed["scenario_id"].nunique()),
            "day_types": sorted(per_seed["day_type"].dropna().astype(str).unique().tolist()),
        }, f, indent=2)
    print(f"Wrote {out_dir}")
    print(pd.read_csv(out_dir / "route_day_method_summary.csv").to_string(index=False))
    if (out_dir / "route_day_overall_delta_summary.csv").exists():
        delta = pd.read_csv(out_dir / "route_day_overall_delta_summary.csv")
        if not delta.empty:
            print(delta.to_string(index=False))


if __name__ == "__main__":
    main()
