#!/usr/bin/env python3
"""Build route-day counterfactual value labels from fixed-headway sweeps.

The input is an external-baseline ``external_baselines_per_seed.csv`` produced
by ``run_freqduet_external_baselines.py``.  Rows are aligned by route-day and
seed, then treated as executable counterfactual candidates.  The script reports
the oracle best fixed headway and a lightweight out-of-fold value selector.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results_freqduet/route_day_policy_matrix_v1/config_setup/config_manifest.csv"
METRICS = ("wait", "cv", "overshoot", "composite")


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        val = float(arr[0])
        return val, val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


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
        if value > 0:
            return value
    return None


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    if "method" in manifest.columns:
        manifest = manifest[manifest["method"].astype(str).eq("main")].copy()
    required = {"config", "scenario_id", "route_id", "route_family", "day_type"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise SystemExit(f"manifest missing columns: {missing}")
    return manifest.drop_duplicates("config")


def load_external(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"external per-seed file does not exist: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"external per-seed file is empty: {path}")
    if "method" not in df.columns and "variant" in df.columns:
        df["method"] = df["variant"]
    required = {"config", "seed", "method", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"external per-seed file missing columns: {missing}")
    meta_cols = [
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
    meta_cols = [col for col in meta_cols if col in manifest.columns]
    out = df.merge(manifest[meta_cols], on="config", how="inner")
    if out.empty:
        raise SystemExit("no external rows matched route-day manifest configs")
    out["method"] = out["method"].astype(str)
    out["target_headway_s"] = out["method"].map(fixed_headway_target_s)
    if out["target_headway_s"].isna().any():
        bad = sorted(out.loc[out["target_headway_s"].isna(), "method"].unique().tolist())
        raise SystemExit(f"non fixed-headway methods in sweep input: {bad}")
    for col in [
        "seed",
        "target_direction",
        "env_direction",
        "demand_scale",
        "service_start_hour",
        "service_end_hour",
        "launch_offset_s",
        "wkdy_boardings",
        "wkdy_mean_load",
        "wkdy_max_load",
        "target_headway_s",
        *METRICS,
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["service_span_h"] = out.get("service_end_hour", 0) - out.get("service_start_hour", 0)
    return out


def aligned_long_wide(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["scenario_id", "day_type", "seed"]
    counts = df.groupby(key_cols)["method"].nunique().reset_index(name="n_methods")
    full_n = int(df["method"].nunique())
    complete = counts[counts["n_methods"].eq(full_n)][key_cols]
    long_df = df.merge(complete, on=key_cols, how="inner").copy()
    if long_df.empty:
        raise SystemExit("no complete route-day/seed candidate groups")
    dup = long_df.duplicated(key_cols + ["method"], keep=False)
    if dup.any():
        bad = long_df.loc[dup, key_cols + ["method", "config"]].head(20)
        raise SystemExit(f"duplicate candidate rows after alignment:\n{bad.to_string(index=False)}")
    cost_wide = long_df.pivot(index=key_cols, columns="method", values=metric)
    metric_wides = []
    for name in METRICS:
        wide = long_df.pivot(index=key_cols, columns="method", values=name)
        wide.columns = [f"{name}_{col}" for col in wide.columns]
        metric_wides.append(wide)
    meta_cols = [
        "agency",
        "route_id",
        "target_direction",
        "env_direction",
        "env_name",
        "route_family",
        "demand_scale",
        "service_start_hour",
        "service_end_hour",
        "service_span_h",
        "launch_offset_s",
        "wkdy_boardings",
        "wkdy_mean_load",
        "wkdy_max_load",
    ]
    meta_cols = [col for col in meta_cols if col in long_df.columns]
    meta = long_df.groupby(key_cols, as_index=False)[meta_cols].first().set_index(key_cols)
    wide_df = pd.concat([meta, cost_wide.add_prefix("cost_"), *metric_wides], axis=1).reset_index()
    cost_cols = [col for col in wide_df.columns if col.startswith("cost_")]
    costs = wide_df[cost_cols].apply(pd.to_numeric, errors="coerce")
    if costs.isna().any().any():
        raise SystemExit("aligned candidate matrix has missing costs")
    wide_df["oracle_best_method"] = costs.idxmin(axis=1).str.replace("cost_", "", regex=False)
    wide_df["oracle_best_cost"] = costs.min(axis=1)
    return long_df, wide_df


def summarize_values(
    rows: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    n_boot: int,
) -> pd.DataFrame:
    out_rows = []
    for keys, group in rows.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = dict(zip(group_cols, keys))
        out["n_rows"] = int(len(group))
        out["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        for col in value_cols:
            vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=np.float64)
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=stable_seed("summary", *group_cols, *keys, col))
            out[f"{col}_mean"] = float(np.nanmean(vals))
            out[f"{col}_ci_low"] = lo
            out[f"{col}_ci_high"] = hi
            if col.startswith("delta_vs_"):
                out[f"{col}_win_rate"] = float(np.nanmean(vals < 0.0))
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def add_oracle_deltas(wide: pd.DataFrame, baseline_method: str) -> pd.DataFrame:
    baseline_col = f"cost_{baseline_method}"
    if baseline_col not in wide.columns:
        raise SystemExit(f"baseline method {baseline_method!r} not in candidates")
    out = wide.copy()
    out["oracle_delta_vs_baseline"] = out["oracle_best_cost"] - out[baseline_col]
    out["oracle_beats_baseline"] = out["oracle_delta_vs_baseline"] < 0.0
    out["baseline_method"] = baseline_method
    out["baseline_cost"] = out[baseline_col]
    for col in [c for c in out.columns if c.startswith("cost_")]:
        method = col.removeprefix("cost_")
        out[f"regret_{method}"] = out[col] - out["oracle_best_cost"]
        out[f"delta_{method}_vs_baseline"] = out[col] - out[baseline_col]
    return out


def make_feature_matrix(long_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = long_df.copy()
    data["target_headway_s2"] = data["target_headway_s"] ** 2
    data["target_delta_s"] = data["target_headway_s"] - 360.0
    data["target_abs_delta_s"] = data["target_delta_s"].abs()
    numeric = [
        "target_headway_s",
        "target_headway_s2",
        "target_delta_s",
        "target_abs_delta_s",
        "demand_scale",
        "service_span_h",
        "launch_offset_s",
        "wkdy_boardings",
        "wkdy_mean_load",
        "wkdy_max_load",
        "target_direction",
        "env_direction",
    ]
    numeric = [col for col in numeric if col in data.columns]
    for base in ["wkdy_boardings", "wkdy_mean_load", "wkdy_max_load", "service_span_h"]:
        if base in data.columns:
            data[f"target_x_{base}"] = data["target_headway_s"] * pd.to_numeric(data[base], errors="coerce")
            numeric.append(f"target_x_{base}")
    cat = [col for col in ["route_family", "day_type", "route_id"] if col in data.columns]
    features = data[numeric + cat].copy()
    features = pd.get_dummies(features, columns=cat, dummy_na=False, dtype=float)
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    feature_cols = features.columns.tolist()
    data = pd.concat([data.reset_index(drop=True), features.reset_index(drop=True).add_prefix("feat__")], axis=1)
    return data, [f"feat__{col}" for col in feature_cols]


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_test = np.where(np.isfinite(x_test), x_test, med)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    train_design = np.column_stack([np.ones(len(x_train)), x_train])
    test_design = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(train_design.shape[1])
    penalty[0, 0] = 0.0
    lhs = train_design.T @ train_design + float(alpha) * penalty
    rhs = train_design.T @ y_train
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    return test_design @ beta


def assign_folds(values: pd.Series, n_folds: int) -> dict[object, int]:
    unique = sorted(values.dropna().unique().tolist(), key=lambda x: str(x))
    n_folds = max(2, min(int(n_folds), len(unique)))
    return {value: idx % n_folds for idx, value in enumerate(unique)}


def run_value_cv(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    metric: str,
    baseline_method: str,
    n_folds: int,
    alpha: float,
    fold_key: str,
) -> pd.DataFrame:
    feature_df, feature_cols = make_feature_matrix(long_df)
    if fold_key not in feature_df.columns:
        raise SystemExit(f"fold key {fold_key!r} not in route-day rows")
    if feature_df[fold_key].nunique() < 2:
        return pd.DataFrame()
    folds = assign_folds(feature_df[fold_key], n_folds=n_folds)
    feature_df["_fold"] = feature_df[fold_key].map(folds)
    group_cols = ["scenario_id", "day_type", "seed"]
    rows = []
    for fold in sorted(feature_df["_fold"].dropna().unique()):
        train = feature_df[feature_df["_fold"].ne(fold)].copy()
        test = feature_df[feature_df["_fold"].eq(fold)].copy()
        if train.empty or test.empty:
            continue
        preds = ridge_predict(
            train[feature_cols].to_numpy(dtype=np.float64),
            pd.to_numeric(train[metric], errors="coerce").to_numpy(dtype=np.float64),
            test[feature_cols].to_numpy(dtype=np.float64),
            alpha=alpha,
        )
        test = test.copy()
        test["pred_cost"] = preds
        idx = test.groupby(group_cols)["pred_cost"].idxmin()
        selected = test.loc[idx].copy()
        selected = selected.merge(
            wide_df,
            on=group_cols,
            how="inner",
            suffixes=("", "_wide"),
        )
        for _, item in selected.iterrows():
            method = str(item["method"])
            selected_cost = float(item[f"cost_{method}"])
            baseline_cost = float(item[f"cost_{baseline_method}"])
            rows.append({
                "cv_protocol": f"{fold_key}_cv",
                "fold": int(fold),
                "scenario_id": item["scenario_id"],
                "route_id": str(item["route_id"]),
                "route_family": str(item["route_family"]),
                "day_type": item["day_type"],
                "seed": int(item["seed"]),
                "selected_method": method,
                "selected_target_headway_s": float(item["target_headway_s"]),
                "selected_pred_cost": float(item["pred_cost"]),
                "selected_cost": selected_cost,
                "baseline_method": baseline_method,
                "baseline_cost": baseline_cost,
                "delta_vs_baseline": selected_cost - baseline_cost,
                "oracle_best_method": item["oracle_best_method"],
                "oracle_best_cost": float(item["oracle_best_cost"]),
                "oracle_regret": selected_cost - float(item["oracle_best_cost"]),
                "oracle_hit": method == item["oracle_best_method"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--external-per-seed", required=True)
    parser.add_argument("--metric", choices=METRICS, default="composite")
    parser.add_argument("--baseline-method", default="fixed_headway_360")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest_path = resolve(args.manifest)
    external_path = resolve(args.external_per_seed)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    external = load_external(external_path, manifest)
    long_df, wide_df = aligned_long_wide(external, metric=args.metric)
    wide_df = add_oracle_deltas(wide_df, baseline_method=args.baseline_method)

    long_df.to_csv(out_dir / "route_day_fixed_headway_candidates_long.csv", index=False)
    wide_df.to_csv(out_dir / "route_day_fixed_headway_oracle_wide.csv", index=False)
    summarize_values(
        wide_df,
        ["baseline_method"] if "baseline_method" in wide_df.columns else ["agency"],
        ["oracle_delta_vs_baseline"],
        n_boot=args.n_boot,
    ).to_csv(out_dir / "route_day_fixed_headway_oracle_summary.csv", index=False)
    summarize_values(
        wide_df,
        ["route_family", "day_type"],
        ["oracle_delta_vs_baseline"],
        n_boot=args.n_boot,
    ).to_csv(out_dir / "route_day_fixed_headway_oracle_family_day_summary.csv", index=False)
    (
        wide_df.groupby(["route_family", "day_type", "oracle_best_method"], as_index=False)
        .agg(
            n_rows=("oracle_best_method", "size"),
            oracle_delta_vs_baseline_mean=("oracle_delta_vs_baseline", "mean"),
            oracle_best_cost_mean=("oracle_best_cost", "mean"),
            baseline_cost_mean=("baseline_cost", "mean"),
        )
        .sort_values(["route_family", "day_type", "n_rows"], ascending=[True, True, False])
        .to_csv(out_dir / "route_day_fixed_headway_best_target_summary.csv", index=False)
    )

    cv_parts = []
    for fold_key in ["seed", "route_id"]:
        cv = run_value_cv(
            long_df=long_df,
            wide_df=wide_df,
            metric=args.metric,
            baseline_method=args.baseline_method,
            n_folds=args.n_folds,
            alpha=args.ridge_alpha,
            fold_key=fold_key,
        )
        if not cv.empty:
            cv_parts.append(cv)
    cv_rows = pd.concat(cv_parts, ignore_index=True, sort=False) if cv_parts else pd.DataFrame()
    if not cv_rows.empty:
        cv_rows.to_csv(out_dir / "route_day_fixed_headway_value_cv_rows.csv", index=False)
        summarize_values(
            cv_rows,
            ["cv_protocol"],
            ["delta_vs_baseline", "oracle_regret"],
            n_boot=args.n_boot,
        ).to_csv(out_dir / "route_day_fixed_headway_value_cv_summary.csv", index=False)
        summarize_values(
            cv_rows,
            ["cv_protocol", "route_family", "day_type"],
            ["delta_vs_baseline", "oracle_regret"],
            n_boot=args.n_boot,
        ).to_csv(out_dir / "route_day_fixed_headway_value_cv_family_day_summary.csv", index=False)
    else:
        (out_dir / "route_day_fixed_headway_value_cv_rows.csv").write_text("")
        (out_dir / "route_day_fixed_headway_value_cv_summary.csv").write_text("")

    with (out_dir / "route_day_value_manifest.json").open("w") as f:
        json.dump({
            "manifest": str(manifest_path),
            "external_per_seed": str(external_path),
            "metric": args.metric,
            "baseline_method": args.baseline_method,
            "n_input_rows": int(len(external)),
            "n_aligned_candidate_rows": int(len(long_df)),
            "n_aligned_route_day_seed_rows": int(len(wide_df)),
            "methods": sorted(long_df["method"].astype(str).unique().tolist()),
            "n_folds": int(args.n_folds),
            "ridge_alpha": float(args.ridge_alpha),
        }, f, indent=2)

    print(f"Wrote {out_dir}")
    print(pd.read_csv(out_dir / "route_day_fixed_headway_oracle_summary.csv").to_string(index=False))
    cv_summary = out_dir / "route_day_fixed_headway_value_cv_summary.csv"
    if cv_summary.exists() and cv_summary.stat().st_size > 0:
        print(pd.read_csv(cv_summary).to_string(index=False))


if __name__ == "__main__":
    main()
