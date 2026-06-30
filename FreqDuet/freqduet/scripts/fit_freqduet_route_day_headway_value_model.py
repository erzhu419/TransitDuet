#!/usr/bin/env python3
"""Fit route-day fixed-headway value selectors from matched rollout labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_freqduet_route_day_value_dataset import (  # noqa: E402
    METRICS,
    add_oracle_deltas,
    aligned_long_wide,
    bootstrap_ci,
    fixed_headway_target_s,
    load_external,
    load_manifest,
)


FEATURE_SETS = {
    "day": ["day_type"],
    "family_day": ["route_family", "day_type"],
    "family_day_route": ["route_family", "day_type", "route_id"],
}
FEATURE_ALIASES = {
    "day": "d",
    "family_day": "fd",
    "family_day_route": "fdr",
}
PROTOCOL_ALIASES = {
    "seed": "seed",
    "route_id": "route",
}


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def assign_folds(values: pd.Series, n_folds: int) -> dict[object, int]:
    unique = sorted(values.dropna().unique().tolist(), key=lambda x: str(x))
    n_folds = max(2, min(int(n_folds), len(unique)))
    return {value: idx % n_folds for idx, value in enumerate(unique)}


def method_target(method: object) -> float:
    target = fixed_headway_target_s(method)
    if target is None:
        raise ValueError(f"cannot parse fixed-headway target from {method!r}")
    return float(target)


def build_features(data: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = data.copy()
    out["target_headway_s2"] = out["target_headway_s"] ** 2
    out["target_log"] = np.log(np.maximum(out["target_headway_s"], 1.0))
    out["target_delta_s"] = out["target_headway_s"] - 540.0
    out["target_abs_delta_s"] = out["target_delta_s"].abs()

    numeric = [
        "target_headway_s",
        "target_headway_s2",
        "target_log",
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
    numeric = [col for col in numeric if col in out.columns]
    for base in ["demand_scale", "wkdy_boardings", "wkdy_mean_load", "wkdy_max_load"]:
        if base in out.columns:
            name = f"target_x_{base}"
            out[name] = out["target_headway_s"] * pd.to_numeric(out[base], errors="coerce")
            numeric.append(name)
    categorical = [col for col in FEATURE_SETS[feature_set] if col in out.columns]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in categorical:
        out[col] = out[col].astype(str)
    return out, numeric, categorical


def make_model(name: str, seed: int, n_jobs: int):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "gbr":
        return GradientBoostingRegressor(
            random_state=seed,
            max_depth=2,
            n_estimators=120,
            learning_rate=0.05,
            min_samples_leaf=8,
        )
    if name == "extra":
        return ExtraTreesRegressor(
            n_estimators=120,
            min_samples_leaf=8,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=8,
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise SystemExit(f"unknown model: {name}")


def make_pipeline(model_name: str, numeric: list[str], categorical: list[str], seed: int, n_jobs: int):
    pre = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", make_model(model_name, seed=seed, n_jobs=n_jobs)),
    ])


def run_model_cv(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    metric: str,
    baseline_method: str,
    fold_key: str,
    feature_set: str,
    model_name: str,
    n_folds: int,
    n_jobs: int,
) -> pd.DataFrame:
    if fold_key not in long_df.columns:
        raise SystemExit(f"fold key {fold_key!r} missing from route-day rows")
    data, numeric, categorical = build_features(long_df, feature_set)
    folds = assign_folds(data[fold_key], n_folds=n_folds)
    data["_fold"] = data[fold_key].map(folds)
    group_cols = ["scenario_id", "day_type", "seed"]
    feature_cols = numeric + categorical
    rows = []
    for fold in sorted(data["_fold"].dropna().unique()):
        train = data[data["_fold"].ne(fold)].copy()
        test = data[data["_fold"].eq(fold)].copy()
        if train.empty or test.empty:
            continue
        pipe = make_pipeline(
            model_name=model_name,
            numeric=numeric,
            categorical=categorical,
            seed=stable_seed(model_name, feature_set, fold_key, fold),
            n_jobs=n_jobs,
        )
        pipe.fit(train[feature_cols], pd.to_numeric(train[metric], errors="coerce"))
        test["pred_cost"] = pipe.predict(test[feature_cols])
        selected_idx = test.groupby(group_cols)["pred_cost"].idxmin()
        selected = test.loc[selected_idx].copy()
        selected = selected.merge(
            wide_df,
            on=group_cols,
            how="inner",
            suffixes=("", "_wide"),
        )
        policy_variant = (
            f"route_value_{model_name}_{FEATURE_ALIASES[feature_set]}_"
            f"{PROTOCOL_ALIASES[fold_key]}_cv"
        )
        for _, item in selected.iterrows():
            method = str(item["method"])
            selected_cost = float(item[f"cost_{method}"])
            baseline_cost = float(item[f"cost_{baseline_method}"])
            rows.append({
                "policy_variant": policy_variant,
                "cv_protocol": f"{fold_key}_cv",
                "model": model_name,
                "feature_set": feature_set,
                "fold": int(fold),
                "config": item["config"],
                "scenario_id": item["scenario_id"],
                "agency": item.get("agency", ""),
                "route_id": str(item["route_id"]),
                "route_family": str(item["route_family"]),
                "day_type": item["day_type"],
                "seed": int(item["seed"]),
                "selected_method": method,
                "target_headway_s": method_target(method),
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


def run_model_train_all_policy(
    long_df: pd.DataFrame,
    model_name: str,
    feature_set: str,
    metric: str,
    n_jobs: int,
) -> pd.DataFrame:
    data, numeric, categorical = build_features(long_df, feature_set)
    feature_cols = numeric + categorical
    pipe = make_pipeline(
        model_name=model_name,
        numeric=numeric,
        categorical=categorical,
        seed=stable_seed(model_name, feature_set, "train_all"),
        n_jobs=n_jobs,
    )
    pipe.fit(data[feature_cols], pd.to_numeric(data[metric], errors="coerce"))
    pred = data.copy()
    pred["pred_cost"] = pipe.predict(pred[feature_cols])
    group_cols = ["scenario_id", "day_type"]
    selected_idx = pred.groupby(group_cols)["pred_cost"].idxmin()
    selected = pred.loc[selected_idx].copy()
    policy_variant = f"route_value_{model_name}_{FEATURE_ALIASES[feature_set]}_trainall"
    selected["policy_variant"] = policy_variant
    selected["policy_mode"] = "train_all"
    selected["model"] = model_name
    selected["feature_set"] = feature_set
    selected["selected_method"] = selected["method"].astype(str)
    selected["target_headway_s"] = selected["selected_method"].map(method_target)
    selected["selected_pred_cost"] = selected["pred_cost"].astype(float)
    cols = [
        "policy_variant",
        "policy_mode",
        "model",
        "feature_set",
        "config",
        "scenario_id",
        "agency",
        "route_id",
        "route_family",
        "day_type",
        "selected_method",
        "target_headway_s",
        "selected_pred_cost",
        "demand_scale",
        "service_start_hour",
        "service_end_hour",
        "launch_offset_s",
        "wkdy_boardings",
        "wkdy_mean_load",
        "wkdy_max_load",
    ]
    cols = [col for col in cols if col in selected.columns]
    return selected[cols].sort_values(["policy_variant", "config"])


def summarize(rows: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    out_rows = []
    group_cols = ["policy_variant", "cv_protocol", "model", "feature_set"]
    for keys, group in rows.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = dict(zip(group_cols, keys))
        out["n_rows"] = int(len(group))
        out["n_seeds"] = int(group["seed"].nunique())
        out["n_configs"] = int(group["config"].nunique())
        for col in ["delta_vs_baseline", "oracle_regret"]:
            vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            lo, hi = bootstrap_ci(
                vals,
                n_boot=n_boot,
                seed=stable_seed("route-day-model", *keys, col),
            )
            out[f"{col}_mean"] = float(np.nanmean(vals)) if vals.size else math.nan
            out[f"{col}_ci_low"] = lo
            out[f"{col}_ci_high"] = hi
        out["delta_vs_baseline_win_rate"] = float(
            np.nanmean(pd.to_numeric(group["delta_vs_baseline"], errors="coerce") < 0.0))
        out["oracle_hit_rate"] = float(np.nanmean(group["oracle_hit"].astype(float)))
        out_rows.append(out)
    return pd.DataFrame(out_rows).sort_values(
        ["cv_protocol", "delta_vs_baseline_mean", "oracle_regret_mean"]
    )


def export_policy(rows: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "policy_variant",
        "cv_protocol",
        "model",
        "feature_set",
        "config",
        "scenario_id",
        "agency",
        "route_id",
        "route_family",
        "day_type",
        "seed",
        "selected_method",
        "target_headway_s",
        "selected_pred_cost",
        "selected_cost",
        "baseline_method",
        "baseline_cost",
        "delta_vs_baseline",
        "oracle_best_method",
        "oracle_best_cost",
        "oracle_regret",
        "oracle_hit",
    ]
    cols = [col for col in cols if col in rows.columns]
    rows[cols].to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--external-per-seed", required=True)
    parser.add_argument("--baseline-method", default="fixed_headway_540")
    parser.add_argument("--metric", choices=METRICS, default="composite")
    parser.add_argument("--fold-keys", default="seed,route_id")
    parser.add_argument("--feature-sets", default="family_day,family_day_route,day")
    parser.add_argument("--models", default="gbr,extra,rf,ridge")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-train-all-policy", action="store_true")
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

    parts = []
    for fold_key in parse_csv(args.fold_keys):
        if fold_key not in {"seed", "route_id"}:
            raise SystemExit(f"unsupported fold key: {fold_key}")
        for feature_set in parse_csv(args.feature_sets):
            if feature_set not in FEATURE_SETS:
                raise SystemExit(f"unsupported feature set: {feature_set}")
            for model_name in parse_csv(args.models):
                cv = run_model_cv(
                    long_df=long_df,
                    wide_df=wide_df,
                    metric=args.metric,
                    baseline_method=args.baseline_method,
                    fold_key=fold_key,
                    feature_set=feature_set,
                    model_name=model_name,
                    n_folds=args.n_folds,
                    n_jobs=args.n_jobs,
                )
                if not cv.empty:
                    parts.append(cv)
    if not parts:
        raise SystemExit("no model CV rows produced")

    rows = pd.concat(parts, ignore_index=True, sort=False)
    summary = summarize(rows, n_boot=args.n_boot)
    rows.to_csv(out_dir / "route_day_headway_value_model_cv_rows.csv", index=False)
    summary.to_csv(out_dir / "route_day_headway_value_model_cv_summary.csv", index=False)
    export_policy(rows, out_dir / "route_day_headway_value_model_policy.csv")
    train_all_parts = []
    if not args.skip_train_all_policy:
        for feature_set in parse_csv(args.feature_sets):
            for model_name in parse_csv(args.models):
                train_all_parts.append(run_model_train_all_policy(
                    long_df=long_df,
                    model_name=model_name,
                    feature_set=feature_set,
                    metric=args.metric,
                    n_jobs=args.n_jobs,
                ))
    if train_all_parts:
        train_all = pd.concat(train_all_parts, ignore_index=True, sort=False)
        train_all.to_csv(
            out_dir / "route_day_headway_value_model_train_all_policy.csv",
            index=False,
        )
        train_summary = (
            train_all.groupby(["policy_variant", "target_headway_s"], as_index=False)
            .agg(n_rows=("config", "size"), n_configs=("config", "nunique"))
            .sort_values(["policy_variant", "target_headway_s"])
        )
        train_summary.to_csv(
            out_dir / "route_day_headway_value_model_train_all_policy_summary.csv",
            index=False,
        )
    with (out_dir / "route_day_headway_value_model_manifest.json").open("w") as f:
        json.dump({
            "manifest": str(manifest_path),
            "external_per_seed": str(external_path),
            "baseline_method": args.baseline_method,
            "metric": args.metric,
            "fold_keys": parse_csv(args.fold_keys),
            "feature_sets": parse_csv(args.feature_sets),
            "models": parse_csv(args.models),
            "n_rows": int(len(rows)),
        }, f, indent=2)

    print(f"Wrote {out_dir}")
    print(summary.to_string(index=False))
    train_summary_path = out_dir / "route_day_headway_value_model_train_all_policy_summary.csv"
    if train_summary_path.exists():
        print("\ntrain-all policy target distribution:")
        print(pd.read_csv(train_summary_path).to_string(index=False))


if __name__ == "__main__":
    main()
