#!/usr/bin/env python3
"""Build trip-level counterfactual value labels from FreqDuet CRN rollouts.

Episode-level action labels were too noisy for selecting terminal/headway
actions. This script aligns `trip_details.csv` from fixed-action common-random
number runs by (domain, seed, episode, trip id), then evaluates an offline
action-conditioned value model on trip-level labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fit_freqduet_action_value_model import (
    add_design_features,
    bootstrap_ci,
    parse_action,
    ridge_fit_predict,
    seed_folds,
    stable_seed,
)


DOMAINS = ("terminal", "highnoise", "odshift", "rushshift")
DEFAULT_CANDIDATES = (
    "target_m20",
    "target0",
    "target_p20",
    "term45_m20",
    "term45_0",
    "term45_p20",
)
RUN_RE = re.compile(r"^(?P<config>.+)_seed(?P<seed>\d+)$")
ACTION_RE = re.compile(
    r"_cfaction_v\d+_(?P<mode>target|terminalhold45)_(?P<delta>d[mp]\d+)_hiro$"
)


def infer_domain(config: str) -> str:
    if "_gen_highnoise_" in config:
        return "highnoise"
    if "_gen_odshift_" in config:
        return "odshift"
    if "_gen_rushshift_" in config:
        return "rushshift"
    if "_terminal_" in config:
        return "terminal"
    return "unknown"


def infer_candidate_method(config: str) -> str:
    match = ACTION_RE.search(config)
    if not match:
        raise ValueError(f"not a cfaction config: {config}")
    mode = "target" if match.group("mode") == "target" else "term45"
    delta = match.group("delta")
    if delta == "dp0":
        return "target0" if mode == "target" else "term45_0"
    suffix = "m" + delta[2:] if delta.startswith("dm") else "p" + delta[2:]
    return f"{mode}_{suffix}"


def parse_run(path: Path) -> tuple[str, int, str]:
    match = RUN_RE.match(path.parent.name)
    if not match:
        raise ValueError(f"cannot parse run dir name: {path.parent.name}")
    config = match.group("config")
    return config, int(match.group("seed")), infer_candidate_method(config)


def find_trip_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "trip_details.csv":
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("trip_details.csv"))
    return sorted(set(files))


def read_trip_rows(paths: list[Path], metric: str, last_k: int, higher_is_better: bool) -> pd.DataFrame:
    parts = []
    required = {"ep", "tid", "dir", metric}
    for path in paths:
        config, seed, method = parse_run(path)
        domain = infer_domain(config)
        if domain not in DOMAINS:
            continue
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise SystemExit(f"{path} missing columns: {missing}")
        if last_k > 0 and not df.empty:
            max_ep = int(pd.to_numeric(df["ep"], errors="coerce").max())
            df = df[pd.to_numeric(df["ep"], errors="coerce") >= max_ep - last_k + 1].copy()
        df["domain"] = domain
        df["seed"] = int(seed)
        df["config"] = config
        df["candidate_method"] = method
        value = pd.to_numeric(df[metric], errors="coerce")
        df["cost"] = -value if higher_is_better else value
        parts.append(df)
    if not parts:
        raise SystemExit("no trip_details.csv rows loaded")
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[np.isfinite(pd.to_numeric(rows["cost"], errors="coerce"))].copy()
    rows["ep"] = rows["ep"].astype(int)
    rows["tid"] = rows["tid"].astype(int)
    rows["seed"] = rows["seed"].astype(int)
    return rows


def build_aligned_long(
    rows: pd.DataFrame,
    candidates: list[str],
    baseline_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["domain", "seed", "ep", "tid"]
    wide = rows.pivot_table(
        index=key_cols,
        columns="candidate_method",
        values="cost",
        aggfunc="mean",
    ).reset_index()
    required = [method for method in candidates if method in wide.columns]
    missing = sorted(set(candidates) - set(required))
    if missing:
        raise SystemExit(f"missing candidate methods in trip rows: {missing}")
    wide = wide.dropna(subset=required).copy()
    if wide.empty:
        raise SystemExit("no fully aligned trip rows across candidates")

    baseline = rows[rows["candidate_method"] == baseline_method].copy()
    if baseline.empty:
        raise SystemExit(f"baseline method {baseline_method!r} missing")
    context = baseline.drop_duplicates(key_cols, keep="first").copy()
    context = context.merge(wide[key_cols], on=key_cols, how="inner")
    add_context_features(context)

    long = wide.melt(
        id_vars=key_cols,
        value_vars=required,
        var_name="candidate_method",
        value_name="cost",
    )
    long = long.merge(context, on=key_cols, how="left", suffixes=("", "_ctx"))
    action_features = pd.DataFrame([parse_action(method) for method in long["candidate_method"]])
    long = pd.concat([long.reset_index(drop=True), action_features], axis=1)
    return long, wide


def add_context_features(df: pd.DataFrame) -> None:
    df["hour_norm"] = pd.to_numeric(df.get("hour", 0.0), errors="coerce") / 24.0
    df["tid_norm"] = pd.to_numeric(df.get("tid", 0.0), errors="coerce") / max(float(df["tid"].max()), 1.0)
    df["ep_norm"] = pd.to_numeric(df.get("ep", 0.0), errors="coerce") / max(float(df["ep"].max()), 1.0)
    df["dir_signed"] = np.where(pd.to_numeric(df.get("dir", 0), errors="coerce") > 0, 1.0, -1.0)
    for period in ("peak", "off", "trans"):
        df[f"period_is_{period}"] = (df.get("period", "").astype(str) == period).astype(float)
    for col in ("base_hw", "eff_hw"):
        if col in df.columns:
            df[f"{col}_norm"] = pd.to_numeric(df[col], errors="coerce") / 600.0
    for col in ("s_hold_mean", "s_hold_std"):
        if col in df.columns:
            df[f"{col}_norm"] = pd.to_numeric(df[col], errors="coerce") / 60.0
    for col in ("terminal_gap_now", "terminal_short_gap", "terminal_over_gap"):
        if col in df.columns:
            df[f"{col}_norm"] = pd.to_numeric(df[col], errors="coerce") / 600.0
    for col in ("fleet_concurrent", "fleet_target", "fleet_pressure"):
        if col in df.columns:
            df[f"{col}_norm"] = pd.to_numeric(df[col], errors="coerce") / 30.0
    if "waiting_total" in df.columns:
        df["waiting_total_norm"] = pd.to_numeric(df["waiting_total"], errors="coerce") / 500.0
    for col in (
        "freq_low_demand",
        "freq_low_forecast",
        "freq_high_energy",
        "freq_middle_energy",
        "freq_od_entropy",
        "freq_promotion_strength",
        "freq_promotion_active",
    ):
        if col in df.columns:
            df[f"{col}_ctx"] = pd.to_numeric(df[col], errors="coerce")


def context_cols(data: pd.DataFrame) -> list[str]:
    cols = [
        "hour_norm",
        "tid_norm",
        "ep_norm",
        "dir_signed",
        "period_is_peak",
        "period_is_off",
        "period_is_trans",
        "base_hw_norm",
        "eff_hw_norm",
        "s_hold_mean_norm",
        "s_hold_std_norm",
        "terminal_gap_now_norm",
        "terminal_short_gap_norm",
        "terminal_over_gap_norm",
        "fleet_concurrent_norm",
        "fleet_target_norm",
        "fleet_pressure_norm",
        "waiting_total_norm",
        "freq_low_demand_ctx",
        "freq_low_forecast_ctx",
        "freq_high_energy_ctx",
        "freq_middle_energy_ctx",
        "freq_od_entropy_ctx",
        "freq_promotion_strength_ctx",
        "freq_promotion_active_ctx",
    ]
    return [col for col in cols if col in data.columns]


def select_trip_predictions(
    test_rows: pd.DataFrame,
    predictions: np.ndarray,
    feature_set: str,
    fold: int,
    baseline_method: str,
) -> pd.DataFrame:
    key_cols = ["domain", "seed", "ep", "tid"]
    pred = test_rows[[
        *key_cols, "candidate_method", "cost"
    ]].copy()
    pred["pred_cost"] = predictions
    selected = (
        pred.sort_values([*key_cols, "pred_cost", "candidate_method"], kind="mergesort")
        .drop_duplicates(key_cols, keep="first")
        [[*key_cols, "candidate_method", "pred_cost", "cost"]]
        .rename(columns={
            "candidate_method": "selected_method",
            "pred_cost": "selected_pred_cost",
            "cost": "selected_cost",
        })
    )
    oracle = (
        pred.sort_values([*key_cols, "cost", "candidate_method"], kind="mergesort")
        .drop_duplicates(key_cols, keep="first")
        [[*key_cols, "candidate_method", "cost"]]
        .rename(columns={
            "candidate_method": "oracle_best_method",
            "cost": "oracle_best_cost",
        })
    )
    baseline = (
        pred[pred["candidate_method"] == baseline_method][[*key_cols, "cost"]]
        .rename(columns={"cost": "baseline_cost"})
    )
    out = selected.merge(oracle, on=key_cols, how="left")
    out = out.merge(baseline, on=key_cols, how="left")
    out["feature_set"] = feature_set
    out["fold"] = int(fold)
    out["oracle_regret"] = out["selected_cost"] - out["oracle_best_cost"]
    out[f"delta_vs_{baseline_method}"] = out["selected_cost"] - out["baseline_cost"]
    return out[[
        "feature_set", "fold", *key_cols,
        "selected_method", "selected_pred_cost", "selected_cost",
        "oracle_best_method", "oracle_best_cost", "oracle_regret",
        f"delta_vs_{baseline_method}", "baseline_cost",
    ]]


def run_trip_model_cv(
    data: pd.DataFrame,
    feature_set: str,
    context_cols: list[str],
    folds: int,
    alpha: float,
    baseline_method: str,
) -> tuple[pd.DataFrame, int]:
    design, feature_cols = add_design_features(data, context_cols, feature_set)
    folds = max(2, min(int(folds), int(design["seed"].nunique())))
    fold_map = seed_folds(design["seed"].unique().tolist(), folds)
    design["_fold"] = design["seed"].map(fold_map)
    rows: list[pd.DataFrame] = []
    x = design[feature_cols].to_numpy(dtype=np.float64)
    y = design["cost"].to_numpy(dtype=np.float64)
    for fold in range(folds):
        train_mask = design["_fold"].to_numpy() != fold
        test_mask = ~train_mask
        if not train_mask.any() or not test_mask.any():
            continue
        pred = ridge_fit_predict(x[train_mask], y[train_mask], x[test_mask], alpha=alpha)
        rows.append(select_trip_predictions(
            design.loc[test_mask].reset_index(drop=True),
            pred,
            feature_set=feature_set,
            fold=fold,
            baseline_method=baseline_method,
        ))
    if not rows:
        return pd.DataFrame(), len(feature_cols)
    return pd.concat(rows, ignore_index=True), len(feature_cols)


def run_trip_mean_cv(data: pd.DataFrame, selector: str, folds: int, baseline_method: str) -> pd.DataFrame:
    folds = max(2, min(int(folds), int(data["seed"].nunique())))
    fold_map = seed_folds(data["seed"].unique().tolist(), folds)
    working = data.copy()
    working["_fold"] = working["seed"].map(fold_map)
    rows: list[pd.DataFrame] = []
    for fold in range(folds):
        train = working[working["_fold"] != fold].copy()
        test = working[working["_fold"] == fold].copy()
        if train.empty or test.empty:
            continue
        if selector == "global_action_mean":
            means = train.groupby("candidate_method")["cost"].mean().to_dict()
            pred = test["candidate_method"].map(means).fillna(train["cost"].mean()).to_numpy(dtype=np.float64)
        elif selector == "domain_action_mean":
            means = train.groupby(["domain", "candidate_method"])["cost"].mean()
            keys = pd.MultiIndex.from_frame(test[["domain", "candidate_method"]])
            pred = means.reindex(keys).to_numpy(dtype=np.float64)
            global_means = train.groupby("candidate_method")["cost"].mean().to_dict()
            fallback = (
                test["candidate_method"]
                .map(global_means)
                .fillna(train["cost"].mean())
                .to_numpy(dtype=np.float64)
            )
            pred = np.where(np.isfinite(pred), pred, fallback)
        else:
            raise SystemExit(f"unknown selector: {selector}")
        rows.append(select_trip_predictions(
            test.reset_index(drop=True),
            pred,
            feature_set=selector,
            fold=fold,
            baseline_method=baseline_method,
        ))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_trip_candidates(data: pd.DataFrame, baseline_method: str, n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    wide = data.pivot_table(
        index=["domain", "seed", "ep", "tid"],
        columns="candidate_method",
        values="cost",
        aggfunc="mean",
    )
    wide["oracle_best_cost"] = wide.min(axis=1)
    methods = sorted(data["candidate_method"].unique())
    for domain in [*DOMAINS, "overall_seed_mean"]:
        if domain == "overall_seed_mean":
            group = wide.reset_index().groupby("seed", as_index=False).mean(numeric_only=True)
        else:
            group = wide.reset_index()
            group = group[group["domain"] == domain]
            group = group.groupby("seed", as_index=False).mean(numeric_only=True)
        if group.empty:
            continue
        for method in methods:
            row = {
                "domain": domain,
                "method": method,
                "cost_mean": float(group[method].mean()),
                "regret_to_oracle_mean": float((group[method] - group["oracle_best_cost"]).mean()),
            }
            if baseline_method in group.columns and method != baseline_method:
                delta = group[method] - group[baseline_method]
                lo, hi = bootstrap_ci(
                    delta.to_numpy(dtype=np.float64),
                    n_boot=n_boot,
                    seed=stable_seed("trip_candidate", domain, method, baseline_method),
                )
                row[f"delta_vs_{baseline_method}_mean"] = float(delta.mean())
                row[f"delta_vs_{baseline_method}_ci95_lo"] = lo
                row[f"delta_vs_{baseline_method}_ci95_hi"] = hi
            rows.append(row)
        rows.append({
            "domain": domain,
            "method": "oracle_best",
            "cost_mean": float(group["oracle_best_cost"].mean()),
            "regret_to_oracle_mean": 0.0,
        })
    return pd.DataFrame(rows)


def summarize_trip_cv_rows(rows: pd.DataFrame, baseline_method: str, n_boot: int) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    delta_col = f"delta_vs_{baseline_method}"
    for feature_set in sorted(rows["feature_set"].unique()):
        sub = rows[rows["feature_set"] == feature_set].copy()
        for domain in [*DOMAINS, "overall_seed_mean"]:
            if domain == "overall_seed_mean":
                seed_group = (
                    sub.groupby("seed", as_index=False)
                    .agg(
                        selected_cost=("selected_cost", "mean"),
                        oracle_best_cost=("oracle_best_cost", "mean"),
                        oracle_regret=("oracle_regret", "mean"),
                        **{delta_col: (delta_col, "mean")},
                    )
                )
                choices = sub["selected_method"].value_counts().sort_index().to_dict()
                n_rows = int(len(seed_group))
            else:
                domain_rows = sub[sub["domain"] == domain].copy()
                seed_group = (
                    domain_rows.groupby("seed", as_index=False)
                    .agg(
                        selected_cost=("selected_cost", "mean"),
                        oracle_best_cost=("oracle_best_cost", "mean"),
                        oracle_regret=("oracle_regret", "mean"),
                        **{delta_col: (delta_col, "mean")},
                    )
                )
                choices = domain_rows["selected_method"].value_counts().sort_index().to_dict()
                n_rows = int(len(domain_rows))
            if seed_group.empty:
                continue
            row = {
                "feature_set": feature_set,
                "domain": domain,
                "n_rows": n_rows,
                "selected_cost_mean": float(seed_group["selected_cost"].mean()),
                "oracle_best_cost_mean": float(seed_group["oracle_best_cost"].mean()),
                "oracle_regret_mean": float(seed_group["oracle_regret"].mean()),
                "selected_method_counts_json": json.dumps(choices, sort_keys=True),
            }
            lo, hi = bootstrap_ci(
                seed_group["oracle_regret"].to_numpy(dtype=np.float64),
                n_boot=n_boot,
                seed=stable_seed("trip_regret", feature_set, domain),
            )
            row["oracle_regret_ci95_lo"] = lo
            row["oracle_regret_ci95_hi"] = hi
            delta = seed_group[delta_col].to_numpy(dtype=np.float64)
            lo, hi = bootstrap_ci(
                delta,
                n_boot=n_boot,
                seed=stable_seed("trip_delta", feature_set, domain, baseline_method),
            )
            row[f"delta_vs_{baseline_method}_mean"] = float(np.nanmean(delta))
            row[f"delta_vs_{baseline_method}_ci95_lo"] = lo
            row[f"delta_vs_{baseline_method}_ci95_hi"] = hi
            row[f"win_vs_{baseline_method}_rate"] = float((delta < 0.0).mean())
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def print_trip_summary(summary: pd.DataFrame, candidate_summary: pd.DataFrame, baseline_method: str) -> None:
    delta_col = f"delta_vs_{baseline_method}_mean"
    lo_col = f"delta_vs_{baseline_method}_ci95_lo"
    hi_col = f"delta_vs_{baseline_method}_ci95_hi"
    overall = summary[summary["domain"] == "overall_seed_mean"].copy()
    cols = [
        "feature_set",
        "selected_cost_mean",
        "oracle_best_cost_mean",
        "oracle_regret_mean",
        delta_col,
        lo_col,
        hi_col,
        f"win_vs_{baseline_method}_rate",
        "selected_method_counts_json",
    ]
    cols = [col for col in cols if col in overall.columns]
    print("\n== trip-level action-value CV summary ==")
    print(overall[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n== trip-level candidate costs, overall ==")
    cand = candidate_summary[candidate_summary["domain"] == "overall_seed_mean"].copy()
    cols = ["method", "cost_mean", "regret_to_oracle_mean", delta_col, lo_col, hi_col]
    cols = [col for col in cols if col in cand.columns]
    print(cand[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", action="append", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--metric", default="gap_dev")
    ap.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--last-k", type=int, default=30)
    ap.add_argument("--baseline-method", default="target0")
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge-alpha", type=float, default=5.0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument(
        "--skip-heavy-exports",
        action="store_true",
        help="Skip raw/long/wide/CV row CSV exports and write summaries only.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [part.strip() for part in args.candidates.split(",") if part.strip()]
    if args.baseline_method not in candidates:
        raise SystemExit(f"baseline method {args.baseline_method!r} not in candidates")

    paths = find_trip_files(args.logs_root)
    rows = read_trip_rows(paths, args.metric, args.last_k, args.higher_is_better)
    long, wide = build_aligned_long(rows, candidates, args.baseline_method)
    ccols = context_cols(long)

    cv_parts = []
    feature_counts = {}
    for feature_set in ("action_domain", "context_action", "context_action_interact"):
        cv, n_features = run_trip_model_cv(
            long,
            feature_set=feature_set,
            context_cols=ccols,
            folds=args.folds,
            alpha=args.ridge_alpha,
            baseline_method=args.baseline_method,
        )
        cv_parts.append(cv)
        feature_counts[feature_set] = n_features
    for selector in ("global_action_mean", "domain_action_mean"):
        cv_parts.append(
            run_trip_mean_cv(
                long,
                selector=selector,
                folds=args.folds,
                baseline_method=args.baseline_method,
            )
        )
    cv_rows = pd.concat(cv_parts, ignore_index=True)
    summary = summarize_trip_cv_rows(cv_rows, args.baseline_method, args.n_boot)
    candidate_summary = summarize_trip_candidates(long, args.baseline_method, args.n_boot)

    if not args.skip_heavy_exports:
        rows.to_csv(out_dir / "trip_counterfactual_raw_rows.csv", index=False)
        long.to_csv(out_dir / "trip_counterfactual_long_rows.csv", index=False)
        wide.to_csv(out_dir / "trip_counterfactual_wide_costs.csv", index=False)
        cv_rows.to_csv(out_dir / "trip_action_value_cv_rows.csv", index=False)
    summary.to_csv(out_dir / "trip_action_value_cv_summary.csv", index=False)
    candidate_summary.to_csv(out_dir / "trip_action_value_candidate_summary.csv", index=False)
    payload = {
        "logs_root": [str(p) for p in args.logs_root],
        "metric": args.metric,
        "higher_is_better": bool(args.higher_is_better),
        "last_k": int(args.last_k),
        "baseline_method": args.baseline_method,
        "candidates": candidates,
        "trip_files": int(len(paths)),
        "raw_rows": int(len(rows)),
        "aligned_context_rows": int(len(wide)),
        "long_rows": int(len(long)),
        "context_features": ccols,
        "feature_counts": feature_counts,
        "heavy_exports_written": not bool(args.skip_heavy_exports),
        "limitation": (
            "Rows are matched by scheduled domain/seed/episode/trip id across "
            "CRN fixed-action rollouts. This is closer to decision-level labels "
            "than episode summaries, but earlier actions can still affect later "
            "trip states, so it is not an exact simulator snapshot replay."
        ),
    }
    (out_dir / "trip_action_value_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_trip_summary(summary, candidate_summary, args.baseline_method)
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
