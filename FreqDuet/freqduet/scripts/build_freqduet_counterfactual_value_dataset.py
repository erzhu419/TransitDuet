#!/usr/bin/env python3
"""Build matched-seed counterfactual value labels for FreqDuet planners.

This script aligns already executed rollouts by ``(domain, seed)`` and treats
each aligned method as an executable candidate action/policy. It is a first
stage value-model audit: the labels are real rollout costs from matched demand
seeds, while the default features intentionally exclude direct outcome columns
such as wait/cv/overshoot/composite.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DOMAINS = ("terminal", "highnoise", "odshift", "rushshift")
METRICS = ("wait", "cv", "overshoot", "composite")

DIAGNOSTIC_FEATURE_CANDIDATES = (
    "upper_hf_power_ratio",
    "lower_lf_drift_ratio",
    "demand_attr_score",
    "lower_action_mean",
    "theta_wait",
    "theta_fleet",
    "theta_cv",
    "w_wait",
    "w_fleet",
    "w_cv",
    "upper_delta_mean",
    "upper_delta_std",
    "upper_delta_min",
    "upper_delta_max",
    "freq_low_demand",
    "freq_low_forecast",
    "freq_high_energy",
    "freq_middle",
    "freq_middle_energy",
    "freq_od_entropy",
    "freq_od_high_energy",
    "freq_od_active",
    "freq_promotion_strength",
    "freq_promotion_age",
    "freq_promotion_score",
    "freq_promotion_active",
    "freq_promotion_persistent",
    "freq_promotion_ratio",
    "freq_promotion_absorptions",
    "freq_promotion_absorbed",
    "demand_attr_mi_score",
    "demand_attr_mi_upper_low",
    "demand_attr_mi_upper_high",
    "demand_attr_mi_lower_high",
    "demand_attr_mi_lower_low",
    "shock_response_hit_rate",
    "shock_events",
    "shock_action_mean_s",
    "lower_drift_penalty_mean",
    "lower_drift_cost_mean",
    "lower_drift_cost_adaptive_gate_mean",
    "upper_hf_penalty_mean",
    "freq_wait_low_share_mean",
    "freq_wait_lower_high_share_mean",
    "freq_wait_lower_raw_credit_weight_mean",
    "terminal_shift_cap_mean",
    "terminal_shift_cap_max",
    "upper_plan_penalty_mean",
    "upper_plan_penalty_max",
    "upper_plan_target_mean",
    "upper_plan_target_std",
    "upper_plan_decisions",
    "upper_plan_reuse_ratio",
    "terminal_launch_shift_mean",
    "terminal_launch_shift_std",
)

DOMAIN_PRIOR_FEATURES = (
    "domain_is_terminal",
    "domain_is_highnoise",
    "domain_is_odshift",
    "domain_is_rushshift",
    "scenario_demand_noise",
    "scenario_od_shift",
    "scenario_rush_shift",
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    path: Path
    method_filter: str | None = None


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def infer_domain(config: object) -> str:
    text = str(config)
    if "_gen_highnoise_" in text or "highnoise" in text:
        return "highnoise"
    if "_gen_odshift_" in text or "odshift" in text:
        return "odshift"
    if "_gen_rushshift_" in text or "rushshift" in text:
        return "rushshift"
    if "_terminal_" in text or "terminal" in text:
        return "terminal"
    return "unknown"


def parse_candidate(text: str) -> CandidateSpec:
    parts = text.split(":")
    if len(parts) < 2 or len(parts) > 3:
        raise argparse.ArgumentTypeError(
            "--candidate must use name:path or name:path:method_filter"
        )
    name, path = parts[0].strip(), parts[1].strip()
    method_filter = parts[2].strip() if len(parts) == 3 and parts[2].strip() else None
    if not name:
        raise argparse.ArgumentTypeError("candidate name is empty")
    if not path:
        raise argparse.ArgumentTypeError(f"candidate {name!r} path is empty")
    return CandidateSpec(name=name, path=Path(path), method_filter=method_filter)


def add_domain_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "domain" not in out.columns:
        if "config" not in out.columns:
            raise SystemExit("missing both domain and config columns")
        out["domain"] = out["config"].map(infer_domain)
    else:
        out["domain"] = out["domain"].astype(str).map(
            lambda value: value if value in DOMAINS else infer_domain(value)
        )
    out = out[out["domain"].isin(DOMAINS)].copy()
    for domain in DOMAINS:
        out[f"domain_is_{domain}"] = (out["domain"] == domain).astype(float)
    out["scenario_demand_noise"] = np.select(
        [out["domain"] == "highnoise", out["domain"] == "odshift"],
        [0.30, 0.15],
        default=0.0,
    )
    out["scenario_od_shift"] = (out["domain"] == "odshift").astype(float)
    out["scenario_rush_shift"] = (out["domain"] == "rushshift").astype(float)
    return out


def read_context(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"reference per-seed file does not exist: {path}")
    df = add_domain_columns(pd.read_csv(path))
    required = {"domain", "seed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"reference file missing columns: {missing}")
    dup = df.duplicated(["domain", "seed"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["domain", "seed", "config"]].head(20)
        raise SystemExit(
            "reference file has duplicate domain/seed rows:\n"
            f"{bad.to_string(index=False)}"
        )
    return df.copy()


def read_candidate(spec: CandidateSpec, metrics: Iterable[str]) -> pd.DataFrame:
    if not spec.path.exists():
        raise SystemExit(f"candidate {spec.name!r} file does not exist: {spec.path}")
    df = add_domain_columns(pd.read_csv(spec.path))
    required = {"domain", "seed", *metrics}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"candidate {spec.name!r} missing columns: {missing}")
    if spec.method_filter:
        if "method" in df.columns:
            df = df[df["method"].astype(str) == spec.method_filter].copy()
        elif "config" in df.columns:
            df = df[df["config"].astype(str).str.contains(spec.method_filter, regex=False)].copy()
        else:
            raise SystemExit(
                f"candidate {spec.name!r} has no method/config column for filter "
                f"{spec.method_filter!r}"
            )
        if df.empty:
            raise SystemExit(
                f"candidate {spec.name!r} has no rows after method filter "
                f"{spec.method_filter!r}"
            )
    dup = df.duplicated(["domain", "seed"], keep=False)
    if dup.any():
        cols = [col for col in ("domain", "seed", "method", "config") if col in df.columns]
        bad = df.loc[dup, cols].sort_values(["domain", "seed"]).head(20)
        raise SystemExit(
            f"candidate {spec.name!r} has duplicate domain/seed rows:\n"
            f"{bad.to_string(index=False)}"
        )
    keep = ["domain", "seed", *metrics]
    out = df[keep].copy()
    out["candidate_method"] = spec.name
    return out


def numeric_features(context: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    available = []
    for col in feature_cols:
        if col not in context.columns:
            continue
        values = pd.to_numeric(context[col], errors="coerce")
        if values.notna().any():
            context[col] = values
            available.append(col)
    if not available:
        raise SystemExit("no usable feature columns found in context rows")
    return context, available


def complete_aligned_context(context: pd.DataFrame, candidates: list[pd.DataFrame]) -> pd.DataFrame:
    key = context[["domain", "seed"]].drop_duplicates()
    common = key.copy()
    for cand in candidates:
        common = common.merge(cand[["domain", "seed"]].drop_duplicates(), on=["domain", "seed"])
    return context.merge(common, on=["domain", "seed"]).copy()


def build_long_and_wide(
    context: pd.DataFrame,
    candidates: list[pd.DataFrame],
    metric: str,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_key = context[["domain", "seed"]].drop_duplicates()
    long_parts = []
    for cand in candidates:
        aligned = context_key.merge(cand, on=["domain", "seed"], how="inner")
        long_parts.append(aligned)
    long_df = pd.concat(long_parts, ignore_index=True)
    long_df["cost"] = pd.to_numeric(long_df[metric], errors="coerce")
    cost_wide = long_df.pivot(index=["domain", "seed"], columns="candidate_method", values="cost")
    metric_wides = []
    for name in metrics:
        wide = long_df.pivot(index=["domain", "seed"], columns="candidate_method", values=name)
        wide.columns = [f"{name}_{col}" for col in wide.columns]
        metric_wides.append(wide)
    wide_df = pd.concat([cost_wide.add_prefix("cost_"), *metric_wides], axis=1).reset_index()
    cost_cols = [col for col in wide_df.columns if col.startswith("cost_")]
    if wide_df[cost_cols].isna().any().any():
        raise SystemExit("aligned candidate matrix still has missing costs")
    costs = wide_df[cost_cols]
    wide_df["oracle_best_method"] = costs.idxmin(axis=1).str.replace("cost_", "", regex=False)
    wide_df["oracle_best_cost"] = costs.min(axis=1)
    for col in cost_cols:
        method = col.removeprefix("cost_")
        wide_df[f"regret_{method}"] = wide_df[col] - wide_df["oracle_best_cost"]
    wide_df["n_candidates"] = len(cost_cols)
    return long_df, wide_df


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


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
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
    penalty = np.eye(train_design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = train_design.T @ train_design + float(alpha) * penalty
    rhs = train_design.T @ y_train
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    return test_design @ beta


def seed_folds(seeds: list[int], n_folds: int) -> dict[int, int]:
    ordered = sorted(int(seed) for seed in seeds)
    return {seed: idx % n_folds for idx, seed in enumerate(ordered)}


def run_cv(
    context: pd.DataFrame,
    wide: pd.DataFrame,
    candidate_methods: list[str],
    feature_cols: list[str],
    metric: str,
    n_folds: int,
    alpha: float,
    feature_set: str,
) -> pd.DataFrame:
    data = context.merge(wide, on=["domain", "seed"], how="inner")
    if data.empty:
        raise SystemExit("no aligned rows for CV")
    n_folds = max(2, min(int(n_folds), int(data["seed"].nunique())))
    folds = seed_folds(data["seed"].unique().tolist(), n_folds)
    data["_fold"] = data["seed"].astype(int).map(folds)
    rows = []
    x_all = data[feature_cols].to_numpy(dtype=np.float64)
    for fold in range(n_folds):
        train_mask = data["_fold"].to_numpy() != fold
        test_mask = ~train_mask
        if not test_mask.any() or not train_mask.any():
            continue
        pred_cols = {}
        for method in candidate_methods:
            y_train = data.loc[train_mask, f"cost_{method}"].to_numpy(dtype=np.float64)
            pred_cols[method] = ridge_fit_predict(
                x_all[train_mask],
                y_train,
                x_all[test_mask],
                alpha=alpha,
            )
        pred_matrix = np.column_stack([pred_cols[method] for method in candidate_methods])
        selected_idx = np.argmin(pred_matrix, axis=1)
        test_data = data.loc[test_mask].reset_index(drop=True)
        for row_idx, method_idx in enumerate(selected_idx):
            method = candidate_methods[int(method_idx)]
            out = {
                "feature_set": feature_set,
                "fold": fold,
                "domain": test_data.loc[row_idx, "domain"],
                "seed": int(test_data.loc[row_idx, "seed"]),
                "selected_method": method,
                "selected_pred_cost": float(pred_matrix[row_idx, method_idx]),
                "selected_cost": float(test_data.loc[row_idx, f"cost_{method}"]),
                "oracle_best_method": test_data.loc[row_idx, "oracle_best_method"],
                "oracle_best_cost": float(test_data.loc[row_idx, "oracle_best_cost"]),
                "oracle_regret": float(
                    test_data.loc[row_idx, f"cost_{method}"]
                    - test_data.loc[row_idx, "oracle_best_cost"]
                ),
            }
            for base in ("main", "fixed", "fixed_headway"):
                if f"cost_{base}" in test_data.columns:
                    out[f"delta_vs_{base}"] = float(
                        test_data.loc[row_idx, f"cost_{method}"]
                        - test_data.loc[row_idx, f"cost_{base}"]
                    )
            for candidate in candidate_methods:
                out[f"pred_cost_{candidate}"] = float(pred_matrix[row_idx, candidate_methods.index(candidate)])
                out[f"true_cost_{candidate}"] = float(test_data.loc[row_idx, f"cost_{candidate}"])
            rows.append(out)
    return pd.DataFrame(rows)


def run_mean_selector_cv(
    context: pd.DataFrame,
    wide: pd.DataFrame,
    candidate_methods: list[str],
    n_folds: int,
    selector: str,
) -> pd.DataFrame:
    data = context[["domain", "seed"]].merge(wide, on=["domain", "seed"], how="inner")
    n_folds = max(2, min(int(n_folds), int(data["seed"].nunique())))
    folds = seed_folds(data["seed"].unique().tolist(), n_folds)
    data["_fold"] = data["seed"].astype(int).map(folds)
    rows = []
    for fold in range(n_folds):
        train = data[data["_fold"] != fold].copy()
        test = data[data["_fold"] == fold].copy().reset_index(drop=True)
        if train.empty or test.empty:
            continue
        global_means = {method: float(train[f"cost_{method}"].mean()) for method in candidate_methods}
        global_choice = min(candidate_methods, key=lambda method: global_means[method])
        domain_choices = {}
        for domain, domain_train in train.groupby("domain"):
            means = {method: float(domain_train[f"cost_{method}"].mean()) for method in candidate_methods}
            domain_choices[domain] = min(candidate_methods, key=lambda method: means[method])
        for row_idx, item in test.iterrows():
            if selector == "global_mean":
                method = global_choice
            elif selector == "domain_mean":
                method = domain_choices.get(item["domain"], global_choice)
            else:
                raise ValueError(selector)
            out = {
                "feature_set": selector,
                "fold": fold,
                "domain": item["domain"],
                "seed": int(item["seed"]),
                "selected_method": method,
                "selected_pred_cost": global_means[method],
                "selected_cost": float(item[f"cost_{method}"]),
                "oracle_best_method": item["oracle_best_method"],
                "oracle_best_cost": float(item["oracle_best_cost"]),
                "oracle_regret": float(item[f"cost_{method}"] - item["oracle_best_cost"]),
            }
            for base in ("main", "fixed", "fixed_headway"):
                if f"cost_{base}" in item.index:
                    out[f"delta_vs_{base}"] = float(item[f"cost_{method}"] - item[f"cost_{base}"])
            rows.append(out)
    return pd.DataFrame(rows)


def derive_margin_fallback_rows(
    cv_rows: pd.DataFrame,
    candidate_methods: list[str],
    fallback_method: str,
    margins: list[float],
) -> pd.DataFrame:
    if fallback_method not in candidate_methods:
        return pd.DataFrame()
    rows = []
    base_rows = cv_rows[
        cv_rows["feature_set"].isin(["diagnostic", "diagnostic_domain_prior"])
    ].copy()
    pred_cols = {method: f"pred_cost_{method}" for method in candidate_methods}
    true_cols = {method: f"true_cost_{method}" for method in candidate_methods}
    for _, item in base_rows.iterrows():
        pred_values = {method: float(item[pred_cols[method]]) for method in candidate_methods}
        best_method = min(candidate_methods, key=lambda method: pred_values[method])
        fallback_pred = pred_values[fallback_method]
        for margin in margins:
            if best_method != fallback_method and (fallback_pred - pred_values[best_method]) > margin:
                method = best_method
            else:
                method = fallback_method
            out = {
                "feature_set": f"{item['feature_set']}_fallback_{fallback_method}_m{margin:.3f}",
                "fold": int(item["fold"]),
                "domain": item["domain"],
                "seed": int(item["seed"]),
                "selected_method": method,
                "selected_pred_cost": pred_values[method],
                "selected_cost": float(item[true_cols[method]]),
                "oracle_best_method": item["oracle_best_method"],
                "oracle_best_cost": float(item["oracle_best_cost"]),
                "oracle_regret": float(item[true_cols[method]] - item["oracle_best_cost"]),
            }
            for base in ("main", "fixed", "fixed_headway"):
                col = true_cols.get(base)
                if col and col in item.index:
                    out[f"delta_vs_{base}"] = float(item[true_cols[method]] - item[col])
            rows.append(out)
    return pd.DataFrame(rows)


def summarize_selector(rows: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    summaries = []
    domains = list(DOMAINS) + ["overall_seed_mean"]
    feature_sets = sorted(rows["feature_set"].unique())
    for feature_set in feature_sets:
        subset_feature = rows[rows["feature_set"] == feature_set].copy()
        for domain in domains:
            if domain == "overall_seed_mean":
                group = (
                    subset_feature.groupby("seed", as_index=False)
                    .agg(
                        selected_cost=("selected_cost", "mean"),
                        oracle_best_cost=("oracle_best_cost", "mean"),
                        oracle_regret=("oracle_regret", "mean"),
                        **{
                            col: (col, "mean")
                            for col in subset_feature.columns
                            if col.startswith("delta_vs_")
                        },
                    )
                )
            else:
                group = subset_feature[subset_feature["domain"] == domain].copy()
            if group.empty:
                continue
            row = {
                "feature_set": feature_set,
                "domain": domain,
                "n_rows": int(len(group)),
                "selected_cost_mean": float(group["selected_cost"].mean()),
                "oracle_best_cost_mean": float(group["oracle_best_cost"].mean()),
                "oracle_regret_mean": float(group["oracle_regret"].mean()),
                "oracle_regret_win_rate": float((group["oracle_regret"] <= 1e-12).mean()),
            }
            lo, hi = bootstrap_ci(
                group["oracle_regret"].to_numpy(),
                n_boot=n_boot,
                seed=stable_seed("selector", feature_set, domain, "oracle_regret"),
            )
            row["oracle_regret_ci95_lo"] = lo
            row["oracle_regret_ci95_hi"] = hi
            for col in [col for col in group.columns if col.startswith("delta_vs_")]:
                delta = group[col].to_numpy(dtype=np.float64)
                lo, hi = bootstrap_ci(
                    delta,
                    n_boot=n_boot,
                    seed=stable_seed("selector", feature_set, domain, col),
                )
                base = col.removeprefix("delta_vs_")
                row[f"delta_vs_{base}_mean"] = float(np.nanmean(delta))
                row[f"delta_vs_{base}_ci95_lo"] = lo
                row[f"delta_vs_{base}_ci95_hi"] = hi
                row[f"win_vs_{base}_rate"] = float((delta < 0.0).mean())
            if domain == "overall_seed_mean":
                choices = subset_feature["selected_method"].value_counts().sort_index().to_dict()
            else:
                choices = group["selected_method"].value_counts().sort_index().to_dict()
            row["selected_method_counts_json"] = json.dumps(choices, sort_keys=True)
            summaries.append(row)
    return pd.DataFrame(summaries)


def summarize_candidates(wide: pd.DataFrame, candidate_methods: list[str], n_boot: int) -> pd.DataFrame:
    rows = []
    for domain in list(DOMAINS) + ["overall_seed_mean"]:
        if domain == "overall_seed_mean":
            group = (
                wide.groupby("seed", as_index=False)
                .agg(
                    oracle_best_cost=("oracle_best_cost", "mean"),
                    **{f"cost_{method}": (f"cost_{method}", "mean") for method in candidate_methods},
                )
            )
        else:
            group = wide[wide["domain"] == domain].copy()
        if group.empty:
            continue
        for method in candidate_methods:
            row = {
                "domain": domain,
                "method": method,
                "n_rows": int(len(group)),
                "cost_mean": float(group[f"cost_{method}"].mean()),
                "regret_to_oracle_mean": float((group[f"cost_{method}"] - group["oracle_best_cost"]).mean()),
            }
            if "cost_main" in group.columns and method != "main":
                delta = group[f"cost_{method}"] - group["cost_main"]
                lo, hi = bootstrap_ci(
                    delta.to_numpy(),
                    n_boot=n_boot,
                    seed=stable_seed("candidate", domain, method, "main"),
                )
                row["delta_vs_main_mean"] = float(delta.mean())
                row["delta_vs_main_ci95_lo"] = lo
                row["delta_vs_main_ci95_hi"] = hi
            if "cost_fixed" in group.columns and method != "fixed":
                delta = group[f"cost_{method}"] - group["cost_fixed"]
                lo, hi = bootstrap_ci(
                    delta.to_numpy(),
                    n_boot=n_boot,
                    seed=stable_seed("candidate", domain, method, "fixed"),
                )
                row["delta_vs_fixed_mean"] = float(delta.mean())
                row["delta_vs_fixed_ci95_lo"] = lo
                row["delta_vs_fixed_ci95_hi"] = hi
            rows.append(row)
        rows.append(
            {
                "domain": domain,
                "method": "oracle_best",
                "n_rows": int(len(group)),
                "cost_mean": float(group["oracle_best_cost"].mean()),
                "regret_to_oracle_mean": 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reference-per-seed",
        type=Path,
        default=Path("results_freqduet/current_main_ep100_reference_only.csv"),
    )
    ap.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        help="Candidate as name:path or name:path:method_filter.",
    )
    ap.add_argument("--metric", default="composite", choices=METRICS)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fallback-method", default="fixed")
    ap.add_argument("--fallback-margins", default="0,0.025,0.05,0.10")
    ap.add_argument(
        "--extra-feature",
        action="append",
        default=[],
        help="Additional reference context feature column; may be repeated.",
    )
    args = ap.parse_args()

    metrics = list(METRICS)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    context = read_context(args.reference_per_seed)
    feature_candidates = list(DIAGNOSTIC_FEATURE_CANDIDATES) + list(args.extra_feature)
    context, diagnostic_features = numeric_features(context, feature_candidates)
    _, domain_prior_features = numeric_features(context, list(DOMAIN_PRIOR_FEATURES))
    candidates = [read_candidate(spec, metrics) for spec in args.candidate]
    context = complete_aligned_context(context, candidates)
    if context.empty:
        raise SystemExit("no common domain/seed rows across reference and candidates")
    long_df, wide_df = build_long_and_wide(context, candidates, args.metric, metrics)
    candidate_methods = sorted(long_df["candidate_method"].unique())

    context_keep = [
        "domain",
        "seed",
        "config",
        *diagnostic_features,
        *domain_prior_features,
    ]
    context_keep = [col for col in context_keep if col in context.columns]
    context_rows = context[context_keep].copy()

    long_df.to_csv(out_dir / "counterfactual_candidate_rows.csv", index=False)
    wide_df.to_csv(out_dir / "counterfactual_context_rows.csv", index=False)
    context_rows.to_csv(out_dir / "counterfactual_reference_context.csv", index=False)

    candidate_summary = summarize_candidates(wide_df, candidate_methods, args.n_boot)
    candidate_summary.to_csv(out_dir / "counterfactual_candidate_summary.csv", index=False)

    cv_parts = []
    cv_specs = [
        ("diagnostic", diagnostic_features),
        ("diagnostic_domain_prior", diagnostic_features + domain_prior_features),
    ]
    for feature_set, features in cv_specs:
        cv = run_cv(
            context_rows,
            wide_df,
            candidate_methods,
            features,
            metric=args.metric,
            n_folds=args.folds,
            alpha=args.ridge_alpha,
            feature_set=feature_set,
        )
        cv_parts.append(cv)
    cv_rows = pd.concat(cv_parts, ignore_index=True)
    for selector in ("global_mean", "domain_mean"):
        cv_parts.append(
            run_mean_selector_cv(
                context_rows,
                wide_df,
                candidate_methods,
                n_folds=args.folds,
                selector=selector,
            )
        )
    margins = [
        float(part.strip())
        for part in str(args.fallback_margins).split(",")
        if part.strip()
    ]
    cv_parts.append(
        derive_margin_fallback_rows(
            cv_rows,
            candidate_methods,
            fallback_method=args.fallback_method,
            margins=margins,
        )
    )
    cv_rows = pd.concat([part for part in cv_parts if not part.empty], ignore_index=True)
    cv_rows.to_csv(out_dir / "counterfactual_value_cv_rows.csv", index=False)
    cv_summary = summarize_selector(cv_rows, args.n_boot)
    cv_summary.to_csv(out_dir / "counterfactual_value_cv_summary.csv", index=False)

    oracle_counts = (
        wide_df.groupby(["domain", "oracle_best_method"]).size().unstack(fill_value=0).to_dict()
    )
    payload = {
        "metric": args.metric,
        "reference_per_seed": str(args.reference_per_seed),
        "candidates": [
            {
                "name": spec.name,
                "path": str(spec.path),
                "method_filter": spec.method_filter,
            }
            for spec in args.candidate
        ],
        "n_context_rows": int(len(context_rows)),
        "domains": {domain: int((context_rows["domain"] == domain).sum()) for domain in DOMAINS},
        "candidate_methods": candidate_methods,
        "diagnostic_features": diagnostic_features,
        "domain_prior_features": domain_prior_features,
        "oracle_best_counts_by_domain": oracle_counts,
        "ridge_alpha": float(args.ridge_alpha),
        "folds": int(args.folds),
        "n_boot": int(args.n_boot),
        "limitation": (
            "Labels are matched-seed rollout costs from completed runs. They are stronger "
            "than same-trajectory proxy labels, but not yet common-random-number action "
            "replay at every decision point."
        ),
    }
    write_json(out_dir / "counterfactual_value_summary.json", payload)

    print(f"wrote {out_dir}")
    print(f"context rows: {len(context_rows)}")
    print(f"candidates: {', '.join(candidate_methods)}")
    print(f"diagnostic features: {len(diagnostic_features)}")
    print("overall selector summary:")
    overall = cv_summary[cv_summary["domain"] == "overall_seed_mean"].copy()
    cols = [
        "feature_set",
        "selected_cost_mean",
        "oracle_best_cost_mean",
        "oracle_regret_mean",
        "delta_vs_main_mean",
        "delta_vs_fixed_mean",
        "selected_method_counts_json",
    ]
    cols = [col for col in cols if col in overall.columns]
    print(overall[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
