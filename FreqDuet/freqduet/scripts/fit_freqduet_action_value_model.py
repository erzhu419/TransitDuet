#!/usr/bin/env python3
"""Fit offline action-conditioned value models on FreqDuet CRN labels.

This is a diagnostic/training bridge between matched rollout labels and an
online value selector. Unlike the earlier per-candidate ridge audit, this model
learns a single value function V(context, action), with explicit action
features and optional context-action interactions.
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("terminal", "highnoise", "odshift", "rushshift")
ACTION_RE = re.compile(r"^(?P<mode>target|term45)_(?P<delta>m20|0|p20)$")
DEFAULT_BASELINE_METHOD = "target0"
DEFAULT_METRIC = "composite"


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def seed_folds(seeds: list[int], n_folds: int) -> dict[int, int]:
    ordered = sorted(int(seed) for seed in seeds)
    return {seed: idx % n_folds for idx, seed in enumerate(ordered)}


def parse_action(method: str) -> dict[str, float | str]:
    text = str(method)
    if text == "target0":
        mode = "target"
        delta_token = "0"
    else:
        match = ACTION_RE.match(text)
        if not match:
            raise SystemExit(
                f"candidate_method {method!r} does not match target_m20/target0/"
                "target_p20/term45_m20/term45_0/term45_p20"
            )
        mode = match.group("mode")
        delta_token = match.group("delta")
    delta_s = {"m20": -20.0, "0": 0.0, "p20": 20.0}[delta_token]
    return {
        "action_mode": mode,
        "action_delta_s": delta_s,
        "action_delta_norm": delta_s / 20.0,
        "action_abs_delta_norm": abs(delta_s) / 20.0,
        "action_positive": 1.0 if delta_s > 0 else 0.0,
        "action_negative": 1.0 if delta_s < 0 else 0.0,
        "action_zero": 1.0 if abs(delta_s) < 1e-9 else 0.0,
        "action_term45": 1.0 if mode == "term45" else 0.0,
        "action_target": 1.0 if mode == "target" else 0.0,
        "action_term45_x_delta": (1.0 if mode == "term45" else 0.0) * delta_s / 20.0,
        "action_term45_x_abs_delta": (1.0 if mode == "term45" else 0.0) * abs(delta_s) / 20.0,
    }


def read_inputs(candidate_rows: Path, context_rows: Path, metric: str) -> pd.DataFrame:
    cand = pd.read_csv(candidate_rows)
    ctx = pd.read_csv(context_rows)
    required_cand = {"domain", "seed", "candidate_method", metric}
    missing = sorted(required_cand - set(cand.columns))
    if missing:
        raise SystemExit(f"candidate rows missing columns: {missing}")
    required_ctx = {"domain", "seed"}
    missing = sorted(required_ctx - set(ctx.columns))
    if missing:
        raise SystemExit(f"context rows missing columns: {missing}")
    action = pd.DataFrame([parse_action(method) for method in cand["candidate_method"]])
    cand = pd.concat([cand.reset_index(drop=True), action], axis=1)
    cand["cost"] = pd.to_numeric(cand[metric], errors="coerce")
    data = cand.merge(ctx, on=["domain", "seed"], how="inner", suffixes=("", "_ctx"))
    if data.empty:
        raise SystemExit("candidate/context merge is empty")
    data = data[data["domain"].isin(DOMAINS)].copy()
    data["seed"] = data["seed"].astype(int)
    return data


def context_feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {
        "domain",
        "seed",
        "config",
        "candidate_method",
        "action_mode",
        "cost",
        "wait",
        "cv",
        "overshoot",
        "composite",
    }
    action_cols = {col for col in data.columns if col.startswith("action_")}
    excluded |= action_cols
    cols = []
    for col in data.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(data[col], errors="coerce")
        if values.notna().any():
            data[col] = values
            cols.append(col)
    return cols


def add_design_features(data: pd.DataFrame, context_cols: list[str], feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    out = data.copy()
    action_cols = [
        "action_delta_norm",
        "action_abs_delta_norm",
        "action_positive",
        "action_negative",
        "action_zero",
        "action_term45",
        "action_target",
        "action_term45_x_delta",
        "action_term45_x_abs_delta",
    ]
    optional_action_cols = [
        "candidate_offset_norm",
        "candidate_abs_offset_norm",
        "candidate_above_actor",
        "candidate_below_actor",
        "candidate_same_as_actor",
        "action_delta_minus_actor_norm",
        "action_abs_delta_minus_actor_norm",
        "action_term45_x_offset",
        "action_term45_x_abs_offset",
    ]
    action_cols.extend([col for col in optional_action_cols if col in out.columns])
    derived: dict[str, pd.Series] = {}
    for domain in DOMAINS:
        col = f"domain_is_{domain}"
        if col not in out.columns:
            derived[col] = (out["domain"] == domain).astype(float)
    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)
        derived = {}
    domain_cols = [f"domain_is_{domain}" for domain in DOMAINS]
    feature_cols = [*action_cols, *domain_cols]
    for domain_col in domain_cols:
        for action_col in action_cols:
            name = f"{domain_col}_x_{action_col}"
            derived[name] = out[domain_col].astype(float) * out[action_col].astype(float)
            feature_cols.append(name)

    if feature_set in {"context_action", "context_action_interact"}:
        feature_cols.extend(context_cols)
    if feature_set == "context_action_interact":
        for ctx_col in context_cols:
            for action_col in ("action_delta_norm", "action_abs_delta_norm", "action_term45"):
                name = f"{ctx_col}_x_{action_col}"
                derived[name] = (
                    pd.to_numeric(out[ctx_col], errors="coerce")
                    * out[action_col].astype(float)
                )
                feature_cols.append(name)
    if feature_set == "action_domain":
        pass
    elif feature_set not in {"context_action", "context_action_interact"}:
        raise SystemExit(f"unknown feature set: {feature_set}")
    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)
    return out, feature_cols


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


def select_from_predictions(
    test_rows: pd.DataFrame,
    predictions: np.ndarray,
    feature_set: str,
    fold: int,
    baseline_method: str,
) -> list[dict[str, object]]:
    pred = test_rows[["domain", "seed", "candidate_method", "cost"]].copy()
    pred["pred_cost"] = predictions
    rows: list[dict[str, object]] = []
    for (domain, seed), group in pred.groupby(["domain", "seed"], sort=False):
        group = group.copy()
        selected = group.loc[group["pred_cost"].idxmin()]
        oracle = group.loc[group["cost"].idxmin()]
        baseline = group[group["candidate_method"] == baseline_method]
        baseline_cost = float(baseline["cost"].iloc[0]) if not baseline.empty else np.nan
        rows.append({
            "feature_set": feature_set,
            "fold": int(fold),
            "domain": domain,
            "seed": int(seed),
            "selected_method": str(selected["candidate_method"]),
            "selected_pred_cost": float(selected["pred_cost"]),
            "selected_cost": float(selected["cost"]),
            "oracle_best_method": str(oracle["candidate_method"]),
            "oracle_best_cost": float(oracle["cost"]),
            "oracle_regret": float(selected["cost"] - oracle["cost"]),
            f"delta_vs_{baseline_method}": float(selected["cost"] - baseline_cost),
            "baseline_cost": baseline_cost,
        })
    return rows


def run_model_cv(
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
    rows: list[dict[str, object]] = []
    x = design[feature_cols].to_numpy(dtype=np.float64)
    y = design["cost"].to_numpy(dtype=np.float64)
    for fold in range(folds):
        train_mask = design["_fold"].to_numpy() != fold
        test_mask = ~train_mask
        if not train_mask.any() or not test_mask.any():
            continue
        pred = ridge_fit_predict(x[train_mask], y[train_mask], x[test_mask], alpha=alpha)
        rows.extend(select_from_predictions(
            design.loc[test_mask].reset_index(drop=True),
            pred,
            feature_set=feature_set,
            fold=fold,
            baseline_method=baseline_method,
        ))
    return pd.DataFrame(rows), len(feature_cols)


def run_mean_cv(data: pd.DataFrame, selector: str, folds: int, baseline_method: str) -> pd.DataFrame:
    folds = max(2, min(int(folds), int(data["seed"].nunique())))
    fold_map = seed_folds(data["seed"].unique().tolist(), folds)
    working = data.copy()
    working["_fold"] = working["seed"].map(fold_map)
    rows: list[dict[str, object]] = []
    for fold in range(folds):
        train = working[working["_fold"] != fold].copy()
        test = working[working["_fold"] == fold].copy()
        if train.empty or test.empty:
            continue
        if selector == "global_action_mean":
            means = train.groupby("candidate_method")["cost"].mean().to_dict()
            test_pred = test["candidate_method"].map(means).fillna(train["cost"].mean()).to_numpy(dtype=np.float64)
        elif selector == "domain_action_mean":
            means = train.groupby(["domain", "candidate_method"])["cost"].mean().to_dict()
            global_means = train.groupby("candidate_method")["cost"].mean().to_dict()
            test_pred = []
            for _, row in test.iterrows():
                test_pred.append(means.get((row["domain"], row["candidate_method"]), global_means.get(row["candidate_method"], train["cost"].mean())))
            test_pred = np.asarray(test_pred, dtype=np.float64)
        else:
            raise SystemExit(f"unknown selector: {selector}")
        rows.extend(select_from_predictions(
            test.reset_index(drop=True),
            test_pred,
            feature_set=selector,
            fold=fold,
            baseline_method=baseline_method,
        ))
    return pd.DataFrame(rows)


def summarize_rows(rows: pd.DataFrame, baseline_method: str, n_boot: int) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    domains = list(DOMAINS) + ["overall_seed_mean"]
    delta_col = f"delta_vs_{baseline_method}"
    for feature_set in sorted(rows["feature_set"].unique()):
        sub = rows[rows["feature_set"] == feature_set].copy()
        for domain in domains:
            if domain == "overall_seed_mean":
                group = (
                    sub.groupby("seed", as_index=False)
                    .agg(
                        selected_cost=("selected_cost", "mean"),
                        oracle_best_cost=("oracle_best_cost", "mean"),
                        oracle_regret=("oracle_regret", "mean"),
                        **{delta_col: (delta_col, "mean")},
                    )
                )
                choices = sub["selected_method"].value_counts().sort_index().to_dict()
            else:
                group = sub[sub["domain"] == domain].copy()
                choices = group["selected_method"].value_counts().sort_index().to_dict()
            if group.empty:
                continue
            row = {
                "feature_set": feature_set,
                "domain": domain,
                "n_rows": int(len(group)),
                "selected_cost_mean": float(group["selected_cost"].mean()),
                "oracle_best_cost_mean": float(group["oracle_best_cost"].mean()),
                "oracle_regret_mean": float(group["oracle_regret"].mean()),
                "selected_method_counts_json": json.dumps(choices, sort_keys=True),
            }
            lo, hi = bootstrap_ci(
                group["oracle_regret"].to_numpy(dtype=np.float64),
                n_boot=n_boot,
                seed=stable_seed("regret", feature_set, domain),
            )
            row["oracle_regret_ci95_lo"] = lo
            row["oracle_regret_ci95_hi"] = hi
            delta = group[delta_col].to_numpy(dtype=np.float64)
            lo, hi = bootstrap_ci(
                delta,
                n_boot=n_boot,
                seed=stable_seed("delta", feature_set, domain, baseline_method),
            )
            row[f"delta_vs_{baseline_method}_mean"] = float(np.nanmean(delta))
            row[f"delta_vs_{baseline_method}_ci95_lo"] = lo
            row[f"delta_vs_{baseline_method}_ci95_hi"] = hi
            row[f"win_vs_{baseline_method}_rate"] = float((delta < 0.0).mean())
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize_candidates(data: pd.DataFrame, baseline_method: str, n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    wide = data.pivot_table(index=["domain", "seed"], columns="candidate_method", values="cost", aggfunc="mean")
    wide["oracle_best_cost"] = wide.min(axis=1)
    domains = list(DOMAINS) + ["overall_seed_mean"]
    methods = sorted(data["candidate_method"].unique())
    for domain in domains:
        if domain == "overall_seed_mean":
            group = wide.reset_index().groupby("seed", as_index=False).mean(numeric_only=True)
        else:
            group = wide.reset_index()
            group = group[group["domain"] == domain]
        if group.empty:
            continue
        for method in methods:
            if method not in group.columns:
                continue
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
                    seed=stable_seed("candidate", domain, method, baseline_method),
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


def print_compact(summary: pd.DataFrame, candidate_summary: pd.DataFrame, baseline_method: str) -> None:
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
    print("\n== overall action-value CV summary ==")
    print(overall[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n== candidate costs, overall ==")
    cand = candidate_summary[candidate_summary["domain"] == "overall_seed_mean"].copy()
    cols = ["method", "cost_mean", "regret_to_oracle_mean", delta_col, lo_col, hi_col]
    cols = [col for col in cols if col in cand.columns]
    print(cand[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-rows", type=Path, required=True)
    ap.add_argument("--context-rows", type=Path, required=True)
    ap.add_argument("--metric", default=DEFAULT_METRIC)
    ap.add_argument("--baseline-method", default=DEFAULT_BASELINE_METHOD)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge-alpha", type=float, default=5.0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data = read_inputs(args.candidate_rows, args.context_rows, args.metric)
    if args.baseline_method not in set(data["candidate_method"]):
        raise SystemExit(f"baseline method {args.baseline_method!r} not in candidate rows")
    context_cols = context_feature_columns(data)

    cv_parts = []
    feature_counts = {}
    for feature_set in ("action_domain", "context_action", "context_action_interact"):
        rows, n_features = run_model_cv(
            data,
            feature_set=feature_set,
            context_cols=context_cols,
            folds=args.folds,
            alpha=args.ridge_alpha,
            baseline_method=args.baseline_method,
        )
        cv_parts.append(rows)
        feature_counts[feature_set] = n_features
    for selector in ("global_action_mean", "domain_action_mean"):
        cv_parts.append(run_mean_cv(data, selector=selector, folds=args.folds, baseline_method=args.baseline_method))
    cv_rows = pd.concat(cv_parts, ignore_index=True)
    summary = summarize_rows(cv_rows, args.baseline_method, args.n_boot)
    candidate_summary = summarize_candidates(data, args.baseline_method, args.n_boot)

    cv_rows.to_csv(out_dir / "action_value_model_cv_rows.csv", index=False)
    summary.to_csv(out_dir / "action_value_model_cv_summary.csv", index=False)
    candidate_summary.to_csv(out_dir / "action_value_model_candidate_summary.csv", index=False)
    payload = {
        "candidate_rows": str(args.candidate_rows),
        "context_rows": str(args.context_rows),
        "metric": args.metric,
        "baseline_method": args.baseline_method,
        "folds": int(args.folds),
        "ridge_alpha": float(args.ridge_alpha),
        "n_boot": int(args.n_boot),
        "n_context_features": int(len(context_cols)),
        "feature_counts": feature_counts,
        "n_cv_rows": int(len(cv_rows)),
        "candidate_methods": sorted(data["candidate_method"].unique()),
        "limitation": (
            "This is an offline seed-held-out value model over aggregate rollout "
            "labels. It is a stronger action-conditioned diagnostic than same-"
            "trajectory online proxy labels, but it is not yet a deployable "
            "per-dispatch estimator."
        ),
    }
    (out_dir / "action_value_model_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_compact(summary, candidate_summary, args.baseline_method)
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
