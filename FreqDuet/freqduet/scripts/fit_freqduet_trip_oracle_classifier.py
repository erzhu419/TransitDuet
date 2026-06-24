#!/usr/bin/env python3
"""Fit a seed-held-out nonlinear selector for trip-level CRN action labels.

The ridge action-value audit expands every context into action rows and then
predicts cost. This script instead predicts the oracle fixed action directly
from the matched trip context. It is a diagnostic bridge for the discrete-action
route: if a nonlinear context -> action classifier cannot beat target0 under
seed-held-out evaluation, the fixed-delta CRN branch should remain diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_freqduet_trip_counterfactual_value_dataset import (
    DEFAULT_CANDIDATES,
    DOMAINS,
    add_context_features,
    bootstrap_ci,
    context_cols,
    find_trip_files,
    read_trip_rows,
    seed_folds,
    stable_seed,
)


def require_sklearn():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:  # pragma: no cover - environment diagnostic.
        raise SystemExit(
            "scikit-learn is required for this audit; install it in the "
            "isolated analysis environment or run on the local workspace Python"
        ) from exc
    return HistGradientBoostingClassifier


def build_context_wide(
    logs_root: list[Path],
    metric: str,
    last_k: int,
    candidates: list[str],
    baseline_method: str,
    higher_is_better: bool,
) -> tuple[pd.DataFrame, list[str]]:
    paths = find_trip_files(logs_root)
    rows = read_trip_rows(paths, metric, last_k, higher_is_better)
    key_cols = ["domain", "seed", "ep", "tid"]
    wide = rows.pivot_table(
        index=key_cols,
        columns="candidate_method",
        values="cost",
        aggfunc="mean",
    ).reset_index()
    missing = sorted(set(candidates) - set(wide.columns))
    if missing:
        raise SystemExit("missing candidate methods in trip rows: " + ", ".join(missing))
    wide = wide.dropna(subset=candidates).copy()
    if wide.empty:
        raise SystemExit("no fully aligned trip rows across candidates")

    baseline = rows[rows["candidate_method"] == baseline_method].copy()
    if baseline.empty:
        raise SystemExit(f"baseline method {baseline_method!r} missing")
    context = baseline.drop_duplicates(key_cols, keep="first").copy()
    context = context.merge(wide[key_cols], on=key_cols, how="inner")
    add_context_features(context)
    features = context_cols(context)
    for domain in DOMAINS:
        name = f"domain_is_{domain}"
        context[name] = (context["domain"] == domain).astype(float)
        features.append(name)
    data = context.merge(wide[key_cols + candidates], on=key_cols, how="inner")
    return data, features


def impute_arrays(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return (
        np.where(np.isfinite(x_train), x_train, med).astype(np.float32, copy=False),
        np.where(np.isfinite(x_test), x_test, med).astype(np.float32, copy=False),
    )


def summarize_rows(rows: pd.DataFrame, baseline_method: str, n_boot: int) -> pd.DataFrame:
    out = []
    delta_col = f"delta_vs_{baseline_method}"
    for domain in [*DOMAINS, "overall_seed_mean"]:
        if domain == "overall_seed_mean":
            group = (
                rows.groupby("seed", as_index=False)
                .agg(
                    selected_cost=("selected_cost", "mean"),
                    oracle_best_cost=("oracle_best_cost", "mean"),
                    oracle_regret=("oracle_regret", "mean"),
                    **{delta_col: (delta_col, "mean")},
                )
            )
            choices = rows["selected_method"].value_counts().sort_index().to_dict()
            n_rows = int(len(group))
        else:
            sub = rows[rows["domain"] == domain].copy()
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
            n_rows = int(len(sub))
        if group.empty:
            continue
        regret = group["oracle_regret"].to_numpy(dtype=np.float64)
        delta = group[delta_col].to_numpy(dtype=np.float64)
        rlo, rhi = bootstrap_ci(
            regret,
            n_boot=n_boot,
            seed=stable_seed("oracle_classifier_regret", domain),
        )
        dlo, dhi = bootstrap_ci(
            delta,
            n_boot=n_boot,
            seed=stable_seed("oracle_classifier_delta", domain, baseline_method),
        )
        out.append({
            "domain": domain,
            "n_rows": n_rows,
            "selected_cost_mean": float(group["selected_cost"].mean()),
            "oracle_best_cost_mean": float(group["oracle_best_cost"].mean()),
            "oracle_regret_mean": float(group["oracle_regret"].mean()),
            "oracle_regret_ci95_lo": rlo,
            "oracle_regret_ci95_hi": rhi,
            f"delta_vs_{baseline_method}_mean": float(np.nanmean(delta)),
            f"delta_vs_{baseline_method}_ci95_lo": dlo,
            f"delta_vs_{baseline_method}_ci95_hi": dhi,
            f"win_vs_{baseline_method}_rate": float((delta < 0.0).mean()),
            "selected_method_counts_json": json.dumps(choices, sort_keys=True),
        })
    return pd.DataFrame(out)


def main() -> int:
    HistGradientBoostingClassifier = require_sklearn()
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", action="append", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--metric", default="gap_dev")
    ap.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--last-k", type=int, default=50)
    ap.add_argument("--baseline-method", default="target0")
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--l2-regularization", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [part.strip() for part in args.candidates.split(",") if part.strip()]
    if args.baseline_method not in candidates:
        raise SystemExit(f"baseline method {args.baseline_method!r} not in candidates")
    data, features = build_context_wide(
        logs_root=args.logs_root,
        metric=args.metric,
        last_k=args.last_k,
        candidates=candidates,
        baseline_method=args.baseline_method,
        higher_is_better=bool(args.higher_is_better),
    )
    oracle_idx = data[candidates].to_numpy(dtype=np.float64).argmin(axis=1)
    labels = np.asarray(candidates, dtype=object)[oracle_idx]
    class_to_id = {method: idx for idx, method in enumerate(candidates)}
    y = np.asarray([class_to_id[str(label)] for label in labels], dtype=np.int64)
    x = data[features].to_numpy(dtype=np.float32)
    folds = max(2, min(int(args.folds), int(data["seed"].nunique())))
    fold_map = seed_folds(data["seed"].unique().tolist(), folds)
    fold_ids = data["seed"].map(fold_map).to_numpy(dtype=np.int64)

    rows = []
    for fold in range(folds):
        train_mask = fold_ids != fold
        test_mask = ~train_mask
        x_train, x_test = impute_arrays(x[train_mask], x[test_mask])
        clf = HistGradientBoostingClassifier(
            max_iter=int(args.max_iter),
            learning_rate=float(args.learning_rate),
            max_leaf_nodes=int(args.max_leaf_nodes),
            l2_regularization=float(args.l2_regularization),
            random_state=stable_seed("oracle_classifier", fold),
        )
        clf.fit(x_train, y[train_mask])
        pred = clf.predict(x_test)
        test = data.loc[test_mask, ["domain", "seed", "ep", "tid", *candidates]].copy()
        pred_methods = np.asarray(candidates, dtype=object)[pred]
        oracle_methods = np.asarray(candidates, dtype=object)[
            test[candidates].to_numpy(dtype=np.float64).argmin(axis=1)
        ]
        selected_cost = np.asarray(
            [test.iloc[i][method] for i, method in enumerate(pred_methods)],
            dtype=np.float64,
        )
        oracle_cost = test[candidates].min(axis=1).to_numpy(dtype=np.float64)
        baseline_cost = test[args.baseline_method].to_numpy(dtype=np.float64)
        fold_rows = test[["domain", "seed", "ep", "tid"]].copy()
        fold_rows["fold"] = int(fold)
        fold_rows["selected_method"] = pred_methods
        fold_rows["oracle_best_method"] = oracle_methods
        fold_rows["selected_cost"] = selected_cost
        fold_rows["oracle_best_cost"] = oracle_cost
        fold_rows["oracle_regret"] = selected_cost - oracle_cost
        fold_rows[f"delta_vs_{args.baseline_method}"] = selected_cost - baseline_cost
        fold_rows["baseline_cost"] = baseline_cost
        rows.append(fold_rows)

    cv_rows = pd.concat(rows, ignore_index=True)
    summary = summarize_rows(cv_rows, args.baseline_method, args.n_boot)
    cv_rows.to_csv(out_dir / "trip_oracle_classifier_cv_rows.csv", index=False)
    summary.to_csv(out_dir / "trip_oracle_classifier_cv_summary.csv", index=False)
    payload = {
        "logs_root": [str(path) for path in args.logs_root],
        "metric": args.metric,
        "last_k": int(args.last_k),
        "baseline_method": args.baseline_method,
        "candidates": candidates,
        "folds": int(folds),
        "max_iter": int(args.max_iter),
        "learning_rate": float(args.learning_rate),
        "max_leaf_nodes": int(args.max_leaf_nodes),
        "l2_regularization": float(args.l2_regularization),
        "n_context_rows": int(len(data)),
        "features": features,
        "limitation": (
            "This is a seed-held-out nonlinear diagnostic over matched fixed-action "
            "CRN trip labels. It predicts oracle action classes from realized "
            "trip context and is not yet a deployable online value estimator."
        ),
    }
    (out_dir / "trip_oracle_classifier_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cols = [
        "domain",
        "selected_cost_mean",
        "oracle_best_cost_mean",
        "oracle_regret_mean",
        f"delta_vs_{args.baseline_method}_mean",
        f"delta_vs_{args.baseline_method}_ci95_lo",
        f"delta_vs_{args.baseline_method}_ci95_hi",
        f"win_vs_{args.baseline_method}_rate",
        "selected_method_counts_json",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
