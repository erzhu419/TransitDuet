#!/usr/bin/env python3
"""Fit and export a lightweight counterfactual action tree selector.

The exported artifact is a JSON decision tree that can be loaded by
``upper.counterfactual_action_selector`` without scikit-learn at simulator
runtime. Labels come from matched fixed-action CRN trip rollouts.
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
    bootstrap_ci,
    seed_folds,
    stable_seed,
)
from fit_freqduet_trip_oracle_classifier import build_context_wide, impute_arrays


def require_sklearn():
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except Exception as exc:  # pragma: no cover - environment diagnostic.
        raise SystemExit(
            "scikit-learn is required for fitting/exporting the tree. "
            "The exported artifact does not require sklearn at runtime."
        ) from exc
    return DecisionTreeClassifier, export_text


def class_ids(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    class_to_id = {name: i for i, name in enumerate(classes)}
    return np.asarray([class_to_id[str(label)] for label in labels], dtype=np.int64)


def summarize_cv_rows(rows: pd.DataFrame, baseline_method: str, n_boot: int) -> pd.DataFrame:
    out = []
    delta_col = f"delta_vs_{baseline_method}"
    for domain in [*DOMAINS, "overall_seed_mean"]:
        if domain == "overall_seed_mean":
            group = rows.groupby("seed", as_index=False).agg(
                selected_cost=("selected_cost", "mean"),
                oracle_best_cost=("oracle_best_cost", "mean"),
                oracle_regret=("oracle_regret", "mean"),
                **{delta_col: (delta_col, "mean")},
            )
            choices = rows["selected_method"].value_counts().sort_index().to_dict()
            n_rows = int(len(group))
        else:
            sub = rows[rows["domain"] == domain].copy()
            group = sub.groupby("seed", as_index=False).agg(
                selected_cost=("selected_cost", "mean"),
                oracle_best_cost=("oracle_best_cost", "mean"),
                oracle_regret=("oracle_regret", "mean"),
                **{delta_col: (delta_col, "mean")},
            )
            choices = sub["selected_method"].value_counts().sort_index().to_dict()
            n_rows = int(len(sub))
        if group.empty:
            continue
        delta = group[delta_col].to_numpy(dtype=np.float64)
        regret = group["oracle_regret"].to_numpy(dtype=np.float64)
        dlo, dhi = bootstrap_ci(
            delta,
            n_boot=n_boot,
            seed=stable_seed("tree_selector_delta", domain, baseline_method),
        )
        rlo, rhi = bootstrap_ci(
            regret,
            n_boot=n_boot,
            seed=stable_seed("tree_selector_regret", domain),
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


def evaluate_seed_folds(
    data: pd.DataFrame,
    features: list[str],
    candidates: list[str],
    baseline_method: str,
    folds: int,
    max_depth: int | None,
    min_samples_leaf: int,
    n_boot: int,
    write_rows: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    DecisionTreeClassifier, _ = require_sklearn()
    costs = data[candidates].to_numpy(dtype=np.float64)
    oracle_idx = costs.argmin(axis=1)
    y = oracle_idx.astype(np.int64)
    x = data[features].to_numpy(dtype=np.float32)
    folds = max(2, min(int(folds), int(data["seed"].nunique())))
    fold_map = seed_folds(data["seed"].unique().tolist(), folds)
    fold_ids = data["seed"].map(fold_map).to_numpy(dtype=np.int64)
    rows = []
    for fold in range(folds):
        train_mask = fold_ids != fold
        test_mask = ~train_mask
        x_train, x_test = impute_arrays(x[train_mask], x[test_mask])
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=stable_seed("trip_action_tree", max_depth, min_samples_leaf, fold),
        )
        clf.fit(x_train, y[train_mask])
        pred = clf.predict(x_test)
        test = data.loc[test_mask, ["domain", "seed", "ep", "tid", *candidates]].copy()
        arr = test[candidates].to_numpy(dtype=np.float64)
        selected_cost = arr[np.arange(arr.shape[0]), pred]
        oracle_cost = arr.min(axis=1)
        baseline_cost = test[baseline_method].to_numpy(dtype=np.float64)
        fold_rows = test[["domain", "seed", "ep", "tid"]].copy()
        fold_rows["fold"] = int(fold)
        fold_rows["selected_method"] = np.asarray(candidates, dtype=object)[pred]
        fold_rows["oracle_best_method"] = np.asarray(candidates, dtype=object)[
            arr.argmin(axis=1)
        ]
        fold_rows["selected_cost"] = selected_cost
        fold_rows["oracle_best_cost"] = oracle_cost
        fold_rows["oracle_regret"] = selected_cost - oracle_cost
        fold_rows[f"delta_vs_{baseline_method}"] = selected_cost - baseline_cost
        fold_rows["baseline_cost"] = baseline_cost
        rows.append(fold_rows)
    cv_rows = pd.concat(rows, ignore_index=True)
    return summarize_cv_rows(cv_rows, baseline_method, n_boot), (
        cv_rows if write_rows else None
    )


def export_tree_artifact(
    clf,
    feature_cols: list[str],
    feature_medians: dict[str, float],
    classes: list[str],
    payload_extra: dict[str, object],
    out_path: Path,
) -> None:
    tree = clf.tree_
    values = tree.value
    if values.ndim == 3:
        values = values[:, 0, :]
    nodes = []
    for node_id in range(tree.node_count):
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        raw = np.asarray(values[node_id], dtype=np.float64).reshape(-1)
        total = float(raw.sum())
        proba = (raw / total).tolist() if total > 0.0 else [0.0] * len(classes)
        nodes.append({
            "id": int(node_id),
            "leaf": bool(left == right),
            "feature": int(tree.feature[node_id]),
            "feature_name": (
                feature_cols[int(tree.feature[node_id])]
                if int(tree.feature[node_id]) >= 0 else ""
            ),
            "threshold": float(tree.threshold[node_id]),
            "left": left,
            "right": right,
            "samples": int(tree.n_node_samples[node_id]),
            "value": raw.tolist(),
            "proba": proba,
        })
    payload = {
        "artifact_type": "freqduet_counterfactual_action_tree",
        "feature_cols": feature_cols,
        "feature_medians": feature_medians,
        "classes": classes,
        "nodes": nodes,
        **payload_extra,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    DecisionTreeClassifier, export_text = require_sklearn()
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", action="append", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--metric", default="gap_dev")
    ap.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--last-k", type=int, default=50)
    ap.add_argument("--baseline-method", default="target0")
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--min-samples-leaf", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--write-cv-rows", action="store_true")
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
    cv_summary, cv_rows = evaluate_seed_folds(
        data=data,
        features=features,
        candidates=candidates,
        baseline_method=args.baseline_method,
        folds=args.folds,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_boot=args.n_boot,
        write_rows=args.write_cv_rows,
    )
    cv_summary.to_csv(out_dir / "trip_action_tree_cv_summary.csv", index=False)
    if cv_rows is not None:
        cv_rows.to_csv(out_dir / "trip_action_tree_cv_rows.csv", index=False)

    costs = data[candidates].to_numpy(dtype=np.float64)
    oracle_methods = np.asarray(candidates, dtype=object)[costs.argmin(axis=1)]
    y = class_ids(oracle_methods, candidates)
    x = data[features].to_numpy(dtype=np.float32)
    x_imp, _ = impute_arrays(x, x)
    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=stable_seed("trip_action_tree_final", args.max_depth, args.min_samples_leaf),
    )
    clf.fit(x_imp, y)
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    feature_medians = {
        name: float(value) for name, value in zip(features, med.tolist())
    }
    artifact_dir = out_dir / "model_artifact"
    export_tree_artifact(
        clf=clf,
        feature_cols=features,
        feature_medians=feature_medians,
        classes=candidates,
        payload_extra={
            "metric": args.metric,
            "last_k": int(args.last_k),
            "baseline_method": args.baseline_method,
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "n_context_rows": int(len(data)),
            "source_logs_root": [str(path) for path in args.logs_root],
            "runtime_note": "Pure JSON tree; no sklearn dependency is required at runtime.",
        },
        out_path=artifact_dir / "tree_selector.json",
    )
    (artifact_dir / "tree_text.txt").write_text(
        export_text(clf, feature_names=features, decimals=4),
        encoding="utf-8",
    )
    pd.DataFrame({
        "feature": features,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        artifact_dir / "feature_importance.csv", index=False
    )
    summary = {
        "logs_root": [str(path) for path in args.logs_root],
        "out_dir": str(out_dir),
        "artifact": str(artifact_dir / "tree_selector.json"),
        "metric": args.metric,
        "last_k": int(args.last_k),
        "baseline_method": args.baseline_method,
        "candidates": candidates,
        "folds": int(args.folds),
        "max_depth": int(args.max_depth),
        "min_samples_leaf": int(args.min_samples_leaf),
        "n_context_rows": int(len(data)),
    }
    (out_dir / "trip_action_tree_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(cv_summary.to_string(index=False))
    print(f"artifact: {artifact_dir / 'tree_selector.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
