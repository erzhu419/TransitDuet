#!/usr/bin/env python3
"""Fit seed-held-out value selectors on snapshot counterfactual labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fit_freqduet_action_value_model import (
    DOMAINS,
    add_design_features,
    bootstrap_ci,
    ridge_fit_predict,
    seed_folds,
    stable_seed,
)


SNAPSHOT_KEYS = ["domain", "config", "seed", "ep", "dispatch_index"]
DEFAULT_CONTEXT_COLS = [
    "dir_signed",
    "dispatch_index_norm",
    "snapshot_time_norm",
    "scheduled_launch_norm",
    "hour_norm",
    "period_is_peak",
    "period_is_off",
    "period_is_trans",
    "base_target_headway_norm",
    "waiting_total_pre_norm",
    "fleet_concurrent_pre_norm",
    "fleet_target_pre_norm",
    "fleet_pressure_pre",
    "headway_cv_active_pre",
    "freq_low_demand",
    "freq_low_forecast",
    "freq_high_energy",
    "freq_middle_energy",
    "freq_od_entropy",
    "freq_promotion_strength",
    "freq_promotion_active",
    "actor_delta_norm",
    "actor_abs_delta_norm",
    "actor_terminal_dispatch",
]


ACTION_RE = re.compile(
    r"^(?P<actor>actor_)?(?P<mode>target|term45)_(?P<delta>m\d+|0|p\d+)$"
)


def parse_action(
    method: str,
    candidate_delta_s: float | None = None,
    candidate_offset_s: float | None = None,
    actor_delta_s: float | None = None,
) -> dict[str, float | str]:
    text = str(method)
    if text == "target0":
        mode = "target"
        token = "0"
        actor_relative = False
    else:
        match = ACTION_RE.match(text)
        if not match:
            raise SystemExit(f"candidate_method {method!r} is not a supported discrete action")
        mode = match.group("mode")
        token = match.group("delta")
        actor_relative = bool(match.group("actor"))
    if candidate_offset_s is None or not np.isfinite(float(candidate_offset_s)):
        if token == "0":
            offset_s = 0.0
        elif token.startswith("m"):
            offset_s = -float(token[1:])
        else:
            offset_s = float(token[1:])
    else:
        offset_s = float(candidate_offset_s)
    if candidate_delta_s is None or not np.isfinite(float(candidate_delta_s)):
        base = float(actor_delta_s or 0.0) if actor_relative else 0.0
        delta_s = base + float(offset_s)
    else:
        delta_s = float(candidate_delta_s)
    actor_delta = float(actor_delta_s or 0.0)
    delta_minus_actor = float(delta_s - actor_delta)
    norm = 60.0
    return {
        "action_mode": mode,
        "action_delta_s": delta_s,
        "action_delta_norm": delta_s / norm,
        "action_abs_delta_norm": abs(delta_s) / norm,
        "action_positive": 1.0 if delta_s > 0 else 0.0,
        "action_negative": 1.0 if delta_s < 0 else 0.0,
        "action_zero": 1.0 if abs(delta_s) < 1e-9 else 0.0,
        "action_term45": 1.0 if mode == "term45" else 0.0,
        "action_target": 1.0 if mode == "target" else 0.0,
        "action_term45_x_delta": (1.0 if mode == "term45" else 0.0) * delta_s / norm,
        "action_term45_x_abs_delta": (1.0 if mode == "term45" else 0.0) * abs(delta_s) / norm,
        "candidate_offset_norm": offset_s / norm,
        "candidate_abs_offset_norm": abs(offset_s) / norm,
        "candidate_above_actor": 1.0 if delta_minus_actor > 1e-9 else 0.0,
        "candidate_below_actor": 1.0 if delta_minus_actor < -1e-9 else 0.0,
        "candidate_same_as_actor": 1.0 if abs(delta_minus_actor) <= 1e-9 else 0.0,
        "action_delta_minus_actor_norm": delta_minus_actor / norm,
        "action_abs_delta_minus_actor_norm": abs(delta_minus_actor) / norm,
        "action_term45_x_offset": (1.0 if mode == "term45" else 0.0) * offset_s / norm,
        "action_term45_x_abs_offset": (1.0 if mode == "term45" else 0.0) * abs(offset_s) / norm,
    }


def read_labels(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("snapshot_counterfactual_labels.csv"))
        if not files and (path / "snapshot_counterfactual_all.csv").exists():
            files = [path / "snapshot_counterfactual_all.csv"]
        if not files:
            raise SystemExit(f"no snapshot labels found under {path}")
        parts = [pd.read_csv(file) for file in files]
        return pd.concat(parts, ignore_index=True)
    return pd.read_csv(path)


def prepare_labels(labels: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "auto":
        metric = "integrated_proxy_cost" if "integrated_proxy_cost" in labels.columns else "proxy_cost"
    elif metric == "integrated_proxy_cost" and metric not in labels.columns and "proxy_cost" in labels.columns:
        print(
            "WARN integrated_proxy_cost missing; falling back to legacy proxy_cost",
            flush=True,
        )
        metric = "proxy_cost"
    required = set(SNAPSHOT_KEYS + ["candidate_method", metric])
    missing = sorted(required - set(labels.columns))
    if missing:
        raise SystemExit(f"snapshot labels missing columns: {missing}")
    data = labels.copy()
    data["seed"] = data["seed"].astype(int)
    data["cost"] = pd.to_numeric(data[metric], errors="coerce")
    data = data[np.isfinite(data["cost"])].copy()
    candidate_delta_col = pd.to_numeric(
        data.get("candidate_delta_s", pd.Series(np.nan, index=data.index)),
        errors="coerce",
    )
    candidate_offset_col = pd.to_numeric(
        data.get("candidate_offset_s", pd.Series(np.nan, index=data.index)),
        errors="coerce",
    )
    actor_delta_col = pd.to_numeric(
        data.get("actor_delta_s", pd.Series(0.0, index=data.index)),
        errors="coerce",
    ).fillna(0.0)
    action = pd.DataFrame([
        parse_action(
            method,
            candidate_delta_s=float(candidate_delta_col.iloc[idx]),
            candidate_offset_s=float(candidate_offset_col.iloc[idx]),
            actor_delta_s=float(actor_delta_col.iloc[idx]),
        )
        for idx, method in enumerate(data["candidate_method"])
    ])
    data = pd.concat([data.reset_index(drop=True), action], axis=1)

    data["dir_signed"] = np.where(pd.to_numeric(data.get("dir", 0), errors="coerce") > 0, 1.0, -1.0)
    data["dispatch_index_norm"] = pd.to_numeric(data.get("dispatch_index", 0.0), errors="coerce") / 262.0
    data["snapshot_time_norm"] = pd.to_numeric(data.get("snapshot_time_s", 0.0), errors="coerce") / 86400.0
    data["scheduled_launch_norm"] = pd.to_numeric(data.get("scheduled_launch_s", 0.0), errors="coerce") / 86400.0
    data["hour_norm"] = pd.to_numeric(data.get("hour", 0.0), errors="coerce") / 24.0
    period = data.get("period", pd.Series("", index=data.index)).astype(str)
    for name in ("peak", "off", "trans"):
        data[f"period_is_{name}"] = (period == name).astype(float)
    data["base_target_headway_norm"] = pd.to_numeric(
        data.get("base_target_headway_s", data.get("base_headway", 360.0)),
        errors="coerce",
    ) / 600.0
    data["actor_delta_s"] = actor_delta_col.reset_index(drop=True)
    data["actor_delta_norm"] = data["actor_delta_s"] / 60.0
    data["actor_abs_delta_norm"] = data["actor_delta_s"].abs() / 60.0
    data["actor_terminal_dispatch"] = pd.to_numeric(
        data.get("actor_terminal_dispatch", 0.0),
        errors="coerce",
    ).fillna(0.0)
    data["waiting_total_pre_norm"] = pd.to_numeric(data.get("waiting_total_pre", 0.0), errors="coerce") / 500.0
    data["fleet_concurrent_pre_norm"] = pd.to_numeric(data.get("fleet_concurrent_pre", 0.0), errors="coerce") / 30.0
    data["fleet_target_pre_norm"] = pd.to_numeric(data.get("fleet_target_pre", 1.0), errors="coerce") / 30.0
    data["fleet_pressure_pre"] = (
        pd.to_numeric(data.get("fleet_concurrent_pre", 0.0), errors="coerce")
        - pd.to_numeric(data.get("fleet_target_pre", 1.0), errors="coerce")
    ) / pd.to_numeric(data.get("fleet_target_pre", 1.0), errors="coerce").clip(lower=1.0)
    for col in DEFAULT_CONTEXT_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def context_cols(data: pd.DataFrame) -> list[str]:
    return [col for col in DEFAULT_CONTEXT_COLS if col in data.columns and data[col].notna().any()]


def add_model_target(data: pd.DataFrame, baseline_method: str, target_mode: str) -> pd.DataFrame:
    out = data.copy()
    if target_mode == "absolute":
        out["model_cost"] = out["cost"].astype(float)
        return out
    if target_mode != "residual":
        raise SystemExit(f"unknown target mode: {target_mode}")
    base = out[out["candidate_method"] == baseline_method][SNAPSHOT_KEYS + ["cost"]].copy()
    if base.empty:
        raise SystemExit(f"baseline method {baseline_method!r} missing for residual target")
    base = base.rename(columns={"cost": "_baseline_snapshot_cost"})
    out = out.merge(base, on=SNAPSHOT_KEYS, how="left")
    if out["_baseline_snapshot_cost"].isna().any():
        raise SystemExit("some snapshot rows are missing baseline cost for residual target")
    out["model_cost"] = out["cost"].astype(float) - out["_baseline_snapshot_cost"].astype(float)
    return out


def select_snapshot_predictions(
    test_rows: pd.DataFrame,
    predictions: np.ndarray,
    feature_set: str,
    fold: int,
    baseline_method: str,
) -> list[dict[str, object]]:
    pred = test_rows[SNAPSHOT_KEYS + ["candidate_method", "cost"]].copy()
    pred["pred_cost"] = predictions
    rows: list[dict[str, object]] = []
    for key, group in pred.groupby(SNAPSHOT_KEYS, sort=False):
        group = group.copy()
        selected = group.loc[group["pred_cost"].idxmin()]
        oracle = group.loc[group["cost"].idxmin()]
        baseline = group[group["candidate_method"] == baseline_method]
        baseline_cost = float(baseline["cost"].iloc[0]) if not baseline.empty else np.nan
        row = {col: value for col, value in zip(SNAPSHOT_KEYS, key)}
        row.update({
            "feature_set": feature_set,
            "fold": int(fold),
            "selected_method": str(selected["candidate_method"]),
            "selected_pred_cost": float(selected["pred_cost"]),
            "selected_cost": float(selected["cost"]),
            "oracle_best_method": str(oracle["candidate_method"]),
            "oracle_best_cost": float(oracle["cost"]),
            "oracle_regret": float(selected["cost"] - oracle["cost"]),
            f"delta_vs_{baseline_method}": float(selected["cost"] - baseline_cost),
            "baseline_cost": baseline_cost,
        })
        rows.append(row)
    return rows


def run_ridge_cv(
    data: pd.DataFrame,
    feature_set: str,
    cols: list[str],
    folds: int,
    alpha: float,
    baseline_method: str,
) -> tuple[pd.DataFrame, int]:
    design, feature_cols = add_design_features(data, cols, feature_set)
    folds = max(2, min(int(folds), int(design["seed"].nunique())))
    fold_map = seed_folds(design["seed"].unique().tolist(), folds)
    design["_fold"] = design["seed"].map(fold_map)
    x = design[feature_cols].to_numpy(dtype=np.float64)
    y_col = "model_cost" if "model_cost" in design.columns else "cost"
    y = design[y_col].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []
    for fold in range(folds):
        train_mask = design["_fold"].to_numpy() != fold
        test_mask = ~train_mask
        if not train_mask.any() or not test_mask.any():
            continue
        pred = ridge_fit_predict(x[train_mask], y[train_mask], x[test_mask], alpha=alpha)
        rows.extend(select_snapshot_predictions(
            design.loc[test_mask].reset_index(drop=True),
            pred,
            feature_set,
            fold,
            baseline_method,
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
            y_col = "model_cost" if "model_cost" in train.columns else "cost"
            means = train.groupby("candidate_method")[y_col].mean().to_dict()
            pred = test["candidate_method"].map(means).fillna(train[y_col].mean()).to_numpy(dtype=np.float64)
        elif selector == "domain_action_mean":
            y_col = "model_cost" if "model_cost" in train.columns else "cost"
            means = train.groupby(["domain", "candidate_method"])[y_col].mean().to_dict()
            global_means = train.groupby("candidate_method")[y_col].mean().to_dict()
            pred = np.asarray([
                means.get((row["domain"], row["candidate_method"]), global_means.get(row["candidate_method"], train[y_col].mean()))
                for _, row in test.iterrows()
            ], dtype=np.float64)
        else:
            raise SystemExit(f"unknown selector: {selector}")
        rows.extend(select_snapshot_predictions(
            test.reset_index(drop=True),
            pred,
            selector,
            fold,
            baseline_method,
        ))
    return pd.DataFrame(rows)


def _finite_matrix(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train = np.where(np.isfinite(train), train, med)
    test = np.where(np.isfinite(test), test, med)
    return train, test


def finite_train_matrix(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train = np.where(np.isfinite(train), train, med)
    return train, med


def make_sklearn_model(model_name: str, random_state: int, n_jobs: int):
    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=6,
            max_features=0.85,
            random_state=random_state,
            n_jobs=max(1, int(n_jobs)),
        )
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=8,
            max_features=0.85,
            random_state=random_state,
            n_jobs=max(1, int(n_jobs)),
        )
    if model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.05,
            min_samples_leaf=15,
            random_state=random_state,
        )
    raise SystemExit(f"unknown sklearn model: {model_name}")


def run_sklearn_cv(
    data: pd.DataFrame,
    base_feature_set: str,
    model_name: str,
    cols: list[str],
    folds: int,
    baseline_method: str,
    n_jobs: int,
) -> tuple[pd.DataFrame, int]:
    design, feature_cols = add_design_features(data, cols, base_feature_set)
    folds = max(2, min(int(folds), int(design["seed"].nunique())))
    fold_map = seed_folds(design["seed"].unique().tolist(), folds)
    design["_fold"] = design["seed"].map(fold_map)
    x = design[feature_cols].to_numpy(dtype=np.float64)
    y_col = "model_cost" if "model_cost" in design.columns else "cost"
    y = design[y_col].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []
    feature_set = f"{model_name}_{base_feature_set}"
    for fold in range(folds):
        train_mask = design["_fold"].to_numpy() != fold
        test_mask = ~train_mask
        if not train_mask.any() or not test_mask.any():
            continue
        x_train, x_test = _finite_matrix(x[train_mask], x[test_mask])
        model = make_sklearn_model(
            model_name,
            random_state=stable_seed("sklearn", model_name, base_feature_set, fold),
            n_jobs=n_jobs,
        )
        model.fit(x_train, y[train_mask])
        pred = np.asarray(model.predict(x_test), dtype=np.float64)
        rows.extend(select_snapshot_predictions(
            design.loc[test_mask].reset_index(drop=True),
            pred,
            feature_set,
            fold,
            baseline_method,
        ))
    return pd.DataFrame(rows), len(feature_cols)


def summarize_selection(rows: pd.DataFrame, baseline_method: str, n_boot: int) -> pd.DataFrame:
    out = []
    delta_col = f"delta_vs_{baseline_method}"
    domains = list(DOMAINS) + ["overall_seed_mean"]
    for feature_set in sorted(rows["feature_set"].unique()):
        sub = rows[rows["feature_set"] == feature_set].copy()
        for domain in domains:
            if domain == "overall_seed_mean":
                seed_group = sub.groupby("seed", as_index=False).agg(
                    selected_cost=("selected_cost", "mean"),
                    oracle_best_cost=("oracle_best_cost", "mean"),
                    oracle_regret=("oracle_regret", "mean"),
                    **{delta_col: (delta_col, "mean")},
                )
                choice_source = sub
            else:
                dom = sub[sub["domain"] == domain].copy()
                seed_group = dom.groupby("seed", as_index=False).agg(
                    selected_cost=("selected_cost", "mean"),
                    oracle_best_cost=("oracle_best_cost", "mean"),
                    oracle_regret=("oracle_regret", "mean"),
                    **{delta_col: (delta_col, "mean")},
                )
                choice_source = dom
            if seed_group.empty:
                continue
            delta = seed_group[delta_col].to_numpy(dtype=np.float64)
            regret = seed_group["oracle_regret"].to_numpy(dtype=np.float64)
            delta_lo, delta_hi = bootstrap_ci(
                delta,
                n_boot,
                stable_seed("snapshot-delta", feature_set, domain, baseline_method),
            )
            regret_lo, regret_hi = bootstrap_ci(
                regret,
                n_boot,
                stable_seed("snapshot-regret", feature_set, domain),
            )
            out.append({
                "feature_set": feature_set,
                "domain": domain,
                "n_snapshots": int(len(choice_source)),
                "n_seeds": int(seed_group["seed"].nunique()),
                "selected_cost_mean": float(seed_group["selected_cost"].mean()),
                "oracle_best_cost_mean": float(seed_group["oracle_best_cost"].mean()),
                "oracle_regret_mean": float(seed_group["oracle_regret"].mean()),
                "oracle_regret_ci95_lo": regret_lo,
                "oracle_regret_ci95_hi": regret_hi,
                f"delta_vs_{baseline_method}_mean": float(delta.mean()),
                f"delta_vs_{baseline_method}_ci95_lo": delta_lo,
                f"delta_vs_{baseline_method}_ci95_hi": delta_hi,
                f"win_vs_{baseline_method}_rate": float((choice_source[delta_col] < 0.0).mean()),
                "selected_method_counts_json": json.dumps(
                    choice_source["selected_method"].value_counts().sort_index().to_dict(),
                    sort_keys=True,
                ),
            })
    return pd.DataFrame(out)


def summarize_oracle_and_candidates(data: pd.DataFrame, baseline_method: str, n_boot: int) -> pd.DataFrame:
    wide = data.pivot_table(
        index=SNAPSHOT_KEYS,
        columns="candidate_method",
        values="cost",
        aggfunc="mean",
    ).reset_index()
    if baseline_method not in wide.columns:
        raise SystemExit(f"baseline method {baseline_method!r} missing from snapshot labels")
    methods = sorted(data["candidate_method"].unique())
    wide["oracle_best_cost"] = wide[methods].min(axis=1)
    rows = []
    for domain in [*DOMAINS, "overall_seed_mean"]:
        sub = wide.copy() if domain == "overall_seed_mean" else wide[wide["domain"] == domain].copy()
        if sub.empty:
            continue
        seed_oracle = sub.groupby("seed")[
            ["oracle_best_cost", baseline_method]
        ].apply(
            lambda g: (g["oracle_best_cost"] - g[baseline_method]).mean()
        ).to_numpy(dtype=np.float64)
        lo, hi = bootstrap_ci(seed_oracle, n_boot, stable_seed("snapshot-oracle", domain, baseline_method))
        rows.append({
            "domain": domain,
            "method": "oracle_best",
            "snapshots": int(len(sub)),
            "seeds": int(sub["seed"].nunique()),
            f"delta_vs_{baseline_method}_mean": float(seed_oracle.mean()),
            f"delta_vs_{baseline_method}_ci95_lo": lo,
            f"delta_vs_{baseline_method}_ci95_hi": hi,
        })
        for method in methods:
            seed_delta = sub.groupby("seed")[
                [method, baseline_method]
            ].apply(
                lambda g, m=method: (g[m] - g[baseline_method]).mean()
            ).to_numpy(dtype=np.float64)
            lo, hi = bootstrap_ci(seed_delta, n_boot, stable_seed("snapshot-candidate", domain, method, baseline_method))
            rows.append({
                "domain": domain,
                "method": method,
                "snapshots": int(len(sub)),
                "seeds": int(sub["seed"].nunique()),
                f"delta_vs_{baseline_method}_mean": float(seed_delta.mean()),
                f"delta_vs_{baseline_method}_ci95_lo": lo,
                f"delta_vs_{baseline_method}_ci95_hi": hi,
            })
    return pd.DataFrame(rows)


def export_sklearn_model(
    data: pd.DataFrame,
    out_dir: Path,
    labels_path: Path,
    metric: str,
    model_name: str,
    base_feature_set: str,
    cols: list[str],
    baseline_method: str,
    n_jobs: int,
) -> None:
    try:
        import joblib
    except Exception as exc:
        raise SystemExit(f"joblib is required for --export-model: {exc}") from exc

    design, feature_cols = add_design_features(data, cols, base_feature_set)
    x = design[feature_cols].to_numpy(dtype=np.float64)
    x_train, med = finite_train_matrix(x)
    y_col = "model_cost" if "model_cost" in design.columns else "cost"
    y = design[y_col].to_numpy(dtype=np.float64)
    model = make_sklearn_model(
        model_name,
        random_state=stable_seed("sklearn-final", model_name, base_feature_set),
        n_jobs=n_jobs,
    )
    model.fit(x_train, y)

    model_dir = out_dir / "model_artifact"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    if hasattr(model, "estimators_"):
        tree_ptr = [0]
        children_left = []
        children_right = []
        feature = []
        threshold = []
        value = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            n_nodes = int(tree.node_count)
            children_left.extend(tree.children_left.astype(np.int32).tolist())
            children_right.extend(tree.children_right.astype(np.int32).tolist())
            feature.extend(tree.feature.astype(np.int32).tolist())
            threshold.extend(tree.threshold.astype(np.float64).tolist())
            value.extend(tree.value.reshape(n_nodes, -1)[:, 0].astype(np.float64).tolist())
            tree_ptr.append(tree_ptr[-1] + n_nodes)
        np.savez_compressed(
            model_dir / "forest_model.npz",
            tree_ptr=np.asarray(tree_ptr, dtype=np.int64),
            children_left=np.asarray(children_left, dtype=np.int32),
            children_right=np.asarray(children_right, dtype=np.int32),
            feature=np.asarray(feature, dtype=np.int32),
            threshold=np.asarray(threshold, dtype=np.float64),
            value=np.asarray(value, dtype=np.float64),
        )
    artifact = {
        "model_name": model_name,
        "feature_set": f"{model_name}_{base_feature_set}",
        "base_feature_set": base_feature_set,
        "baseline_method": baseline_method,
        "target_mode": "residual" if "model_cost" in design.columns else "absolute",
        "candidate_methods": sorted(str(x) for x in data["candidate_method"].unique()),
        "context_cols": cols,
        "feature_cols": feature_cols,
        "feature_medians": {
            col: float(value) for col, value in zip(feature_cols, med)
        },
        "action_scale_s": 60.0,
        "training_labels": str(labels_path),
        "training_metric": metric,
        "n_rows": int(len(design)),
        "n_snapshots": int(design[SNAPSHOT_KEYS].drop_duplicates().shape[0]),
        "n_seeds": int(design["seed"].nunique()),
        "metric": "model_cost",
        "limitation": (
            "Trained on offline snapshot replay labels. Use only behind a "
            "config-gated online evaluation wrapper until paired online tests "
            "against current main and fixed-headway are complete."
        ),
    }
    (model_dir / "model_artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_compact(summary: pd.DataFrame, candidate_summary: pd.DataFrame, baseline_method: str) -> None:
    delta = f"delta_vs_{baseline_method}_mean"
    lo = f"delta_vs_{baseline_method}_ci95_lo"
    hi = f"delta_vs_{baseline_method}_ci95_hi"
    overall = summary[summary["domain"] == "overall_seed_mean"].copy()
    cols = [
        "feature_set",
        "selected_cost_mean",
        "oracle_best_cost_mean",
        "oracle_regret_mean",
        delta,
        lo,
        hi,
        f"win_vs_{baseline_method}_rate",
        "selected_method_counts_json",
    ]
    print("\n== snapshot value selector CV, overall seed-held-out ==")
    print(overall[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n== snapshot candidate/oracle, overall ==")
    cand = candidate_summary[candidate_summary["domain"] == "overall_seed_mean"].copy()
    print(cand[["method", delta, lo, hi]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--metric",
        default="auto",
        help="Cost label column. Use auto to prefer integrated_proxy_cost and fall back to proxy_cost.",
    )
    parser.add_argument("--baseline-method", default="target_0")
    parser.add_argument("--target-mode", choices=["absolute", "residual"], default="absolute")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument(
        "--sklearn-models",
        default="extra_trees,random_forest,hist_gradient_boosting",
        help="Comma-separated sklearn regressors to add; empty disables them.",
    )
    parser.add_argument("--sklearn-n-jobs", type=int, default=4)
    parser.add_argument(
        "--export-model",
        default="",
        help="Optional final sklearn model to train on all labels and save under out-dir/model_artifact.",
    )
    parser.add_argument("--export-feature-set", default="context_action_interact")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = prepare_labels(read_labels(args.labels), args.metric)
    if args.baseline_method not in set(labels["candidate_method"]):
        raise SystemExit(f"baseline method {args.baseline_method!r} not in labels")
    if int(labels["seed"].nunique()) < 2:
        raise SystemExit("snapshot value-model CV requires at least 2 distinct seeds")
    labels = add_model_target(labels, args.baseline_method, args.target_mode)
    cols = context_cols(labels)
    cv_parts = []
    feature_counts = {}
    for feature_set in ("action_domain", "context_action", "context_action_interact"):
        rows, n_features = run_ridge_cv(
            labels,
            feature_set,
            cols,
            args.folds,
            args.ridge_alpha,
            args.baseline_method,
        )
        cv_parts.append(rows)
        feature_counts[feature_set] = int(n_features)
    for selector in ("global_action_mean", "domain_action_mean"):
        cv_parts.append(run_mean_cv(labels, selector, args.folds, args.baseline_method))
    sklearn_models = [part.strip() for part in str(args.sklearn_models).split(",") if part.strip()]
    for model_name in sklearn_models:
        rows, n_features = run_sklearn_cv(
            labels,
            "context_action_interact",
            model_name,
            cols,
            args.folds,
            args.baseline_method,
            args.sklearn_n_jobs,
        )
        cv_parts.append(rows)
        feature_counts[f"{model_name}_context_action_interact"] = int(n_features)

    cv_rows = pd.concat(cv_parts, ignore_index=True)
    summary = summarize_selection(cv_rows, args.baseline_method, args.n_boot)
    candidate_summary = summarize_oracle_and_candidates(labels, args.baseline_method, args.n_boot)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv_rows.to_csv(args.out_dir / "snapshot_value_model_cv_rows.csv", index=False)
    summary.to_csv(args.out_dir / "snapshot_value_model_cv_summary.csv", index=False)
    candidate_summary.to_csv(args.out_dir / "snapshot_value_model_candidate_summary.csv", index=False)
    meta = {
        "labels": str(args.labels),
        "metric": args.metric,
        "baseline_method": args.baseline_method,
        "target_mode": args.target_mode,
        "folds": int(args.folds),
        "ridge_alpha": float(args.ridge_alpha),
        "sklearn_models": sklearn_models,
        "sklearn_n_jobs": int(args.sklearn_n_jobs),
        "n_boot": int(args.n_boot),
        "n_rows": int(len(labels)),
        "n_snapshots": int(labels[SNAPSHOT_KEYS].drop_duplicates().shape[0]),
        "n_seeds": int(labels["seed"].nunique()),
        "context_cols": cols,
        "feature_counts": feature_counts,
        "limitation": (
            "Offline seed-held-out selector over short-horizon snapshot replay "
            "labels. It estimates whether a per-dispatch value model can close "
            "the fixed-headway gap; it is not yet wired into online FreqDuet."
        ),
    }
    (args.out_dir / "snapshot_value_model_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if str(args.export_model).strip():
        export_sklearn_model(
            labels,
            args.out_dir,
            args.labels,
            args.metric,
            str(args.export_model).strip(),
            str(args.export_feature_set).strip(),
            cols,
            args.baseline_method,
            args.sklearn_n_jobs,
        )
    print_compact(summary, candidate_summary, args.baseline_method)
    print(f"\nWrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
