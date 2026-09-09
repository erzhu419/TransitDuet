#!/usr/bin/env python3
"""Fit a deployable, train-seed-held-out V28 prefix value model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_freqduet_prefix_counterfactual import bootstrap_ci
from scripts.audit_protocol_v6_v28_prefix_common import (
    DECISION_INDICES,
    EVAL_SEEDS,
    EXPECTED_METHODS,
    MATRIX_PROTOCOL_VERSION,
    MODEL_PROTOCOL_VERSION,
    OUTCOME_DELTAS,
    PRIMARY_DELTA,
    TRAIN_SEEDS,
)
from scripts.run_freqduet_protocol_v2_matrix import git_provenance


CONTEXT_KEYS = ["train_seed", "scenario_seed", "decision_index", "eval_episode"]
MODEL_METHODS = [method for method in EXPECTED_METHODS if method != "actor_firstknot_0"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
GUARD_MARGINS = [0.0, 0.0005, 0.001, 0.002]
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _fit_source_commit() -> str:
    source = git_provenance()
    commit = str(source.get("commit", "")).lower()
    _require(bool(COMMIT_RE.fullmatch(commit)),
             f"invalid fitter source commit {commit!r}")
    _require(source.get("tracked_dirty") is False,
             "fitter source snapshot is tracked-dirty")
    return commit


def _array(text: str, label: str) -> np.ndarray:
    try:
        value = np.asarray(json.loads(str(text)), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"invalid {label} JSON") from exc
    _require(value.size > 0 and np.isfinite(value).all(), f"non-finite {label}")
    return value


def load_labels(aggregate_dir: Path) -> tuple[dict, pd.DataFrame, list[str]]:
    root = Path(aggregate_dir).resolve()
    manifest = json.loads((root / "matrix_manifest.json").read_text())
    _require(manifest.get("protocol_version") == MATRIX_PROTOCOL_VERSION,
             "wrong V28 matrix protocol")
    _require(manifest.get("strict_complete") is True, "V28 matrix is not strict-complete")
    labels = pd.read_csv(root / "prefix_counterfactual_all.csv")
    labels = labels[labels["candidate_method"].isin(MODEL_METHODS)].copy()
    expected_rows = len(TRAIN_SEEDS) * len(EVAL_SEEDS) * len(DECISION_INDICES) * len(MODEL_METHODS)
    _require(len(labels) == expected_rows,
             f"expected {expected_rows} model rows, found {len(labels)}")
    actual_methods = set(labels["candidate_method"].astype(str))
    _require(actual_methods == set(MODEL_METHODS),
             f"model candidate roster mismatch: {sorted(actual_methods)}")
    counts = labels.groupby(CONTEXT_KEYS)["candidate_method"].nunique()
    _require(len(counts) == len(TRAIN_SEEDS) * len(EVAL_SEEDS) * len(DECISION_INDICES),
             "model context count mismatch")
    _require(counts.eq(len(MODEL_METHODS)).all(), "a context has incomplete candidates")

    state_cols = sorted(
        column for column in labels.columns if column.startswith("upper_state_")
        and column != "upper_state_dim"
    )
    _require(bool(state_cols), "no causal upper-state columns found")
    dimensions = pd.to_numeric(labels["upper_state_dim"], errors="coerce")
    _require(dimensions.notna().all() and dimensions.eq(len(state_cols)).all(),
             "upper-state dimensions are inconsistent")
    for column in state_cols + list(OUTCOME_DELTAS.values()):
        values = pd.to_numeric(labels[column], errors="coerce")
        _require(np.isfinite(values.to_numpy(dtype=np.float64)).all(),
                 f"non-finite model column {column}")
        labels[column] = values
    for column in CONTEXT_KEYS + ["candidate_offset_s"]:
        labels[column] = pd.to_numeric(labels[column], errors="raise")
    labels["actor_action"] = [
        _array(value, "actor action") for value in labels["actor_action_json"]
    ]
    labels["candidate_action"] = [
        _array(value, "candidate action") for value in labels["candidate_action_json"]
    ]
    action_dims = {value.size for value in labels["actor_action"]}
    action_dims.update(value.size for value in labels["candidate_action"])
    _require(len(action_dims) == 1, f"inconsistent action dimensions: {sorted(action_dims)}")
    _require(not labels.duplicated(CONTEXT_KEYS + ["candidate_method"]).any(),
             "duplicate model rows")
    return manifest, labels.reset_index(drop=True), state_cols


def build_features(
    labels: pd.DataFrame,
    state_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    state = labels[state_cols].to_numpy(dtype=np.float64)
    actor = np.stack(labels["actor_action"].to_list()).astype(np.float64)
    candidate = np.stack(labels["candidate_action"].to_list()).astype(np.float64)
    delta = candidate - actor
    offset = labels["candidate_offset_s"].to_numpy(dtype=np.float64)[:, None]
    offset_norm = offset / 30.0
    abs_offset_norm = np.abs(offset_norm)

    action_dim = actor.shape[1]
    matrices = [
        state,
        actor / 60.0,
        candidate / 60.0,
        delta / 30.0,
        np.abs(delta) / 30.0,
        offset_norm,
        abs_offset_norm,
        state * offset_norm,
        state * abs_offset_norm,
    ]
    names = (
        list(state_cols)
        + [f"actor_action_{index:02d}" for index in range(action_dim)]
        + [f"candidate_action_{index:02d}" for index in range(action_dim)]
        + [f"action_delta_{index:02d}" for index in range(action_dim)]
        + [f"action_abs_delta_{index:02d}" for index in range(action_dim)]
        + ["candidate_offset_norm", "candidate_abs_offset_norm"]
        + [f"{column}_x_candidate_offset" for column in state_cols]
        + [f"{column}_x_candidate_abs_offset" for column in state_cols]
    )
    design = np.concatenate(matrices, axis=1)
    _require(design.shape[1] == len(names), "feature name/design mismatch")
    _require(np.isfinite(design).all(), "non-finite deployable features")
    prohibited = ("domain", "config", "seed", "future", "outcome", "episode_")
    bad = [name for name in names if any(token in name.lower() for token in prohibited)]
    _require(not bad, f"prohibited deployable features: {bad}")
    return design, names


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        coefficient = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "coefficient": coefficient,
        "alpha": float(alpha),
    }


def predict(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    return design @ coefficient


def select_rows(
    labels: pd.DataFrame,
    predictions: np.ndarray,
    *,
    guard_margin: float,
) -> pd.DataFrame:
    frame = labels.copy()
    frame["predicted_delta"] = np.asarray(predictions, dtype=np.float64)
    selected: list[pd.Series] = []
    for _, group in frame.groupby(CONTEXT_KEYS, sort=True):
        actor = group[group["candidate_method"] == "actor"]
        _require(len(actor) == 1, "context has no unique actor row")
        actor_row = actor.iloc[0]
        candidates = group[group["candidate_method"] != "actor"]
        best = candidates.loc[candidates["predicted_delta"].idxmin()]
        if float(best["predicted_delta"]) + float(guard_margin) < float(
                actor_row["predicted_delta"]):
            selected.append(best)
        else:
            selected.append(actor_row)
    return pd.DataFrame(selected).reset_index(drop=True)


def inner_score(
    labels: pd.DataFrame,
    x: np.ndarray,
    *,
    outer_train_seeds: list[int],
    alpha: float,
    guard_margin: float,
) -> float:
    scores = []
    seeds = labels["train_seed"].to_numpy(dtype=int)
    target = labels[PRIMARY_DELTA].to_numpy(dtype=np.float64)
    for heldout in outer_train_seeds:
        train_mask = np.isin(seeds, [seed for seed in outer_train_seeds if seed != heldout])
        test_mask = seeds == heldout
        model = fit_ridge(x[train_mask], target[train_mask], alpha)
        selected = select_rows(
            labels.loc[test_mask].reset_index(drop=True),
            predict(model, x[test_mask]),
            guard_margin=guard_margin,
        )
        scores.append(float(selected[PRIMARY_DELTA].mean()))
    return float(np.mean(scores))


def choose_hyperparameters(
    labels: pd.DataFrame,
    x: np.ndarray,
    train_seeds: list[int],
) -> tuple[float, float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        for margin in GUARD_MARGINS:
            score = inner_score(
                labels,
                x,
                outer_train_seeds=train_seeds,
                alpha=alpha,
                guard_margin=margin,
            )
            rows.append({"alpha": alpha, "guard_margin": margin, "score": score})
    best = min(rows, key=lambda row: (
        row["score"], -row["guard_margin"], -row["alpha"]
    ))
    return float(best["alpha"]), float(best["guard_margin"]), rows


def nested_predictions(
    labels: pd.DataFrame,
    x: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selected_parts = []
    folds: list[dict[str, object]] = []
    seeds = labels["train_seed"].to_numpy(dtype=int)
    target = labels[PRIMARY_DELTA].to_numpy(dtype=np.float64)
    for heldout in TRAIN_SEEDS:
        outer_train = [seed for seed in TRAIN_SEEDS if seed != heldout]
        alpha, margin, scores = choose_hyperparameters(labels, x, outer_train)
        train_mask = np.isin(seeds, outer_train)
        test_mask = seeds == heldout
        model = fit_ridge(x[train_mask], target[train_mask], alpha)
        selected = select_rows(
            labels.loc[test_mask].reset_index(drop=True),
            predict(model, x[test_mask]),
            guard_margin=margin,
        )
        selected["outer_holdout_train_seed"] = int(heldout)
        selected["selected_alpha"] = alpha
        selected["selected_guard_margin"] = margin
        selected_parts.append(selected)
        folds.append({
            "heldout_train_seed": int(heldout),
            "inner_train_seeds": outer_train,
            "selected_alpha": alpha,
            "selected_guard_margin": margin,
            "inner_grid": scores,
        })
    return pd.concat(selected_parts, ignore_index=True), folds


def global_mean_predictions(labels: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for heldout in TRAIN_SEEDS:
        train = labels[labels["train_seed"] != heldout]
        test = labels[labels["train_seed"] == heldout]
        means = train.groupby("candidate_method")[PRIMARY_DELTA].mean()
        method = str(means.idxmin())
        chosen = test[test["candidate_method"] == method].copy()
        chosen["outer_holdout_train_seed"] = int(heldout)
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True)


def oracle_predictions(labels: pd.DataFrame) -> pd.DataFrame:
    ordered = labels.copy()
    ordered["_actor_tie"] = np.where(ordered["candidate_method"].eq("actor"), 0, 1)
    ordered = ordered.sort_values(
        CONTEXT_KEYS + [PRIMARY_DELTA, "_actor_tie"], kind="mergesort"
    )
    return (
        ordered.groupby(CONTEXT_KEYS, sort=True, as_index=False).first()
        .drop(columns=["_actor_tie"])
    )


def summarize_selection(frame: pd.DataFrame, *, seed: int) -> dict[str, object]:
    outcomes = {}
    for index, (name, column) in enumerate(OUTCOME_DELTAS.items()):
        blocks = (
            frame.groupby(["train_seed", "scenario_seed"], sort=True)[column]
            .mean()
            .to_numpy(dtype=np.float64)
        )
        low, high = bootstrap_ci(blocks, seed=seed + index)
        outcomes[name] = {
            "mean": float(np.mean(blocks)),
            "ci_low": low,
            "ci_high": high,
            "blocks": int(blocks.size),
        }
    return {
        "contexts": int(len(frame)),
        "non_actor_fraction": float((frame["candidate_method"] != "actor").mean()),
        "method_counts": {
            str(key): int(value)
            for key, value in frame["candidate_method"].value_counts().sort_index().items()
        },
        "outcomes": outcomes,
    }


def final_model(
    labels: pd.DataFrame,
    x: np.ndarray,
    feature_names: list[str],
    *,
    fit_source_commit: str,
) -> dict[str, object]:
    alpha, margin, grid = choose_hyperparameters(labels, x, TRAIN_SEEDS)
    model = fit_ridge(
        x, labels[PRIMARY_DELTA].to_numpy(dtype=np.float64), alpha
    )
    return {
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "fit_source_commit": fit_source_commit,
        "feature_contract": "causal_s_upper_plus_candidate_action_v1",
        "target": PRIMARY_DELTA,
        "candidate_methods": MODEL_METHODS,
        "feature_names": feature_names,
        "feature_mean": np.asarray(model["feature_mean"]).tolist(),
        "feature_scale": np.asarray(model["feature_scale"]).tolist(),
        "coefficient": np.asarray(model["coefficient"]).tolist(),
        "alpha": alpha,
        "guard_margin": margin,
        "full_seed_cv_grid": grid,
    }


def fit(aggregate_dir: Path, out_dir: Path) -> dict[str, object]:
    fit_source_commit = _fit_source_commit()
    manifest, labels, state_cols = load_labels(aggregate_dir)
    x, feature_names = build_features(labels, state_cols)
    selected, folds = nested_predictions(labels, x)
    global_mean = global_mean_predictions(labels)
    oracle = oracle_predictions(labels)
    summaries = {
        "nested_causal_ridge": summarize_selection(selected, seed=28280),
        "global_action_mean": summarize_selection(global_mean, seed=28380),
        "oracle": summarize_selection(oracle, seed=28480),
    }
    primary = summaries["nested_causal_ridge"]["outcomes"]["service_cost_restricted"]
    journey = summaries["nested_causal_ridge"]["outcomes"]["journey_min"]
    cv = summaries["nested_causal_ridge"]["outcomes"]["headway_cv"]
    fleet = summaries["nested_causal_ridge"]["outcomes"]["fleet_overshoot"]
    completion = summaries["nested_causal_ridge"]["outcomes"]["trip_completion_rate"]
    unserved = summaries["nested_causal_ridge"]["outcomes"]["passenger_unserved_rate"]
    per_train = selected.groupby("train_seed")[PRIMARY_DELTA].mean()
    global_primary = summaries["global_action_mean"]["outcomes"]["service_cost_restricted"]
    checks = {
        "strict_complete_input": manifest.get("strict_complete") is True,
        "four_outer_train_seed_folds": set(selected["outer_holdout_train_seed"].astype(int))
        == set(TRAIN_SEEDS),
        "causal_features_only": True,
        "nontrivial_override_fraction": summaries["nested_causal_ridge"][
            "non_actor_fraction"] >= 0.05,
        "primary_crossed_ci_below_zero": float(primary["ci_high"]) < 0.0,
        "primary_improves_in_three_train_seeds": int((per_train < 0.0).sum()) >= 3,
        "beats_global_action_mean": float(primary["mean"]) < float(global_primary["mean"]),
        "journey_noninferior": float(journey["ci_high"]) <= 0.10,
        "headway_cv_noninferior": bool(
            float(cv["mean"]) <= 0.0 and float(cv["ci_high"]) <= 0.003
        ),
        "fleet_overshoot_noninferior": float(fleet["ci_high"]) <= 0.25,
        "completion_not_reduced": float(completion["ci_low"]) >= -1e-9,
        "unserved_not_increased": float(unserved["ci_high"]) <= 1e-9,
    }
    eligible = all(checks.values())
    report: dict[str, object] = {
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "status": "eligible_for_untouched_online_screen" if eligible else "no_pass",
        "claim_eligible": False,
        "source_commit": manifest.get("source_commit"),
        "rollout_source_commit": manifest.get("rollout_source_commit"),
        "aggregation_source_commit": manifest.get("aggregation_source_commit"),
        "fit_source_commit": fit_source_commit,
        "feature_contract": "causal_s_upper_plus_candidate_action_v1",
        "prohibited_features": [
            "domain", "config", "seed identity", "future trace", "outcome-derived"
        ],
        "state_dimension": len(state_cols),
        "feature_dimension": len(feature_names),
        "outer_folds": folds,
        "per_train_seed_primary_mean": {
            str(key): float(value) for key, value in per_train.items()
        },
        "summaries": summaries,
        "gate_checks": checks,
        "gate_note": (
            "Passing permits only a separately frozen untouched-seed online screen; "
            "offline cross-validation cannot promote the controller."
        ),
    }
    model = final_model(
        labels,
        x,
        feature_names,
        fit_source_commit=fit_source_commit,
    )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "nested_selected_contexts.csv", index=False)
    global_mean.to_csv(out_dir / "global_mean_selected_contexts.csv", index=False)
    oracle.to_csv(out_dir / "oracle_selected_contexts.csv", index=False)
    (out_dir / "value_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "value_model_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = fit(args.aggregate_dir, args.out_dir)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "value_model_invalid.json").write_text(json.dumps({
            "protocol_version": MODEL_PROTOCOL_VERSION,
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2, sort_keys=True) + "\n")
        raise
    print(
        f"DONE V28 value gate status={report['status']} "
        f"features={report['feature_dimension']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
