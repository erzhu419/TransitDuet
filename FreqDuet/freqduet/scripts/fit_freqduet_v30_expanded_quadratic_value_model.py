#!/usr/bin/env python3
"""Develop the frozen low-capacity V30 expanded-observation value model."""

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

from scripts.audit_protocol_v6_v30_expanded_prefix_common import (
    CONFIRMATION_DECISION_INDICES,
    CONFIRMATION_EVAL_EPISODE,
    CONFIRMATION_POLICY_SEEDS,
    CONFIRMATION_REPLAY_SEED,
    CONFIRMATION_SCENARIO_SEEDS,
    DISCOVERY_DECISION_INDICES,
    DISCOVERY_EVAL_EPISODE,
    DISCOVERY_REPLAY_SEED,
    DISCOVERY_SCENARIO_SEEDS,
    DISCOVERY_TRAIN_SEEDS,
    EXPANDED_CONTEXT_COLUMNS,
    EXPECTED_METHODS,
    LABEL_PROTOCOL_VERSION,
    MATRIX_PROTOCOL_VERSION,
    MODEL_PROTOCOL_VERSION,
    OFFSETS_S,
    OUTCOME_DELTAS,
    PRIMARY_DELTA,
)
from scripts.fit_freqduet_prefix_quadratic_value_model import (
    ALPHAS,
    GLOBAL_COMPARATOR,
    GUARD_MARGINS,
    INTERIOR_METHODS,
    _action_arrays,
    fit_zero_baseline_ridge,
    fixed_method,
    paired_summary,
    predict,
)
from scripts.fit_freqduet_prefix_value_model import (
    CONTEXT_KEYS,
    oracle_predictions,
    select_rows,
    summarize_selection,
)
from scripts.run_freqduet_protocol_v2_matrix import git_provenance


FEATURE_CONTRACT = "waiting_cv_signed_quadratic_v1"
MODEL_METHODS = [
    method for method in EXPECTED_METHODS
    if method != "actor_firstknot_0"
]
BASIS_NAMES = ["positive", "negative", "positive_sq", "negative_sq"]
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_commit() -> str:
    source = git_provenance()
    commit = str(source.get("commit", "")).lower()
    _require(bool(COMMIT_RE.fullmatch(commit)),
             f"invalid V30 development source commit {commit!r}")
    _require(source.get("tracked_dirty") is False,
             "V30 development source snapshot is tracked-dirty")
    return commit


def _array(text: str, label: str) -> np.ndarray:
    try:
        value = np.asarray(json.loads(str(text)), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"invalid {label} JSON") from exc
    _require(value.size > 0 and np.isfinite(value).all(), f"non-finite {label}")
    return value


def load_labels(aggregate_dir: Path) -> tuple[dict, pd.DataFrame]:
    root = Path(aggregate_dir).resolve()
    manifest = json.loads((root / "matrix_manifest.json").read_text())
    _require(manifest.get("protocol_version") == MATRIX_PROTOCOL_VERSION,
             "wrong V30 discovery matrix protocol")
    _require(manifest.get("strict_complete") is True,
             "V30 discovery matrix is not strict-complete")
    expected_manifest = {
        "label_protocol_version": LABEL_PROTOCOL_VERSION,
        "train_seeds": DISCOVERY_TRAIN_SEEDS,
        "eval_seeds": DISCOVERY_SCENARIO_SEEDS,
        "decision_indices": DISCOVERY_DECISION_INDICES,
        "offsets_s": OFFSETS_S,
        "eval_episode": DISCOVERY_EVAL_EPISODE,
        "replay_seed": DISCOVERY_REPLAY_SEED,
        "required_context_columns": EXPANDED_CONTEXT_COLUMNS,
    }
    for key, expected in expected_manifest.items():
        _require(manifest.get(key) == expected,
                 f"V30 manifest {key} does not match frozen discovery roster")

    labels = pd.read_csv(root / "prefix_counterfactual_all.csv")
    labels = labels[labels["candidate_method"].isin(MODEL_METHODS)].copy()
    context_count = (
        len(DISCOVERY_TRAIN_SEEDS)
        * len(DISCOVERY_SCENARIO_SEEDS)
        * len(DISCOVERY_DECISION_INDICES)
    )
    _require(len(labels) == context_count * len(MODEL_METHODS),
             "V30 model row count does not match the frozen discovery matrix")
    _require(set(labels["candidate_method"].astype(str)) == set(MODEL_METHODS),
             "V30 model candidate roster mismatch")
    counts = labels.groupby(CONTEXT_KEYS)["candidate_method"].nunique()
    _require(len(counts) == context_count, "V30 model context count mismatch")
    _require(counts.eq(len(MODEL_METHODS)).all(),
             "a V30 context has incomplete candidates")
    _require(not labels.duplicated(CONTEXT_KEYS + ["candidate_method"]).any(),
             "duplicate V30 model rows")

    numeric_columns = (
        list(CONTEXT_KEYS)
        + ["candidate_offset_s"]
        + EXPANDED_CONTEXT_COLUMNS
        + list(OUTCOME_DELTAS.values())
    )
    for column in numeric_columns:
        values = pd.to_numeric(labels[column], errors="coerce")
        _require(np.isfinite(values.to_numpy(dtype=np.float64)).all(),
                 f"non-finite V30 model column {column}")
        labels[column] = values
    labels["actor_action"] = [
        _array(value, "actor action") for value in labels["actor_action_json"]
    ]
    labels["candidate_action"] = [
        _array(value, "candidate action")
        for value in labels["candidate_action_json"]
    ]
    action_dims = {value.size for value in labels["actor_action"]}
    action_dims.update(value.size for value in labels["candidate_action"])
    _require(len(action_dims) == 1,
             f"inconsistent V30 action dimensions: {sorted(action_dims)}")
    return manifest, labels.reset_index(drop=True)


def treatment_basis(labels: pd.DataFrame) -> np.ndarray:
    _, signed_delta_s = _action_arrays(labels)
    signed = signed_delta_s / 30.0
    positive = np.maximum(signed, 0.0)
    negative = np.maximum(-signed, 0.0)
    basis = np.column_stack([
        positive,
        negative,
        np.square(positive),
        np.square(negative),
    ])
    _require(np.isfinite(basis).all(), "non-finite V30 treatment basis")
    return basis


def fit_context_scaler(
    labels: pd.DataFrame,
    train_mask: np.ndarray,
) -> dict[str, object]:
    mask = np.asarray(train_mask, dtype=bool)
    _require(mask.shape == (len(labels),), "V30 scaler train mask mismatch")
    context = labels[EXPANDED_CONTEXT_COLUMNS].to_numpy(dtype=np.float64)
    actor_mask = labels["candidate_method"].eq("actor").to_numpy() & mask
    train_contexts = labels.loc[mask].groupby(CONTEXT_KEYS).ngroups
    _require(int(actor_mask.sum()) == int(train_contexts),
             "V30 scaler requires one actor row per training context")
    _require(train_contexts >= 2, "V30 scaler has fewer than two contexts")
    actor_context = context[actor_mask]
    mean = actor_context.mean(axis=0)
    scale = actor_context.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return {
        "context_columns": list(EXPANDED_CONTEXT_COLUMNS),
        "context_mean": mean,
        "context_scale": scale,
        "fitted_actor_contexts": int(actor_mask.sum()),
    }


def transform_features(
    labels: pd.DataFrame,
    scaler: dict[str, object],
) -> tuple[np.ndarray, list[str]]:
    _require(
        list(scaler["context_columns"]) == EXPANDED_CONTEXT_COLUMNS,
        "V30 context schema mismatch",
    )
    context = labels[EXPANDED_CONTEXT_COLUMNS].to_numpy(dtype=np.float64)
    mean = np.asarray(scaler["context_mean"], dtype=np.float64)
    scale = np.asarray(scaler["context_scale"], dtype=np.float64)
    _require(mean.shape == scale.shape == (len(EXPANDED_CONTEXT_COLUMNS),),
             "V30 context normalization shape mismatch")
    standardized = (context - mean) / scale
    basis = treatment_basis(labels)
    interactions = (basis[:, :, None] * standardized[:, None, :]).reshape(
        len(labels), -1
    )
    design = np.concatenate([basis, interactions], axis=1)
    names = list(BASIS_NAMES)
    names.extend(
        f"{basis_name}_x_{context_name}"
        for basis_name in BASIS_NAMES
        for context_name in EXPANDED_CONTEXT_COLUMNS
    )
    _require(design.shape == (len(labels), 12),
             "V30 treatment design is not exactly 12-dimensional")
    _require(len(names) == 12, "V30 feature name mismatch")
    _require(np.isfinite(design).all(), "non-finite V30 features")
    identity = np.all(basis == 0.0, axis=1)
    _require(np.array_equal(
        design[identity], np.zeros_like(design[identity])
    ), "zero treatment has nonzero V30 features")
    return design, names


def fit_pipeline(
    labels: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    alpha: float,
) -> tuple[dict[str, object], np.ndarray, list[str]]:
    scaler = fit_context_scaler(labels, train_mask)
    design, names = transform_features(labels, scaler)
    target = labels[PRIMARY_DELTA].to_numpy(dtype=np.float64)
    ridge = fit_zero_baseline_ridge(
        design[train_mask], target[train_mask], alpha
    )
    pipeline = {
        **scaler,
        "feature_names": names,
        "feature_scale": np.asarray(ridge["feature_scale"], dtype=np.float64),
        "coefficient": np.asarray(ridge["coefficient"], dtype=np.float64),
        "alpha": float(alpha),
    }
    return pipeline, design, names


def predict_pipeline(pipeline: dict[str, object], design: np.ndarray) -> np.ndarray:
    return predict({
        "feature_scale": pipeline["feature_scale"],
        "coefficient": pipeline["coefficient"],
    }, design)


def inner_score(
    labels: pd.DataFrame,
    *,
    outer_train_seeds: list[int],
    alpha: float,
    guard_margin: float,
) -> float:
    scores: list[float] = []
    seeds = labels["train_seed"].to_numpy(dtype=int)
    for heldout in outer_train_seeds:
        inner_train = [seed for seed in outer_train_seeds if seed != heldout]
        train_mask = np.isin(seeds, inner_train)
        test_mask = seeds == heldout
        pipeline, design, _ = fit_pipeline(
            labels, train_mask, alpha=alpha
        )
        selected = select_rows(
            labels.loc[test_mask].reset_index(drop=True),
            predict_pipeline(pipeline, design[test_mask]),
            guard_margin=guard_margin,
        )
        scores.append(float(selected[PRIMARY_DELTA].mean()))
    return float(np.mean(scores))


def choose_hyperparameters(
    labels: pd.DataFrame,
    train_seeds: list[int],
) -> tuple[float, float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        for margin in GUARD_MARGINS:
            rows.append({
                "alpha": float(alpha),
                "guard_margin": float(margin),
                "score": inner_score(
                    labels,
                    outer_train_seeds=train_seeds,
                    alpha=alpha,
                    guard_margin=margin,
                ),
            })
    best = min(rows, key=lambda row: (
        row["score"], -row["guard_margin"], -row["alpha"]
    ))
    return float(best["alpha"]), float(best["guard_margin"]), rows


def nested_predictions(
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selected_parts: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    seeds = labels["train_seed"].to_numpy(dtype=int)
    for heldout in DISCOVERY_TRAIN_SEEDS:
        outer_train = [
            seed for seed in DISCOVERY_TRAIN_SEEDS if seed != heldout
        ]
        alpha, margin, grid = choose_hyperparameters(labels, outer_train)
        train_mask = np.isin(seeds, outer_train)
        test_mask = seeds == heldout
        pipeline, design, names = fit_pipeline(
            labels, train_mask, alpha=alpha
        )
        selected = select_rows(
            labels.loc[test_mask].reset_index(drop=True),
            predict_pipeline(pipeline, design[test_mask]),
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
            "feature_dimension": len(names),
            "scaler_actor_contexts": pipeline["fitted_actor_contexts"],
            "inner_grid": grid,
        })
    return pd.concat(selected_parts, ignore_index=True), folds


def final_model(
    labels: pd.DataFrame,
    *,
    fit_source_commit: str,
    input_manifest: dict,
) -> dict[str, object]:
    alpha, margin, grid = choose_hyperparameters(
        labels, DISCOVERY_TRAIN_SEEDS
    )
    train_mask = np.ones(len(labels), dtype=bool)
    pipeline, _, names = fit_pipeline(labels, train_mask, alpha=alpha)
    return {
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "development_only": True,
        "claim_eligible": False,
        "feature_contract": FEATURE_CONTRACT,
        "fit_source_commit": fit_source_commit,
        "training_rollout_source_commit": input_manifest.get(
            "rollout_source_commit"
        ),
        "training_aggregation_source_commit": input_manifest.get(
            "aggregation_source_commit"
        ),
        "target": PRIMARY_DELTA,
        "candidate_methods": MODEL_METHODS,
        "global_comparator": GLOBAL_COMPARATOR,
        "context_columns": list(EXPANDED_CONTEXT_COLUMNS),
        "context_mean": np.asarray(pipeline["context_mean"]).tolist(),
        "context_scale": np.asarray(pipeline["context_scale"]).tolist(),
        "feature_names": names,
        "feature_scale": np.asarray(pipeline["feature_scale"]).tolist(),
        "coefficient": np.asarray(pipeline["coefficient"]).tolist(),
        "alpha": alpha,
        "guard_margin": margin,
        "full_seed_cv_grid": grid,
    }


def develop(aggregate_dir: Path, out_dir: Path) -> dict[str, object]:
    fit_commit = source_commit()
    manifest, labels = load_labels(aggregate_dir)
    selected, folds = nested_predictions(labels)
    p30 = fixed_method(labels, GLOBAL_COMPARATOR)
    oracle = oracle_predictions(labels)
    summaries = {
        "nested_expanded_quadratic": summarize_selection(selected, seed=30030),
        "global_p30": summarize_selection(p30, seed=30130),
        "oracle": summarize_selection(oracle, seed=30230),
        "nested_minus_global_p30": paired_summary(
            selected, p30, seed=30330
        ),
    }
    primary = summaries["nested_expanded_quadratic"]["outcomes"][
        "service_cost_restricted"
    ]
    paired_primary = summaries["nested_minus_global_p30"]["outcomes"][
        "service_cost_restricted"
    ]
    journey = summaries["nested_expanded_quadratic"]["outcomes"]["journey_min"]
    cv = summaries["nested_expanded_quadratic"]["outcomes"]["headway_cv"]
    fleet = summaries["nested_expanded_quadratic"]["outcomes"]["fleet_overshoot"]
    completion = summaries["nested_expanded_quadratic"]["outcomes"][
        "trip_completion_rate"
    ]
    unserved = summaries["nested_expanded_quadratic"]["outcomes"][
        "passenger_unserved_rate"
    ]
    per_train = selected.groupby("train_seed")[PRIMARY_DELTA].mean()
    interior_fraction = float(
        selected["candidate_method"].isin(INTERIOR_METHODS).mean()
    )
    checks = {
        "strict_complete_v30_discovery_input": (
            manifest.get("strict_complete") is True
        ),
        "four_outer_train_seed_folds": set(
            selected["outer_holdout_train_seed"].astype(int)
        ) == set(DISCOVERY_TRAIN_SEEDS),
        "expanded_context_exact": list(EXPANDED_CONTEXT_COLUMNS)
        == ["waiting_total_pre", "headway_cv_active_pre"],
        "zero_actor_feature_contract": True,
        "scaler_fit_inside_fold": True,
        "low_capacity_exactly_12_features": all(
            int(fold["feature_dimension"]) == 12 for fold in folds
        ),
        "nontrivial_override_fraction": float(
            (selected["candidate_method"] != "actor").mean()
        ) >= 0.05,
        "interior_action_fraction": interior_fraction >= 0.05,
        "primary_crossed_ci_below_zero": float(primary["ci_high"]) < 0.0,
        "primary_improves_in_three_train_seeds": int(
            (per_train < 0.0).sum()
        ) >= 3,
        "mean_beats_global_p30": float(paired_primary["mean"]) < 0.0,
        "paired_ci_beats_global_p30": float(paired_primary["ci_high"]) < 0.0,
        "journey_noninferior": float(journey["ci_high"]) <= 0.10,
        "headway_cv_noninferior": bool(
            float(cv["mean"]) <= 0.0 and float(cv["ci_high"]) <= 0.003
        ),
        "fleet_overshoot_noninferior": float(fleet["ci_high"]) <= 0.25,
        "completion_not_reduced": float(completion["ci_low"]) >= -1e-9,
        "unserved_not_increased": float(unserved["ci_high"]) <= 1e-9,
    }
    ready = all(checks.values())
    report: dict[str, object] = {
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "status": (
            "ready_to_freeze_v30_confirmation"
            if ready else "development_no_pass"
        ),
        "claim_eligible": False,
        "development_dataset": "V30 fresh-context exact-prefix labels",
        "fit_source_commit": fit_commit,
        "input_manifest": manifest,
        "feature_contract": FEATURE_CONTRACT,
        "feature_dimension": 12,
        "global_comparator": GLOBAL_COMPARATOR,
        "interior_action_fraction": interior_fraction,
        "per_train_seed_primary_mean": {
            str(key): float(value) for key, value in per_train.items()
        },
        "outer_folds": folds,
        "summaries": summaries,
        "development_checks": checks,
        "frozen_confirmation_roster": {
            "policy_seeds": CONFIRMATION_POLICY_SEEDS,
            "scenario_seeds": CONFIRMATION_SCENARIO_SEEDS,
            "decision_indices": CONFIRMATION_DECISION_INDICES,
            "eval_episode": CONFIRMATION_EVAL_EPISODE,
            "replay_seed": CONFIRMATION_REPLAY_SEED,
        },
        "interpretation": (
            "Development evidence only. Passing freezes this exact 12-feature "
            "pipeline and permits one confirmation on the already frozen fresh "
            "policy/scenario/decision roster; it cannot promote the controller."
        ),
    }
    model = final_model(
        labels,
        fit_source_commit=fit_commit,
        input_manifest=manifest,
    )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "nested_selected_contexts.csv", index=False)
    p30.to_csv(out_dir / "global_p30_contexts.csv", index=False)
    oracle.to_csv(out_dir / "oracle_selected_contexts.csv", index=False)
    (out_dir / "expanded_quadratic_value_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "development_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = develop(args.aggregate_dir, args.out_dir)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "development_invalid.json").write_text(json.dumps({
            "protocol_version": MODEL_PROTOCOL_VERSION,
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2, sort_keys=True) + "\n")
        raise
    print(
        f"DONE V30 development status={report['status']} "
        f"features={report['feature_dimension']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
