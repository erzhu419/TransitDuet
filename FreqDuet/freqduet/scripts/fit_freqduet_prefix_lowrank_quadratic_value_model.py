#!/usr/bin/env python3
"""Develop the fold-local low-rank V29 quadratic treatment model on V28."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_protocol_v6_v28_prefix_common import (
    OUTCOME_DELTAS,
    PRIMARY_DELTA,
    TRAIN_SEEDS,
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
    source_commit,
)
from scripts.fit_freqduet_prefix_value_model import (
    CONTEXT_KEYS,
    MODEL_METHODS,
    load_labels,
    oracle_predictions,
    select_rows,
    summarize_selection,
)


PROTOCOL_VERSION = "freqduet-v29-lowrank-quadratic-development-v2"
FEATURE_CONTRACT = "causal_foldlocal_pca_signed_quadratic_v1"
CONTEXT_RANKS = [0, 1, 2, 4]
BASIS_NAMES = ["positive", "negative", "positive_sq", "negative_sq"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def raw_context(
    labels: pd.DataFrame,
    state_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    state = labels[state_cols].to_numpy(dtype=np.float64)
    actor = np.stack(labels["actor_action"].to_list()).astype(np.float64)
    context = np.concatenate([state, actor / 60.0], axis=1)
    names = list(state_cols) + [
        f"actor_action_{index:02d}" for index in range(actor.shape[1])
    ]
    _require(context.shape[1] == len(names), "raw context name mismatch")
    _require(np.isfinite(context).all(), "non-finite V29 context")
    return context, names


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
    _require(np.isfinite(basis).all(), "non-finite treatment basis")
    return basis


def _fix_component_signs(components: np.ndarray) -> np.ndarray:
    fixed = np.asarray(components, dtype=np.float64).copy()
    for row in fixed:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return fixed


def fit_context_projection(
    labels: pd.DataFrame,
    state_cols: list[str],
    train_mask: np.ndarray,
    *,
    rank: int,
) -> dict[str, object]:
    rank = int(rank)
    _require(rank in CONTEXT_RANKS, f"unregistered context rank {rank}")
    mask = np.asarray(train_mask, dtype=bool)
    _require(mask.shape == (len(labels),), "projection train mask mismatch")
    context, context_names = raw_context(labels, state_cols)
    actor_mask = labels["candidate_method"].eq("actor").to_numpy() & mask
    train_contexts = labels.loc[mask].groupby(CONTEXT_KEYS).ngroups
    _require(int(actor_mask.sum()) == int(train_contexts),
             "projection requires one actor row per training context")
    _require(train_contexts >= 2, "projection has fewer than two contexts")

    actor_context = context[actor_mask]
    mean = actor_context.mean(axis=0)
    scale = actor_context.std(axis=0)
    active = scale > 1e-12
    scale = np.where(active, scale, 1.0)
    standardized = (actor_context - mean) / scale
    max_rank = min(
        standardized.shape[0] - 1,
        int(np.count_nonzero(active)),
        standardized.shape[1],
    )
    _require(rank <= max_rank, f"rank {rank} exceeds fitted rank {max_rank}")

    if rank == 0:
        components = np.zeros((0, standardized.shape[1]), dtype=np.float64)
        explained = np.zeros(0, dtype=np.float64)
    else:
        _, singular, right = np.linalg.svd(standardized, full_matrices=False)
        components = _fix_component_signs(right[:rank])
        energy = np.square(singular)
        total = float(energy.sum())
        explained = (
            energy[:rank] / total if total > 1e-12
            else np.zeros(rank, dtype=np.float64)
        )
    return {
        "context_columns": context_names,
        "context_mean": mean,
        "context_scale": scale,
        "context_components": components,
        "context_rank": rank,
        "explained_variance_fraction": explained,
        "fitted_actor_contexts": int(actor_mask.sum()),
    }


def transform_features(
    labels: pd.DataFrame,
    state_cols: list[str],
    projection: dict[str, object],
) -> tuple[np.ndarray, list[str]]:
    context, context_names = raw_context(labels, state_cols)
    expected_names = [str(name) for name in projection["context_columns"]]
    _require(context_names == expected_names, "projection context schema mismatch")
    mean = np.asarray(projection["context_mean"], dtype=np.float64)
    scale = np.asarray(projection["context_scale"], dtype=np.float64)
    components = np.asarray(
        projection["context_components"], dtype=np.float64
    )
    rank = int(projection["context_rank"])
    _require(mean.shape == scale.shape == (context.shape[1],),
             "projection normalization shape mismatch")
    _require(components.shape == (rank, context.shape[1]),
             "projection component shape mismatch")

    basis = treatment_basis(labels)
    projected = ((context - mean) / scale) @ components.T
    interactions = (basis[:, :, None] * projected[:, None, :]).reshape(
        len(labels), -1
    )
    design = np.concatenate([basis, interactions], axis=1)
    names = list(BASIS_NAMES)
    names.extend(
        f"{basis_name}_x_context_pc_{index:02d}"
        for basis_name in BASIS_NAMES
        for index in range(rank)
    )
    _require(design.shape == (len(labels), 4 * (rank + 1)),
             "low-rank treatment design shape mismatch")
    _require(design.shape[1] == len(names), "low-rank feature name mismatch")
    _require(np.isfinite(design).all(), "non-finite low-rank features")
    identity = np.all(basis == 0.0, axis=1)
    _require(np.array_equal(
        design[identity], np.zeros_like(design[identity])
    ), "zero treatment has nonzero low-rank features")
    prohibited = ("domain", "config", "seed", "future", "outcome", "episode_")
    bad = [name for name in names if any(
        token in name.lower() for token in prohibited
    )]
    _require(not bad, f"prohibited low-rank features: {bad}")
    return design, names


def fit_pipeline(
    labels: pd.DataFrame,
    state_cols: list[str],
    train_mask: np.ndarray,
    *,
    rank: int,
    alpha: float,
) -> tuple[dict[str, object], np.ndarray, list[str]]:
    projection = fit_context_projection(
        labels, state_cols, train_mask, rank=rank
    )
    design, names = transform_features(labels, state_cols, projection)
    target = labels[PRIMARY_DELTA].to_numpy(dtype=np.float64)
    ridge = fit_zero_baseline_ridge(
        design[train_mask], target[train_mask], alpha
    )
    pipeline = {
        **projection,
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
    state_cols: list[str],
    *,
    outer_train_seeds: list[int],
    rank: int,
    alpha: float,
    guard_margin: float,
) -> float:
    scores = []
    seeds = labels["train_seed"].to_numpy(dtype=int)
    for heldout in outer_train_seeds:
        inner_train = [seed for seed in outer_train_seeds if seed != heldout]
        train_mask = np.isin(seeds, inner_train)
        test_mask = seeds == heldout
        pipeline, design, _ = fit_pipeline(
            labels,
            state_cols,
            train_mask,
            rank=rank,
            alpha=alpha,
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
    state_cols: list[str],
    train_seeds: list[int],
) -> tuple[int, float, float, list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    for rank in CONTEXT_RANKS:
        for alpha in ALPHAS:
            for margin in GUARD_MARGINS:
                score = inner_score(
                    labels,
                    state_cols,
                    outer_train_seeds=train_seeds,
                    rank=rank,
                    alpha=alpha,
                    guard_margin=margin,
                )
                rows.append({
                    "context_rank": int(rank),
                    "alpha": float(alpha),
                    "guard_margin": float(margin),
                    "score": score,
                })
    best = min(rows, key=lambda row: (
        row["score"], row["context_rank"],
        -row["guard_margin"], -row["alpha"],
    ))
    return (
        int(best["context_rank"]),
        float(best["alpha"]),
        float(best["guard_margin"]),
        rows,
    )


def nested_predictions(
    labels: pd.DataFrame,
    state_cols: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selected_parts = []
    folds: list[dict[str, object]] = []
    seeds = labels["train_seed"].to_numpy(dtype=int)
    for heldout in TRAIN_SEEDS:
        outer_train = [seed for seed in TRAIN_SEEDS if seed != heldout]
        rank, alpha, margin, grid = choose_hyperparameters(
            labels, state_cols, outer_train
        )
        train_mask = np.isin(seeds, outer_train)
        test_mask = seeds == heldout
        pipeline, design, names = fit_pipeline(
            labels,
            state_cols,
            train_mask,
            rank=rank,
            alpha=alpha,
        )
        selected = select_rows(
            labels.loc[test_mask].reset_index(drop=True),
            predict_pipeline(pipeline, design[test_mask]),
            guard_margin=margin,
        )
        selected["outer_holdout_train_seed"] = int(heldout)
        selected["selected_context_rank"] = int(rank)
        selected["selected_alpha"] = alpha
        selected["selected_guard_margin"] = margin
        selected_parts.append(selected)
        folds.append({
            "heldout_train_seed": int(heldout),
            "inner_train_seeds": outer_train,
            "selected_context_rank": int(rank),
            "selected_alpha": alpha,
            "selected_guard_margin": margin,
            "feature_dimension": len(names),
            "projection_actor_contexts": pipeline["fitted_actor_contexts"],
            "projection_explained_variance_fraction": np.asarray(
                pipeline["explained_variance_fraction"]
            ).tolist(),
            "inner_grid": grid,
        })
    return pd.concat(selected_parts, ignore_index=True), folds


def final_model(
    labels: pd.DataFrame,
    state_cols: list[str],
    *,
    fit_source_commit: str,
    input_manifest: dict,
) -> dict[str, object]:
    rank, alpha, margin, grid = choose_hyperparameters(
        labels, state_cols, TRAIN_SEEDS
    )
    train_mask = np.ones(len(labels), dtype=bool)
    pipeline, _, names = fit_pipeline(
        labels,
        state_cols,
        train_mask,
        rank=rank,
        alpha=alpha,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
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
        "state_columns": state_cols,
        "context_columns": pipeline["context_columns"],
        "context_mean": np.asarray(pipeline["context_mean"]).tolist(),
        "context_scale": np.asarray(pipeline["context_scale"]).tolist(),
        "context_components": np.asarray(
            pipeline["context_components"]
        ).tolist(),
        "context_rank": int(rank),
        "projection_explained_variance_fraction": np.asarray(
            pipeline["explained_variance_fraction"]
        ).tolist(),
        "feature_names": names,
        "feature_scale": np.asarray(pipeline["feature_scale"]).tolist(),
        "coefficient": np.asarray(pipeline["coefficient"]).tolist(),
        "alpha": alpha,
        "guard_margin": margin,
        "full_seed_cv_grid": grid,
    }


def develop(aggregate_dir: Path, out_dir: Path) -> dict[str, object]:
    fit_commit = source_commit()
    manifest, labels, state_cols = load_labels(aggregate_dir)
    selected, folds = nested_predictions(labels, state_cols)
    p30 = fixed_method(labels, GLOBAL_COMPARATOR)
    oracle = oracle_predictions(labels)
    model = final_model(
        labels,
        state_cols,
        fit_source_commit=fit_commit,
        input_manifest=manifest,
    )
    summaries = {
        "nested_lowrank_quadratic": summarize_selection(selected, seed=29429),
        "global_p30": summarize_selection(p30, seed=29529),
        "oracle": summarize_selection(oracle, seed=29629),
        "nested_minus_global_p30": paired_summary(
            selected, p30, seed=29729
        ),
    }
    primary = summaries["nested_lowrank_quadratic"]["outcomes"][
        "service_cost_restricted"
    ]
    paired_primary = summaries["nested_minus_global_p30"]["outcomes"][
        "service_cost_restricted"
    ]
    journey = summaries["nested_lowrank_quadratic"]["outcomes"]["journey_min"]
    cv = summaries["nested_lowrank_quadratic"]["outcomes"]["headway_cv"]
    fleet = summaries["nested_lowrank_quadratic"]["outcomes"]["fleet_overshoot"]
    completion = summaries["nested_lowrank_quadratic"]["outcomes"][
        "trip_completion_rate"
    ]
    unserved = summaries["nested_lowrank_quadratic"]["outcomes"][
        "passenger_unserved_rate"
    ]
    per_train = selected.groupby("train_seed")[PRIMARY_DELTA].mean()
    interior_fraction = float(
        selected["candidate_method"].isin(INTERIOR_METHODS).mean()
    )
    feature_dimensions = [int(fold["feature_dimension"]) for fold in folds]
    checks = {
        "strict_complete_v28_input": manifest.get("strict_complete") is True,
        "four_outer_train_seed_folds": set(
            selected["outer_holdout_train_seed"].astype(int)
        ) == set(TRAIN_SEEDS),
        "zero_actor_feature_contract": True,
        "causal_features_only": True,
        "projection_fit_inside_fold": True,
        "low_capacity_at_most_20_features": max(feature_dimensions) <= 20,
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
        "protocol_version": PROTOCOL_VERSION,
        "status": (
            "ready_to_preregister_v29_confirmation"
            if ready else "development_no_pass"
        ),
        "claim_eligible": False,
        "development_dataset": "V28 exact-prefix labels",
        "fit_source_commit": fit_commit,
        "input_manifest": manifest,
        "feature_contract": FEATURE_CONTRACT,
        "state_dimension": len(state_cols),
        "candidate_context_ranks": CONTEXT_RANKS,
        "outer_feature_dimensions": feature_dimensions,
        "global_comparator": GLOBAL_COMPARATOR,
        "interior_action_fraction": interior_fraction,
        "per_train_seed_primary_mean": {
            str(key): float(value) for key, value in per_train.items()
        },
        "outer_folds": folds,
        "summaries": summaries,
        "development_checks": checks,
        "interpretation": (
            "Development evidence only. Passing freezes this exact low-rank "
            "pipeline and permits one external V29 label confirmation; it is "
            "not an effect claim or online-policy promotion."
        ),
    }

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "nested_selected_contexts.csv", index=False)
    p30.to_csv(out_dir / "global_p30_contexts.csv", index=False)
    oracle.to_csv(out_dir / "oracle_selected_contexts.csv", index=False)
    (out_dir / "lowrank_quadratic_value_model.json").write_text(
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
            "protocol_version": PROTOCOL_VERSION,
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2, sort_keys=True) + "\n")
        raise
    print(
        f"DONE V29 D2 development status={report['status']} "
        f"outer_dims={report['outer_feature_dimensions']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
