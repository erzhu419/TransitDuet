#!/usr/bin/env python3
"""Develop the V29 zero-baseline quadratic treatment-effect model on V28."""

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
    OUTCOME_DELTAS,
    PRIMARY_DELTA,
    TRAIN_SEEDS,
)
from scripts.fit_freqduet_prefix_value_model import (
    CONTEXT_KEYS,
    MODEL_METHODS,
    load_labels,
    oracle_predictions,
    select_rows,
    summarize_selection,
)
from scripts.run_freqduet_protocol_v2_matrix import git_provenance


PROTOCOL_VERSION = "freqduet-v29-quadratic-treatment-development-v1"
FEATURE_CONTRACT = "causal_zero_baseline_signed_quadratic_v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
GUARD_MARGINS = [0.0, 0.0001, 0.00025, 0.0005]
GLOBAL_COMPARATOR = "actor_firstknot_p30"
INTERIOR_METHODS = {"actor_firstknot_m15", "actor_firstknot_p15"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_commit() -> str:
    source = git_provenance()
    commit = str(source.get("commit", "")).lower()
    _require(bool(COMMIT_RE.fullmatch(commit)),
             f"invalid V29 development source commit {commit!r}")
    _require(source.get("tracked_dirty") is False,
             "V29 development source snapshot is tracked-dirty")
    return commit


def _action_arrays(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    actor = np.stack(labels["actor_action"].to_list()).astype(np.float64)
    candidate = np.stack(labels["candidate_action"].to_list()).astype(np.float64)
    _require(actor.shape == candidate.shape and actor.ndim == 2,
             "actor/candidate action shapes differ")
    difference = candidate - actor
    changed = np.sum(np.abs(difference) > 1e-7, axis=1)
    _require(np.all(changed <= 1),
             "V29 treatment basis requires at most one changed action coordinate")
    signed_delta_s = difference.sum(axis=1)
    requested = pd.to_numeric(
        labels["candidate_offset_s"], errors="raise"
    ).to_numpy(dtype=np.float64)
    _require(np.all(requested * signed_delta_s >= -1e-8),
             "executed treatment sign disagrees with requested offset")
    zero_requested = np.abs(requested) <= 1e-12
    _require(np.array_equal(
        signed_delta_s[zero_requested], np.zeros(zero_requested.sum())
    ), "zero-offset treatment changed the executed action")
    identity = labels["candidate_method"].eq("actor").to_numpy()
    _require(np.array_equal(signed_delta_s[identity], np.zeros(identity.sum())),
             "actor treatment is not exactly zero")
    return actor, signed_delta_s


def build_treatment_features(
    labels: pd.DataFrame,
    state_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    state = labels[state_cols].to_numpy(dtype=np.float64)
    actor, signed_delta_s = _action_arrays(labels)
    context = np.concatenate([state, actor / 60.0], axis=1)
    context_names = list(state_cols) + [
        f"actor_action_{index:02d}" for index in range(actor.shape[1])
    ]

    signed = signed_delta_s / 30.0
    positive = np.maximum(signed, 0.0)
    negative = np.maximum(-signed, 0.0)
    basis = np.column_stack([
        positive,
        negative,
        positive ** 2,
        negative ** 2,
    ])
    basis_names = ["positive", "negative", "positive_sq", "negative_sq"]
    interactions = (basis[:, :, None] * context[:, None, :]).reshape(
        len(labels), -1
    )
    names = list(basis_names)
    names.extend(
        f"{basis_name}_x_{context_name}"
        for basis_name in basis_names
        for context_name in context_names
    )
    design = np.concatenate([basis, interactions], axis=1)

    _require(design.shape[1] == len(names), "feature name/design mismatch")
    _require(np.isfinite(design).all(), "non-finite V29 treatment features")
    actor_rows = labels["candidate_method"].eq("actor").to_numpy()
    _require(np.array_equal(
        design[actor_rows], np.zeros_like(design[actor_rows])
    ), "actor features are not exactly zero")
    prohibited = ("domain", "config", "seed", "future", "outcome", "episode_")
    bad = [name for name in names if any(
        token in name.lower() for token in prohibited
    )]
    _require(not bad, f"prohibited V29 deployable features: {bad}")
    return design, names


def fit_zero_baseline_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    scale = np.sqrt(np.mean(np.square(x), axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = x / scale
    lhs = normalized.T @ normalized
    lhs += np.eye(normalized.shape[1], dtype=np.float64) * float(alpha)
    rhs = normalized.T @ np.asarray(y, dtype=np.float64)
    try:
        coefficient = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return {
        "feature_scale": scale,
        "coefficient": coefficient,
        "alpha": float(alpha),
    }


def predict(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    return (x / scale) @ coefficient


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
        train_mask = np.isin(
            seeds, [seed for seed in outer_train_seeds if seed != heldout]
        )
        test_mask = seeds == heldout
        model = fit_zero_baseline_ridge(x[train_mask], target[train_mask], alpha)
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
            rows.append({
                "alpha": float(alpha),
                "guard_margin": float(margin),
                "score": score,
            })
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
        alpha, margin, grid = choose_hyperparameters(labels, x, outer_train)
        train_mask = np.isin(seeds, outer_train)
        test_mask = seeds == heldout
        model = fit_zero_baseline_ridge(x[train_mask], target[train_mask], alpha)
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
            "inner_grid": grid,
        })
    return pd.concat(selected_parts, ignore_index=True), folds


def fixed_method(labels: pd.DataFrame, method: str) -> pd.DataFrame:
    selected = labels[labels["candidate_method"].eq(str(method))].copy()
    expected_contexts = labels.groupby(CONTEXT_KEYS).ngroups
    _require(len(selected) == expected_contexts,
             f"fixed comparator {method} is incomplete")
    return selected.reset_index(drop=True)


def paired_summary(
    selected: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, object]:
    columns = list(OUTCOME_DELTAS.values())
    left = selected[CONTEXT_KEYS + columns].copy()
    right = reference[CONTEXT_KEYS + columns].copy()
    paired = left.merge(
        right,
        on=CONTEXT_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_selected", "_reference"),
    )
    _require(len(paired) == len(selected) == len(reference),
             "selected/reference contexts do not align")
    outcomes: dict[str, dict[str, float | int]] = {}
    for index, (name, column) in enumerate(OUTCOME_DELTAS.items()):
        delta_column = f"{name}_selected_minus_reference"
        paired[delta_column] = (
            paired[f"{column}_selected"] - paired[f"{column}_reference"]
        )
        blocks = (
            paired.groupby(["train_seed", "scenario_seed"], sort=True)[
                delta_column
            ].mean().to_numpy(dtype=np.float64)
        )
        low, high = bootstrap_ci(blocks, seed=seed + index)
        outcomes[name] = {
            "mean": float(np.mean(blocks)),
            "ci_low": low,
            "ci_high": high,
            "blocks": int(blocks.size),
        }
    return {"contexts": int(len(paired)), "outcomes": outcomes}


def final_model(
    labels: pd.DataFrame,
    x: np.ndarray,
    feature_names: list[str],
    state_cols: list[str],
    *,
    fit_source_commit: str,
    input_manifest: dict,
) -> dict[str, object]:
    alpha, margin, grid = choose_hyperparameters(labels, x, TRAIN_SEEDS)
    model = fit_zero_baseline_ridge(
        x, labels[PRIMARY_DELTA].to_numpy(dtype=np.float64), alpha
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
        "feature_names": feature_names,
        "feature_scale": np.asarray(model["feature_scale"]).tolist(),
        "coefficient": np.asarray(model["coefficient"]).tolist(),
        "alpha": alpha,
        "guard_margin": margin,
        "full_seed_cv_grid": grid,
    }


def develop(aggregate_dir: Path, out_dir: Path) -> dict[str, object]:
    fit_commit = source_commit()
    manifest, labels, state_cols = load_labels(aggregate_dir)
    x, feature_names = build_treatment_features(labels, state_cols)
    selected, folds = nested_predictions(labels, x)
    p30 = fixed_method(labels, GLOBAL_COMPARATOR)
    oracle = oracle_predictions(labels)
    summaries = {
        "nested_quadratic": summarize_selection(selected, seed=29029),
        "global_p30": summarize_selection(p30, seed=29129),
        "oracle": summarize_selection(oracle, seed=29229),
        "nested_minus_global_p30": paired_summary(
            selected, p30, seed=29329
        ),
    }
    primary = summaries["nested_quadratic"]["outcomes"][
        "service_cost_restricted"
    ]
    paired_primary = summaries["nested_minus_global_p30"]["outcomes"][
        "service_cost_restricted"
    ]
    journey = summaries["nested_quadratic"]["outcomes"]["journey_min"]
    cv = summaries["nested_quadratic"]["outcomes"]["headway_cv"]
    fleet = summaries["nested_quadratic"]["outcomes"]["fleet_overshoot"]
    completion = summaries["nested_quadratic"]["outcomes"][
        "trip_completion_rate"
    ]
    unserved = summaries["nested_quadratic"]["outcomes"][
        "passenger_unserved_rate"
    ]
    per_train = selected.groupby("train_seed")[PRIMARY_DELTA].mean()
    interior_fraction = float(
        selected["candidate_method"].isin(INTERIOR_METHODS).mean()
    )
    checks = {
        "strict_complete_v28_input": manifest.get("strict_complete") is True,
        "four_outer_train_seed_folds": set(
            selected["outer_holdout_train_seed"].astype(int)
        ) == set(TRAIN_SEEDS),
        "zero_actor_feature_contract": True,
        "causal_features_only": True,
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
            "ready_to_preregister_v29" if ready else "development_no_pass"
        ),
        "claim_eligible": False,
        "development_dataset": "V28 exact-prefix labels",
        "fit_source_commit": fit_commit,
        "input_manifest": manifest,
        "feature_contract": FEATURE_CONTRACT,
        "state_dimension": len(state_cols),
        "feature_dimension": len(feature_names),
        "global_comparator": GLOBAL_COMPARATOR,
        "interior_action_fraction": interior_fraction,
        "per_train_seed_primary_mean": {
            str(key): float(value) for key, value in per_train.items()
        },
        "outer_folds": folds,
        "summaries": summaries,
        "development_checks": checks,
        "interpretation": (
            "Development evidence only. Passing freezes this exact model and "
            "permits one external V29 label confirmation; it is not an effect "
            "claim or online-policy promotion."
        ),
    }
    model = final_model(
        labels,
        x,
        feature_names,
        state_cols,
        fit_source_commit=fit_commit,
        input_manifest=manifest,
    )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "nested_selected_contexts.csv", index=False)
    p30.to_csv(out_dir / "global_p30_contexts.csv", index=False)
    oracle.to_csv(out_dir / "oracle_selected_contexts.csv", index=False)
    (out_dir / "quadratic_value_model.json").write_text(
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
        f"DONE V29 development status={report['status']} "
        f"features={report['feature_dimension']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
