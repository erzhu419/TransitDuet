#!/usr/bin/env python3
"""Develop the V32 pairwise composite-service value planner."""

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
    DISCOVERY_TRAIN_SEEDS,
    EXPANDED_CONTEXT_COLUMNS,
    OUTCOME_DELTAS,
    PRIMARY_DELTA,
)
from scripts.audit_protocol_v6_v32_pairwise_composite_common import (
    CONTEXT_RANKS,
    FEATURE_CONTRACT,
    MODEL_PROTOCOL_VERSION,
    SERVICE_REFERENCE_METHOD,
    confirmation_is_fresh_from_v31,
    frozen_confirmation_roster,
)
from scripts.fit_freqduet_prefix_quadratic_value_model import (
    ALPHAS,
    GUARD_MARGINS,
    INTERIOR_METHODS,
    fit_zero_baseline_ridge,
    fixed_method,
    paired_summary,
    predict,
)
from scripts.fit_freqduet_prefix_value_model import (
    CONTEXT_KEYS,
    oracle_predictions,
    summarize_selection,
)
from scripts.fit_freqduet_v30_expanded_quadratic_value_model import (
    MODEL_METHODS,
)
from scripts.fit_freqduet_v31_pairwise_safe_value_model import (
    contrast_to_method,
    fit_context_projection,
    joint_safe_oracle,
    load_labels,
    selection_metrics,
    transform_features,
)
from scripts.run_freqduet_protocol_v2_matrix import git_provenance


HEADWAY_DELTA = OUTCOME_DELTAS["headway_cv"]
UNSERVED_DELTA = OUTCOME_DELTAS["passenger_unserved_rate"]
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
COMPOSITE_OUTCOME_COMPONENTS = {
    "restricted_wait": 1.0,
    "fleet_overshoot": 1.0,
    "headway_cv": 1.0,
    "passenger_unserved_rate": 5.0,
    "incomplete_service": 5.0,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_commit() -> str:
    source = git_provenance()
    commit = str(source.get("commit", "")).lower()
    _require(bool(COMMIT_RE.fullmatch(commit)),
             f"invalid V32 development source commit {commit!r}")
    _require(source.get("tracked_dirty") is False,
             "V32 development source snapshot is tracked-dirty")
    return commit


def fit_service_pipeline(
    labels: pd.DataFrame,
    state_cols: list[str],
    train_mask: np.ndarray,
    *,
    rank: int,
    alpha: float,
) -> tuple[dict[str, object], np.ndarray, list[str]]:
    mask = np.asarray(train_mask, dtype=bool)
    projection = fit_context_projection(
        labels, state_cols, mask, rank=rank
    )
    absolute, names = transform_features(labels, state_cols, projection)
    service_design = contrast_to_method(
        labels, absolute, SERVICE_REFERENCE_METHOD
    )
    service_target = contrast_to_method(
        labels,
        labels[PRIMARY_DELTA].to_numpy(dtype=np.float64),
        SERVICE_REFERENCE_METHOD,
    )
    ridge = fit_zero_baseline_ridge(
        service_design[mask], service_target[mask], alpha
    )
    pipeline = {
        **projection,
        "feature_names": names,
        "feature_scale": np.asarray(ridge["feature_scale"], dtype=np.float64),
        "coefficient": np.asarray(ridge["coefficient"], dtype=np.float64),
        "alpha": float(alpha),
    }
    return pipeline, service_design, names


def predict_service(
    pipeline: dict[str, object],
    service_design: np.ndarray,
) -> np.ndarray:
    return predict({
        "feature_scale": pipeline["feature_scale"],
        "coefficient": pipeline["coefficient"],
    }, service_design)


def select_pairwise_rows(
    labels: pd.DataFrame,
    predictions: np.ndarray,
    *,
    service_margin: float,
) -> pd.DataFrame:
    frame = labels.reset_index(drop=True).copy()
    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    _require(values.shape == (len(frame),), "V32 prediction shape mismatch")
    _require(np.isfinite(values).all(), "non-finite V32 predictions")
    frame["predicted_service_vs_p30"] = values
    p30 = frame[frame["candidate_method"].eq(
        SERVICE_REFERENCE_METHOD
    )].copy()
    contexts = frame.groupby(CONTEXT_KEYS).ngroups
    _require(len(p30) == contexts, "V32 has no unique p30 default")
    _require(np.array_equal(
        p30["predicted_service_vs_p30"].to_numpy(dtype=np.float64),
        np.zeros(contexts, dtype=np.float64),
    ), "V32 p30 service prediction is not exactly zero")

    eligible = frame[
        frame["predicted_service_vs_p30"].add(service_margin).lt(0.0)
    ].copy()
    if eligible.empty:
        return p30.sort_values(CONTEXT_KEYS).reset_index(drop=True)
    eligible["candidate_abs_offset_s"] = eligible[
        "candidate_offset_s"
    ].abs()
    ranked = eligible.sort_values(
        CONTEXT_KEYS + [
            "predicted_service_vs_p30",
            "candidate_abs_offset_s",
            "candidate_method",
        ],
        kind="mergesort",
    )
    best = ranked.loc[
        ~ranked.duplicated(CONTEXT_KEYS, keep="first")
    ].copy()
    covered = set(best[CONTEXT_KEYS].itertuples(index=False, name=None))
    fallback = p30[[
        tuple(row) not in covered
        for row in p30[CONTEXT_KEYS].itertuples(index=False, name=None)
    ]]
    selected = pd.concat([best, fallback], ignore_index=True)
    _require(len(selected) == contexts, "V32 selected context count mismatch")
    _require(not selected.duplicated(CONTEXT_KEYS).any(),
             "V32 selected duplicate contexts")
    return selected.sort_values(CONTEXT_KEYS).reset_index(drop=True)


def _inner_fold_predictions(
    labels: pd.DataFrame,
    state_cols: list[str],
    train_seeds: list[int],
    *,
    rank: int,
    alpha: float,
) -> list[tuple[pd.DataFrame, np.ndarray]]:
    folds: list[tuple[pd.DataFrame, np.ndarray]] = []
    seeds = labels["train_seed"].to_numpy(dtype=int)
    for heldout in train_seeds:
        fit_seeds = [seed for seed in train_seeds if seed != heldout]
        train_mask = np.isin(seeds, fit_seeds)
        test_mask = seeds == heldout
        pipeline, design, _ = fit_service_pipeline(
            labels,
            state_cols,
            train_mask,
            rank=rank,
            alpha=alpha,
        )
        prediction = predict_service(pipeline, design)
        folds.append((
            labels.loc[test_mask].reset_index(drop=True),
            prediction[test_mask],
        ))
    return folds


def choose_hyperparameters(
    labels: pd.DataFrame,
    state_cols: list[str],
    train_seeds: list[int],
) -> tuple[dict[str, float | int] | None, list[dict[str, object]]]:
    seeds = labels["train_seed"].to_numpy(dtype=int)
    evaluation_labels = labels.loc[
        np.isin(seeds, train_seeds)
    ].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for rank in CONTEXT_RANKS:
        for alpha in ALPHAS:
            fold_predictions = _inner_fold_predictions(
                labels,
                state_cols,
                train_seeds,
                rank=rank,
                alpha=alpha,
            )
            for margin in GUARD_MARGINS:
                selected = pd.concat([
                    select_pairwise_rows(
                        fold_labels,
                        fold_prediction,
                        service_margin=margin,
                    )
                    for fold_labels, fold_prediction in fold_predictions
                ], ignore_index=True)
                metrics = selection_metrics(selected, evaluation_labels)
                feasible = bool(
                    metrics["service_vs_p30_mean"] < 0.0
                    and metrics["headway_vs_actor_mean"] <= 0.0
                    and metrics["unserved_vs_actor_mean"] <= 0.0
                    and metrics["default_override_fraction"] >= 0.05
                    and metrics["interior_action_fraction"] >= 0.05
                )
                rows.append({
                    "context_rank": int(rank),
                    "alpha": float(alpha),
                    "service_margin": float(margin),
                    "feasible": feasible,
                    **metrics,
                })
    feasible_rows = [row for row in rows if bool(row["feasible"])]
    if not feasible_rows:
        return None, rows
    best = min(feasible_rows, key=lambda row: (
        float(row["service_vs_p30_mean"]),
        float(row["headway_vs_actor_mean"]),
        float(row["unserved_vs_actor_mean"]),
        int(row["context_rank"]),
        -float(row["service_margin"]),
        -float(row["alpha"]),
    ))
    return {
        "context_rank": int(best["context_rank"]),
        "alpha": float(best["alpha"]),
        "service_margin": float(best["service_margin"]),
    }, rows


def nested_predictions(
    labels: pd.DataFrame,
    state_cols: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selected_parts: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    seeds = labels["train_seed"].to_numpy(dtype=int)
    for heldout in DISCOVERY_TRAIN_SEEDS:
        outer_train = [
            seed for seed in DISCOVERY_TRAIN_SEEDS if seed != heldout
        ]
        chosen, grid = choose_hyperparameters(labels, state_cols, outer_train)
        test_mask = seeds == heldout
        test_labels = labels.loc[test_mask].reset_index(drop=True)
        if chosen is None:
            selected = fixed_method(test_labels, SERVICE_REFERENCE_METHOD)
            feature_dimension = 0
            projection_contexts = 0
            explained: list[float] = []
        else:
            train_mask = np.isin(seeds, outer_train)
            pipeline, design, names = fit_service_pipeline(
                labels,
                state_cols,
                train_mask,
                rank=int(chosen["context_rank"]),
                alpha=float(chosen["alpha"]),
            )
            selected = select_pairwise_rows(
                test_labels,
                predict_service(pipeline, design)[test_mask],
                service_margin=float(chosen["service_margin"]),
            )
            feature_dimension = len(names)
            projection_contexts = int(pipeline["fitted_actor_contexts"])
            explained = np.asarray(
                pipeline["explained_variance_fraction"]
            ).tolist()
        selected["outer_holdout_train_seed"] = int(heldout)
        selected["selection_enabled"] = chosen is not None
        selected_parts.append(selected)
        folds.append({
            "heldout_train_seed": int(heldout),
            "inner_train_seeds": outer_train,
            "selected_configuration": chosen,
            "feature_dimension": feature_dimension,
            "projection_actor_contexts": projection_contexts,
            "projection_explained_variance_fraction": explained,
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
    chosen, grid = choose_hyperparameters(
        labels, state_cols, DISCOVERY_TRAIN_SEEDS
    )
    base: dict[str, object] = {
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "development_only": True,
        "claim_eligible": False,
        "enabled": chosen is not None,
        "feature_contract": FEATURE_CONTRACT,
        "fit_source_commit": fit_source_commit,
        "training_rollout_source_commit": input_manifest.get(
            "rollout_source_commit"
        ),
        "training_aggregation_source_commit": input_manifest.get(
            "aggregation_source_commit"
        ),
        "candidate_methods": MODEL_METHODS,
        "service_reference_method": SERVICE_REFERENCE_METHOD,
        "service_target": f"{PRIMARY_DELTA}_minus_p30",
        "composite_outcome_components": COMPOSITE_OUTCOME_COMPONENTS,
        "selection_configuration": chosen,
        "full_seed_cv_grid": grid,
    }
    if chosen is None:
        base["fallback"] = SERVICE_REFERENCE_METHOD
        return base

    pipeline, _, names = fit_service_pipeline(
        labels,
        state_cols,
        np.ones(len(labels), dtype=bool),
        rank=int(chosen["context_rank"]),
        alpha=float(chosen["alpha"]),
    )
    base.update({
        "state_columns": state_cols,
        "expanded_context_columns": list(EXPANDED_CONTEXT_COLUMNS),
        "context_columns": pipeline["context_columns"],
        "context_mean": np.asarray(pipeline["context_mean"]).tolist(),
        "context_scale": np.asarray(pipeline["context_scale"]).tolist(),
        "context_components": np.asarray(
            pipeline["context_components"]
        ).tolist(),
        "projection_explained_variance_fraction": np.asarray(
            pipeline["explained_variance_fraction"]
        ).tolist(),
        "feature_names": names,
        "feature_scale": np.asarray(pipeline["feature_scale"]).tolist(),
        "coefficient": np.asarray(pipeline["coefficient"]).tolist(),
    })
    return base


def develop(aggregate_dir: Path, out_dir: Path) -> dict[str, object]:
    fit_commit = source_commit()
    manifest, labels, state_cols = load_labels(aggregate_dir)
    selected, folds = nested_predictions(labels, state_cols)
    p30 = fixed_method(labels, SERVICE_REFERENCE_METHOD)
    oracle = oracle_predictions(labels)
    safe_oracle = joint_safe_oracle(labels)
    model = final_model(
        labels,
        state_cols,
        fit_source_commit=fit_commit,
        input_manifest=manifest,
    )
    summaries = {
        "nested_pairwise_composite": summarize_selection(selected, seed=32132),
        "global_p30": summarize_selection(p30, seed=32232),
        "oracle": summarize_selection(oracle, seed=32332),
        "joint_safe_oracle": summarize_selection(safe_oracle, seed=32432),
        "nested_minus_global_p30": paired_summary(
            selected, p30, seed=32532
        ),
    }
    primary = summaries["nested_pairwise_composite"]["outcomes"][
        "service_cost_restricted"
    ]
    paired_primary = summaries["nested_minus_global_p30"]["outcomes"][
        "service_cost_restricted"
    ]
    journey = summaries["nested_pairwise_composite"]["outcomes"][
        "journey_min"
    ]
    cv = summaries["nested_pairwise_composite"]["outcomes"]["headway_cv"]
    fleet = summaries["nested_pairwise_composite"]["outcomes"][
        "fleet_overshoot"
    ]
    completion = summaries["nested_pairwise_composite"]["outcomes"][
        "trip_completion_rate"
    ]
    unserved = summaries["nested_pairwise_composite"]["outcomes"][
        "passenger_unserved_rate"
    ]
    per_train = selected.groupby("train_seed")[PRIMARY_DELTA].mean()
    default_override_fraction = float(
        selected["candidate_method"].ne(SERVICE_REFERENCE_METHOD).mean()
    )
    interior_fraction = float(
        selected["candidate_method"].isin(INTERIOR_METHODS).mean()
    )
    feature_dimensions = [int(fold["feature_dimension"]) for fold in folds]
    checks = {
        "strict_complete_v30_discovery_input": (
            manifest.get("strict_complete") is True
        ),
        "four_outer_train_seed_folds": set(
            selected["outer_holdout_train_seed"].astype(int)
        ) == set(DISCOVERY_TRAIN_SEEDS),
        "fresh_v32_confirmation_roster_frozen": (
            confirmation_is_fresh_from_v31()
        ),
        "pairwise_service_reference_exactly_p30": (
            SERVICE_REFERENCE_METHOD == "actor_firstknot_p30"
        ),
        "composite_service_soft_risk_semantics": True,
        "no_pointwise_post_policy_guard": True,
        "projection_fit_inside_fold": True,
        "low_capacity_at_most_36_features": max(feature_dimensions) <= 36,
        "all_outer_selectors_enabled": all(
            bool(fold["selected_configuration"]) for fold in folds
        ),
        "final_selector_enabled": bool(model["enabled"]),
        "nontrivial_default_override_fraction": (
            default_override_fraction >= 0.05
        ),
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
            "ready_to_freeze_v32_confirmation"
            if ready else "development_no_pass"
        ),
        "claim_eligible": False,
        "development_dataset": "V30 fresh-context exact-prefix labels",
        "fit_source_commit": fit_commit,
        "input_manifest": manifest,
        "feature_contract": FEATURE_CONTRACT,
        "state_dimension": len(state_cols),
        "expanded_context_columns": list(EXPANDED_CONTEXT_COLUMNS),
        "candidate_context_ranks": CONTEXT_RANKS,
        "outer_feature_dimensions": feature_dimensions,
        "service_reference_method": SERVICE_REFERENCE_METHOD,
        "composite_outcome_components": COMPOSITE_OUTCOME_COMPONENTS,
        "default_override_fraction": default_override_fraction,
        "interior_action_fraction": interior_fraction,
        "per_train_seed_primary_mean": {
            str(key): float(value) for key, value in per_train.items()
        },
        "outer_folds": folds,
        "summaries": summaries,
        "development_checks": checks,
        "frozen_confirmation_roster": frozen_confirmation_roster(),
        "interpretation": (
            "Development evidence only. V32 uses the registered composite "
            "service outcome as the soft reliability objective and applies "
            "headway/unserved requirements only to held-out aggregate model "
            "selection and the unchanged outer gate. Passing permits one "
            "unchanged fresh-roster confirmation; it does not itself promote "
            "the controller."
        ),
    }

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "nested_selected_contexts.csv", index=False)
    p30.to_csv(out_dir / "global_p30_contexts.csv", index=False)
    safe_oracle.to_csv(out_dir / "joint_safe_oracle_contexts.csv", index=False)
    (out_dir / "pairwise_composite_value_model.json").write_text(
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
        f"DONE V32 development status={report['status']} "
        f"outer_dims={report['outer_feature_dimensions']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
