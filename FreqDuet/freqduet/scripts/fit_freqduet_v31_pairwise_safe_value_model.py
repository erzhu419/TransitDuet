#!/usr/bin/env python3
"""Develop the V31 pairwise service and absolute-risk value planner."""

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
from scripts.audit_protocol_v6_v31_pairwise_common import (
    CONTEXT_RANKS,
    FEATURE_CONTRACT,
    MODEL_PROTOCOL_VERSION,
    RISK_THRESHOLD,
    SERVICE_REFERENCE_METHOD,
    confirmation_is_disjoint_from_development,
    frozen_confirmation_roster,
)
from scripts.fit_freqduet_prefix_lowrank_quadratic_value_model import (
    BASIS_NAMES,
    _fix_component_signs,
    treatment_basis,
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
    load_labels as load_v30_labels,
)
from scripts.run_freqduet_protocol_v2_matrix import git_provenance


HEADWAY_DELTA = OUTCOME_DELTAS["headway_cv"]
UNSERVED_DELTA = OUTCOME_DELTAS["passenger_unserved_rate"]
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_commit() -> str:
    source = git_provenance()
    commit = str(source.get("commit", "")).lower()
    _require(bool(COMMIT_RE.fullmatch(commit)),
             f"invalid V31 development source commit {commit!r}")
    _require(source.get("tracked_dirty") is False,
             "V31 development source snapshot is tracked-dirty")
    return commit


def load_labels(
    aggregate_dir: Path,
) -> tuple[dict, pd.DataFrame, list[str]]:
    manifest, labels = load_v30_labels(aggregate_dir)
    state_cols = sorted(
        column for column in labels.columns
        if column.startswith("upper_state_") and column != "upper_state_dim"
    )
    _require(bool(state_cols), "no causal upper-state columns in V30 labels")
    dimensions = pd.to_numeric(labels["upper_state_dim"], errors="coerce")
    _require(
        dimensions.notna().all() and dimensions.eq(len(state_cols)).all(),
        "V31 upper-state dimensions are inconsistent",
    )
    for column in state_cols:
        values = pd.to_numeric(labels[column], errors="coerce")
        _require(np.isfinite(values.to_numpy(dtype=np.float64)).all(),
                 f"non-finite V31 context column {column}")
        labels[column] = values

    causal_columns = state_cols + list(EXPANDED_CONTEXT_COLUMNS)
    varying = (
        labels.groupby(CONTEXT_KEYS, sort=False)[causal_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    _require(not varying.any(),
             "V31 causal context changes across candidate branches")
    actor_actions = labels.groupby(CONTEXT_KEYS, sort=False)[
        "actor_action_json"
    ].nunique(dropna=False)
    _require(actor_actions.eq(1).all(),
             "V31 actor action changes across candidate branches")
    return manifest, labels.reset_index(drop=True), state_cols


def raw_context(
    labels: pd.DataFrame,
    state_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    state = labels[state_cols].to_numpy(dtype=np.float64)
    expanded = labels[EXPANDED_CONTEXT_COLUMNS].to_numpy(dtype=np.float64)
    actor = np.stack(labels["actor_action"].to_list()).astype(np.float64)
    context = np.concatenate([state, expanded, actor / 60.0], axis=1)
    names = (
        list(state_cols)
        + list(EXPANDED_CONTEXT_COLUMNS)
        + [f"actor_action_{index:02d}" for index in range(actor.shape[1])]
    )
    _require(context.shape[1] == len(names), "V31 context name mismatch")
    _require(np.isfinite(context).all(), "non-finite V31 causal context")
    return context, names


def fit_context_projection(
    labels: pd.DataFrame,
    state_cols: list[str],
    train_mask: np.ndarray,
    *,
    rank: int,
) -> dict[str, object]:
    rank = int(rank)
    _require(rank in CONTEXT_RANKS, f"unregistered V31 context rank {rank}")
    mask = np.asarray(train_mask, dtype=bool)
    _require(mask.shape == (len(labels),), "V31 projection mask mismatch")
    context, context_names = raw_context(labels, state_cols)
    actor_mask = labels["candidate_method"].eq("actor").to_numpy() & mask
    train_contexts = labels.loc[mask].groupby(CONTEXT_KEYS).ngroups
    _require(int(actor_mask.sum()) == int(train_contexts),
             "V31 projection requires one actor row per training context")
    _require(train_contexts >= 2, "V31 projection has fewer than two contexts")

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
    _require(rank <= max_rank,
             f"V31 rank {rank} exceeds fitted rank {max_rank}")
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
    _require(
        context_names == [str(value) for value in projection["context_columns"]],
        "V31 projection context schema mismatch",
    )
    mean = np.asarray(projection["context_mean"], dtype=np.float64)
    scale = np.asarray(projection["context_scale"], dtype=np.float64)
    components = np.asarray(
        projection["context_components"], dtype=np.float64
    )
    rank = int(projection["context_rank"])
    _require(mean.shape == scale.shape == (context.shape[1],),
             "V31 context normalization shape mismatch")
    _require(components.shape == (rank, context.shape[1]),
             "V31 projection component shape mismatch")

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
             "V31 treatment design shape mismatch")
    _require(design.shape[1] == len(names), "V31 feature name mismatch")
    _require(np.isfinite(design).all(), "non-finite V31 treatment design")
    actor = labels["candidate_method"].eq("actor").to_numpy()
    _require(np.array_equal(
        design[actor], np.zeros_like(design[actor])
    ), "V31 actor treatment features are not exactly zero")
    return design, names


def reference_indices(labels: pd.DataFrame, method: str) -> np.ndarray:
    _require(np.array_equal(
        labels.index.to_numpy(), np.arange(len(labels))
    ), "V31 labels require a contiguous positional index")
    methods = labels["candidate_method"].astype(str).to_numpy()
    references = np.full(len(labels), -1, dtype=np.int64)
    for positions in labels.groupby(CONTEXT_KEYS, sort=False).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        matches = positions[methods[positions] == str(method)]
        _require(len(matches) == 1,
                 f"context has no unique V31 reference {method}")
        references[positions] = int(matches[0])
    _require(np.all(references >= 0), "incomplete V31 reference alignment")
    return references


def contrast_to_method(
    labels: pd.DataFrame,
    values: np.ndarray,
    method: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _require(array.shape[0] == len(labels), "V31 contrast row mismatch")
    contrasted = array - array[reference_indices(labels, method)]
    reference = labels["candidate_method"].eq(str(method)).to_numpy()
    _require(np.array_equal(
        contrasted[reference], np.zeros_like(contrasted[reference])
    ), f"V31 {method} contrast is not exactly zero")
    return contrasted


def fit_pipeline(
    labels: pd.DataFrame,
    state_cols: list[str],
    train_mask: np.ndarray,
    *,
    rank: int,
    alpha: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, list[str]]:
    mask = np.asarray(train_mask, dtype=bool)
    projection = fit_context_projection(
        labels, state_cols, mask, rank=rank
    )
    absolute_design, names = transform_features(labels, state_cols, projection)
    service_design = contrast_to_method(
        labels, absolute_design, SERVICE_REFERENCE_METHOD
    )
    service_target = contrast_to_method(
        labels,
        labels[PRIMARY_DELTA].to_numpy(dtype=np.float64),
        SERVICE_REFERENCE_METHOD,
    )
    targets = {
        "service_vs_p30": (service_design, service_target),
        "headway_vs_actor": (
            absolute_design,
            labels[HEADWAY_DELTA].to_numpy(dtype=np.float64),
        ),
        "unserved_vs_actor": (
            absolute_design,
            labels[UNSERVED_DELTA].to_numpy(dtype=np.float64),
        ),
    }
    heads: dict[str, dict[str, np.ndarray | float]] = {}
    for name, (design, target) in targets.items():
        heads[name] = fit_zero_baseline_ridge(
            design[mask], target[mask], alpha
        )
    pipeline = {
        **projection,
        "feature_names": names,
        "alpha": float(alpha),
        "heads": heads,
    }
    return pipeline, absolute_design, service_design, names


def predict_pipeline(
    pipeline: dict[str, object],
    absolute_design: np.ndarray,
    service_design: np.ndarray,
) -> dict[str, np.ndarray]:
    heads = pipeline["heads"]
    _require(isinstance(heads, dict), "V31 pipeline heads are invalid")
    return {
        "service_vs_p30": predict(heads["service_vs_p30"], service_design),
        "headway_vs_actor": predict(
            heads["headway_vs_actor"], absolute_design
        ),
        "unserved_vs_actor": predict(
            heads["unserved_vs_actor"], absolute_design
        ),
    }


def select_constrained_rows(
    labels: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    service_margin: float,
    risk_threshold: float = RISK_THRESHOLD,
) -> pd.DataFrame:
    frame = labels.reset_index(drop=True).copy()
    prediction_columns = {
        "service_vs_p30": "predicted_service_vs_p30",
        "headway_vs_actor": "predicted_headway_vs_actor",
        "unserved_vs_actor": "predicted_unserved_vs_actor",
    }
    for key, column in prediction_columns.items():
        values = np.asarray(predictions[key], dtype=np.float64).reshape(-1)
        _require(values.shape == (len(frame),),
                 f"V31 prediction shape mismatch for {key}")
        _require(np.isfinite(values).all(),
                 f"non-finite V31 predictions for {key}")
        frame[column] = values

    selected: list[pd.Series] = []
    for _, group in frame.groupby(CONTEXT_KEYS, sort=True):
        actor = group[group["candidate_method"].eq("actor")]
        p30 = group[group["candidate_method"].eq(SERVICE_REFERENCE_METHOD)]
        _require(len(actor) == 1, "V31 context has no unique actor")
        _require(len(p30) == 1, "V31 context has no unique p30 default")
        actor_row = actor.iloc[0]
        p30_row = p30.iloc[0]
        _require(
            float(actor_row["predicted_headway_vs_actor"]) == 0.0
            and float(actor_row["predicted_unserved_vs_actor"]) == 0.0,
            "V31 actor risk prediction is not exactly zero",
        )
        _require(float(p30_row["predicted_service_vs_p30"]) == 0.0,
                 "V31 p30 service prediction is not exactly zero")

        safe = group[
            group["predicted_headway_vs_actor"].le(risk_threshold)
            & group["predicted_unserved_vs_actor"].le(risk_threshold)
        ]
        improving = safe[
            safe["predicted_service_vs_p30"].add(service_margin).lt(0.0)
        ].copy()
        if not improving.empty:
            improving["candidate_abs_offset_s"] = improving[
                "candidate_offset_s"
            ].abs()
            chosen = improving.sort_values([
                "predicted_service_vs_p30",
                "candidate_abs_offset_s",
                "candidate_method",
            ], kind="mergesort").iloc[0]
        elif (
            float(p30_row["predicted_headway_vs_actor"]) <= risk_threshold
            and float(p30_row["predicted_unserved_vs_actor"]) <= risk_threshold
        ):
            chosen = p30_row
        else:
            chosen = actor_row
        selected.append(chosen)
    return pd.DataFrame(selected).reset_index(drop=True)


def _paired_primary_mean(
    selected: pd.DataFrame,
    reference: pd.DataFrame,
) -> float:
    paired = selected[CONTEXT_KEYS + [PRIMARY_DELTA]].merge(
        reference[CONTEXT_KEYS + [PRIMARY_DELTA]],
        on=CONTEXT_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_selected", "_reference"),
    )
    _require(len(paired) == len(selected) == len(reference),
             "V31 paired inner contexts do not align")
    return float((
        paired[f"{PRIMARY_DELTA}_selected"]
        - paired[f"{PRIMARY_DELTA}_reference"]
    ).mean())


def selection_metrics(
    selected: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
) -> dict[str, float]:
    p30 = fixed_method(evaluation_labels, SERVICE_REFERENCE_METHOD)
    return {
        "service_vs_p30_mean": _paired_primary_mean(selected, p30),
        "service_vs_actor_mean": float(selected[PRIMARY_DELTA].mean()),
        "headway_vs_actor_mean": float(selected[HEADWAY_DELTA].mean()),
        "unserved_vs_actor_mean": float(selected[UNSERVED_DELTA].mean()),
        "default_override_fraction": float(
            selected["candidate_method"].ne(SERVICE_REFERENCE_METHOD).mean()
        ),
        "interior_action_fraction": float(
            selected["candidate_method"].isin(INTERIOR_METHODS).mean()
        ),
    }


def _inner_fold_predictions(
    labels: pd.DataFrame,
    state_cols: list[str],
    train_seeds: list[int],
    *,
    rank: int,
    alpha: float,
) -> list[tuple[pd.DataFrame, dict[str, np.ndarray]]]:
    folds: list[tuple[pd.DataFrame, dict[str, np.ndarray]]] = []
    seeds = labels["train_seed"].to_numpy(dtype=int)
    for heldout in train_seeds:
        fit_seeds = [seed for seed in train_seeds if seed != heldout]
        train_mask = np.isin(seeds, fit_seeds)
        test_mask = seeds == heldout
        pipeline, absolute, service, _ = fit_pipeline(
            labels,
            state_cols,
            train_mask,
            rank=rank,
            alpha=alpha,
        )
        predictions = predict_pipeline(pipeline, absolute, service)
        folds.append((
            labels.loc[test_mask].reset_index(drop=True),
            {key: value[test_mask] for key, value in predictions.items()},
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
                    select_constrained_rows(
                        fold_labels,
                        fold_values,
                        service_margin=margin,
                    )
                    for fold_labels, fold_values in fold_predictions
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
            selected = fixed_method(test_labels, "actor")
            feature_dimension = 0
            projection_contexts = 0
            explained: list[float] = []
        else:
            train_mask = np.isin(seeds, outer_train)
            pipeline, absolute, service, names = fit_pipeline(
                labels,
                state_cols,
                train_mask,
                rank=int(chosen["context_rank"]),
                alpha=float(chosen["alpha"]),
            )
            predictions = predict_pipeline(pipeline, absolute, service)
            selected = select_constrained_rows(
                test_labels,
                {key: value[test_mask] for key, value in predictions.items()},
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


def _serialized_head(head: dict[str, np.ndarray | float]) -> dict[str, object]:
    return {
        "feature_scale": np.asarray(head["feature_scale"]).tolist(),
        "coefficient": np.asarray(head["coefficient"]).tolist(),
        "alpha": float(head["alpha"]),
    }


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
        "risk_targets": [HEADWAY_DELTA, UNSERVED_DELTA],
        "risk_threshold": RISK_THRESHOLD,
        "selection_configuration": chosen,
        "full_seed_cv_grid": grid,
    }
    if chosen is None:
        base["fallback"] = "actor"
        return base

    train_mask = np.ones(len(labels), dtype=bool)
    pipeline, _, _, names = fit_pipeline(
        labels,
        state_cols,
        train_mask,
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
        "heads": {
            name: _serialized_head(head)
            for name, head in pipeline["heads"].items()
        },
    })
    return base


def joint_safe_oracle(labels: pd.DataFrame) -> pd.DataFrame:
    safe = labels[
        labels[HEADWAY_DELTA].le(0.0) & labels[UNSERVED_DELTA].le(0.0)
    ].copy()
    selected = (
        safe.sort_values(
            CONTEXT_KEYS + [PRIMARY_DELTA, "candidate_offset_s"],
            kind="mergesort",
        )
        .groupby(CONTEXT_KEYS, sort=True, as_index=False)
        .first()
    )
    _require(len(selected) == labels.groupby(CONTEXT_KEYS).ngroups,
             "V31 joint-safe oracle has incomplete contexts")
    return selected


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
        "nested_pairwise_safe": summarize_selection(selected, seed=31131),
        "global_p30": summarize_selection(p30, seed=31231),
        "oracle": summarize_selection(oracle, seed=31331),
        "joint_safe_oracle": summarize_selection(safe_oracle, seed=31431),
        "nested_minus_global_p30": paired_summary(
            selected, p30, seed=31531
        ),
    }
    primary = summaries["nested_pairwise_safe"]["outcomes"][
        "service_cost_restricted"
    ]
    paired_primary = summaries["nested_minus_global_p30"]["outcomes"][
        "service_cost_restricted"
    ]
    journey = summaries["nested_pairwise_safe"]["outcomes"]["journey_min"]
    cv = summaries["nested_pairwise_safe"]["outcomes"]["headway_cv"]
    fleet = summaries["nested_pairwise_safe"]["outcomes"]["fleet_overshoot"]
    completion = summaries["nested_pairwise_safe"]["outcomes"][
        "trip_completion_rate"
    ]
    unserved = summaries["nested_pairwise_safe"]["outcomes"][
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
        "fresh_confirmation_roster_frozen_and_disjoint": (
            confirmation_is_disjoint_from_development()
        ),
        "pairwise_service_reference_exactly_p30": (
            SERVICE_REFERENCE_METHOD == "actor_firstknot_p30"
        ),
        "absolute_actor_risk_zero_contract": True,
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
            "ready_to_freeze_v31_confirmation"
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
        "risk_threshold": RISK_THRESHOLD,
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
            "Development evidence only. V31 removes context-level service "
            "nuisance by learning candidate-minus-p30 effects and makes "
            "headway and unserved risk explicit value heads. Passing permits "
            "one unchanged evaluation on the frozen fresh confirmation "
            "roster; it does not itself promote the controller."
        ),
    }

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "nested_selected_contexts.csv", index=False)
    p30.to_csv(out_dir / "global_p30_contexts.csv", index=False)
    safe_oracle.to_csv(out_dir / "joint_safe_oracle_contexts.csv", index=False)
    (out_dir / "pairwise_safe_value_model.json").write_text(
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
        f"DONE V31 development status={report['status']} "
        f"outer_dims={report['outer_feature_dimensions']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
