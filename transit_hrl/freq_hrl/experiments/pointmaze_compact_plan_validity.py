"""Compact nonlinear plan-validity qualification on paired PointMaze branches."""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

import numpy as np

from .pointmaze_goal_validation import _json_ready
from .pointmaze_plan_validity_branching import (
    BRANCH_CATEGORIES,
    POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH,
    POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION,
    _prediction_metrics,
    _ridge_fit_predict,
    _task_options,
    _validate_branch_seed_roles,
    build_parser as build_stage8b_parser,
    train_pointmaze_plan_validity_cell,
)


POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION = (
    "pointmaze_compact_plan_validity_stage8c_v1"
)
POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH = (
    "paired_keep_renew_compact_nonlinear_plan_validity_qualification"
)
POINTMAZE_COMPACT_PLAN_VALIDITY_POLICY = "hrl_regime_history"
COMPACT_PLAN_VALIDITY_PREDICTORS = (
    "current_compact_quadratic",
    "causal_dynamic_quadratic",
    "causal_validity_interactions",
)
DEFAULT_RIDGE_ALPHA_GRID = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
)

_CURRENT_COMPACT_NAMES = (
    "physical_0",
    "physical_1",
    "physical_2",
    "physical_3",
    "target_error_0",
    "target_error_1",
    "waypoint_error_0",
    "waypoint_error_1",
    "tracking_distance",
    "waypoint_distance",
    "plan_age_seconds",
    "plan_age_fraction",
    "target_current_0",
    "target_current_1",
    "force_current_0",
    "force_current_1",
    "force_current_norm",
)
_CAUSAL_DYNAMIC_NAMES = (
    "target_error_0",
    "target_error_1",
    "waypoint_error_0",
    "waypoint_error_1",
    "tracking_distance",
    "waypoint_distance",
    "plan_age_fraction",
    "target_velocity_001_0",
    "target_velocity_001_1",
    "target_velocity_010_0",
    "target_velocity_010_1",
    "target_velocity_025_0",
    "target_velocity_025_1",
    "target_velocity_050_0",
    "target_velocity_050_1",
    "target_velocity_change_norm",
    "force_current_norm",
)


def _base_matrix(
    rows: list[dict[str, Any]],
) -> tuple[tuple[str, ...], np.ndarray]:
    if not rows:
        raise ValueError("compact plan-validity features require rows")
    schema = tuple(map(str, rows[0]["feature_names"]))
    matrix = np.asarray([row["causal_features"] for row in rows], dtype=np.float64)
    if (
        len(set(schema)) != len(schema)
        or matrix.shape != (len(rows), len(schema))
        or not np.all(np.isfinite(matrix))
        or any(tuple(map(str, row["feature_names"])) != schema for row in rows)
        or any(bool(row.get("candidate_feature_has_future_access")) for row in rows)
        or any(bool(row.get("candidate_feature_has_regime_label")) for row in rows)
    ):
        raise ValueError("compact plan-validity causal feature contract changed")
    return schema, matrix


def _select_named_features(
    schema: tuple[str, ...],
    matrix: np.ndarray,
    names: tuple[str, ...],
) -> np.ndarray:
    indices = {name: index for index, name in enumerate(schema)}
    missing = [name for name in names if name not in indices]
    if missing:
        raise ValueError(f"compact plan-validity features are missing: {missing}")
    return matrix[:, [indices[name] for name in names]]


def _quadratic_features(
    matrix: np.ndarray,
    names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = [matrix, np.square(matrix)]
    feature_names = [*names, *(f"{name}__squared" for name in names)]
    interactions = []
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            interactions.append(matrix[:, left] * matrix[:, right])
            feature_names.append(f"{names[left]}__x__{names[right]}")
    if interactions:
        values.append(np.column_stack(interactions))
    return np.column_stack(values), tuple(feature_names)


def _causal_validity_interactions(
    schema: tuple[str, ...],
    matrix: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    index = {name: position for position, name in enumerate(schema)}

    def column(name: str) -> np.ndarray:
        if name not in index:
            raise ValueError(f"compact plan-validity feature is missing: {name}")
        return matrix[:, index[name]]

    age = column("plan_age_fraction")
    tracking = column("tracking_distance")
    waypoint_distance = column("waypoint_distance")
    velocity_change = column("target_velocity_change_norm")
    force_norm = column("force_current_norm")
    target_error = np.column_stack((
        column("target_error_0"), column("target_error_1")
    ))
    waypoint_error = np.column_stack((
        column("waypoint_error_0"), column("waypoint_error_1")
    ))
    values = [age, tracking, waypoint_distance, velocity_change, force_norm]
    names = [
        "plan_age_fraction",
        "tracking_distance",
        "waypoint_distance",
        "target_velocity_change_norm",
        "force_current_norm",
    ]
    values.extend((target_error[:, 0], target_error[:, 1]))
    values.extend((waypoint_error[:, 0], waypoint_error[:, 1]))
    names.extend((
        "target_error_0",
        "target_error_1",
        "waypoint_error_0",
        "waypoint_error_1",
    ))

    velocities: list[np.ndarray] = []
    for lag in ("001", "010", "025", "050"):
        velocity = np.column_stack((
            column(f"target_velocity_{lag}_0"),
            column(f"target_velocity_{lag}_1"),
        ))
        velocities.append(velocity)
        norm = np.linalg.norm(velocity, axis=1)
        values.extend((
            velocity[:, 0],
            velocity[:, 1],
            norm,
            np.sum(target_error * velocity, axis=1),
            np.sum(waypoint_error * velocity, axis=1),
            age * norm,
        ))
        names.extend((
            f"target_velocity_{lag}_0",
            f"target_velocity_{lag}_1",
            f"target_velocity_{lag}_norm",
            f"target_error_dot_velocity_{lag}",
            f"waypoint_error_dot_velocity_{lag}",
            f"plan_age_x_velocity_{lag}_norm",
        ))

    short_minus_long = velocities[0] - velocities[-1]
    values.extend((
        np.linalg.norm(short_minus_long, axis=1),
        np.sum(target_error * short_minus_long, axis=1),
        np.sum(waypoint_error * short_minus_long, axis=1),
        age * velocity_change,
        age * tracking,
        age * waypoint_distance,
    ))
    names.extend((
        "velocity_001_minus_050_norm",
        "target_error_dot_velocity_001_minus_050",
        "waypoint_error_dot_velocity_001_minus_050",
        "plan_age_x_velocity_change_norm",
        "plan_age_x_tracking_distance",
        "plan_age_x_waypoint_distance",
    ))
    transformed = np.column_stack(values)
    if transformed.shape[1] != 39 or not np.all(np.isfinite(transformed)):
        raise RuntimeError("compact plan-validity interaction features are invalid")
    return transformed, tuple(names)


def compact_plan_validity_feature_views(
    rows: list[dict[str, Any]],
) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    schema, matrix = _base_matrix(rows)
    current = _select_named_features(schema, matrix, _CURRENT_COMPACT_NAMES)
    dynamic = _select_named_features(schema, matrix, _CAUSAL_DYNAMIC_NAMES)
    current_quadratic = _quadratic_features(current, _CURRENT_COMPACT_NAMES)
    dynamic_quadratic = _quadratic_features(dynamic, _CAUSAL_DYNAMIC_NAMES)
    interactions = _causal_validity_interactions(schema, matrix)
    return {
        "current_compact_quadratic": current_quadratic,
        "causal_dynamic_quadratic": dynamic_quadratic,
        "causal_validity_interactions": interactions,
    }


def _grouped_ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    groups: np.ndarray,
    eval_x: np.ndarray,
    *,
    alpha_grid: Iterable[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    alphas = tuple(sorted(set(map(float, alpha_grid))))
    group = np.asarray(groups, dtype=np.int64).reshape(-1)
    if (
        not alphas
        or any(not np.isfinite(alpha) or alpha <= 0.0 for alpha in alphas)
        or group.shape != (len(train_x),)
        or np.unique(group).size < 2
    ):
        raise ValueError("grouped plan-validity ridge contract is invalid")
    losses: dict[str, float] = {}
    for alpha in alphas:
        fold_losses = []
        for held_out in np.unique(group):
            train = group != held_out
            validation = ~train
            prediction, _ = _ridge_fit_predict(
                train_x[train],
                train_y[train],
                train_x[validation],
                alpha=alpha,
            )
            fold_losses.append(float(np.mean(np.square(
                train_y[validation] - prediction
            ))))
        losses[str(alpha)] = float(np.mean(fold_losses))
    selected_alpha = min(alphas, key=lambda alpha: (losses[str(alpha)], alpha))
    prediction, model = _ridge_fit_predict(
        train_x, train_y, eval_x, alpha=selected_alpha
    )
    model.update({
        "alpha_selection": "leave_one_path_seed_out_mean_mse",
        "alpha_grid": list(alphas),
        "group_count": int(np.unique(group).size),
        "cross_validation_mse": losses,
    })
    return prediction, model


def fit_compact_plan_validity_predictors(
    fit_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    ridge_alpha_grid: Iterable[float] = DEFAULT_RIDGE_ALPHA_GRID,
    selection_rate: float,
) -> dict[str, Any]:
    if not fit_rows or not eval_rows or not 0.0 < float(selection_rate) <= 1.0:
        raise ValueError("compact plan-validity predictor split is invalid")
    fit_views = compact_plan_validity_feature_views(fit_rows)
    eval_views = compact_plan_validity_feature_views(eval_rows)
    target = np.asarray([
        row["renew_ise_advantage"] for row in fit_rows
    ], dtype=np.float64)
    groups = np.asarray([row["seed"] for row in fit_rows], dtype=np.int64)
    output: dict[str, Any] = {}
    for name in COMPACT_PLAN_VALIDITY_PREDICTORS:
        fit_x, feature_names = fit_views[name]
        eval_x, eval_names = eval_views[name]
        if feature_names != eval_names:
            raise ValueError("compact plan-validity transformed schema changed")
        prediction, model = _grouped_ridge_fit_predict(
            fit_x,
            target,
            groups,
            eval_x,
            alpha_grid=ridge_alpha_grid,
        )
        for row, value in zip(eval_rows, prediction, strict=True):
            row[f"prediction_{name}"] = float(value)
        output[name] = {
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
            "model": model,
            "evaluation": _prediction_metrics(
                eval_rows, prediction, selection_rate=selection_rate
            ),
        }
    return {
        "fit_row_count": len(fit_rows),
        "evaluation_row_count": len(eval_rows),
        "fit_group_count": int(np.unique(groups).size),
        "target": "renew_ise_advantage_keep_minus_renew",
        "selection_utility_is_local_counterfactual_not_closed_loop_return": True,
        "predictors": output,
    }


def train_pointmaze_compact_plan_validity_cell(
    *,
    ridge_alpha_grid: Iterable[float] = DEFAULT_RIDGE_ALPHA_GRID,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = train_pointmaze_plan_validity_cell(
        ridge_alpha=1.0,
        **kwargs,
    )
    fit_rows = payload["branch_fit_rows"]
    eval_rows = payload["branch_evaluation_rows"]
    schema = payload["branch_feature_names"]
    masks = payload["branch_feature_masks"]
    for row in (*fit_rows, *eval_rows):
        row["feature_names"] = schema
        row["feature_masks"] = masks
        for key in tuple(row):
            if key.startswith("prediction_"):
                row.pop(key)
    predictor = fit_compact_plan_validity_predictors(
        fit_rows,
        eval_rows,
        ridge_alpha_grid=ridge_alpha_grid,
        selection_rate=float(kwargs["selection_rate"]),
    )
    for row in (*fit_rows, *eval_rows):
        row.pop("feature_names")
        row.pop("feature_masks")
        row.pop("oracle_regime_context", None)
        row.pop("oracle_regime_used_only_by_diagnostic_predictor", None)
        row["protocol_version"] = POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION
        row["algorithm_path"] = POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH
        row["privileged_regime_context_present"] = False

    payload["source_branch_protocol_version"] = (
        POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION
    )
    payload["source_branch_algorithm_path"] = POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH
    payload["protocol_version"] = POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION
    payload["algorithm_path"] = POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH
    payload["evidence_role"] = "compact_plan_validity_predictor_qualification"
    payload["plan_validity_predictor"] = predictor
    payload["predictor_development_source"] = (
        "stage8b_formal_results_used_for_structure_selection_only"
    )
    payload["trigger_training"] = "disabled_qualification_only"
    payload["plan_validity_predictor_deployment"] = "disabled"
    return payload


def resolved_compact_plan_validity_protocol(
    *,
    args: argparse.Namespace,
    task_options: dict[str, Any],
) -> dict[str, Any]:
    training, selection, branch_fit, branch_eval = _validate_branch_seed_roles(
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        branch_fit_seeds=args.branch_fit_seeds,
        branch_eval_seeds=args.branch_eval_seeds,
    )
    return {
        "protocol_version": POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION,
        "source_branch_protocol_version": POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH,
        "evidence_role": "compact_plan_validity_predictor_qualification",
        "policy": POINTMAZE_COMPACT_PLAN_VALIDITY_POLICY,
        "environment_id": str(args.env_id),
        "optimizer_seed": int(args.optimizer_seed),
        "iterations": int(args.iterations),
        "horizon": int(args.horizon),
        "upper_period_seconds": float(args.upper_period_seconds),
        "history_seconds": float(args.history_seconds),
        "fast_period_seconds": float(args.fast_period_seconds),
        "maximum_subgoal_delta": float(args.maximum_subgoal_delta),
        "reference_hidden_dim": int(args.reference_hidden_dim),
        "learning_rate": float(args.learning_rate),
        "checkpoint_evaluation_interval": int(
            args.checkpoint_evaluation_interval
        ),
        "branch_window_seconds": float(args.branch_window_seconds),
        "max_events_per_class": int(args.max_events_per_class),
        "ridge_alpha_grid": list(map(float, args.ridge_alpha_grid)),
        "ridge_alpha_selection": "leave_one_path_seed_out_mean_mse",
        "selection_rate": float(args.selection_rate),
        "train_seeds": list(training),
        "selection_seeds": list(selection),
        "branch_fit_seeds": list(branch_fit),
        "branch_eval_seeds": list(branch_eval),
        "branch_categories": list(BRANCH_CATEGORIES),
        "predictors": list(COMPACT_PLAN_VALIDITY_PREDICTORS),
        "primary_endpoint": (
            "causal_validity_interactions_selected_utility_minus_"
            "current_compact_quadratic"
        ),
        "task_options": _json_ready(task_options),
        "future_access": "none_for_candidate_features",
        "regime_label_access": "none_for_all_stage8c_predictors",
        "branch_supervision_budget": "reported_separately_from_control_training",
        "disabled": [
            "deployed_trigger",
            "joint_controller_predictor_training",
            "future_event_input_to_candidate",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_stage8b_parser()
    parser.description = __doc__
    parser.add_argument(
        "--ridge-alpha-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_RIDGE_ALPHA_GRID),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < float(args.selection_rate) <= 1.0:
        raise ValueError("selection rate must be in (0, 1]")
    task_options = _task_options(args)
    protocol = resolved_compact_plan_validity_protocol(
        args=args, task_options=task_options
    )
    output: dict[str, Any] = {
        "protocol": protocol,
        "status": "dry_run" if args.dry_run else "complete",
        "cells": [],
    }
    if not args.dry_run:
        output["cells"].append(train_pointmaze_compact_plan_validity_cell(
            env_id=args.env_id,
            train_seeds=args.train_seeds,
            selection_seeds=args.selection_seeds,
            branch_fit_seeds=args.branch_fit_seeds,
            branch_eval_seeds=args.branch_eval_seeds,
            iterations=args.iterations,
            horizon=args.horizon,
            optimizer_seed=args.optimizer_seed,
            upper_period_seconds=args.upper_period_seconds,
            history_seconds=args.history_seconds,
            fast_period_seconds=args.fast_period_seconds,
            maximum_subgoal_delta=args.maximum_subgoal_delta,
            reference_hidden_dim=args.reference_hidden_dim,
            learning_rate=args.learning_rate,
            checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
            branch_window_seconds=args.branch_window_seconds,
            max_events_per_class=args.max_events_per_class,
            selection_rate=args.selection_rate,
            task_options=task_options,
            ridge_alpha_grid=args.ridge_alpha_grid,
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
