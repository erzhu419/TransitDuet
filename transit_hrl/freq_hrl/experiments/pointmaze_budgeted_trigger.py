"""Causal, fixed-budget PointMaze planner timing with a frozen controller."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter

from .pointmaze_compact_plan_validity import (
    DEFAULT_RIDGE_ALPHA_GRID,
    _CURRENT_COMPACT_NAMES,
    _base_matrix,
    _causal_validity_interactions,
    _grouped_ridge_fit_predict,
    _quadratic_features,
    _select_named_features,
)
from .pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    POINTMAZE_LOWER_ACTION_COST,
    _json_ready,
    pointmaze_goal_bounds,
    squash_box_action,
)
from .pointmaze_plan_validity_branching import (
    PointMazeRegimeFeatureBuilder,
    _causal_plan_features,
    _ridge_fit_predict,
    _validate_branch_seed_roles,
    _task_options,
    evaluate_plan_renewal_dataset,
)
from .pointmaze_plan_value_qualification import (
    DEFAULT_DISTRACTOR_AMPLITUDE,
    DEFAULT_DISTRACTOR_DWELL_SECONDS,
    DEFAULT_FORCE_PULSE_AMPLITUDE,
    DEFAULT_FORCE_PULSE_DURATION_SECONDS,
    DEFAULT_FORCE_PULSE_GAP_SECONDS,
    DEFAULT_REGIME_DWELL_SECONDS,
    _make_task,
    train_pointmaze_plan_value_cell,
)
from freq_hrl.domains.mujoco import POINTMAZE_REGIME_SPEEDS


POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION = (
    "pointmaze_budgeted_plan_trigger_stage9_v1"
)
POINTMAZE_BUDGETED_TRIGGER_ALGORITHM_PATH = (
    "causal_plan_value_trigger_one_call_per_period_variable_duration_controller"
)
TRIGGER_MODES = (
    "fixed",
    "balanced_jitter",
    "current_compact_quadratic",
    "causal_validity_interactions",
)
PREDICTOR_MODES = TRIGGER_MODES[2:]


def balanced_jitter_schedule(
    *,
    seed: int,
    horizon: int,
    period_steps: int,
    max_offset_steps: int,
) -> tuple[int, ...]:
    if (
        int(horizon) < int(period_steps)
        or int(period_steps) < 2
        or not 0 < int(max_offset_steps) < int(period_steps)
    ):
        raise ValueError("budgeted trigger schedule dimensions are invalid")
    starts = range(0, int(horizon), int(period_steps))
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 9_001_009])
    )
    return tuple(
        0 if start == 0 else start + int(rng.integers(
            0, min(int(max_offset_steps), int(horizon) - start - 1) + 1
        ))
        for start in starts
    )


def _predictor_vector(
    names: Iterable[str],
    values: Iterable[float],
    predictor: str,
) -> np.ndarray:
    row = {
        "feature_names": list(names),
        "causal_features": list(values),
        "candidate_feature_has_future_access": False,
        "candidate_feature_has_regime_label": False,
    }
    schema, matrix = _base_matrix([row])
    if predictor == "causal_validity_interactions":
        transformed, _ = _causal_validity_interactions(schema, matrix)
    elif predictor == "current_compact_quadratic":
        current = _select_named_features(schema, matrix, _CURRENT_COMPACT_NAMES)
        transformed, _ = _quadratic_features(current, _CURRENT_COMPACT_NAMES)
    else:
        raise ValueError(f"unknown budgeted trigger predictor: {predictor}")
    return transformed[0]


def fit_budgeted_trigger_predictors(
    fit_rows: list[dict[str, Any]],
    *,
    alpha_grid: Iterable[float] = DEFAULT_RIDGE_ALPHA_GRID,
    threshold_quantile: float = 0.75,
) -> dict[str, dict[str, Any]]:
    if not 0.0 < float(threshold_quantile) < 1.0:
        raise ValueError("trigger threshold quantile must be in (0, 1)")
    schema, matrix = _base_matrix(fit_rows)
    target = np.asarray(
        [row["renew_ise_advantage"] for row in fit_rows], dtype=np.float64
    )
    groups = np.asarray([row["seed"] for row in fit_rows], dtype=np.int64)
    output: dict[str, dict[str, Any]] = {}
    for name in PREDICTOR_MODES:
        if name == "causal_validity_interactions":
            x, feature_names = _causal_validity_interactions(schema, matrix)
        else:
            current = _select_named_features(
                schema, matrix, _CURRENT_COMPACT_NAMES
            )
            x, feature_names = _quadratic_features(
                current, _CURRENT_COMPACT_NAMES
            )
        _, model = _grouped_ridge_fit_predict(
            x, target, groups, x, alpha_grid=alpha_grid
        )
        out_of_fold = np.empty(len(fit_rows), dtype=np.float64)
        for held_out in np.unique(groups):
            train = groups != held_out
            out_of_fold[~train], _ = _ridge_fit_predict(
                x[train], target[train], x[~train], alpha=model["alpha"]
            )
        output[name] = {
            "feature_names": list(feature_names),
            "model": model,
            "threshold": float(np.quantile(
                out_of_fold, float(threshold_quantile)
            )),
            "threshold_quantile": float(threshold_quantile),
            "threshold_source": "branch_fit_leave_one_path_out_predictions",
        }
    return output


def _predict_renewal_advantage(
    *,
    feature_names: Iterable[str],
    features: Iterable[float],
    predictor: str,
    fitted: dict[str, Any],
) -> float:
    vector = _predictor_vector(feature_names, features, predictor)
    model = fitted["model"]
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if vector.shape != mean.shape or mean.shape != scale.shape:
        raise ValueError("budgeted trigger feature dimensions changed")
    score = float(weights[0] + np.dot((vector - mean) / scale, weights[1:]))
    if not np.isfinite(score):
        raise ValueError("budgeted trigger score is non-finite")
    return score


def rollout_budgeted_trigger(
    model: Any,
    *,
    seed: int,
    mode: str,
    fitted_predictors: dict[str, dict[str, Any]],
    env_id: str,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float,
    max_offset_steps: int,
    check_stride_steps: int,
    task_options: dict[str, Any],
) -> dict[str, Any]:
    if mode not in TRIGGER_MODES:
        raise ValueError(f"unknown budgeted trigger mode: {mode}")
    period = time_scale.upper_period_steps
    if (
        int(horizon) < period
        or int(horizon) % period
        or not 0 < int(max_offset_steps) < period
        or int(check_stride_steps) < 1
        or int(max_offset_steps) % int(check_stride_steps)
    ):
        raise ValueError("budgeted trigger timing is invalid")
    random_schedule = set(balanced_jitter_schedule(
        seed=seed,
        horizon=horizon,
        period_steps=period,
        max_offset_steps=max_offset_steps,
    ))
    task = _make_task(
        env_id=env_id, seed=seed, horizon=horizon, **task_options
    )
    try:
        observation = task.reset()
        world_low, world_high = pointmaze_goal_bounds(task.environment)
        maximum_delta = np.full(
            observation.achieved_goal.size,
            float(maximum_subgoal_delta),
            dtype=np.float32,
        )
        adapter = RelativeSubgoalAdapter(
            maximum_delta=maximum_delta,
            world_low=world_low,
            world_high=world_high,
            action_cost=POINTMAZE_LOWER_ACTION_COST,
        )
        history = PointMazeRegimeFeatureBuilder(
            time_scale=time_scale,
            task_dim=int(observation.task_measurement.size),
        )
        history.reset(observation)
        model.reset_recurrent_inference()
        achieved_before = observation.achieved_goal.copy()
        subgoal = achieved_before.copy()
        last_plan_step = -1
        planned_bin = -1
        decisions: list[int] = []
        score_checks = 0
        early_calls = 0
        rewards: list[float] = []
        squared_errors: list[float] = []
        for step in range(int(horizon)):
            bin_index = step // period
            bin_start = bin_index * period
            plan_now = step == 0
            if mode == "fixed":
                plan_now = step == bin_start
            elif mode == "balanced_jitter":
                plan_now = step in random_schedule
            elif step > 0 and bin_index != planned_bin:
                offset = step - bin_start
                if offset <= max_offset_steps and offset % check_stride_steps == 0:
                    names, values, _ = _causal_plan_features(
                        observation=observation,
                        feature_builder=history,
                        subgoal=subgoal,
                        plan_age_steps=step - last_plan_step,
                        time_scale=time_scale,
                    )
                    score = _predict_renewal_advantage(
                        feature_names=names,
                        features=values,
                        predictor=mode,
                        fitted=fitted_predictors[mode],
                    )
                    score_checks += 1
                    plan_now = (
                        score >= fitted_predictors[mode]["threshold"]
                        or offset == max_offset_steps
                    )
                    if plan_now and offset < max_offset_steps:
                        early_calls += 1
            if plan_now:
                upper_state = history.upper_state(
                    observation, oracle_context=None
                )
                output = model.plan_goal(upper_state, sample=False)
                subgoal = adapter.decode(
                    np.asarray(output["action"], dtype=np.float32),
                    achieved_before,
                )
                decisions.append(step)
                planned_bin = bin_index
                last_plan_step = step
            lower_state = history.lower_state(observation, subgoal=subgoal)
            output = model.act_conditioned(lower_state, sample=False)
            action = squash_box_action(
                np.asarray(output["action"], dtype=np.float32),
                task.action_low,
                task.action_high,
            )
            next_observation, reward, terminated, truncated, info = task.step(
                action
            )
            if (terminated or truncated) and step + 1 != int(horizon):
                raise RuntimeError("budgeted trigger episode ended early")
            rewards.append(float(reward))
            squared_errors.append(float(info["tracking_distance"]) ** 2)
            achieved_before = next_observation.achieved_goal.copy()
            observation = next_observation
            history.update(observation)
        durations = np.diff(np.asarray([*decisions, horizon], dtype=np.int64))
        fixed_budget = len(range(0, horizon, period))
        per_bin = Counter(step // period for step in decisions)
        if (
            len(decisions) != fixed_budget
            or len(per_bin) != fixed_budget
            or set(per_bin.values()) != {1}
            or np.min(durations) < period - max_offset_steps
            or np.max(durations) > period + max_offset_steps
        ):
            raise RuntimeError("budgeted trigger changed the planning budget")
        return {
            "protocol_version": POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION,
            "algorithm_path": POINTMAZE_BUDGETED_TRIGGER_ALGORITHM_PATH,
            "seed": int(seed),
            "mode": mode,
            "episode_return": float(np.sum(rewards)),
            "tracking_squared_error_integral": float(
                np.sum(squared_errors) * time_scale.dt_seconds
            ),
            "tracking_rmse": float(np.sqrt(np.mean(squared_errors))),
            "episode_length": int(horizon),
            "upper_decision_count": len(decisions),
            "fixed_budget_upper_decision_count": fixed_budget,
            "decision_steps": decisions,
            "option_duration_steps_min": int(np.min(durations)),
            "option_duration_steps_max": int(np.max(durations)),
            "option_duration_steps_sum": int(np.sum(durations)),
            "trigger_score_checks": score_checks,
            "trigger_early_calls": early_calls,
            "planner_called_only_on_decision": True,
            "has_privileged_regime_input": False,
            "protocol_valid": True,
        }
    finally:
        task.environment.close()


def train_pointmaze_budgeted_trigger_cell(
    *,
    env_id: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    branch_fit_seeds: Iterable[int],
    trigger_eval_seeds: Iterable[int],
    iterations: int,
    horizon: int,
    optimizer_seed: int,
    upper_period_seconds: float,
    history_seconds: float,
    fast_period_seconds: float,
    maximum_subgoal_delta: float,
    reference_hidden_dim: int,
    learning_rate: float,
    checkpoint_evaluation_interval: int,
    branch_window_seconds: float,
    max_events_per_class: int,
    ridge_alpha_grid: Iterable[float],
    threshold_quantile: float,
    max_offset_steps: int,
    check_stride_steps: int,
    task_options: dict[str, Any],
) -> dict[str, Any]:
    training, selection, branch_fit, trigger_eval = _validate_branch_seed_roles(
        train_seeds=train_seeds,
        selection_seeds=selection_seeds,
        branch_fit_seeds=branch_fit_seeds,
        branch_eval_seeds=trigger_eval_seeds,
    )
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=0.01,
        upper_period_seconds=upper_period_seconds,
        history_seconds=history_seconds,
        fast_period_seconds=fast_period_seconds,
    )
    def decision_schedule(seed: int) -> tuple[int, ...]:
        return balanced_jitter_schedule(
            seed=seed,
            horizon=horizon,
            period_steps=time_scale.upper_period_steps,
            max_offset_steps=max_offset_steps,
        )

    payload, controller = train_pointmaze_plan_value_cell(
        method="hrl_regime_history",
        env_id=env_id,
        train_seeds=training,
        selection_seeds=selection,
        eval_seeds=(*branch_fit, *trigger_eval),
        iterations=iterations,
        horizon=horizon,
        optimizer_seed=optimizer_seed,
        upper_period_seconds=upper_period_seconds,
        history_seconds=history_seconds,
        fast_period_seconds=fast_period_seconds,
        maximum_subgoal_delta=maximum_subgoal_delta,
        reference_hidden_dim=reference_hidden_dim,
        learning_rate=learning_rate,
        checkpoint_evaluation_interval=checkpoint_evaluation_interval,
        waypoint_perturbation=0.25,
        event_window_seconds=1.0,
        task_options=task_options,
        diagnostic_schedules=("fixed",),
        training_decision_steps_fn=decision_schedule,
        training_schedule_name="balanced_jitter",
    )
    if float(payload["time_scale"]["env_dt_seconds"]) != 0.01:
        raise RuntimeError("budgeted trigger task time step changed")
    fit_rows = evaluate_plan_renewal_dataset(
        controller,
        seeds=branch_fit,
        split="trigger_fit",
        optimizer_seed=optimizer_seed,
        branch_window_seconds=branch_window_seconds,
        max_events_per_class=max_events_per_class,
        env_id=env_id,
        horizon=horizon,
        time_scale=time_scale,
        maximum_subgoal_delta=maximum_subgoal_delta,
        task_options=task_options,
        prefix_decision_steps_fn=decision_schedule,
    )
    predictors = fit_budgeted_trigger_predictors(
        fit_rows,
        alpha_grid=ridge_alpha_grid,
        threshold_quantile=threshold_quantile,
    )
    evaluation = [
        rollout_budgeted_trigger(
            controller,
            seed=seed,
            mode=mode,
            fitted_predictors=predictors,
            env_id=env_id,
            horizon=horizon,
            time_scale=time_scale,
            maximum_subgoal_delta=maximum_subgoal_delta,
            max_offset_steps=max_offset_steps,
            check_stride_steps=check_stride_steps,
            task_options=task_options,
        )
        for seed in trigger_eval
        for mode in TRIGGER_MODES
    ]
    reference_rows = {
        (int(row["seed"]), "balanced_jitter"): row
        for row in payload["canonical_evaluation_rows"]
    }
    reference_rows.update({
        (int(row["seed"]), "fixed"): row
        for row in payload["evaluation_rows"]
    })
    for row in evaluation:
        if row["mode"] not in ("fixed", "balanced_jitter"):
            continue
        reference = reference_rows[(row["seed"], row["mode"])]
        if (
            row["decision_steps"] != reference["decision_steps"]
            or abs(
                row["tracking_squared_error_integral"]
                - reference["tracking_squared_error_integral"]
            ) > 1e-8
            or abs(row["episode_return"] - reference["episode_return"]) > 1e-8
        ):
            raise RuntimeError("budgeted trigger replay differs from controller")
    for row in fit_rows:
        row.pop("oracle_regime_context", None)
        row.pop("oracle_regime_used_only_by_diagnostic_predictor", None)
        row["privileged_regime_context_present"] = False
        row["protocol_version"] = POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION
        row["algorithm_path"] = POINTMAZE_BUDGETED_TRIGGER_ALGORITHM_PATH
    payload.update({
        "protocol_version": POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_BUDGETED_TRIGGER_ALGORITHM_PATH,
        "evidence_role": "budgeted_closed_loop_trigger_development",
        "trigger_training": "branch_supervised_ridge_with_grouped_fit_cv",
        "controller_training_schedule": "balanced_jitter",
        "branch_fit_seeds": list(branch_fit),
        "trigger_eval_seeds": list(trigger_eval),
        "branch_fit_rows": fit_rows,
        "branch_feature_names": fit_rows[0]["feature_names"],
        "branch_window_seconds": float(branch_window_seconds),
        "max_events_per_class": int(max_events_per_class),
        "trigger_predictors": predictors,
        "trigger_evaluation_rows": evaluation,
        "trigger_max_offset_steps": int(max_offset_steps),
        "trigger_check_stride_steps": int(check_stride_steps),
        "trigger_threshold_quantile": float(threshold_quantile),
        "branch_fit_primitive_steps_replayed": int(sum(
            row["keep_primitive_steps_replayed"]
            + row["renew_primitive_steps_replayed"]
            for row in fit_rows
        )),
    })
    payload["training_schedule"] = "balanced_jitter"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--iterations", type=int, default=384)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--upper-period-seconds", type=float, default=0.50)
    parser.add_argument("--history-seconds", type=float, default=0.64)
    parser.add_argument("--fast-period-seconds", type=float, default=0.04)
    parser.add_argument("--maximum-subgoal-delta", type=float, default=0.75)
    parser.add_argument("--reference-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=8)
    parser.add_argument("--branch-window-seconds", type=float, default=0.50)
    parser.add_argument("--max-events-per-class", type=int, default=4)
    parser.add_argument(
        "--ridge-alpha-grid", nargs="+", type=float,
        default=list(DEFAULT_RIDGE_ALPHA_GRID),
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.75)
    parser.add_argument("--max-offset-steps", type=int, default=25)
    parser.add_argument("--check-stride-steps", type=int, default=5)
    parser.add_argument(
        "--regime-dwell-seconds", nargs=2, type=float,
        default=list(DEFAULT_REGIME_DWELL_SECONDS),
    )
    parser.add_argument(
        "--target-speed-modes", nargs="+", type=float,
        default=list(POINTMAZE_REGIME_SPEEDS),
    )
    parser.add_argument(
        "--force-pulse-amplitude", type=float,
        default=DEFAULT_FORCE_PULSE_AMPLITUDE,
    )
    parser.add_argument(
        "--force-pulse-duration-seconds", nargs=2, type=float,
        default=list(DEFAULT_FORCE_PULSE_DURATION_SECONDS),
    )
    parser.add_argument(
        "--force-pulse-gap-seconds", nargs=2, type=float,
        default=list(DEFAULT_FORCE_PULSE_GAP_SECONDS),
    )
    parser.add_argument(
        "--distractor-amplitude", type=float,
        default=DEFAULT_DISTRACTOR_AMPLITUDE,
    )
    parser.add_argument(
        "--distractor-dwell-seconds", nargs=2, type=float,
        default=list(DEFAULT_DISTRACTOR_DWELL_SECONDS),
    )
    for role in (
        "train", "selection", "branch-fit", "trigger-eval"
    ):
        parser.add_argument(f"--{role}-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    training, selection, branch_fit, trigger_eval = _validate_branch_seed_roles(
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        branch_fit_seeds=args.branch_fit_seeds,
        branch_eval_seeds=args.trigger_eval_seeds,
    )
    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("threshold quantile must be in (0, 1)")
    task_options = _task_options(args)
    protocol = {
        "protocol_version": POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_BUDGETED_TRIGGER_ALGORITHM_PATH,
        "evidence_role": "budgeted_closed_loop_trigger_development",
        "optimizer_seed": args.optimizer_seed,
        "environment_id": args.env_id,
        "iterations": args.iterations,
        "horizon": args.horizon,
        "upper_period_seconds": args.upper_period_seconds,
        "history_seconds": args.history_seconds,
        "fast_period_seconds": args.fast_period_seconds,
        "maximum_subgoal_delta": args.maximum_subgoal_delta,
        "reference_hidden_dim": args.reference_hidden_dim,
        "learning_rate": args.learning_rate,
        "checkpoint_evaluation_interval": args.checkpoint_evaluation_interval,
        "branch_window_seconds": args.branch_window_seconds,
        "max_events_per_class": args.max_events_per_class,
        "ridge_alpha_grid": args.ridge_alpha_grid,
        "threshold_quantile": args.threshold_quantile,
        "max_offset_steps": args.max_offset_steps,
        "check_stride_steps": args.check_stride_steps,
        "train_seeds": list(training),
        "selection_seeds": list(selection),
        "branch_fit_seeds": list(branch_fit),
        "trigger_eval_seeds": list(trigger_eval),
        "task_options": task_options,
        "methods": list(TRIGGER_MODES),
        "planning_budget": "one_upper_call_per_50_step_bin",
        "predictor_threshold_source": "branch_fit_grouped_out_of_fold_only",
        "candidate_future_or_regime_access": False,
    }
    output: dict[str, Any] = {
        "protocol": protocol,
        "status": "dry_run" if args.dry_run else "complete",
        "cells": [],
    }
    if not args.dry_run:
        output["cells"].append(train_pointmaze_budgeted_trigger_cell(
            env_id=args.env_id,
            train_seeds=training,
            selection_seeds=selection,
            branch_fit_seeds=branch_fit,
            trigger_eval_seeds=trigger_eval,
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
            ridge_alpha_grid=args.ridge_alpha_grid,
            threshold_quantile=args.threshold_quantile,
            max_offset_steps=args.max_offset_steps,
            check_stride_steps=args.check_stride_steps,
            task_options=task_options,
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
