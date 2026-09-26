"""Same-budget paired timing supervision for the PointMaze causal trigger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter

from .pointmaze_budgeted_trigger import (
    balanced_jitter_schedule,
    build_parser as build_stage9_parser,
    fit_budgeted_trigger_predictors,
    rollout_budgeted_trigger,
)
from .pointmaze_goal_validation import (
    POINTMAZE_LOWER_ACTION_COST,
    _json_ready,
    pointmaze_goal_bounds,
    squash_box_action,
)
from .pointmaze_plan_validity_branching import (
    PointMazeRegimeFeatureBuilder,
    _causal_plan_features,
    _task_options,
    _validate_branch_seed_roles,
)
from .pointmaze_plan_value_qualification import (
    _make_task,
    train_pointmaze_plan_value_cell,
)


PROTOCOL_VERSION = "pointmaze_timing_pair_stage11_v1_development"
ALGORITHM_PATH = "same_budget_grid_timing_pairs_branch_supervised_ridge"


def timing_pair_schedule(
    *, horizon: int, period_steps: int, max_offset_steps: int,
    bin_index: int, offset: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        horizon % period_steps or not 1 <= bin_index < horizon // period_steps
        or not 0 <= offset < max_offset_steps < period_steps
    ):
        raise ValueError("timing-pair schedule is invalid")
    wait = tuple(
        0 if index == 0 else index * period_steps + max_offset_steps
        for index in range(horizon // period_steps)
    )
    now = list(wait)
    now[bin_index] = bin_index * period_steps + offset
    return wait, tuple(now)


def timing_pair_opportunities(
    *, seed: int, optimizer_seed: int, horizon: int, period_steps: int,
    max_offset_steps: int, check_stride_steps: int, pairs_per_seed: int,
) -> tuple[tuple[int, int], ...]:
    bins = np.arange(1, horizon // period_steps)
    if (
        not 1 <= pairs_per_seed <= len(bins)
        or max_offset_steps % check_stride_steps
    ):
        raise ValueError("timing-pair opportunity count or stride is invalid")
    rng = np.random.default_rng(np.random.SeedSequence([
        int(optimizer_seed), int(seed), 11_071,
    ]))
    selected = sorted(map(int, rng.choice(
        bins, size=pairs_per_seed, replace=False,
    )))
    offset_count = max_offset_steps // check_stride_steps
    return tuple(
        (bin_index, ((index + seed) % offset_count) * check_stride_steps)
        for index, bin_index in enumerate(selected)
    )


def rollout_timing_schedule(
    controller: Any,
    *, seed: int, decision_steps: tuple[int, ...], capture_step: int,
    env_id: str, horizon: int, time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float, task_options: dict[str, Any],
) -> dict[str, Any]:
    period = time_scale.upper_period_steps
    if (
        len(decision_steps) != horizon // period
        or tuple(step // period for step in decision_steps)
        != tuple(range(horizon // period))
        or capture_step < 0 or capture_step >= horizon
    ):
        raise ValueError("timing-pair rollout violates its call budget")
    task = _make_task(env_id=env_id, seed=seed, horizon=horizon, **task_options)
    try:
        observation = task.reset()
        world_low, world_high = pointmaze_goal_bounds(task.environment)
        adapter = RelativeSubgoalAdapter(
            maximum_delta=np.full(
                observation.achieved_goal.size, maximum_subgoal_delta,
                dtype=np.float32,
            ),
            world_low=world_low,
            world_high=world_high,
            action_cost=POINTMAZE_LOWER_ACTION_COST,
        )
        history = PointMazeRegimeFeatureBuilder(
            time_scale=time_scale,
            task_dim=int(observation.task_measurement.size),
        )
        history.reset(observation)
        controller.reset_recurrent_inference()
        achieved_before = observation.achieved_goal.copy()
        subgoal = achieved_before.copy()
        decision_set = set(decision_steps)
        last_plan_step = -1
        episode_return = 0.0
        episode_ise = 0.0
        prefix_snapshot = None
        feature_names = None
        causal_features = None
        for step in range(horizon):
            if step == capture_step:
                feature_names, causal_features, _ = _causal_plan_features(
                    observation=observation,
                    feature_builder=history,
                    subgoal=subgoal,
                    plan_age_steps=step - last_plan_step,
                    time_scale=time_scale,
                )
                prefix_snapshot = np.concatenate((
                    observation.physical,
                    observation.achieved_goal,
                    observation.target,
                    observation.task_measurement,
                    history.history,
                    subgoal,
                    np.asarray([last_plan_step], dtype=np.float64),
                )).astype(np.float64)
            if step in decision_set:
                upper_state = history.upper_state(
                    observation, oracle_context=None,
                )
                output = controller.plan_goal(upper_state, sample=False)
                subgoal = adapter.decode(
                    np.asarray(output["action"], dtype=np.float32),
                    achieved_before,
                )
                last_plan_step = step
            lower_state = history.lower_state(observation, subgoal=subgoal)
            output = controller.act_conditioned(lower_state, sample=False)
            action = squash_box_action(
                np.asarray(output["action"], dtype=np.float32),
                task.action_low,
                task.action_high,
            )
            next_observation, reward, terminated, truncated, info = task.step(
                action,
            )
            if (terminated or truncated) and step + 1 != horizon:
                raise RuntimeError("timing-pair episode ended early")
            episode_return += float(reward)
            episode_ise += (
                float(info["tracking_distance"]) ** 2
                * time_scale.dt_seconds
            )
            achieved_before = next_observation.achieved_goal.copy()
            observation = next_observation
            history.update(observation)
        if prefix_snapshot is None or causal_features is None:
            raise RuntimeError("timing-pair prefix was not captured")
        return {
            "episode_return": episode_return,
            "tracking_squared_error_integral": episode_ise,
            "decision_steps": decision_steps,
            "upper_decision_count": len(decision_steps),
            "feature_names": feature_names,
            "causal_features": causal_features,
            "prefix_snapshot": prefix_snapshot,
        }
    finally:
        task.environment.close()


def evaluate_timing_pair(
    controller: Any,
    *, seed: int, bin_index: int, offset: int, env_id: str, horizon: int,
    time_scale: PhysicalTimeScaleContract, maximum_subgoal_delta: float,
    max_offset_steps: int, task_options: dict[str, Any],
) -> dict[str, Any]:
    wait_schedule, now_schedule = timing_pair_schedule(
        horizon=horizon,
        period_steps=time_scale.upper_period_steps,
        max_offset_steps=max_offset_steps,
        bin_index=bin_index,
        offset=offset,
    )
    opportunity_step = bin_index * time_scale.upper_period_steps + offset
    common = {
        "seed": seed, "capture_step": opportunity_step,
        "env_id": env_id, "horizon": horizon, "time_scale": time_scale,
        "maximum_subgoal_delta": maximum_subgoal_delta,
        "task_options": task_options,
    }
    wait = rollout_timing_schedule(
        controller, decision_steps=wait_schedule, **common,
    )
    now = rollout_timing_schedule(
        controller, decision_steps=now_schedule, **common,
    )
    prefix_difference = float(np.max(np.abs(
        wait["prefix_snapshot"] - now["prefix_snapshot"]
    )))
    if (
        prefix_difference > 1e-10
        or wait["feature_names"] != now["feature_names"]
        or not np.array_equal(
            wait["causal_features"], now["causal_features"]
        )
    ):
        raise RuntimeError("timing-pair arms lack an identical causal prefix")
    return {
        "seed": int(seed),
        "bin_index": int(bin_index),
        "offset": int(offset),
        "opportunity_step": int(opportunity_step),
        "feature_names": list(wait["feature_names"]),
        "causal_features": wait["causal_features"].tolist(),
        "candidate_feature_has_future_access": False,
        "candidate_feature_has_regime_label": False,
        "keep_tracking_squared_error_integral": wait[
            "tracking_squared_error_integral"
        ],
        "renew_tracking_squared_error_integral": now[
            "tracking_squared_error_integral"
        ],
        "renew_ise_advantage": (
            wait["tracking_squared_error_integral"]
            - now["tracking_squared_error_integral"]
        ),
        "keep_return": wait["episode_return"],
        "renew_return": now["episode_return"],
        "prefix_max_abs_difference": prefix_difference,
        "keep_upper_decision_count": wait["upper_decision_count"],
        "renew_upper_decision_count": now["upper_decision_count"],
        "pair_primitive_steps_replayed": 2 * horizon,
    }


def train_cell(args: argparse.Namespace) -> dict[str, Any]:
    training, selection, branch_fit, trigger_eval = _validate_branch_seed_roles(
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        branch_fit_seeds=args.branch_fit_seeds,
        branch_eval_seeds=args.trigger_eval_seeds,
    )
    task_options = _task_options(args)
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=0.01,
        upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds,
    )

    def training_schedule(seed: int) -> tuple[int, ...]:
        return balanced_jitter_schedule(
            seed=seed,
            horizon=args.horizon,
            period_steps=time_scale.upper_period_steps,
            max_offset_steps=args.max_offset_steps,
        )

    payload, controller = train_pointmaze_plan_value_cell(
        method="hrl_regime_history",
        env_id=args.env_id,
        train_seeds=training,
        selection_seeds=selection,
        eval_seeds=(*branch_fit, *trigger_eval),
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
        waypoint_perturbation=0.25,
        event_window_seconds=1.0,
        task_options=task_options,
        diagnostic_schedules=("fixed",),
        training_decision_steps_fn=training_schedule,
        training_schedule_name="balanced_jitter",
    )
    fixed_schedule = tuple(range(0, args.horizon, time_scale.upper_period_steps))
    sanity_seed = trigger_eval[0]
    sanity = rollout_timing_schedule(
        controller,
        seed=sanity_seed,
        decision_steps=fixed_schedule,
        capture_step=time_scale.upper_period_steps,
        env_id=args.env_id,
        horizon=args.horizon,
        time_scale=time_scale,
        maximum_subgoal_delta=args.maximum_subgoal_delta,
        task_options=task_options,
    )
    reference = next(
        row for row in payload["evaluation_rows"]
        if row["seed"] == sanity_seed
    )
    if (
        sanity["decision_steps"] != tuple(reference["decision_steps"])
        or abs(
            sanity["tracking_squared_error_integral"]
            - reference["tracking_squared_error_integral"]
        ) > 1e-8
        or abs(sanity["episode_return"] - reference["episode_return"]) > 1e-8
    ):
        raise RuntimeError("timing-pair evaluator differs from fixed controller")

    fit_rows = [
        evaluate_timing_pair(
            controller,
            seed=seed,
            bin_index=bin_index,
            offset=offset,
            env_id=args.env_id,
            horizon=args.horizon,
            time_scale=time_scale,
            maximum_subgoal_delta=args.maximum_subgoal_delta,
            max_offset_steps=args.max_offset_steps,
            task_options=task_options,
        )
        for seed in branch_fit
        for bin_index, offset in timing_pair_opportunities(
            seed=seed,
            optimizer_seed=args.optimizer_seed,
            horizon=args.horizon,
            period_steps=time_scale.upper_period_steps,
            max_offset_steps=args.max_offset_steps,
            check_stride_steps=args.check_stride_steps,
            pairs_per_seed=args.pairs_per_seed,
        )
    ]
    predictors = fit_budgeted_trigger_predictors(
        fit_rows,
        alpha_grid=args.ridge_alpha_grid,
        threshold_quantile=args.threshold_quantile,
    )
    candidate_rows = [
        rollout_budgeted_trigger(
            controller,
            seed=seed,
            mode="causal_validity_interactions",
            fitted_predictors=predictors,
            env_id=args.env_id,
            horizon=args.horizon,
            time_scale=time_scale,
            maximum_subgoal_delta=args.maximum_subgoal_delta,
            max_offset_steps=args.max_offset_steps,
            check_stride_steps=args.check_stride_steps,
            task_options=task_options,
        )
        for seed in trigger_eval
    ]
    fixed_rows = [
        {
            key: row[key]
            for key in (
                "seed", "episode_return", "tracking_squared_error_integral",
                "decision_steps", "upper_decision_count",
            )
        }
        for row in payload["evaluation_rows"]
        if row["seed"] in set(trigger_eval)
    ]
    return {
        "optimizer_seed": args.optimizer_seed,
        "protocol_version": PROTOCOL_VERSION,
        "algorithm_path": ALGORITHM_PATH,
        "controller_selected_iteration": payload["selected_checkpoint_iteration"],
        "controller_parameter_count": payload["capacity"]["actual_parameter_count"],
        "train_seeds": list(training),
        "selection_seeds": list(selection),
        "branch_fit_seeds": list(branch_fit),
        "trigger_eval_seeds": list(trigger_eval),
        "pairs_per_seed": args.pairs_per_seed,
        "branch_fit_primitive_steps_replayed": sum(
            row["pair_primitive_steps_replayed"] for row in fit_rows
        ),
        "fixed_sanity_primitive_steps_replayed": args.horizon,
        "branch_fit_rows": fit_rows,
        "trigger_predictor": predictors["causal_validity_interactions"],
        "fixed_replay_rows": fixed_rows,
        "aligned_candidate_rows": candidate_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_stage9_parser()
    parser.description = __doc__
    parser.add_argument("--pairs-per-seed", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    period = int(round(args.upper_period_seconds / 0.01))
    if (
        args.horizon % period
        or not 1 <= args.pairs_per_seed < args.horizon // period
        or args.max_offset_steps != 25
        or args.check_stride_steps != 5
    ):
        raise ValueError("Stage-11 timing-pair protocol was changed")
    output = {
        "status": "dry_run" if args.dry_run else "complete",
        "protocol": {
            "protocol_version": PROTOCOL_VERSION,
            "algorithm_path": ALGORITHM_PATH,
            "optimizer_seed": args.optimizer_seed,
            "environment_id": args.env_id,
            "iterations": args.iterations,
            "horizon": args.horizon,
            "pairs_per_seed": args.pairs_per_seed,
            "max_offset_steps": args.max_offset_steps,
            "check_stride_steps": args.check_stride_steps,
            "threshold_quantile": args.threshold_quantile,
            "ridge_alpha_grid": list(args.ridge_alpha_grid),
            "train_seeds": args.train_seeds,
            "selection_seeds": args.selection_seeds,
            "branch_fit_seeds": args.branch_fit_seeds,
            "trigger_eval_seeds": args.trigger_eval_seeds,
            "task_options": _task_options(args),
            "training_target": "same_budget_full_episode_ise_wait_minus_now",
        },
        "cells": [] if args.dry_run else [train_cell(args)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
