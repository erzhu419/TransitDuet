"""Compare one-check and deadline deferral with adaptive trigger continuation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from .pointmaze_budgeted_trigger import (
    _predict_renewal_advantage,
    build_parser,
)
from .pointmaze_deployed_pair_diagnostic import replay_source_controller, select_bins
from .pointmaze_goal_validation import (
    POINTMAZE_LOWER_ACTION_COST, _json_ready, pointmaze_goal_bounds, squash_box_action,
)
from .pointmaze_plan_validity_branching import (
    PointMazeRegimeFeatureBuilder, _causal_plan_features, _task_options,
)
from .pointmaze_plan_value_qualification import _make_task


PROTOCOL_VERSION = "pointmaze_adaptive_pair_stage14_v1_diagnostic"
ARMS = ("now", "wait_one_check", "wait_deadline")


def rollout_intervention(
    controller, *, seed, intervention_step, arm, predictor, args, time_scale,
    score_fn=None, collect_value_trace=False, capture_endpoint_history=False,
    continuation_seed=None,
):
    period = time_scale.upper_period_steps
    if intervention_step is None:
        if arm is not None or continuation_seed is not None:
            raise ValueError("a factual rollout has no intervention arm")
    elif (
        arm not in ARMS or not period <= intervention_step < args.horizon - period
        or intervention_step % period >= args.max_offset_steps
        or intervention_step % period % args.check_stride_steps
    ):
        raise ValueError("intervention must be an eligible nondeadline check")
    task = _make_task(
        env_id=args.env_id, seed=seed, horizon=args.horizon, **_task_options(args),
    )
    try:
        observation = task.reset()
        low, high = pointmaze_goal_bounds(task.environment)
        adapter = RelativeSubgoalAdapter(
            maximum_delta=np.full(observation.achieved_goal.size,
                                  args.maximum_subgoal_delta, dtype=np.float32),
            world_low=low, world_high=high, action_cost=POINTMAZE_LOWER_ACTION_COST,
        )
        history = PointMazeRegimeFeatureBuilder(
            time_scale=time_scale, task_dim=int(observation.task_measurement.size),
        )
        history.reset(observation)
        controller.reset_recurrent_inference()
        subgoal = observation.achieved_goal.copy()
        last_plan_step = -1
        decisions, rewards, errors = [], [], []
        captured = None
        value_trace = []
        trace_start = 0 if intervention_step is None else intervention_step + period
        for step in range(args.horizon):
            bin_index, offset = divmod(step, period)
            if continuation_seed is not None and step == trace_start:
                task.driver = task.driver.conditional_future(step=step, seed=continuation_seed)
                if not np.array_equal(np.concatenate(task.driver.sample(step)), observation.task_measurement):
                    raise RuntimeError("conditional future changed the current observation")
            if collect_value_trace and step >= trace_start and (step - trace_start) % period == 0:
                names, values, _ = _causal_plan_features(
                    observation=observation, feature_builder=history,
                    subgoal=subgoal, plan_age_steps=step - last_plan_step,
                    time_scale=time_scale,
                )
                value_trace.append({
                    "step": step,
                    "feature_names": [*names, "within_bin_fraction", "remaining_fraction", "budget_spent"],
                    "features": [*map(float, values), offset / period,
                                 (args.horizon - step) / args.horizon,
                                 float(bool(decisions) and decisions[-1] // period == bin_index)],
                })
                if capture_endpoint_history and step == trace_start:
                    value_trace[-1]["controller_history"] = history.history.tolist()
            plan_now = step == 0
            if step > 0 and (not decisions or decisions[-1] // period != bin_index):
                if offset <= args.max_offset_steps and offset % args.check_stride_steps == 0:
                    names, values, _ = _causal_plan_features(
                        observation=observation, feature_builder=history,
                        subgoal=subgoal, plan_age_steps=step - last_plan_step,
                        time_scale=time_scale,
                    )
                    score = _predict_renewal_advantage(
                        feature_names=names, features=values,
                        predictor="causal_validity_interactions", fitted=predictor,
                    ) if score_fn is None else score_fn(
                        names, values, offset / args.max_offset_steps,
                        (args.horizon - step) / args.horizon,
                    )
                    if step == intervention_step:
                        captured = {
                            "score": score, "features": np.asarray(values),
                            "feature_names": names,
                            "prefix": np.concatenate((
                                observation.physical, observation.achieved_goal,
                                observation.target, observation.task_measurement,
                                history.history, subgoal, [last_plan_step],
                            )).astype(np.float64),
                        }
                    plan_now = score >= predictor["threshold"] or offset == args.max_offset_steps
                    if step == intervention_step:
                        plan_now = arm == "now"
                    elif (
                        arm == "wait_deadline" and step > intervention_step
                        and bin_index == intervention_step // period
                    ):
                        plan_now = offset == args.max_offset_steps
            if plan_now:
                output = controller.plan_goal(
                    history.upper_state(observation, oracle_context=None), sample=False,
                )
                subgoal = adapter.decode(np.asarray(output["action"], dtype=np.float32),
                                         observation.achieved_goal)
                last_plan_step = step
                decisions.append(step)
            output = controller.act_conditioned(
                history.lower_state(observation, subgoal=subgoal), sample=False,
            )
            action = squash_box_action(np.asarray(output["action"], dtype=np.float32),
                                       task.action_low, task.action_high)
            observation, reward, terminated, truncated, info = task.step(action)
            if (terminated or truncated) and step + 1 != args.horizon:
                raise RuntimeError("intervention episode ended early")
            rewards.append(float(reward))
            errors.append(float(info["tracking_distance"]) ** 2)
            history.update(observation)
        if (
            (intervention_step is not None and captured is None)
            or [s // period for s in decisions] != list(range(args.horizon // period))
        ):
            raise RuntimeError("intervention prefix or one-call-per-bin budget is invalid")
        if collect_value_trace:
            suffix = np.cumsum(np.asarray(errors)[::-1])[::-1] * time_scale.dt_seconds
            for point in value_trace:
                point["cost_to_go"] = float(suffix[point["step"]])
        return {
            **(captured or {}),
            **({"value_trace": value_trace} if collect_value_trace else {}),
            "decision_steps": decisions,
            "episode_return": float(np.sum(rewards)),
            "tracking_squared_error_integral": float(np.sum(errors) * time_scale.dt_seconds),
            "window_ise": (float(np.sum(errors[intervention_step:intervention_step + period])
                                 * time_scale.dt_seconds)
                           if intervention_step is not None else None),
        }
    finally:
        task.environment.close()


def run_cell(args):
    source, controller, time_scale = replay_source_controller(args)
    period = time_scale.upper_period_steps
    predictor = source["trigger_predictor"]
    rows = []
    for candidate in source["aligned_candidate_rows"]:
        seed, schedule = candidate["seed"], candidate["decision_steps"]
        for bin_index in select_bins(
            root=args.optimizer_seed, seed=seed, decision_steps=schedule,
            period=period, deadline=args.max_offset_steps,
            pairs_per_class=args.pairs_per_class,
        ):
            early = schedule[bin_index] % period < args.max_offset_steps
            check = schedule[bin_index] if early else bin_index * period
            arms = {name: rollout_intervention(
                controller, seed=seed, intervention_step=check, arm=name,
                predictor=predictor, args=args, time_scale=time_scale,
            ) for name in ARMS}
            now = arms["now"]
            factual = now if early else arms["wait_one_check"]
            if (
                factual["decision_steps"] != schedule
                or abs(factual["episode_return"] - candidate["episode_return"]) > 1e-8
                or abs(factual["tracking_squared_error_integral"]
                       - candidate["tracking_squared_error_integral"]) > 1e-8
                or (now["score"] >= predictor["threshold"]) != early
                or any(not np.array_equal(value[key], now[key])
                       for value in arms.values() for key in ("prefix", "features"))
            ):
                raise RuntimeError("adaptive factual replay or causal prefix differs")
            rows.append({
                "seed": seed, "bin_index": bin_index, "check_step": check,
                "factual_early": early, "score": now["score"],
                "threshold": predictor["threshold"], "prefix_max_abs_difference": 0.0,
                "arms": {name: {key: value[key] for key in (
                    "decision_steps", "episode_return",
                    "tracking_squared_error_integral", "window_ise",
                )} for name, value in arms.items()},
            })
    return {
        "optimizer_seed": args.optimizer_seed,
        "paths": len(source["aligned_candidate_rows"]), "opportunities": len(rows),
        "selected_checkpoint_iteration": source["controller_selected_iteration"],
        "intervention_primitive_steps_replayed": len(rows) * len(ARMS) * args.horizon,
        "rows": rows,
    }


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--pairs-per-class", type=int, default=2)
    args = parser.parse_args(argv)
    output = {
        "status": "dry_run" if args.dry_run else "complete",
        "protocol": {
            "protocol_version": PROTOCOL_VERSION, "source_result": str(args.source_result),
            "optimizer_seed": args.optimizer_seed, "pairs_per_class": args.pairs_per_class,
            "arms": ARMS, "continuation": "frozen_stage12_trigger_on_each_arms_own_observations",
            "evidence_role": "mechanism_diagnostic_only",
        },
        "cells": [] if args.dry_run else [run_cell(args)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
