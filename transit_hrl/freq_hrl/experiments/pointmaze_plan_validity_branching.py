"""Paired keep/renew qualification for causal plan-validity learning.

This is a successor to the failed Stage-8 event-timing gate.  It does not run
a trigger in the control loop.  Instead, it replays the same deterministic
prefix twice and changes only whether the upper planner renews the current
waypoint at a registered opportunity.  Both branches then keep their waypoint
for the same physical-time window while the lower controller remains closed
loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import (
    POINTMAZE_REGIME_SPEEDS,
    POINTMAZE_U_ROUTE_XY,
    RelativeSubgoalAdapter,
)

from .pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    POINTMAZE_LOWER_ACTION_COST,
    _json_ready,
    pointmaze_goal_bounds,
    squash_box_action,
)
from .pointmaze_plan_value_qualification import (
    DEFAULT_DISTRACTOR_AMPLITUDE,
    DEFAULT_DISTRACTOR_DWELL_SECONDS,
    DEFAULT_FORCE_PULSE_AMPLITUDE,
    DEFAULT_FORCE_PULSE_DURATION_SECONDS,
    DEFAULT_FORCE_PULSE_GAP_SECONDS,
    DEFAULT_REGIME_DWELL_SECONDS,
    PointMazeRegimeFeatureBuilder,
    _make_task,
    fixed_replan_steps,
    train_pointmaze_plan_value_cell,
)


POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION = (
    "pointmaze_counterfactual_plan_validity_stage8b_v1"
)
POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH = (
    "paired_keep_renew_plan_value_qualification_not_deployed_trigger"
)
POINTMAZE_PLAN_VALIDITY_POLICY = "hrl_regime_history"
BRANCH_CATEGORIES = (
    "regime_lag_010ms",
    "regime_lag_100ms",
    "regime_lag_250ms",
    "force_pulse_010ms",
    "distractor_change_010ms",
    "neutral_matched",
)
PREDICTOR_NAMES = (
    "age_only",
    "plan_state",
    "change_magnitude",
    "causal_history",
    "causal_history_plus_regime",
)


@dataclass(frozen=True)
class PlanRenewalOpportunity:
    category: str
    step: int
    source_step: int | None
    lag_steps: int | None


def _evenly_spaced(values: Iterable[int], *, count: int) -> tuple[int, ...]:
    ordered = tuple(sorted(set(map(int, values))))
    if int(count) < 1 or not ordered:
        return ()
    if len(ordered) <= int(count):
        return ordered
    indices = np.rint(
        np.linspace(0, len(ordered) - 1, num=int(count))
    ).astype(np.int64)
    return tuple(ordered[int(index)] for index in indices)


def plan_renewal_opportunities(
    *,
    horizon: int,
    period_steps: int,
    branch_window_steps: int,
    regime_change_steps: Iterable[int],
    force_pulse_steps: Iterable[int],
    distractor_change_steps: Iterable[int],
    seed: int,
    max_events_per_class: int,
) -> tuple[PlanRenewalOpportunity, ...]:
    """Build bounded causal opportunities without using future actor input."""

    if min(
        int(horizon),
        int(period_steps),
        int(branch_window_steps),
        int(max_events_per_class),
    ) < 1:
        raise ValueError("plan-validity opportunity counts must be positive")
    latest = int(horizon) - int(branch_window_steps)
    fixed = set(range(0, int(horizon), int(period_steps)))

    def valid(step: int) -> bool:
        return 1 <= int(step) <= latest and int(step) not in fixed

    opportunities: list[PlanRenewalOpportunity] = []
    occupied: set[int] = set()

    def add_event_class(
        category: str,
        sources: Iterable[int],
        lag_steps: int,
    ) -> None:
        candidates = [
            (int(source) + int(lag_steps), int(source))
            for source in sorted(set(map(int, sources)))
            if valid(int(source) + int(lag_steps))
            and int(source) + int(lag_steps) not in occupied
        ]
        selected_steps = set(_evenly_spaced(
            (step for step, _ in candidates), count=max_events_per_class
        ))
        for step, source in candidates:
            if step not in selected_steps:
                continue
            opportunities.append(PlanRenewalOpportunity(
                category=category,
                step=step,
                source_step=source,
                lag_steps=int(lag_steps),
            ))
            occupied.add(step)

    changes = tuple(sorted(set(map(int, regime_change_steps))))
    add_event_class("regime_lag_010ms", changes, 1)
    add_event_class("regime_lag_100ms", changes, 10)
    add_event_class("regime_lag_250ms", changes, 25)
    add_event_class("force_pulse_010ms", force_pulse_steps, 1)
    add_event_class(
        "distractor_change_010ms", distractor_change_steps, 1
    )

    consequential_events = tuple(sorted(set(
        changes + tuple(map(int, force_pulse_steps))
    )))
    distractor_events = tuple(sorted(set(map(int, distractor_change_steps))))
    neutral_candidates = []
    for step in range(1, latest + 1):
        if not valid(step) or step in occupied:
            continue
        if any(
            step - 10 <= event <= step + int(branch_window_steps)
            for event in consequential_events
        ):
            continue
        if any(abs(event - step) < 10 for event in distractor_events):
            continue
        neutral_candidates.append(step)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 8_208_031])
    )
    if neutral_candidates:
        order = rng.permutation(len(neutral_candidates))
        selected = sorted(
            neutral_candidates[int(index)]
            for index in order[:int(max_events_per_class)]
        )
        for step in selected:
            opportunities.append(PlanRenewalOpportunity(
                category="neutral_matched",
                step=int(step),
                source_step=None,
                lag_steps=None,
            ))
            occupied.add(int(step))

    result = tuple(sorted(
        opportunities, key=lambda item: (item.step, item.category)
    ))
    observed = {item.category for item in result}
    if set(BRANCH_CATEGORIES) - observed:
        missing = sorted(set(BRANCH_CATEGORIES) - observed)
        raise ValueError(f"plan-validity opportunity class missing: {missing}")
    if len({item.step for item in result}) != len(result):
        raise RuntimeError("plan-validity opportunities must have unique steps")
    return result


def _causal_plan_features(
    *,
    observation: Any,
    feature_builder: PointMazeRegimeFeatureBuilder,
    subgoal: np.ndarray,
    plan_age_steps: int,
    time_scale: PhysicalTimeScaleContract,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, tuple[int, ...]]]:
    history = feature_builder.history.reshape(
        time_scale.history_steps, -1
    ).astype(np.float64)
    dt = float(time_scale.dt_seconds)
    target = history[:, :2]
    force = history[:, 2:4]
    distractor = history[:, 4:6]

    names: list[str] = []
    values: list[float] = []

    def append_vector(prefix: str, vector: np.ndarray) -> None:
        flat = np.asarray(vector, dtype=np.float64).reshape(-1)
        for index, value in enumerate(flat):
            names.append(f"{prefix}_{index}")
            values.append(float(value))

    append_vector("physical", observation.physical)
    append_vector("target_error", observation.target_error)
    waypoint_error = np.asarray(
        subgoal - observation.achieved_goal, dtype=np.float64
    )
    append_vector("waypoint_error", waypoint_error)
    names.extend((
        "tracking_distance",
        "waypoint_distance",
        "plan_age_seconds",
        "plan_age_fraction",
    ))
    values.extend((
        float(np.linalg.norm(observation.target_error)),
        float(np.linalg.norm(waypoint_error)),
        float(plan_age_steps * dt),
        float(plan_age_steps / time_scale.upper_period_steps),
    ))
    append_vector("target_current", target[-1])
    append_vector("force_current", force[-1])
    append_vector("distractor_current", distractor[-1])

    velocity: dict[int, np.ndarray] = {}
    for lag in (1, 10, 25, 50):
        if lag >= history.shape[0]:
            raise ValueError("plan-validity history is shorter than feature lag")
        estimate = (target[-1] - target[-1 - lag]) / (lag * dt)
        velocity[lag] = estimate
        append_vector(f"target_velocity_{lag:03d}", estimate)
    for window in (10, 25):
        rms = np.sqrt(np.mean(np.square(force[-window:]), axis=0))
        append_vector(f"force_rms_{window:03d}", rms)
        append_vector(
            f"distractor_delta_{window:03d}",
            distractor[-1] - distractor[-window],
        )
    names.extend((
        "target_velocity_change_norm",
        "force_current_norm",
        "distractor_change_norm",
    ))
    values.extend((
        float(np.linalg.norm(velocity[1] - velocity[25])),
        float(np.linalg.norm(force[-1])),
        float(np.linalg.norm(distractor[-1] - distractor[-10])),
    ))

    name_to_index = {name: index for index, name in enumerate(names)}
    if len(name_to_index) != len(names):
        raise RuntimeError("plan-validity feature names must be unique")
    age_only = (name_to_index["plan_age_fraction"],)
    plan_state = tuple(
        index
        for index, name in enumerate(names)
        if name.startswith((
            "physical_",
            "target_error_",
            "waypoint_error_",
            "target_current_",
            "force_current_",
            "distractor_current_",
        ))
        or name in {
            "tracking_distance",
            "waypoint_distance",
            "plan_age_seconds",
            "plan_age_fraction",
            "force_current_norm",
        }
    )
    change_magnitude = tuple(name_to_index[name] for name in (
        "plan_age_fraction",
        "target_velocity_change_norm",
        "force_current_norm",
        "distractor_change_norm",
    ))
    masks = {
        "age_only": age_only,
        "plan_state": plan_state,
        "change_magnitude": change_magnitude,
        "causal_history": tuple(range(len(names))),
    }
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(names),) or not np.all(np.isfinite(vector)):
        raise RuntimeError("plan-validity causal feature vector is invalid")
    return tuple(names), vector, masks


def _run_branch(
    model: Any,
    *,
    seed: int,
    opportunity: PlanRenewalOpportunity,
    renew: bool,
    branch_window_steps: int,
    env_id: str,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float,
    task_options: dict[str, Any],
) -> dict[str, Any]:
    if str(model.config.state_encoder) != "mlp":
        raise ValueError("paired branch replay currently requires stateless MLPs")
    task = _make_task(
        env_id=env_id, seed=int(seed), horizon=int(horizon), **task_options
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
        features = PointMazeRegimeFeatureBuilder(
            time_scale=time_scale,
            task_dim=int(observation.task_measurement.size),
        )
        features.reset(observation)
        model.reset_recurrent_inference()
        fixed = set(fixed_replan_steps(
            horizon=horizon,
            period_steps=time_scale.upper_period_steps,
        ))
        if opportunity.step in fixed:
            raise ValueError("branch opportunity collides with a fixed replan")

        achieved_before = observation.achieved_goal.copy()
        subgoal = achieved_before.copy()
        last_plan_step = -1
        branch_losses: list[float] = []
        branch_rewards: list[float] = []
        branch_requested: list[np.ndarray] = []
        feature_names: tuple[str, ...] | None = None
        feature_vector: np.ndarray | None = None
        feature_masks: dict[str, tuple[int, ...]] | None = None
        context: np.ndarray | None = None
        old_subgoal: np.ndarray | None = None
        renewed_subgoal: np.ndarray | None = None
        prefix_snapshot: np.ndarray | None = None
        near_route_corner: bool | None = None
        end_step = int(opportunity.step) + int(branch_window_steps)

        for step in range(end_step):
            if step < opportunity.step and step in fixed:
                upper_state = features.upper_state(
                    observation, oracle_context=None
                )
                output = model.plan_goal(upper_state, sample=False)
                subgoal = adapter.decode(
                    np.asarray(output["action"], dtype=np.float32),
                    achieved_before,
                )
                last_plan_step = int(step)

            if step == opportunity.step:
                if last_plan_step < 0:
                    raise RuntimeError("branch prefix has no active plan")
                old_subgoal = subgoal.copy()
                feature_names, feature_vector, feature_masks = (
                    _causal_plan_features(
                        observation=observation,
                        feature_builder=features,
                        subgoal=subgoal,
                        plan_age_steps=step - last_plan_step,
                        time_scale=time_scale,
                    )
                )
                context = task.privileged_context().astype(
                    np.float64, copy=True
                )
                near_route_corner = bool(
                    np.min(np.linalg.norm(
                        np.asarray(observation.target, dtype=np.float64)
                        - POINTMAZE_U_ROUTE_XY[[2, 4]],
                        axis=1,
                    )) <= 0.35
                )
                prefix_snapshot = np.concatenate((
                    observation.physical,
                    observation.achieved_goal,
                    observation.task_measurement,
                    features.history,
                    subgoal,
                )).astype(np.float64)
                if renew:
                    upper_state = features.upper_state(
                        observation, oracle_context=None
                    )
                    output = model.plan_goal(upper_state, sample=False)
                    subgoal = adapter.decode(
                        np.asarray(output["action"], dtype=np.float32),
                        achieved_before,
                    )
                renewed_subgoal = subgoal.copy()

            lower_state = features.lower_state(
                observation, subgoal=subgoal
            )
            output = model.act_conditioned(lower_state, sample=False)
            requested = squash_box_action(
                np.asarray(output["action"], dtype=np.float32),
                task.action_low,
                task.action_high,
            )
            next_observation, reward, terminated, truncated, info = task.step(
                requested
            )
            if terminated or truncated:
                raise RuntimeError("plan-validity branch ended before its window")
            if step >= opportunity.step:
                branch_losses.append(
                    float(info["tracking_distance"]) ** 2
                    * time_scale.dt_seconds
                )
                branch_rewards.append(float(reward))
                branch_requested.append(requested.copy())
            achieved_before = next_observation.achieved_goal.copy()
            observation = next_observation
            features.update(observation)

        if any(value is None for value in (
            feature_names,
            feature_vector,
            feature_masks,
            context,
            old_subgoal,
            renewed_subgoal,
            prefix_snapshot,
            near_route_corner,
        )):
            raise RuntimeError("plan-validity branch snapshot was not captured")
        requested_array = np.asarray(branch_requested, dtype=np.float64)
        return {
            "feature_names": feature_names,
            "feature_vector": feature_vector,
            "feature_masks": feature_masks,
            "oracle_regime_context": context,
            "old_subgoal": old_subgoal,
            "branch_subgoal": renewed_subgoal,
            "prefix_snapshot": prefix_snapshot,
            "branch_tracking_squared_error_integral": float(
                np.sum(branch_losses)
            ),
            "branch_return": float(np.sum(branch_rewards)),
            "branch_requested_action_rms": float(np.sqrt(
                np.mean(np.square(requested_array))
            )),
            "near_route_corner": bool(near_route_corner),
            "regime_id": int(np.argmax(context)),
            "primitive_steps_replayed": int(end_step),
            "task_diagnostics": task.diagnostics(),
        }
    finally:
        task.environment.close()


def evaluate_plan_renewal_pair(
    model: Any,
    *,
    seed: int,
    opportunity: PlanRenewalOpportunity,
    branch_window_steps: int,
    env_id: str,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float,
    task_options: dict[str, Any],
    optimizer_seed: int,
    split: str,
) -> dict[str, Any]:
    common = {
        "seed": int(seed),
        "opportunity": opportunity,
        "branch_window_steps": int(branch_window_steps),
        "env_id": str(env_id),
        "horizon": int(horizon),
        "time_scale": time_scale,
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "task_options": dict(task_options),
    }
    keep = _run_branch(model, renew=False, **common)
    renew = _run_branch(model, renew=True, **common)
    prefix_difference = float(np.max(np.abs(
        keep["prefix_snapshot"] - renew["prefix_snapshot"]
    )))
    feature_difference = float(np.max(np.abs(
        keep["feature_vector"] - renew["feature_vector"]
    )))
    if prefix_difference > 1e-10 or feature_difference > 1e-10:
        raise RuntimeError("keep/renew branches do not share an exact prefix")
    if keep["feature_names"] != renew["feature_names"]:
        raise RuntimeError("keep/renew feature schemas differ")
    if keep["feature_masks"] != renew["feature_masks"]:
        raise RuntimeError("keep/renew feature masks differ")
    if not np.array_equal(
        keep["oracle_regime_context"], renew["oracle_regime_context"]
    ):
        raise RuntimeError("keep/renew privileged contexts differ")
    if keep["task_diagnostics"] != renew["task_diagnostics"]:
        raise RuntimeError("keep/renew external paths differ")

    keep_loss = float(keep["branch_tracking_squared_error_integral"])
    renew_loss = float(renew["branch_tracking_squared_error_integral"])
    keep_return = float(keep["branch_return"])
    renew_return = float(renew["branch_return"])
    old_subgoal = np.asarray(keep["old_subgoal"], dtype=np.float64)
    new_subgoal = np.asarray(renew["branch_subgoal"], dtype=np.float64)
    diagnostics = keep["task_diagnostics"]
    row = {
        "protocol_version": POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH,
        "policy": POINTMAZE_PLAN_VALIDITY_POLICY,
        "optimizer_seed": int(optimizer_seed),
        "split": str(split),
        "seed": int(seed),
        "category": opportunity.category,
        "opportunity_step": int(opportunity.step),
        "source_step": (
            None if opportunity.source_step is None
            else int(opportunity.source_step)
        ),
        "lag_steps": (
            None if opportunity.lag_steps is None
            else int(opportunity.lag_steps)
        ),
        "branch_window_steps": int(branch_window_steps),
        "branch_window_seconds": float(
            branch_window_steps * time_scale.dt_seconds
        ),
        "feature_names": list(keep["feature_names"]),
        "causal_features": np.asarray(
            keep["feature_vector"], dtype=np.float64
        ).tolist(),
        "feature_masks": {
            name: list(indices)
            for name, indices in keep["feature_masks"].items()
        },
        "oracle_regime_context": np.asarray(
            keep["oracle_regime_context"], dtype=np.float64
        ).tolist(),
        "keep_tracking_squared_error_integral": keep_loss,
        "renew_tracking_squared_error_integral": renew_loss,
        "renew_ise_advantage": keep_loss - renew_loss,
        "keep_return": keep_return,
        "renew_return": renew_return,
        "renew_return_advantage": renew_return - keep_return,
        "renew_waypoint_delta": float(np.linalg.norm(
            new_subgoal - old_subgoal
        )),
        "keep_requested_action_rms": float(
            keep["branch_requested_action_rms"]
        ),
        "renew_requested_action_rms": float(
            renew["branch_requested_action_rms"]
        ),
        "near_route_corner": bool(keep["near_route_corner"]),
        "regime_id": int(keep["regime_id"]),
        "prefix_max_abs_difference": prefix_difference,
        "feature_max_abs_difference": feature_difference,
        "keep_primitive_steps_replayed": int(
            keep["primitive_steps_replayed"]
        ),
        "renew_primitive_steps_replayed": int(
            renew["primitive_steps_replayed"]
        ),
        "branch_pair_extra_supervision": True,
        "branch_pair_counted_outside_controller_training_budget": True,
        "keep_upper_calls_at_opportunity": 0,
        "renew_upper_calls_at_opportunity": 1,
        "downstream_upper_call_count_per_branch": 0,
        "lower_controller_remains_closed_loop": True,
        "candidate_feature_has_future_access": False,
        "candidate_feature_has_regime_label": False,
        "oracle_regime_used_only_by_diagnostic_predictor": True,
        "protocol_valid": True,
        "regime_change_steps": diagnostics["regime_change_steps"],
        "force_pulse_start_steps": diagnostics["force_pulse_start_steps"],
        "distractor_change_steps": diagnostics[
            "distractor_change_steps"
        ],
    }
    numeric = (
        keep_loss,
        renew_loss,
        keep_return,
        renew_return,
        row["renew_ise_advantage"],
        row["renew_return_advantage"],
        row["renew_waypoint_delta"],
    )
    if not all(np.isfinite(float(value)) for value in numeric):
        raise RuntimeError("plan-validity branch row contains non-finite data")
    return row


def evaluate_plan_renewal_dataset(
    model: Any,
    *,
    seeds: Iterable[int],
    split: str,
    optimizer_seed: int,
    branch_window_seconds: float,
    max_events_per_class: int,
    env_id: str,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float,
    task_options: dict[str, Any],
) -> list[dict[str, Any]]:
    branch_window_steps = max(1, int(round(
        float(branch_window_seconds) / time_scale.dt_seconds
    )))
    rows: list[dict[str, Any]] = []
    for seed in map(int, seeds):
        task = _make_task(
            env_id=env_id, seed=seed, horizon=horizon, **task_options
        )
        try:
            opportunities = plan_renewal_opportunities(
                horizon=horizon,
                period_steps=time_scale.upper_period_steps,
                branch_window_steps=branch_window_steps,
                regime_change_steps=task.driver.regime_change_steps,
                force_pulse_steps=task.driver.pulse_start_steps,
                distractor_change_steps=task.driver.distractor_change_steps,
                seed=seed,
                max_events_per_class=max_events_per_class,
            )
        finally:
            task.environment.close()
        for opportunity in opportunities:
            rows.append(evaluate_plan_renewal_pair(
                model,
                seed=seed,
                opportunity=opportunity,
                branch_window_steps=branch_window_steps,
                env_id=env_id,
                horizon=horizon,
                time_scale=time_scale,
                maximum_subgoal_delta=maximum_subgoal_delta,
                task_options=task_options,
                optimizer_seed=optimizer_seed,
                split=split,
            ))
    return rows


def _ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = np.asarray(train_x, dtype=np.float64)
    x_eval = np.asarray(eval_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    if (
        x_train.ndim != 2
        or x_eval.ndim != 2
        or x_train.shape[0] != y.size
        or x_train.shape[1] != x_eval.shape[1]
        or y.size < 2
        or not np.all(np.isfinite(x_train))
        or not np.all(np.isfinite(x_eval))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("plan-validity ridge data are invalid")
    if not np.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("plan-validity ridge alpha must be positive")
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z_train = (x_train - mean) / scale
    z_eval = (x_eval - mean) / scale
    design = np.column_stack((np.ones(y.size), z_train))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )
    prediction = np.column_stack((
        np.ones(x_eval.shape[0]), z_eval
    )) @ weights
    return prediction, {
        "alpha": float(alpha),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights.tolist(),
    }


def _prediction_metrics(
    rows: list[dict[str, Any]],
    prediction: np.ndarray,
    *,
    selection_rate: float,
) -> dict[str, Any]:
    target = np.asarray([
        float(row["renew_ise_advantage"]) for row in rows
    ], dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != estimate.shape or target.size < 2:
        raise ValueError("plan-validity prediction shape mismatch")
    pearson = float(stats.pearsonr(target, estimate).statistic)
    spearman = float(stats.spearmanr(target, estimate).statistic)
    if not np.isfinite(pearson):
        pearson = 0.0
    if not np.isfinite(spearman):
        spearman = 0.0
    mse = float(np.mean(np.square(target - estimate)))
    variance = float(np.sum(np.square(target - np.mean(target))))
    r2 = float(
        1.0 - np.sum(np.square(target - estimate)) / variance
        if variance > 1e-15 else 0.0
    )

    grouped: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(int(row["seed"]), []).append(index)
    selected: list[int] = []
    per_seed_utility: dict[str, float] = {}
    for seed, indices in sorted(grouped.items()):
        count = max(1, int(round(float(selection_rate) * len(indices))))
        ranked = sorted(
            indices, key=lambda index: (estimate[index], -index), reverse=True
        )[:count]
        selected.extend(ranked)
        per_seed_utility[str(seed)] = float(np.mean(target[ranked]))
    categories = {
        category: float(np.mean([
            rows[index]["category"] == category for index in selected
        ]))
        for category in BRANCH_CATEGORIES
    }
    return {
        "mse": mse,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "selected_mean_renew_ise_advantage": float(np.mean(target[selected])),
        "selected_positive_rate": float(np.mean(target[selected] > 0.0)),
        "selected_count": len(selected),
        "selection_rate": float(selection_rate),
        "selected_category_fraction": categories,
        "per_seed_selected_mean_renew_ise_advantage": per_seed_utility,
    }


def fit_plan_validity_predictors(
    fit_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    ridge_alpha: float,
    selection_rate: float,
) -> dict[str, Any]:
    if not fit_rows or not eval_rows:
        raise ValueError("plan-validity predictor requires both splits")
    schema = tuple(map(str, fit_rows[0]["feature_names"]))
    masks = {
        name: tuple(map(int, indices))
        for name, indices in fit_rows[0]["feature_masks"].items()
    }
    for row in (*fit_rows, *eval_rows):
        if tuple(map(str, row["feature_names"])) != schema:
            raise ValueError("plan-validity feature schema changed")
        observed_masks = {
            name: tuple(map(int, indices))
            for name, indices in row["feature_masks"].items()
        }
        if observed_masks != masks:
            raise ValueError("plan-validity feature masks changed")

    fit_base = np.asarray([
        row["causal_features"] for row in fit_rows
    ], dtype=np.float64)
    eval_base = np.asarray([
        row["causal_features"] for row in eval_rows
    ], dtype=np.float64)
    fit_regime = np.asarray([
        row["oracle_regime_context"] for row in fit_rows
    ], dtype=np.float64)
    eval_regime = np.asarray([
        row["oracle_regime_context"] for row in eval_rows
    ], dtype=np.float64)
    target = np.asarray([
        row["renew_ise_advantage"] for row in fit_rows
    ], dtype=np.float64)

    predictor_indices = {
        "age_only": masks["age_only"],
        "plan_state": masks["plan_state"],
        "change_magnitude": masks["change_magnitude"],
        "causal_history": masks["causal_history"],
    }
    output: dict[str, Any] = {}
    for name, indices in predictor_indices.items():
        prediction, model = _ridge_fit_predict(
            fit_base[:, indices],
            target,
            eval_base[:, indices],
            alpha=ridge_alpha,
        )
        for row, value in zip(eval_rows, prediction, strict=True):
            row[f"prediction_{name}"] = float(value)
        output[name] = {
            "feature_indices": list(indices),
            "feature_names": [schema[index] for index in indices],
            "model": model,
            "evaluation": _prediction_metrics(
                eval_rows, prediction, selection_rate=selection_rate
            ),
        }

    fit_oracle = np.column_stack((fit_base, fit_regime))
    eval_oracle = np.column_stack((eval_base, eval_regime))
    prediction, model = _ridge_fit_predict(
        fit_oracle,
        target,
        eval_oracle,
        alpha=ridge_alpha,
    )
    for row, value in zip(eval_rows, prediction, strict=True):
        row["prediction_causal_history_plus_regime"] = float(value)
    output["causal_history_plus_regime"] = {
        "feature_indices": list(range(fit_oracle.shape[1])),
        "feature_names": [
            *schema,
            *(f"oracle_regime_{index}" for index in range(fit_regime.shape[1])),
        ],
        "model": model,
        "evaluation": _prediction_metrics(
            eval_rows, prediction, selection_rate=selection_rate
        ),
    }
    return {
        "fit_row_count": len(fit_rows),
        "evaluation_row_count": len(eval_rows),
        "target": "renew_ise_advantage_keep_minus_renew",
        "selection_utility_is_local_counterfactual_not_closed_loop_return": True,
        "predictors": output,
    }


def _validate_branch_seed_roles(
    *,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    branch_fit_seeds: Iterable[int],
    branch_eval_seeds: Iterable[int],
) -> tuple[tuple[int, ...], ...]:
    roles = tuple(
        tuple(map(int, values))
        for values in (
            train_seeds,
            selection_seeds,
            branch_fit_seeds,
            branch_eval_seeds,
        )
    )
    if any(not role or len(set(role)) != len(role) for role in roles):
        raise ValueError("plan-validity seed roles must be nonempty and unique")
    for left in range(len(roles)):
        for right in range(left + 1, len(roles)):
            if set(roles[left]) & set(roles[right]):
                raise ValueError("plan-validity seed roles must be disjoint")
    return roles


def train_pointmaze_plan_validity_cell(
    *,
    env_id: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    branch_fit_seeds: Iterable[int],
    branch_eval_seeds: Iterable[int],
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
    ridge_alpha: float,
    selection_rate: float,
    task_options: dict[str, Any],
) -> dict[str, Any]:
    training, selection, branch_fit, branch_eval = (
        _validate_branch_seed_roles(
            train_seeds=train_seeds,
            selection_seeds=selection_seeds,
            branch_fit_seeds=branch_fit_seeds,
            branch_eval_seeds=branch_eval_seeds,
        )
    )
    controller_eval = (*branch_fit, *branch_eval)
    payload, model = train_pointmaze_plan_value_cell(
        method=POINTMAZE_PLAN_VALIDITY_POLICY,
        env_id=env_id,
        train_seeds=training,
        selection_seeds=selection,
        eval_seeds=controller_eval,
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
    )
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=float(payload["time_scale"]["env_dt_seconds"]),
        upper_period_seconds=upper_period_seconds,
        history_seconds=history_seconds,
        fast_period_seconds=fast_period_seconds,
    )
    fit_rows = evaluate_plan_renewal_dataset(
        model,
        seeds=branch_fit,
        split="predictor_fit",
        optimizer_seed=optimizer_seed,
        branch_window_seconds=branch_window_seconds,
        max_events_per_class=max_events_per_class,
        env_id=env_id,
        horizon=horizon,
        time_scale=time_scale,
        maximum_subgoal_delta=maximum_subgoal_delta,
        task_options=task_options,
    )
    eval_rows = evaluate_plan_renewal_dataset(
        model,
        seeds=branch_eval,
        split="qualification_eval",
        optimizer_seed=optimizer_seed,
        branch_window_seconds=branch_window_seconds,
        max_events_per_class=max_events_per_class,
        env_id=env_id,
        horizon=horizon,
        time_scale=time_scale,
        maximum_subgoal_delta=maximum_subgoal_delta,
        task_options=task_options,
    )
    predictor = fit_plan_validity_predictors(
        fit_rows,
        eval_rows,
        ridge_alpha=ridge_alpha,
        selection_rate=selection_rate,
    )
    feature_names = list(fit_rows[0]["feature_names"])
    feature_masks = dict(fit_rows[0]["feature_masks"])
    path_manifest: dict[str, dict[str, Any]] = {}
    for row in (*fit_rows, *eval_rows):
        seed_key = str(int(row["seed"]))
        path = {
            "regime_change_steps": row.pop("regime_change_steps"),
            "force_pulse_start_steps": row.pop("force_pulse_start_steps"),
            "distractor_change_steps": row.pop("distractor_change_steps"),
        }
        previous = path_manifest.setdefault(seed_key, path)
        if previous != path:
            raise RuntimeError("plan-validity path manifest changed within seed")
        if row.pop("feature_names") != feature_names:
            raise RuntimeError("plan-validity compact feature schema changed")
        if row.pop("feature_masks") != feature_masks:
            raise RuntimeError("plan-validity compact feature masks changed")

    payload["source_controller_protocol_version"] = payload[
        "protocol_version"
    ]
    payload["source_controller_algorithm_path"] = payload["algorithm_path"]
    payload["protocol_version"] = POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION
    payload["algorithm_path"] = POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH
    payload["evidence_role"] = "counterfactual_plan_validity_qualification"
    payload["controller_training_schedule"] = payload.pop(
        "training_schedule"
    )
    payload["controller_evaluation_rows"] = payload.pop("evaluation_rows")
    payload["branch_fit_seeds"] = list(branch_fit)
    payload["branch_eval_seeds"] = list(branch_eval)
    payload["branch_fit_rows"] = fit_rows
    payload["branch_evaluation_rows"] = eval_rows
    payload["branch_feature_names"] = feature_names
    payload["branch_feature_masks"] = feature_masks
    payload["branch_path_manifest"] = path_manifest
    payload["plan_validity_predictor"] = predictor
    payload["branch_window_seconds"] = float(branch_window_seconds)
    payload["max_events_per_class"] = int(max_events_per_class)
    payload["paired_branch_transition_budget"] = {
        "fit_primitive_steps_replayed": int(sum(
            int(row["keep_primitive_steps_replayed"])
            + int(row["renew_primitive_steps_replayed"])
            for row in fit_rows
        )),
        "evaluation_primitive_steps_replayed": int(sum(
            int(row["keep_primitive_steps_replayed"])
            + int(row["renew_primitive_steps_replayed"])
            for row in eval_rows
        )),
        "counted_as_extra_supervision": True,
    }
    payload["belief_training"] = "disabled"
    payload["trigger_training"] = "disabled_qualification_only"
    payload["plan_validity_predictor_deployment"] = "disabled"
    return payload


def resolved_plan_validity_protocol(
    *,
    args: argparse.Namespace,
    task_options: dict[str, Any],
) -> dict[str, Any]:
    training, selection, branch_fit, branch_eval = (
        _validate_branch_seed_roles(
            train_seeds=args.train_seeds,
            selection_seeds=args.selection_seeds,
            branch_fit_seeds=args.branch_fit_seeds,
            branch_eval_seeds=args.branch_eval_seeds,
        )
    )
    return {
        "protocol_version": POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH,
        "evidence_role": "counterfactual_plan_validity_qualification",
        "policy": POINTMAZE_PLAN_VALIDITY_POLICY,
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
        "ridge_alpha": float(args.ridge_alpha),
        "selection_rate": float(args.selection_rate),
        "train_seeds": list(training),
        "selection_seeds": list(selection),
        "branch_fit_seeds": list(branch_fit),
        "branch_eval_seeds": list(branch_eval),
        "branch_categories": list(BRANCH_CATEGORIES),
        "predictors": list(PREDICTOR_NAMES),
        "primary_endpoint": "renew_ise_advantage_keep_minus_renew",
        "task_options": _json_ready(task_options),
        "future_access": "none_for_candidate_features",
        "oracle_regime_role": "diagnostic_predictor_only",
        "branch_supervision_budget": "reported_separately_from_control_training",
        "disabled": [
            "deployed_trigger",
            "joint_controller_predictor_training",
            "future_event_input_to_candidate",
        ],
    }


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
    parser.add_argument(
        "--checkpoint-evaluation-interval", type=int, default=8
    )
    parser.add_argument("--branch-window-seconds", type=float, default=0.50)
    parser.add_argument("--max-events-per-class", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--selection-rate", type=float, default=0.25)
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
    parser.add_argument("--train-seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--selection-seeds", nargs="+", type=int, required=True
    )
    parser.add_argument(
        "--branch-fit-seeds", nargs="+", type=int, required=True
    )
    parser.add_argument(
        "--branch-eval-seeds", nargs="+", type=int, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _task_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "regime_dwell_seconds": tuple(map(
            float, args.regime_dwell_seconds
        )),
        "target_speed_modes": tuple(map(float, args.target_speed_modes)),
        "force_pulse_amplitude": float(args.force_pulse_amplitude),
        "force_pulse_duration_seconds": tuple(map(
            float, args.force_pulse_duration_seconds
        )),
        "force_pulse_gap_seconds": tuple(map(
            float, args.force_pulse_gap_seconds
        )),
        "distractor_amplitude": float(args.distractor_amplitude),
        "distractor_dwell_seconds": tuple(map(
            float, args.distractor_dwell_seconds
        )),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < float(args.selection_rate) <= 1.0:
        raise ValueError("selection rate must be in (0, 1]")
    task_options = _task_options(args)
    protocol = resolved_plan_validity_protocol(
        args=args, task_options=task_options
    )
    output: dict[str, Any] = {
        "protocol": protocol,
        "status": "dry_run" if args.dry_run else "complete",
        "cells": [],
    }
    if not args.dry_run:
        output["cells"].append(train_pointmaze_plan_validity_cell(
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
            checkpoint_evaluation_interval=(
                args.checkpoint_evaluation_interval
            ),
            branch_window_seconds=args.branch_window_seconds,
            max_events_per_class=args.max_events_per_class,
            ridge_alpha=args.ridge_alpha,
            selection_rate=args.selection_rate,
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
