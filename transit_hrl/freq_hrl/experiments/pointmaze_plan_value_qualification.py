"""Stage-8 qualification of plan dependence and oracle replan value.

This module does not implement a learned trigger.  It asks whether the task and
the existing goal-conditioned hierarchy contain enough plan value to justify
learning one.  Oracle schedules use hidden switch times only as a privileged
reference and always preserve the fixed scheduler's call count.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import (
    POINTMAZE_REGIME_SPEEDS,
    PointMazeRegimeObservation,
    PointMazeRegimeTask,
    RelativeSubgoalAdapter,
    goal_environment_contract,
)
from freq_hrl.rl import (
    GoalConditionedActorCriticPPO,
    GoalConditionedPPOConfig,
    HierarchicalRolloutBuilder,
    matched_hierarchical_hidden_dim,
    summarize_numeric_rows,
    train_frequency_separated_ppo,
)

from .pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    POINTMAZE_LOWER_ACTION_COST,
    _json_ready,
    _training_seed,
    _validate_seed_roles,
    make_pointmaze_environment,
    pointmaze_goal_bounds,
    pointmaze_runtime_versions,
    squash_box_action,
)


POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION = (
    "pointmaze_plan_value_qualification_stage8_v1"
)
POINTMAZE_PLAN_VALUE_ALGORITHM_PATH = (
    "belief_replanning_task_qualification_not_learned_trigger"
)
POINTMAZE_PLAN_VALUE_METHODS = (
    "hrl_regime_history",
    "hrl_regime_oracle_context",
)
POINTMAZE_PLAN_VALUE_SCHEDULES = (
    "fixed",
    "stale_plan",
    "fixed_waypoint_perturbed",
    "oracle_event_delay_000ms",
    "oracle_event_delay_100ms",
    "oracle_event_delay_250ms",
    "oracle_event_delay_500ms",
)
ORACLE_EVENT_DELAY_SECONDS = {
    "oracle_event_delay_000ms": 0.0,
    "oracle_event_delay_100ms": 0.10,
    "oracle_event_delay_250ms": 0.25,
    "oracle_event_delay_500ms": 0.50,
}
DEFAULT_REGIME_DWELL_SECONDS = (0.80, 1.60)
DEFAULT_FORCE_PULSE_AMPLITUDE = 0.18
DEFAULT_FORCE_PULSE_DURATION_SECONDS = (0.04, 0.10)
DEFAULT_FORCE_PULSE_GAP_SECONDS = (0.45, 1.10)
DEFAULT_DISTRACTOR_AMPLITUDE = 0.50
DEFAULT_DISTRACTOR_DWELL_SECONDS = (0.35, 0.85)
DEFAULT_WAYPOINT_PERTURBATION = 0.25
DEFAULT_EVENT_WINDOW_SECONDS = 1.00


@dataclass(frozen=True)
class PointMazePlanValueDimensions:
    physical: int
    goal: int
    task: int
    history: int
    oracle_context: int
    base_upper: int
    oracle_upper: int
    lower: int
    action: int


class PointMazeRegimeFeatureBuilder:
    """Causal raw-history features with optional oracle upper context."""

    def __init__(
        self,
        *,
        time_scale: PhysicalTimeScaleContract,
        task_dim: int = 6,
    ) -> None:
        if int(task_dim) < 1:
            raise ValueError("task_dim must be positive")
        self.time_scale = time_scale
        self.task_dim = int(task_dim)
        self._history: np.ndarray | None = None

    @property
    def history(self) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("regime feature builder must be reset")
        return self._history.astype(np.float32, copy=True).reshape(-1)

    def reset(self, observation: PointMazeRegimeObservation) -> None:
        sample = self._sample(observation)
        self._history = np.repeat(
            sample.reshape(1, -1), self.time_scale.history_steps, axis=0
        )

    def update(self, observation: PointMazeRegimeObservation) -> None:
        sample = self._sample(observation)
        if self._history is None:
            self.reset(observation)
            return
        self._history[:-1] = self._history[1:]
        self._history[-1] = sample

    def _sample(self, observation: PointMazeRegimeObservation) -> np.ndarray:
        sample = np.asarray(
            observation.task_measurement, dtype=np.float32
        ).reshape(-1)
        if sample.shape != (self.task_dim,) or not np.all(np.isfinite(sample)):
            raise ValueError("regime task measurement shape changed")
        return sample

    def upper_state(
        self,
        observation: PointMazeRegimeObservation,
        *,
        oracle_context: np.ndarray | None,
    ) -> np.ndarray:
        pieces = [
            observation.physical,
            observation.target_error,
            self.history,
        ]
        if oracle_context is not None:
            context = np.asarray(oracle_context, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(context)):
                raise ValueError("oracle context must be finite")
            pieces.append(context)
        return np.concatenate(pieces).astype(np.float32, copy=False)

    def lower_state(
        self,
        observation: PointMazeRegimeObservation,
        *,
        subgoal: np.ndarray,
    ) -> np.ndarray:
        waypoint_error = (
            np.asarray(subgoal, dtype=np.float32).reshape(-1)
            - observation.achieved_goal
        )
        if waypoint_error.shape != observation.achieved_goal.shape:
            raise ValueError("regime PointMaze subgoal dimension mismatch")
        return np.concatenate((
            observation.physical,
            waypoint_error,
            self.history,
        )).astype(np.float32, copy=False)


def fixed_replan_steps(*, horizon: int, period_steps: int) -> tuple[int, ...]:
    if int(horizon) < 1 or int(period_steps) < 1:
        raise ValueError("horizon and period_steps must be positive")
    return tuple(range(0, int(horizon), int(period_steps)))


def relocate_replan_steps(
    *,
    fixed_steps: Iterable[int],
    event_steps: Iterable[int],
    delay_steps: int,
    horizon: int,
) -> tuple[int, ...]:
    """Move nearest periodic calls to delayed events without changing budget."""

    if int(delay_steps) < 0:
        raise ValueError("event delay cannot be negative")
    fixed = sorted(set(map(int, fixed_steps)))
    if not fixed or fixed[0] != 0:
        raise ValueError("fixed schedule must contain the initial decision")
    if fixed[-1] >= int(horizon):
        raise ValueError("fixed schedule extends beyond the episode")
    schedule = set(fixed)
    protected = {0}
    for event in sorted(set(map(int, event_steps))):
        target = int(event) + int(delay_steps)
        if target <= 0 or target >= int(horizon):
            continue
        if target in schedule:
            protected.add(target)
            continue
        removable = [step for step in schedule if step not in protected]
        if not removable:
            raise ValueError("event schedule exceeds the fixed planning budget")
        removed = min(
            removable,
            key=lambda step: (abs(step - target), step < target, step),
        )
        schedule.remove(removed)
        schedule.add(target)
        protected.add(target)
    result = tuple(sorted(schedule))
    if len(result) != len(fixed) or result[0] != 0:
        raise RuntimeError("oracle schedule changed the planning budget")
    return result


def schedule_for_task(
    task: PointMazeRegimeTask,
    *,
    time_scale: PhysicalTimeScaleContract,
    schedule_mode: str,
) -> tuple[int, ...]:
    mode = str(schedule_mode)
    if mode not in POINTMAZE_PLAN_VALUE_SCHEDULES:
        raise ValueError(f"unknown plan-value schedule: {mode}")
    fixed = fixed_replan_steps(
        horizon=task.horizon,
        period_steps=time_scale.upper_period_steps,
    )
    if mode == "stale_plan":
        return (0,)
    if mode in ("fixed", "fixed_waypoint_perturbed"):
        return fixed
    delay_steps = int(round(
        ORACLE_EVENT_DELAY_SECONDS[mode] / time_scale.dt_seconds
    ))
    return relocate_replan_steps(
        fixed_steps=fixed,
        event_steps=task.driver.regime_change_steps,
        delay_steps=delay_steps,
        horizon=task.horizon,
    )


def _make_task(
    *,
    env_id: str,
    seed: int,
    horizon: int,
    regime_dwell_seconds: tuple[float, float],
    target_speed_modes: tuple[float, ...],
    force_pulse_amplitude: float,
    force_pulse_duration_seconds: tuple[float, float],
    force_pulse_gap_seconds: tuple[float, float],
    distractor_amplitude: float,
    distractor_dwell_seconds: tuple[float, float],
) -> PointMazeRegimeTask:
    environment = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    return PointMazeRegimeTask(
        environment,
        seed=seed,
        horizon=horizon,
        regime_dwell_seconds=regime_dwell_seconds,
        target_speed_modes=target_speed_modes,
        force_pulse_amplitude=force_pulse_amplitude,
        force_pulse_duration_seconds=force_pulse_duration_seconds,
        force_pulse_gap_seconds=force_pulse_gap_seconds,
        distractor_amplitude=distractor_amplitude,
        distractor_dwell_seconds=distractor_dwell_seconds,
    )


def pointmaze_plan_value_dimensions(
    *,
    env_id: str,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    task_options: dict[str, Any],
) -> PointMazePlanValueDimensions:
    task = _make_task(
        env_id=env_id, seed=0, horizon=horizon, **task_options
    )
    try:
        observation = task.reset()
        features = PointMazeRegimeFeatureBuilder(
            time_scale=time_scale,
            task_dim=int(observation.task_measurement.size),
        )
        features.reset(observation)
        physical = int(observation.physical.size)
        goal = int(observation.achieved_goal.size)
        task_dim = int(observation.task_measurement.size)
        history = int(features.history.size)
        oracle_context = int(task.privileged_context().size)
        shared = physical + goal + history
        return PointMazePlanValueDimensions(
            physical=physical,
            goal=goal,
            task=task_dim,
            history=history,
            oracle_context=oracle_context,
            base_upper=shared,
            oracle_upper=shared + oracle_context,
            lower=shared,
            action=int(task.action_low.size),
        )
    finally:
        task.environment.close()


def build_pointmaze_plan_value_model(
    *,
    method: str,
    dimensions: PointMazePlanValueDimensions,
    reference_hidden_dim: int,
    learning_rate: float,
    optimizer_seed: int,
) -> tuple[GoalConditionedActorCriticPPO, dict[str, Any]]:
    name = str(method)
    if name not in POINTMAZE_PLAN_VALUE_METHODS:
        raise ValueError(f"unknown plan-value method: {name}")

    def make(*, upper_state_dim: int, hidden_dim: int):
        return GoalConditionedActorCriticPPO(GoalConditionedPPOConfig(
            upper_state_dim=int(upper_state_dim),
            lower_state_dim=int(dimensions.lower),
            goal_dim=int(dimensions.goal),
            action_dim=int(dimensions.action),
            hidden_dim=int(hidden_dim),
            learning_rate=float(learning_rate),
            epochs=4,
            minibatch_size=1024,
            init_log_std=-0.7,
        ))

    torch.manual_seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed) % (2**32 - 1))
    reference = make(
        upper_state_dim=dimensions.base_upper,
        hidden_dim=int(reference_hidden_dim),
    )
    parameter_budget = int(reference.trainable_parameter_count)
    if name == "hrl_regime_history":
        return reference, {
            "reference_parameter_budget": parameter_budget,
            "actual_parameter_count": parameter_budget,
            "parameter_budget_ratio": 1.0,
            "hidden_dim": int(reference_hidden_dim),
            "oracle_context_visible_to_upper": False,
            **reference.mainline_contract(),
        }

    hidden_dim, expected, ratio = matched_hierarchical_hidden_dim(
        target_parameter_count=parameter_budget,
        upper_state_dim=dimensions.oracle_upper,
        lower_state_dim=dimensions.lower,
        goal_dim=dimensions.goal,
        action_dim=dimensions.action,
    )
    torch.manual_seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed) % (2**32 - 1))
    model = make(
        upper_state_dim=dimensions.oracle_upper,
        hidden_dim=hidden_dim,
    )
    if int(model.trainable_parameter_count) != int(expected):
        raise RuntimeError("plan-value parameter accounting changed")
    return model, {
        "reference_parameter_budget": parameter_budget,
        "actual_parameter_count": int(model.trainable_parameter_count),
        "parameter_budget_ratio": float(ratio),
        "hidden_dim": int(hidden_dim),
        "oracle_context_visible_to_upper": True,
        **model.mainline_contract(),
    }


def _event_diagnostics(
    *,
    distances: np.ndarray,
    successes: np.ndarray,
    event_steps: Iterable[int],
    decision_steps: tuple[int, ...],
    dt_seconds: float,
    window_seconds: float,
) -> dict[str, Any]:
    window_steps = max(1, int(round(float(window_seconds) / dt_seconds)))
    valid_events = [
        int(step)
        for step in event_steps
        if 0 <= int(step) < int(distances.size)
    ]
    aligned: list[list[float]] = [[] for _ in range(window_steps)]
    recovery: list[float] = []
    recovery_censored = 0
    replan_delay: list[float] = []
    for event in valid_events:
        for offset in range(window_steps):
            index = event + offset
            if index < distances.size:
                aligned[offset].append(float(distances[index] ** 2))
        success_offsets = np.flatnonzero(
            successes[event:min(distances.size, event + window_steps)] > 0.5
        )
        if success_offsets.size:
            recovery.append(float(success_offsets[0] * dt_seconds))
        else:
            recovery.append(float(window_steps * dt_seconds))
            recovery_censored += 1
        following = [step for step in decision_steps if step >= event]
        if following:
            replan_delay.append(float((following[0] - event) * dt_seconds))
        else:
            replan_delay.append(float((distances.size - event) * dt_seconds))
    aligned_mean = [
        float(np.mean(values)) if values else 0.0 for values in aligned
    ]
    aligned_count = [len(values) for values in aligned]
    post_values = [value for values in aligned for value in values]
    return {
        "scored_regime_change_count": len(valid_events),
        "event_window_seconds": float(window_steps * dt_seconds),
        "event_post_tracking_mse": (
            float(np.mean(post_values)) if post_values else 0.0
        ),
        "event_recovery_seconds_mean": (
            float(np.mean(recovery)) if recovery else 0.0
        ),
        "event_recovery_censored_rate": (
            float(recovery_censored / len(recovery)) if recovery else 0.0
        ),
        "event_to_replan_delay_seconds_mean": (
            float(np.mean(replan_delay)) if replan_delay else 0.0
        ),
        "event_aligned_tracking_mse": aligned_mean,
        "event_aligned_sample_count": aligned_count,
    }


def _causal_distinguishability_diagnostics(
    task: PointMazeRegimeTask,
    *,
    comparison_steps: int = 5,
    velocity_change_threshold: float = 0.20,
) -> dict[str, float]:
    """Measure when target observations first reveal a registered switch.

    This is an event-conditioned identifiability witness, not a deployable
    change detector: true event times select the windows, while every velocity
    used inside a window is computed only after its endpoint is observed.
    """

    if int(comparison_steps) < 1:
        raise ValueError("comparison_steps must be positive")
    if (
        not np.isfinite(float(velocity_change_threshold))
        or float(velocity_change_threshold) <= 0.0
    ):
        raise ValueError("velocity_change_threshold must be positive")
    targets = np.asarray([
        task.driver.sample(step)[0]
        for step in range(task.horizon + 1)
    ], dtype=np.float64)
    velocity = np.diff(targets, axis=0) / float(task.dt_seconds)
    delays: list[float] = []
    censored = 0
    for event in task.driver.regime_change_steps:
        if event <= 0 or event >= task.horizon:
            continue
        pre = velocity[max(0, event - int(comparison_steps)):event]
        if pre.size == 0:
            continue
        reference = np.median(pre, axis=0)
        stop = min(task.horizon, event + int(comparison_steps))
        detected = None
        for action_step in range(event, stop):
            if float(np.linalg.norm(velocity[action_step] - reference)) > float(
                velocity_change_threshold
            ):
                # velocity[action_step] becomes available with target[t + 1].
                detected = float(
                    (action_step - event + 1) * task.dt_seconds
                )
                break
        if detected is None:
            detected = float((stop - event) * task.dt_seconds)
            censored += 1
        delays.append(detected)
    return {
        "causal_distinguishability_delay_seconds_mean": (
            float(np.mean(delays)) if delays else 0.0
        ),
        "causal_distinguishability_delay_seconds_max": (
            float(np.max(delays)) if delays else 0.0
        ),
        "causal_distinguishability_censored_rate": (
            float(censored / len(delays)) if delays else 0.0
        ),
        "causal_distinguishability_event_count": len(delays),
        "causal_distinguishability_is_privileged_windowed_diagnostic": True,
    }


def rollout_hrl_pointmaze_plan_value(
    model: GoalConditionedActorCriticPPO,
    *,
    method: str,
    schedule_mode: str,
    env_id: str,
    seed: int,
    horizon: int,
    sample: bool,
    parameter_budget: int,
    time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float,
    waypoint_perturbation: float,
    event_window_seconds: float,
    task_options: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    name = str(method)
    if name not in POINTMAZE_PLAN_VALUE_METHODS:
        raise ValueError(f"unknown plan-value method: {name}")
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
        subgoal_adapter = RelativeSubgoalAdapter(
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
        builder = HierarchicalRolloutBuilder(gamma=float(model.config.gamma))
        schedule = schedule_for_task(
            task, time_scale=time_scale, schedule_mode=schedule_mode
        )
        decision_set = set(schedule)
        fixed_budget = len(fixed_replan_steps(
            horizon=horizon, period_steps=time_scale.upper_period_steps
        ))
        perturb_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 8_170_031])
        )

        rewards: list[float] = []
        distances: list[float] = []
        successes: list[float] = []
        requested_actions: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        intrinsic_rewards: list[float] = []
        subgoal_distances: list[float] = []
        actual_decision_steps: list[int] = []
        planner_wall_seconds = 0.0
        lower_wall_seconds = 0.0
        terminated = truncated = False
        achieved_before = observation.achieved_goal.copy()
        subgoal = achieved_before.copy()
        oracle_context_visible = name == "hrl_regime_oracle_context"

        for step in range(int(horizon)):
            if step in decision_set:
                context = (
                    task.privileged_context()
                    if oracle_context_visible else None
                )
                upper_state = features.upper_state(
                    observation, oracle_context=context
                )
                started = time.perf_counter()
                upper_output = model.plan_goal(upper_state, sample=sample)
                planner_wall_seconds += time.perf_counter() - started
                raw_goal = np.asarray(
                    upper_output["action"], dtype=np.float32
                ).reshape(-1)
                subgoal = subgoal_adapter.decode(raw_goal, achieved_before)
                if schedule_mode == "fixed_waypoint_perturbed":
                    direction = perturb_rng.normal(size=subgoal.size)
                    norm = float(np.linalg.norm(direction))
                    if norm <= 1e-12:
                        direction[0] = 1.0
                        norm = 1.0
                    delta = (
                        direction / norm * float(waypoint_perturbation)
                    ).astype(np.float32)
                    local_low = np.maximum(
                        world_low, achieved_before - maximum_delta
                    )
                    local_high = np.minimum(
                        world_high, achieved_before + maximum_delta
                    )
                    subgoal = np.clip(
                        subgoal + delta, local_low, local_high
                    ).astype(np.float32)
                builder.begin_upper(
                    state=upper_state,
                    action=raw_goal,
                    logp=float(upper_output["logp"]),
                    value=float(upper_output["value"]),
                )
                actual_decision_steps.append(int(step))

            lower_state = features.lower_state(
                observation, subgoal=subgoal
            )
            started = time.perf_counter()
            lower_output = model.act_conditioned(lower_state, sample=sample)
            lower_wall_seconds += time.perf_counter() - started
            raw_action = np.asarray(
                lower_output["action"], dtype=np.float32
            ).reshape(-1)
            requested = squash_box_action(
                raw_action, task.action_low, task.action_high
            )
            next_observation, task_reward, terminated, truncated, info = (
                task.step(requested)
            )
            done = bool(terminated or truncated)
            subgoal_distance = float(np.linalg.norm(
                (subgoal - next_observation.achieved_goal) / maximum_delta
            ) / np.sqrt(observation.achieved_goal.size))
            intrinsic_reward = subgoal_adapter.intrinsic_reward(
                achieved_before=achieved_before,
                achieved_after=next_observation.achieved_goal,
                subgoal=subgoal,
                action=requested,
            )
            option_boundary = (step + 1) in decision_set
            lower_terminal = bool(done or option_boundary)
            builder.add_lower(
                state=lower_state,
                action=raw_action,
                logp=float(lower_output["logp"]),
                value=float(lower_output["value"]),
                reward=float(intrinsic_reward),
                upper_reward=float(task_reward),
                done=done,
                lower_done=lower_terminal,
                cost=0.0,
            )
            rewards.append(float(task_reward))
            distances.append(float(info["tracking_distance"]))
            successes.append(float(info["tracking_success"]))
            requested_actions.append(requested)
            executed_actions.append(np.asarray(info["executed_action"]))
            intrinsic_rewards.append(float(intrinsic_reward))
            subgoal_distances.append(subgoal_distance)
            achieved_before = next_observation.achieved_goal.copy()
            observation = next_observation
            features.update(observation)
            if done:
                break

        builder.finish(terminal=True)
        batch = builder.build() if sample else None
        reward = np.asarray(rewards, dtype=np.float64)
        distance = np.asarray(distances, dtype=np.float64)
        success = np.asarray(successes, dtype=np.float64)
        requested_array = np.asarray(requested_actions, dtype=np.float64)
        executed_array = np.asarray(executed_actions, dtype=np.float64)
        intrinsic = np.asarray(intrinsic_rewards, dtype=np.float64)
        subgoal_distance_array = np.asarray(
            subgoal_distances, dtype=np.float64
        )
        decision_steps = tuple(actual_decision_steps)
        option_edges = (*decision_steps, int(reward.size))
        option_durations = np.diff(np.asarray(option_edges, dtype=np.int64))
        budget_matched = (
            len(decision_steps) == fixed_budget
            if schedule_mode != "stale_plan" else False
        )
        protocol_valid = bool(
            reward.size == int(horizon)
            and reward.size
            == distance.size
            == success.size
            == requested_array.shape[0]
            == executed_array.shape[0]
            == intrinsic.size
            == subgoal_distance_array.size
            and decision_steps == schedule
            and option_durations.size == len(decision_steps)
            and int(np.sum(option_durations)) == int(reward.size)
            and np.all(option_durations > 0)
            and np.all(np.isfinite(reward))
            and np.all(np.isfinite(distance))
            and (
                batch is None
                or (
                    int(np.sum(batch.upper.duration)) == int(reward.size)
                    and batch.lower.size == int(reward.size)
                )
            )
        )
        event = _event_diagnostics(
            distances=distance,
            successes=success,
            event_steps=task.driver.regime_change_steps,
            decision_steps=decision_steps,
            dt_seconds=time_scale.dt_seconds,
            window_seconds=event_window_seconds,
        )
        distinguishability = _causal_distinguishability_diagnostics(task)
        oracle_schedule = schedule_mode.startswith("oracle_event_")
        row = {
            "protocol_version": POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION,
            "algorithm_path": POINTMAZE_PLAN_VALUE_ALGORITHM_PATH,
            "method": name,
            "schedule_mode": str(schedule_mode),
            "seed": int(seed),
            "episode_return": float(np.sum(reward)),
            "tracking_success_rate": float(np.mean(success)),
            "tracking_rmse": float(np.sqrt(np.mean(np.square(distance)))),
            "tracking_mae": float(np.mean(distance)),
            "tracking_squared_error_integral": float(
                np.sum(np.square(distance)) * time_scale.dt_seconds
            ),
            "final_tracking_distance": float(distance[-1]),
            "maximum_tracking_distance": float(np.max(distance)),
            "subgoal_tracking_rmse": float(np.sqrt(
                np.mean(np.square(subgoal_distance_array))
            )),
            "lower_intrinsic_return": float(np.sum(intrinsic)),
            "requested_action_rms": float(np.sqrt(
                np.mean(np.square(requested_array))
            )),
            "executed_action_rms": float(np.sqrt(
                np.mean(np.square(executed_array))
            )),
            "episode_length": int(reward.size),
            "upper_decision_count": len(decision_steps),
            "fixed_budget_upper_decision_count": int(fixed_budget),
            "upper_calls_per_second": float(
                len(decision_steps)
                / (reward.size * time_scale.dt_seconds)
            ),
            "planning_budget_matched": float(budget_matched),
            "decision_steps": list(decision_steps),
            "option_duration_steps_min": int(np.min(option_durations)),
            "option_duration_steps_max": int(np.max(option_durations)),
            "option_duration_steps_mean": float(np.mean(option_durations)),
            "option_duration_steps_sum": int(np.sum(option_durations)),
            "planner_wall_seconds": float(planner_wall_seconds),
            "lower_wall_seconds": float(lower_wall_seconds),
            "policy_has_current_regime_access": bool(
                oracle_context_visible
            ),
            "policy_has_future_regime_access": False,
            "schedule_has_regime_event_access": bool(oracle_schedule),
            "schedule_has_future_regime_access": bool(oracle_schedule),
            "event_delay_seconds": (
                float(ORACLE_EVENT_DELAY_SECONDS[schedule_mode])
                if oracle_schedule else None
            ),
            "waypoint_perturbation": (
                float(waypoint_perturbation)
                if schedule_mode == "fixed_waypoint_perturbed" else 0.0
            ),
            "trainable_parameter_count": int(
                model.trainable_parameter_count
            ),
            "parameter_budget": int(parameter_budget),
            "parameter_budget_ratio": float(
                model.trainable_parameter_count / parameter_budget
            ),
            "protocol_valid": float(protocol_valid),
            "terminated": float(bool(terminated)),
            "truncated": float(bool(truncated)),
            **event,
            **distinguishability,
            **time_scale.metadata(),
            **task.diagnostics(),
        }
        return batch, row
    finally:
        task.environment.close()


def _compact_history(
    history: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    retained = {
        "iteration",
        "train_objective",
        "validation_score",
        "validation_score_raw",
        "checkpoint_selected",
        "checkpoint_eligible",
        "mean_episode_return",
        "mean_tracking_success_rate",
        "mean_tracking_rmse",
        "mean_tracking_squared_error_integral",
        "upper_loss",
        "lower_loss",
    }
    return [
        {key: value for key, value in row.items() if key in retained}
        for row in history
    ]


def pointmaze_plan_value_checkpoint_rank(
    rows: list[dict[str, Any]],
) -> tuple[float, float]:
    if not rows:
        raise ValueError("plan-value checkpoint requires rows")
    tracking_loss = float(np.mean([
        float(row["tracking_squared_error_integral"]) for row in rows
    ]))
    episode_return = float(np.mean([
        float(row["episode_return"]) for row in rows
    ]))
    if not np.isfinite(tracking_loss) or not np.isfinite(episode_return):
        raise ValueError("plan-value checkpoint rank must be finite")
    return -tracking_loss, episode_return


def train_pointmaze_plan_value_cell(
    *,
    method: str,
    env_id: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
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
    waypoint_perturbation: float,
    event_window_seconds: float,
    task_options: dict[str, Any],
) -> tuple[dict[str, Any], GoalConditionedActorCriticPPO]:
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    probe = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        environment_contract = goal_environment_contract(probe)
        world_low, world_high = pointmaze_goal_bounds(probe)
    finally:
        probe.close()
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=float(environment_contract["env_dt_seconds"]),
        upper_period_seconds=float(upper_period_seconds),
        history_seconds=float(history_seconds),
        fast_period_seconds=float(fast_period_seconds),
    )
    if time_scale.upper_period_steps >= int(horizon):
        raise ValueError("plan-value upper period must be below horizon")
    dimensions = pointmaze_plan_value_dimensions(
        env_id=env_id,
        horizon=horizon,
        time_scale=time_scale,
        task_options=task_options,
    )
    model, capacity = build_pointmaze_plan_value_model(
        method=method,
        dimensions=dimensions,
        reference_hidden_dim=reference_hidden_dim,
        learning_rate=learning_rate,
        optimizer_seed=optimizer_seed,
    )
    parameter_budget = int(capacity["reference_parameter_budget"])
    common_rollout = {
        "method": str(method),
        "env_id": str(env_id),
        "horizon": int(horizon),
        "parameter_budget": parameter_budget,
        "time_scale": time_scale,
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "waypoint_perturbation": float(waypoint_perturbation),
        "event_window_seconds": float(event_window_seconds),
        "task_options": dict(task_options),
    }
    canonical_rollout = lambda policy, seed, sample: (
        rollout_hrl_pointmaze_plan_value(
            policy,
            seed=seed,
            sample=sample,
            schedule_mode="fixed",
            **common_rollout,
        )
    )
    untrained_rows = [
        canonical_rollout(model, int(seed), False)[1]
        for seed in evaluation
    ]
    for row in untrained_rows:
        row["training_replicate_seed"] = int(optimizer_seed)
    metadata = {
        "protocol_version": POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_PLAN_VALUE_ALGORITHM_PATH,
        "environment_id": str(env_id),
        "optimizer_seed": int(optimizer_seed),
        "evidence_role": "task_qualification_development",
        "primary_endpoint": "tracking_squared_error_integral",
        "training_schedule": "fixed_period",
        "diagnostic_schedule_contract": (
            "same_frozen_policy_with_fixed_budget_event_relocation_v1"
        ),
        "oracle_contract": (
            "current_regime_only_no_future_for_policy; switch_times_only_for_"
            "privileged_schedule_reference"
        ),
        "trigger_training": "disabled_not_yet_authorized",
        "belief_training": "disabled_not_yet_authorized",
        "dimensions": dimensions.__dict__,
        "capacity": capacity,
        "environment_contract": environment_contract,
        "runtime_versions": pointmaze_runtime_versions(),
        "world_low": world_low.tolist(),
        "world_high": world_high.tolist(),
        "time_scale": time_scale.metadata(),
        "task_options": _json_ready(task_options),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "waypoint_perturbation": float(waypoint_perturbation),
        "event_window_seconds": float(event_window_seconds),
        "lower_intrinsic_reward_contract": (
            "waypoint_distance_progress_minus_action_cost_v1"
        ),
        "trigger_reward_contract": "not_applicable",
    }
    seed_fn = lambda root, iteration: _training_seed(
        optimizer_seed=optimizer_seed,
        rollout_root=root,
        iteration=iteration,
    )
    payload, canonical_rows, trained = train_frequency_separated_ppo(
        model=model,
        train_seeds=training,
        selection_seeds=selection,
        eval_seeds=evaluation,
        iterations=int(iterations),
        rollout_fn=canonical_rollout,
        objective_fn=lambda row: float(row["episode_return"]),
        summary_fn=summarize_numeric_rows,
        training_seed_fn=seed_fn,
        policy=str(method),
        domain="pointmaze_hidden_regime_plan_value",
        metadata=metadata,
        checkpoint_score_contract=(
            "mean_dense_episode_return_secondary_to_lexicographic_rank"
        ),
        checkpoint_rank_fn=pointmaze_plan_value_checkpoint_rank,
        checkpoint_rank_names=(
            "negative_mean_tracking_squared_error_integral",
            "mean_dense_episode_return",
        ),
        checkpoint_rank_contract=(
            "lexicographic_negative_mean_tracking_squared_error_integral_"
            "then_mean_dense_return_v1"
        ),
        checkpoint_minimum_iteration=0,
        checkpoint_evaluation_interval=int(checkpoint_evaluation_interval),
    )
    for row in canonical_rows:
        row["training_replicate_seed"] = int(optimizer_seed)
    diagnostic_rows: list[dict[str, Any]] = []
    for seed in evaluation:
        for schedule_mode in POINTMAZE_PLAN_VALUE_SCHEDULES:
            _, row = rollout_hrl_pointmaze_plan_value(
                trained,
                seed=int(seed),
                sample=False,
                schedule_mode=schedule_mode,
                **common_rollout,
            )
            row["training_replicate_seed"] = int(optimizer_seed)
            diagnostic_rows.append(row)
    payload["training_core"] = payload["trainer"]
    payload["trainer"] = "goal_conditioned_variable_duration_smdp_ppo_v1"
    payload["history"] = _compact_history(payload["history"])
    payload["history_schema"] = "pointmaze_plan_value_compact_history_v1"
    payload["optimizer_seed"] = int(optimizer_seed)
    payload["untrained_evaluation_rows"] = untrained_rows
    payload["canonical_evaluation_rows"] = canonical_rows
    payload["evaluation_rows"] = diagnostic_rows
    payload["trajectory_contract"]["upper"] = (
        "one transition per actual replan interval with gamma^duration bootstrap"
    )
    payload["trajectory_contract"]["lower"] = (
        "one primitive transition with intrinsic GAE ending at actual waypoint change"
    )
    return payload, trained


def resolved_pointmaze_plan_value_protocol(
    *,
    methods: Iterable[str],
    env_id: str,
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
    waypoint_perturbation: float,
    event_window_seconds: float,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
    task_options: dict[str, Any],
) -> dict[str, Any]:
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    names = tuple(map(str, methods))
    if not names or any(
        name not in POINTMAZE_PLAN_VALUE_METHODS for name in names
    ):
        raise ValueError("plan-value protocol has an unknown method")
    return {
        "protocol_version": POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_PLAN_VALUE_ALGORITHM_PATH,
        "evidence_role": "task_qualification_development",
        "methods": list(names),
        "schedule_modes": list(POINTMAZE_PLAN_VALUE_SCHEDULES),
        "environment_id": str(env_id),
        "iterations": int(iterations),
        "horizon": int(horizon),
        "optimizer_seed": int(optimizer_seed),
        "upper_period_seconds": float(upper_period_seconds),
        "history_seconds": float(history_seconds),
        "fast_period_seconds": float(fast_period_seconds),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "reference_hidden_dim": int(reference_hidden_dim),
        "learning_rate": float(learning_rate),
        "checkpoint_evaluation_interval": int(
            checkpoint_evaluation_interval
        ),
        "waypoint_perturbation": float(waypoint_perturbation),
        "event_window_seconds": float(event_window_seconds),
        "train_seeds": training,
        "selection_seeds": selection,
        "eval_seeds": evaluation,
        "task_options": _json_ready(task_options),
        "primary_endpoint": "tracking_squared_error_integral",
        "disabled_unqualified_mechanisms": [
            "learned_belief",
            "learned_plan_validity_critic",
            "learned_event_trigger",
            "joint_finetuning",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=POINTMAZE_PLAN_VALUE_METHODS,
        default=list(POINTMAZE_PLAN_VALUE_METHODS),
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--iterations", type=int, default=384)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--optimizer-seed", type=int, default=204007)
    parser.add_argument("--upper-period-seconds", type=float, default=0.50)
    parser.add_argument("--history-seconds", type=float, default=0.64)
    parser.add_argument("--fast-period-seconds", type=float, default=0.04)
    parser.add_argument("--maximum-subgoal-delta", type=float, default=0.75)
    parser.add_argument("--reference-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=8)
    parser.add_argument(
        "--waypoint-perturbation",
        type=float,
        default=DEFAULT_WAYPOINT_PERTURBATION,
    )
    parser.add_argument(
        "--event-window-seconds",
        type=float,
        default=DEFAULT_EVENT_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--regime-dwell-seconds",
        nargs=2,
        type=float,
        default=list(DEFAULT_REGIME_DWELL_SECONDS),
    )
    parser.add_argument(
        "--target-speed-modes",
        nargs="+",
        type=float,
        default=list(POINTMAZE_REGIME_SPEEDS),
    )
    parser.add_argument(
        "--force-pulse-amplitude",
        type=float,
        default=DEFAULT_FORCE_PULSE_AMPLITUDE,
    )
    parser.add_argument(
        "--force-pulse-duration-seconds",
        nargs=2,
        type=float,
        default=list(DEFAULT_FORCE_PULSE_DURATION_SECONDS),
    )
    parser.add_argument(
        "--force-pulse-gap-seconds",
        nargs=2,
        type=float,
        default=list(DEFAULT_FORCE_PULSE_GAP_SECONDS),
    )
    parser.add_argument(
        "--distractor-amplitude",
        type=float,
        default=DEFAULT_DISTRACTOR_AMPLITUDE,
    )
    parser.add_argument(
        "--distractor-dwell-seconds",
        nargs=2,
        type=float,
        default=list(DEFAULT_DISTRACTOR_DWELL_SECONDS),
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--selection-seeds", nargs="+", type=int, required=True
    )
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _task_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "regime_dwell_seconds": tuple(map(float, args.regime_dwell_seconds)),
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
    task_options = _task_options_from_args(args)
    protocol = resolved_pointmaze_plan_value_protocol(
        methods=args.methods,
        env_id=args.env_id,
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
        waypoint_perturbation=args.waypoint_perturbation,
        event_window_seconds=args.event_window_seconds,
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        eval_seeds=args.eval_seeds,
        task_options=task_options,
    )
    output: dict[str, Any] = {
        "protocol": protocol,
        "status": "dry_run" if args.dry_run else "complete",
        "cells": [],
    }
    if not args.dry_run:
        for method in args.methods:
            payload, _ = train_pointmaze_plan_value_cell(
                method=method,
                env_id=args.env_id,
                train_seeds=args.train_seeds,
                selection_seeds=args.selection_seeds,
                eval_seeds=args.eval_seeds,
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
                waypoint_perturbation=args.waypoint_perturbation,
                event_window_seconds=args.event_window_seconds,
                task_options=task_options,
            )
            output["cells"].append(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
