"""Stage-2 gate for ordinary goal-conditioned HRL on PointMaze."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.domains.mujoco import (
    RelativeSubgoalAdapter,
    environment_timing,
    goal_environment_contract,
    parse_goal_observation,
)
from freq_hrl.rl import (
    GoalConditionedActorCriticPPO,
    GoalConditionedPPOConfig,
    HierarchicalRolloutBuilder,
    JointActorCriticPPO,
    JointPPOConfig,
    JointTrajectoryBatch,
    flat_actor_critic_parameter_count,
    matched_hierarchical_hidden_dim,
    summarize_numeric_rows,
    train_frequency_separated_ppo,
    train_joint_ppo,
)


POINTMAZE_GOAL_PROTOCOL_VERSION = "pointmaze_goal_control_stage2_v1"
POINTMAZE_METHODS = ("flat_goal_ppo", "hrl_goal_ppo")
DEFAULT_ENV_ID = "PointMaze_UMaze-v3"
DEFAULT_TRAIN_SEEDS = (51011, 51017, 51031, 51047)
DEFAULT_SELECTION_SEEDS = (52009, 52021, 52027, 52051)
DEFAULT_EVAL_SEEDS = (
    53003,
    53017,
    53029,
    53047,
    53069,
    53077,
    53089,
    53101,
)
POINTMAZE_CHECKPOINT_RANK_NAMES = (
    "mean_success_rate",
    "mean_dense_episode_return",
)
POINTMAZE_CHECKPOINT_RANK_CONTRACT = (
    "lexicographic_mean_success_then_mean_dense_episode_return_v1"
)


@dataclass(frozen=True)
class PointMazeDimensions:
    physical: int
    goal: int
    action: int
    flat: int
    upper: int
    lower: int


def _load_gymnasium() -> tuple[Any, Any]:
    try:
        import gymnasium as gym
        import gymnasium_robotics
    except ImportError as exc:
        raise RuntimeError(
            "PointMaze stage-2 requires gymnasium and gymnasium-robotics"
        ) from exc
    gym.register_envs(gymnasium_robotics)
    return gym, gymnasium_robotics


def make_pointmaze_environment(
    *,
    env_id: str,
    horizon: int,
) -> Any:
    gym, _ = _load_gymnasium()
    return gym.make(
        str(env_id),
        reward_type="dense",
        continuing_task=False,
        reset_target=False,
        max_episode_steps=int(horizon),
    )


def pointmaze_dimensions(*, env_id: str, horizon: int) -> PointMazeDimensions:
    environment = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        observation, _ = environment.reset(seed=0)
        parsed = parse_goal_observation(observation)
        contract = goal_environment_contract(environment)
    finally:
        environment.close()
    physical = int(parsed.physical.size)
    goal = int(parsed.achieved_goal.size)
    action = int(contract["action_dim"])
    return PointMazeDimensions(
        physical=physical,
        goal=goal,
        action=action,
        flat=physical + goal,
        upper=physical + goal,
        lower=physical + goal,
    )


def pointmaze_goal_bounds(environment: Any) -> tuple[np.ndarray, np.ndarray]:
    maze = getattr(environment.unwrapped, "maze", None)
    locations = []
    for name in ("unique_goal_locations", "unique_reset_locations"):
        locations.extend(list(getattr(maze, name, ()) or ()))
    if not locations:
        raise ValueError("PointMaze environment does not expose valid locations")
    points = np.asarray(locations, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 1 or not np.all(np.isfinite(points)):
        raise ValueError("PointMaze locations must form a finite matrix")
    margin = 0.5 * float(getattr(maze, "maze_size_scaling", 1.0))
    return (
        np.min(points, axis=0) - margin,
        np.max(points, axis=0) + margin,
    )


def flat_goal_state(observation: Any) -> np.ndarray:
    parsed = parse_goal_observation(observation)
    return np.concatenate((parsed.physical, parsed.goal_error)).astype(
        np.float32, copy=False
    )


def upper_goal_state(observation: Any) -> np.ndarray:
    return flat_goal_state(observation)


def lower_goal_state(observation: Any, subgoal: np.ndarray) -> np.ndarray:
    parsed = parse_goal_observation(observation)
    subgoal_error = np.asarray(
        np.asarray(subgoal, dtype=np.float32).reshape(-1) - parsed.achieved_goal,
        dtype=np.float32,
    )
    if subgoal_error.shape != parsed.achieved_goal.shape:
        raise ValueError("PointMaze subgoal dimension mismatch")
    return np.concatenate((parsed.physical, subgoal_error)).astype(
        np.float32, copy=False
    )


def squash_box_action(
    raw_action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    lower = np.asarray(low, dtype=np.float32).reshape(-1)
    upper = np.asarray(high, dtype=np.float32).reshape(-1)
    if raw.shape != lower.shape or upper.shape != lower.shape:
        raise ValueError("raw action and Box bounds must have equal dimensions")
    if not np.all(np.isfinite(raw)) or np.any(upper <= lower):
        raise ValueError("action and bounds must be finite and ordered")
    midpoint = 0.5 * (upper + lower)
    half_range = 0.5 * (upper - lower)
    return midpoint + half_range * np.tanh(raw)


def _episode_row(
    *,
    method: str,
    seed: int,
    rewards: list[float],
    goal_distances: list[float],
    actions: list[np.ndarray],
    path_length: float,
    success: bool,
    terminated: bool,
    truncated: bool,
    subgoal_distances: list[float],
    subgoals: list[np.ndarray],
    upper_decisions: int,
    parameter_count: int,
    parameter_budget: int,
    env_dt_seconds: float,
    upper_period_steps: int,
    maximum_subgoal_delta: float,
) -> dict[str, Any]:
    reward_array = np.asarray(rewards, dtype=np.float64)
    distance_array = np.asarray(goal_distances, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    subgoal_distance_array = np.asarray(subgoal_distances, dtype=np.float64)
    subgoal_array = np.asarray(subgoals, dtype=np.float64)
    protocol_valid = bool(
        reward_array.size > 0
        and reward_array.size == distance_array.size == action_array.shape[0]
        and np.all(np.isfinite(reward_array))
        and np.all(np.isfinite(distance_array))
        and np.all(np.isfinite(action_array))
        and (method == "flat_goal_ppo" or 0 < upper_decisions < reward_array.size)
    )
    return {
        "protocol_version": POINTMAZE_GOAL_PROTOCOL_VERSION,
        "algorithm_path": "goal_conditioned_hrl_mainline",
        "method": str(method),
        "seed": int(seed),
        "episode_return": float(np.sum(reward_array)),
        "reward_mean": float(np.mean(reward_array)),
        "success": float(bool(success)),
        "final_goal_distance": float(distance_array[-1]),
        "minimum_goal_distance": float(np.min(distance_array)),
        "path_length": float(path_length),
        "episode_length": int(reward_array.size),
        "terminated": float(bool(terminated)),
        "truncated": float(bool(truncated)),
        "action_rms": float(np.sqrt(np.mean(np.square(action_array)))),
        "action_saturation_rate": float(np.mean(np.abs(action_array) >= 0.99)),
        "upper_decision_count": int(upper_decisions),
        "subgoal_tracking_rmse": (
            float(np.sqrt(np.mean(np.square(subgoal_distance_array))))
            if subgoal_distance_array.size else 0.0
        ),
        "subgoal_reached_rate": (
            float(np.mean(subgoal_distance_array <= 0.20))
            if subgoal_distance_array.size else 0.0
        ),
        "subgoal_rms": (
            float(np.sqrt(np.mean(np.square(subgoal_array))))
            if subgoal_array.size else 0.0
        ),
        "env_dt_seconds": float(env_dt_seconds),
        "upper_period_steps": int(upper_period_steps),
        "upper_period_seconds": float(upper_period_steps * env_dt_seconds),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "trainable_parameter_count": int(parameter_count),
        "parameter_budget": int(parameter_budget),
        "parameter_budget_ratio": float(parameter_count / parameter_budget),
        "projector_enabled": 0.0,
        "promotion_enabled": 0.0,
        "leakage_loss_enabled": 0.0,
        "responsibility_gauge_enabled": 0.0,
        "protocol_valid": float(protocol_valid),
    }


def rollout_flat_pointmaze(
    model: JointActorCriticPPO,
    *,
    env_id: str,
    seed: int,
    horizon: int,
    sample: bool,
    parameter_budget: int,
) -> tuple[JointTrajectoryBatch | None, dict[str, Any]]:
    environment = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        observation, _ = environment.reset(seed=int(seed))
        timing = environment_timing(environment)
        action_low = np.asarray(environment.action_space.low, dtype=np.float32)
        action_high = np.asarray(environment.action_space.high, dtype=np.float32)
        model.reset_recurrent_inference()
        states: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[float] = []
        logps: list[float] = []
        values: list[float] = []
        distances: list[float] = []
        actions: list[np.ndarray] = []
        path_length = 0.0
        success = False
        terminated = truncated = False
        achieved_before = parse_goal_observation(observation).achieved_goal
        for _ in range(int(horizon)):
            state = flat_goal_state(observation)
            output = model.act(state, sample=sample)
            raw_action = np.asarray(output["action"], dtype=np.float32).reshape(-1)
            action = squash_box_action(raw_action, action_low, action_high)
            next_observation, reward, terminated, truncated, info = environment.step(action)
            done = bool(terminated or truncated)
            parsed_next = parse_goal_observation(next_observation)
            distance = float(np.linalg.norm(parsed_next.goal_error))
            path_length += float(np.linalg.norm(
                parsed_next.achieved_goal - achieved_before
            ))
            achieved_before = parsed_next.achieved_goal
            success = bool(success or info.get("success", False))
            states.append(state)
            raw_actions.append(raw_action)
            rewards.append(float(reward))
            dones.append(float(done))
            logps.append(float(output["logp"]))
            values.append(float(output["value"]))
            distances.append(distance)
            actions.append(action)
            observation = next_observation
            if done:
                break
        parameter_count = int(sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ))
        row = _episode_row(
            method="flat_goal_ppo",
            seed=seed,
            rewards=rewards,
            goal_distances=distances,
            actions=actions,
            path_length=path_length,
            success=success,
            terminated=terminated,
            truncated=truncated,
            subgoal_distances=[],
            subgoals=[],
            upper_decisions=0,
            parameter_count=parameter_count,
            parameter_budget=parameter_budget,
            env_dt_seconds=timing.control_dt_seconds,
            upper_period_steps=0,
            maximum_subgoal_delta=0.0,
        )
        batch = None
        if sample:
            batch = JointTrajectoryBatch(
                state=np.asarray(states, dtype=np.float32),
                action=np.asarray(raw_actions, dtype=np.float32),
                reward=np.asarray(rewards, dtype=np.float32),
                done=np.asarray(dones, dtype=np.float32),
                old_logp=np.asarray(logps, dtype=np.float32),
                old_value=np.asarray(values, dtype=np.float32),
            )
        return batch, row
    finally:
        environment.close()


def rollout_hrl_pointmaze(
    model: GoalConditionedActorCriticPPO,
    *,
    env_id: str,
    seed: int,
    horizon: int,
    sample: bool,
    parameter_budget: int,
    upper_period_steps: int,
    maximum_subgoal_delta: float,
) -> tuple[Any, dict[str, Any]]:
    environment = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        observation, _ = environment.reset(seed=int(seed))
        timing = environment_timing(environment)
        action_low = np.asarray(environment.action_space.low, dtype=np.float32)
        action_high = np.asarray(environment.action_space.high, dtype=np.float32)
        world_low, world_high = pointmaze_goal_bounds(environment)
        goal_dim = int(parse_goal_observation(observation).achieved_goal.size)
        maximum_delta = np.full(
            goal_dim, float(maximum_subgoal_delta), dtype=np.float32
        )
        subgoal_adapter = RelativeSubgoalAdapter(
            maximum_delta=maximum_delta,
            world_low=world_low,
            world_high=world_high,
            action_cost=0.0,
        )
        model.reset_recurrent_inference()
        builder = HierarchicalRolloutBuilder(gamma=float(model.config.gamma))
        rewards: list[float] = []
        distances: list[float] = []
        actions: list[np.ndarray] = []
        subgoal_distances: list[float] = []
        subgoals: list[np.ndarray] = []
        path_length = 0.0
        success = False
        terminated = truncated = False
        upper_decisions = 0
        parsed = parse_goal_observation(observation)
        achieved_before = parsed.achieved_goal
        subgoal = achieved_before.copy()
        for step in range(int(horizon)):
            if step % int(upper_period_steps) == 0:
                upper_state = upper_goal_state(observation)
                upper_output = model.plan_goal(upper_state, sample=sample)
                raw_goal = np.asarray(
                    upper_output["action"], dtype=np.float32
                ).reshape(-1)
                subgoal = subgoal_adapter.decode(raw_goal, achieved_before)
                builder.begin_upper(
                    state=upper_state,
                    action=raw_goal,
                    logp=float(upper_output["logp"]),
                    value=float(upper_output["value"]),
                )
                subgoals.append(subgoal.copy())
                upper_decisions += 1
            lower_state = lower_goal_state(observation, subgoal)
            lower_output = model.act_conditioned(lower_state, sample=sample)
            raw_action = np.asarray(
                lower_output["action"], dtype=np.float32
            ).reshape(-1)
            action = squash_box_action(raw_action, action_low, action_high)
            next_observation, task_reward, terminated, truncated, info = environment.step(action)
            done = bool(terminated or truncated)
            parsed_next = parse_goal_observation(next_observation)
            subgoal_distance = float(np.linalg.norm(
                (subgoal - parsed_next.achieved_goal) / maximum_delta
            ) / np.sqrt(goal_dim))
            intrinsic_reward = (
                -subgoal_distance
                - 0.005 * float(np.mean(np.square(action)))
            )
            builder.add_lower(
                state=lower_state,
                action=raw_action,
                logp=float(lower_output["logp"]),
                value=float(lower_output["value"]),
                reward=float(intrinsic_reward),
                upper_reward=float(task_reward),
                done=done,
                cost=0.0,
            )
            goal_distance = float(np.linalg.norm(parsed_next.goal_error))
            path_length += float(np.linalg.norm(
                parsed_next.achieved_goal - achieved_before
            ))
            success = bool(success or info.get("success", False))
            rewards.append(float(task_reward))
            distances.append(goal_distance)
            actions.append(action)
            subgoal_distances.append(subgoal_distance)
            achieved_before = parsed_next.achieved_goal
            observation = next_observation
            if done:
                break
        builder.finish(terminal=True)
        batch = builder.build() if sample else None
        row = _episode_row(
            method="hrl_goal_ppo",
            seed=seed,
            rewards=rewards,
            goal_distances=distances,
            actions=actions,
            path_length=path_length,
            success=success,
            terminated=terminated,
            truncated=truncated,
            subgoal_distances=subgoal_distances,
            subgoals=subgoals,
            upper_decisions=upper_decisions,
            parameter_count=model.trainable_parameter_count,
            parameter_budget=parameter_budget,
            env_dt_seconds=timing.control_dt_seconds,
            upper_period_steps=upper_period_steps,
            maximum_subgoal_delta=maximum_subgoal_delta,
        )
        return batch, row
    finally:
        environment.close()


def build_pointmaze_model(
    *,
    method: str,
    dimensions: PointMazeDimensions,
    reference_hidden_dim: int,
    learning_rate: float,
    optimizer_seed: int,
) -> tuple[Any, dict[str, Any]]:
    name = str(method)
    if name not in POINTMAZE_METHODS:
        raise ValueError(f"unknown PointMaze method: {name}")
    target_parameters = flat_actor_critic_parameter_count(
        dimensions.flat,
        dimensions.action,
        int(reference_hidden_dim),
    )
    torch.manual_seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed) % (2**32 - 1))
    if name == "flat_goal_ppo":
        model = JointActorCriticPPO(JointPPOConfig(
            state_dim=dimensions.flat,
            action_dim=dimensions.action,
            hidden_dim=int(reference_hidden_dim),
            learning_rate=float(learning_rate),
            epochs=4,
            minibatch_size=1024,
            init_log_std=-0.7,
        ))
        actual = int(sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ))
        return model, {
            "reference_parameter_budget": target_parameters,
            "actual_parameter_count": actual,
            "parameter_budget_ratio": float(actual / target_parameters),
            "hidden_dim": int(reference_hidden_dim),
        }
    hidden_dim, expected, ratio = matched_hierarchical_hidden_dim(
        target_parameter_count=target_parameters,
        upper_state_dim=dimensions.upper,
        lower_state_dim=dimensions.lower,
        goal_dim=dimensions.goal,
        action_dim=dimensions.action,
    )
    model = GoalConditionedActorCriticPPO(GoalConditionedPPOConfig(
        upper_state_dim=dimensions.upper,
        lower_state_dim=dimensions.lower,
        goal_dim=dimensions.goal,
        action_dim=dimensions.action,
        hidden_dim=hidden_dim,
        learning_rate=float(learning_rate),
        epochs=4,
        minibatch_size=1024,
        init_log_std=-0.7,
    ))
    actual = model.trainable_parameter_count
    if actual != expected:
        raise RuntimeError("PointMaze hierarchical parameter accounting changed")
    return model, {
        "reference_parameter_budget": target_parameters,
        "actual_parameter_count": actual,
        "parameter_budget_ratio": ratio,
        "hidden_dim": hidden_dim,
        **model.mainline_contract(),
    }


def _training_seed(
    *, optimizer_seed: int, rollout_root: int, iteration: int
) -> int:
    return int((
        int(rollout_root) * 22_695_477
        + int(optimizer_seed) * 1_103_515_245
        + (int(iteration) + 1) * 12_345
    ) % (2**32 - 1))


def _validate_seed_roles(
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
) -> tuple[list[int], list[int], list[int]]:
    roles = [
        list(map(int, train_seeds)),
        list(map(int, selection_seeds)),
        list(map(int, eval_seeds)),
    ]
    if any(not values or len(values) != len(set(values)) for values in roles):
        raise ValueError("each PointMaze seed role must be non-empty and unique")
    if any(set(roles[i]) & set(roles[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("PointMaze train, selection, and evaluation seeds must be disjoint")
    return roles[0], roles[1], roles[2]


def pointmaze_checkpoint_rank(
    rows: list[dict[str, Any]],
) -> tuple[float, float]:
    if not rows:
        raise ValueError("PointMaze checkpoint ranking requires validation rows")
    success = float(np.mean([float(row["success"]) for row in rows]))
    episode_return = float(np.mean([
        float(row["episode_return"]) for row in rows
    ]))
    if not np.isfinite(success) or not np.isfinite(episode_return):
        raise ValueError("PointMaze checkpoint rank must be finite")
    return success, episode_return


def train_pointmaze_cell(
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
    maximum_subgoal_delta: float,
    reference_hidden_dim: int = 128,
    learning_rate: float = 3e-4,
    checkpoint_evaluation_interval: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    dimensions = pointmaze_dimensions(env_id=env_id, horizon=horizon)
    probe = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        contract = goal_environment_contract(probe)
        world_low, world_high = pointmaze_goal_bounds(probe)
    finally:
        probe.close()
    dt = float(contract["env_dt_seconds"])
    upper_period_steps = int(round(float(upper_period_seconds) / dt))
    if upper_period_steps < 2 or upper_period_steps >= int(horizon):
        raise ValueError("PointMaze upper period must span 2..horizon-1 steps")
    if not np.isclose(upper_period_steps * dt, float(upper_period_seconds)):
        raise ValueError("PointMaze upper period must resolve exactly in env steps")
    if not np.isfinite(float(maximum_subgoal_delta)) or maximum_subgoal_delta <= 0.0:
        raise ValueError("maximum_subgoal_delta must be positive and finite")
    model, capacity = build_pointmaze_model(
        method=method,
        dimensions=dimensions,
        reference_hidden_dim=reference_hidden_dim,
        learning_rate=learning_rate,
        optimizer_seed=optimizer_seed,
    )
    parameter_budget = int(capacity["reference_parameter_budget"])
    common_metadata = {
        "protocol_version": POINTMAZE_GOAL_PROTOCOL_VERSION,
        "algorithm_path": "goal_conditioned_hrl_mainline",
        "environment_id": str(env_id),
        "optimizer_seed": int(optimizer_seed),
        "reward_type": "dense",
        "success_is_primary_endpoint": True,
        "checkpoint_objective": POINTMAZE_CHECKPOINT_RANK_CONTRACT,
        "goal_semantics": "upper_relative_xy_waypoint_lower_physical_acceleration",
        "lower_final_goal_visibility": "hidden; lower receives only waypoint error",
        "projector": "disabled",
        "promotion": "disabled",
        "leakage_loss": "disabled",
        "responsibility_gauge": "disabled",
        "dimensions": dimensions.__dict__,
        "capacity": capacity,
        "environment_contract": contract,
        "world_low": world_low.tolist(),
        "world_high": world_high.tolist(),
        "upper_period_steps": int(upper_period_steps),
        "upper_period_seconds": float(upper_period_seconds),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
    }
    seed_fn = lambda root, iteration: _training_seed(
        optimizer_seed=optimizer_seed,
        rollout_root=root,
        iteration=iteration,
    )
    common_rollout = {
        "env_id": str(env_id),
        "horizon": int(horizon),
        "parameter_budget": parameter_budget,
    }
    if str(method) == "hrl_goal_ppo":
        payload, rows, trained = train_frequency_separated_ppo(
            model=model,
            train_seeds=training,
            selection_seeds=selection,
            eval_seeds=evaluation,
            iterations=int(iterations),
            rollout_fn=lambda policy, seed, sample: rollout_hrl_pointmaze(
                policy,
                seed=seed,
                sample=sample,
                upper_period_steps=upper_period_steps,
                maximum_subgoal_delta=maximum_subgoal_delta,
                **common_rollout,
            ),
            objective_fn=lambda row: float(row["episode_return"]),
            summary_fn=summarize_numeric_rows,
            training_seed_fn=seed_fn,
            policy=str(method),
            domain="pointmaze_goal_control",
            metadata=common_metadata,
            checkpoint_score_contract="mean_dense_episode_return",
            checkpoint_rank_fn=pointmaze_checkpoint_rank,
            checkpoint_rank_names=POINTMAZE_CHECKPOINT_RANK_NAMES,
            checkpoint_rank_contract=POINTMAZE_CHECKPOINT_RANK_CONTRACT,
            checkpoint_minimum_iteration=0,
            checkpoint_evaluation_interval=int(checkpoint_evaluation_interval),
        )
        payload["training_core"] = payload["trainer"]
        payload["trainer"] = "goal_conditioned_smdp_ppo_v1"
    elif str(method) == "flat_goal_ppo":
        payload, rows, trained = train_joint_ppo(
            model=model,
            train_seeds=training,
            selection_seeds=selection,
            eval_seeds=evaluation,
            iterations=int(iterations),
            rollout_fn=lambda policy, seed, sample: rollout_flat_pointmaze(
                policy,
                seed=seed,
                sample=sample,
                **common_rollout,
            ),
            objective_fn=lambda row: float(row["episode_return"]),
            summary_fn=summarize_numeric_rows,
            training_seed_fn=seed_fn,
            policy=str(method),
            domain="pointmaze_goal_control",
            metadata=common_metadata,
            checkpoint_score_contract="mean_dense_episode_return",
            checkpoint_rank_fn=pointmaze_checkpoint_rank,
            checkpoint_rank_names=POINTMAZE_CHECKPOINT_RANK_NAMES,
            checkpoint_rank_contract=POINTMAZE_CHECKPOINT_RANK_CONTRACT,
            checkpoint_minimum_iteration=0,
            checkpoint_evaluation_interval=int(checkpoint_evaluation_interval),
        )
    else:
        raise ValueError(f"unknown PointMaze method: {method}")
    for row in rows:
        row["training_replicate_seed"] = int(optimizer_seed)
    payload["optimizer_seed"] = int(optimizer_seed)
    payload["evaluation_rows"] = rows
    return payload, rows, trained


def resolved_pointmaze_protocol(
    *,
    methods: Iterable[str],
    env_id: str,
    iterations: int,
    horizon: int,
    optimizer_seed: int,
    upper_period_seconds: float,
    maximum_subgoal_delta: float,
    reference_hidden_dim: int,
    learning_rate: float,
    checkpoint_evaluation_interval: int,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
) -> dict[str, Any]:
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    method_names = list(map(str, methods))
    if not method_names or any(name not in POINTMAZE_METHODS for name in method_names):
        raise ValueError("PointMaze protocol contains an unknown or empty method set")
    return {
        "protocol_version": POINTMAZE_GOAL_PROTOCOL_VERSION,
        "methods": method_names,
        "environment_id": str(env_id),
        "reward_type": "dense",
        "primary_endpoint": "success_rate",
        "checkpoint_rank_contract": POINTMAZE_CHECKPOINT_RANK_CONTRACT,
        "iterations": int(iterations),
        "horizon": int(horizon),
        "optimizer_seed": int(optimizer_seed),
        "upper_period_seconds": float(upper_period_seconds),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "reference_hidden_dim": int(reference_hidden_dim),
        "learning_rate": float(learning_rate),
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
        "train_seeds": training,
        "selection_seeds": selection,
        "eval_seeds": evaluation,
        "disabled_legacy_mechanisms": [
            "action_spectrum_projector",
            "promotion",
            "leakage_loss",
            "responsibility_gauge",
            "projection_consistency",
            "multiscale_representation",
        ],
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PointMaze ordinary goal-conditioned HRL gate."
    )
    parser.add_argument("--methods", nargs="+", choices=POINTMAZE_METHODS, default=list(POINTMAZE_METHODS))
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--iterations", type=int, default=768)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--optimizer-seed", type=int, default=54007)
    parser.add_argument("--upper-period-seconds", type=float, default=0.25)
    parser.add_argument("--maximum-subgoal-delta", type=float, default=0.75)
    parser.add_argument("--reference-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=16)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--selection-seeds", nargs="+", type=int, default=list(DEFAULT_SELECTION_SEEDS))
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = resolved_pointmaze_protocol(
        methods=args.methods,
        env_id=args.env_id,
        iterations=args.iterations,
        horizon=args.horizon,
        optimizer_seed=args.optimizer_seed,
        upper_period_seconds=args.upper_period_seconds,
        maximum_subgoal_delta=args.maximum_subgoal_delta,
        reference_hidden_dim=args.reference_hidden_dim,
        learning_rate=args.learning_rate,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        eval_seeds=args.eval_seeds,
    )
    output: dict[str, Any] = {
        "protocol": protocol,
        "status": "dry_run" if args.dry_run else "complete",
        "cells": [],
    }
    if not args.dry_run:
        for method in args.methods:
            payload, _, _ = train_pointmaze_cell(
                method=method,
                env_id=args.env_id,
                train_seeds=args.train_seeds,
                selection_seeds=args.selection_seeds,
                eval_seeds=args.eval_seeds,
                iterations=args.iterations,
                horizon=args.horizon,
                optimizer_seed=args.optimizer_seed,
                upper_period_seconds=args.upper_period_seconds,
                maximum_subgoal_delta=args.maximum_subgoal_delta,
                reference_hidden_dim=args.reference_hidden_dim,
                learning_rate=args.learning_rate,
                checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
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
