"""Stage-5 substrate for HRL with a separate exogenous PointMaze stream."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.core import CausalHaarMultiscaleEncoder, PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import (
    PointMazeExternalObservation,
    PointMazeExternalTask,
    RelativeSubgoalAdapter,
    environment_timing,
    goal_environment_contract,
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

from .pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    POINTMAZE_HISTORY_FIELDS,
    POINTMAZE_LOWER_ACTION_COST,
    _json_ready,
    _training_seed,
    _validate_seed_roles,
    make_pointmaze_environment,
    pointmaze_goal_bounds,
    pointmaze_runtime_versions,
    squash_box_action,
)


POINTMAZE_EXOGENOUS_PROTOCOL_VERSION = "pointmaze_exogenous_control_stage5_v1"
POINTMAZE_EXOGENOUS_ALGORITHM_PATH = "exogenous_goal_conditioned_hrl_mainline"
POINTMAZE_EXOGENOUS_METHODS = (
    "flat_exogenous_history",
    "hrl_exogenous_history",
)
POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODES = (
    "success_then_return",
    "return_then_success",
)
DEFAULT_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODE = "success_then_return"
_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_NAMES = {
    "success_then_return": (
        "mean_tracking_success_rate",
        "mean_dense_episode_return",
    ),
    "return_then_success": (
        "mean_dense_episode_return",
        "mean_tracking_success_rate",
    ),
}
_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_CONTRACTS = {
    "success_then_return": (
        "lexicographic_mean_tracking_success_then_mean_dense_return_v1"
    ),
    "return_then_success": (
        "lexicographic_mean_dense_return_then_mean_tracking_success_v1"
    ),
}
POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_NAMES = (
    _POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_NAMES[
        DEFAULT_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODE
    ]
)
POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_CONTRACT = (
    _POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_CONTRACTS[
        DEFAULT_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODE
    ]
)
DEFAULT_TARGET_SPEED = 1.0
DEFAULT_FORCE_RMS = 0.12
DEFAULT_FORCE_PERIOD_SECONDS = (0.04, 0.04)
POINTMAZE_EXOGENOUS_REPRESENTATIONS = (
    "history",
    "filtered",
    "multiscale_all",
    "multiscale_routed_masked",
    "multiscale_swapped_masked",
)

POINTMAZE_EXOGENOUS_HISTORY_FIELDS = POINTMAZE_HISTORY_FIELDS | frozenset({
    "tracking_success_rate_mean",
    "tracking_rmse_mean",
    "tracking_mae_mean",
    "requested_action_rms_mean",
    "executed_action_rms_mean",
    "force_rms_mean",
})


@dataclass(frozen=True)
class PointMazeExogenousDimensions:
    physical: int
    goal: int
    task: int
    history: int
    action: int
    flat: int
    upper: int
    lower: int


class PointMazeExogenousFeatureBuilder:
    """Keep physical feedback current while buffering only the external stream."""

    def __init__(self, *, time_scale: PhysicalTimeScaleContract) -> None:
        self.encoder = CausalHaarMultiscaleEncoder(
            feature_dim=4,
            time_scale=time_scale,
        )
        self._snapshot = None

    @property
    def snapshot(self):
        if self._snapshot is None:
            raise RuntimeError("external feature builder must be reset")
        return self._snapshot

    def reset(self, observation: PointMazeExternalObservation):
        self._snapshot = self.encoder.reset(observation.task_measurement)
        return self._snapshot

    def update(self, observation: PointMazeExternalObservation):
        self._snapshot = self.encoder.update(observation.task_measurement)
        return self._snapshot

    def dimensions(
        self,
        observation: PointMazeExternalObservation,
        *,
        action_dim: int,
        representation: str = "history",
    ) -> PointMazeExogenousDimensions:
        physical = int(observation.physical.size)
        goal = int(observation.achieved_goal.size)
        task = int(observation.task_measurement.size)
        history = int(self.snapshot.history.size)
        encoded = self._task_features(representation, level="flat")
        if int(encoded.size) != history:
            raise RuntimeError("external representation changed fixed state shape")
        shared = physical + goal + history
        return PointMazeExogenousDimensions(
            physical=physical,
            goal=goal,
            task=task,
            history=history,
            action=int(action_dim),
            flat=shared,
            upper=shared,
            lower=shared,
        )

    def _task_features(self, representation: str, *, level: str) -> np.ndarray:
        name = str(representation)
        if level not in ("flat", "upper", "lower"):
            raise ValueError(f"unknown external hierarchy level: {level}")
        if name == "history":
            return self.snapshot.history
        if name == "filtered":
            return self.snapshot.filtered
        if name == "multiscale_all" or level == "flat":
            if name not in POINTMAZE_EXOGENOUS_REPRESENTATIONS:
                raise ValueError(
                    f"unknown external PointMaze representation: {name}"
                )
            return self.snapshot.multiscale
        if name == "multiscale_routed_masked":
            blocks = (
                (self.snapshot.slow, self.snapshot.mid, np.zeros_like(self.snapshot.high))
                if level == "upper"
                else (np.zeros_like(self.snapshot.slow), self.snapshot.mid, self.snapshot.high)
            )
            return np.concatenate(blocks)
        if name == "multiscale_swapped_masked":
            blocks = (
                (np.zeros_like(self.snapshot.slow), self.snapshot.mid, self.snapshot.high)
                if level == "upper"
                else (self.snapshot.slow, self.snapshot.mid, np.zeros_like(self.snapshot.high))
            )
            return np.concatenate(blocks)
        raise ValueError(f"unknown external PointMaze representation: {name}")

    def flat_state(
        self,
        observation: PointMazeExternalObservation,
        *,
        representation: str = "history",
    ) -> np.ndarray:
        return np.concatenate((
            observation.physical,
            observation.target_error,
            self._task_features(representation, level="flat"),
        )).astype(np.float32, copy=False)

    def upper_state(
        self,
        observation: PointMazeExternalObservation,
        *,
        representation: str = "history",
    ) -> np.ndarray:
        return np.concatenate((
            observation.physical,
            observation.target_error,
            self._task_features(representation, level="upper"),
        )).astype(np.float32, copy=False)

    def lower_state(
        self,
        observation: PointMazeExternalObservation,
        *,
        subgoal: np.ndarray,
        representation: str = "history",
    ) -> np.ndarray:
        waypoint_error = np.asarray(
            np.asarray(subgoal, dtype=np.float32).reshape(-1)
            - observation.achieved_goal,
            dtype=np.float32,
        )
        if waypoint_error.shape != observation.achieved_goal.shape:
            raise ValueError("external PointMaze subgoal dimension mismatch")
        return np.concatenate((
            observation.physical,
            waypoint_error,
            self._task_features(representation, level="lower"),
        )).astype(np.float32, copy=False)


def _make_task(
    *,
    env_id: str,
    seed: int,
    horizon: int,
    target_speed: float,
    force_rms: float,
    force_period_seconds: tuple[float, float],
) -> PointMazeExternalTask:
    environment = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    return PointMazeExternalTask(
        environment,
        seed=seed,
        horizon=horizon,
        target_speed=target_speed,
        force_rms=force_rms,
        force_period_seconds=force_period_seconds,
    )


def pointmaze_exogenous_dimensions(
    *,
    env_id: str,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    target_speed: float = DEFAULT_TARGET_SPEED,
    force_rms: float = DEFAULT_FORCE_RMS,
    force_period_seconds: tuple[float, float] = DEFAULT_FORCE_PERIOD_SECONDS,
    representation: str = "history",
) -> PointMazeExogenousDimensions:
    task = _make_task(
        env_id=env_id,
        seed=0,
        horizon=horizon,
        target_speed=target_speed,
        force_rms=force_rms,
        force_period_seconds=force_period_seconds,
    )
    try:
        observation = task.reset()
        features = PointMazeExogenousFeatureBuilder(time_scale=time_scale)
        features.reset(observation)
        return features.dimensions(
            observation,
            action_dim=int(task.action_low.size),
            representation=representation,
        )
    finally:
        task.environment.close()


def _episode_row(
    *,
    method: str,
    seed: int,
    rewards: list[float],
    distances: list[float],
    successes: list[float],
    requested_actions: list[np.ndarray],
    executed_actions: list[np.ndarray],
    forces: list[np.ndarray],
    path_length: float,
    terminated: bool,
    truncated: bool,
    subgoal_distances: list[float],
    subgoals: list[np.ndarray],
    intrinsic_rewards: list[float],
    lower_option_boundaries: int,
    upper_decisions: int,
    parameter_count: int,
    parameter_budget: int,
    time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float,
    task: PointMazeExternalTask,
    protocol_version: str = POINTMAZE_EXOGENOUS_PROTOCOL_VERSION,
    algorithm_path: str = POINTMAZE_EXOGENOUS_ALGORITHM_PATH,
) -> dict[str, Any]:
    reward = np.asarray(rewards, dtype=np.float64)
    distance = np.asarray(distances, dtype=np.float64)
    success = np.asarray(successes, dtype=np.float64)
    requested = np.asarray(requested_actions, dtype=np.float64)
    executed = np.asarray(executed_actions, dtype=np.float64)
    force = np.asarray(forces, dtype=np.float64)
    subgoal_distance = np.asarray(subgoal_distances, dtype=np.float64)
    subgoal_array = np.asarray(subgoals, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_rewards, dtype=np.float64)
    hierarchical = str(method).startswith("hrl_exogenous")
    protocol_valid = bool(
        reward.size == task.horizon
        and reward.size
        == distance.size
        == success.size
        == requested.shape[0]
        == executed.shape[0]
        == force.shape[0]
        and np.all(np.isfinite(reward))
        and np.all(np.isfinite(distance))
        and np.all(np.isfinite(requested))
        and np.all(np.isfinite(executed))
        and np.all(np.isfinite(force))
        and (
            not hierarchical
            or (
                0 < upper_decisions < reward.size
                and intrinsic.size == reward.size
                and np.all(np.isfinite(intrinsic))
                and 0 < lower_option_boundaries <= upper_decisions
            )
        )
    )
    return {
        "protocol_version": str(protocol_version),
        "algorithm_path": str(algorithm_path),
        "method": str(method),
        "seed": int(seed),
        "episode_return": float(np.sum(reward)),
        "reward_mean": float(np.mean(reward)),
        "tracking_success_rate": float(np.mean(success)),
        "episode_tracking_success": float(np.mean(success) >= 0.50),
        "tracking_rmse": float(np.sqrt(np.mean(np.square(distance)))),
        "tracking_mae": float(np.mean(distance)),
        "final_tracking_distance": float(distance[-1]),
        "maximum_tracking_distance": float(np.max(distance)),
        "path_length": float(path_length),
        "episode_length": int(reward.size),
        "terminated": float(bool(terminated)),
        "truncated": float(bool(truncated)),
        "requested_action_rms": float(np.sqrt(np.mean(np.square(requested)))),
        "executed_action_rms": float(np.sqrt(np.mean(np.square(executed)))),
        "action_saturation_rate": float(np.mean(np.abs(executed) >= 0.99)),
        "force_rms": float(np.sqrt(np.mean(np.square(force)))),
        "upper_decision_count": int(upper_decisions),
        "subgoal_tracking_rmse": (
            float(np.sqrt(np.mean(np.square(subgoal_distance))))
            if subgoal_distance.size else 0.0
        ),
        "subgoal_reached_rate": (
            float(np.mean(subgoal_distance <= 0.20))
            if subgoal_distance.size else 0.0
        ),
        "subgoal_rms": (
            float(np.sqrt(np.mean(np.square(subgoal_array))))
            if subgoal_array.size else 0.0
        ),
        "lower_intrinsic_return": (
            float(np.sum(intrinsic)) if intrinsic.size else 0.0
        ),
        "lower_intrinsic_reward_mean": (
            float(np.mean(intrinsic)) if intrinsic.size else 0.0
        ),
        "lower_option_boundary_count": int(lower_option_boundaries),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "trainable_parameter_count": int(parameter_count),
        "parameter_budget": int(parameter_budget),
        "parameter_budget_ratio": float(parameter_count / parameter_budget),
        "projector_enabled": 0.0,
        "promotion_enabled": 0.0,
        "leakage_loss_enabled": 0.0,
        "responsibility_gauge_enabled": 0.0,
        "protocol_valid": float(protocol_valid),
        **time_scale.metadata(),
        **task.diagnostics(),
    }


def rollout_flat_pointmaze_exogenous(
    model: JointActorCriticPPO,
    *,
    env_id: str,
    seed: int,
    horizon: int,
    sample: bool,
    parameter_budget: int,
    time_scale: PhysicalTimeScaleContract,
    target_speed: float,
    force_rms: float,
    force_period_seconds: tuple[float, float],
    method: str = "flat_exogenous_history",
    representation: str = "history",
    protocol_version: str = POINTMAZE_EXOGENOUS_PROTOCOL_VERSION,
    algorithm_path: str = POINTMAZE_EXOGENOUS_ALGORITHM_PATH,
) -> tuple[JointTrajectoryBatch | None, dict[str, Any]]:
    task = _make_task(
        env_id=env_id,
        seed=seed,
        horizon=horizon,
        target_speed=target_speed,
        force_rms=force_rms,
        force_period_seconds=force_period_seconds,
    )
    try:
        observation = task.reset()
        features = PointMazeExogenousFeatureBuilder(time_scale=time_scale)
        features.reset(observation)
        model.reset_recurrent_inference()
        states: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[float] = []
        logps: list[float] = []
        values: list[float] = []
        distances: list[float] = []
        successes: list[float] = []
        requested_actions: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        path_length = 0.0
        terminated = truncated = False
        achieved_before = observation.achieved_goal.copy()
        for _ in range(int(horizon)):
            state = features.flat_state(
                observation, representation=representation
            )
            output = model.act(state, sample=sample)
            raw_action = np.asarray(output["action"], dtype=np.float32).reshape(-1)
            requested = squash_box_action(
                raw_action, task.action_low, task.action_high
            )
            next_observation, reward, terminated, truncated, info = task.step(
                requested
            )
            done = bool(terminated or truncated)
            path_length += float(np.linalg.norm(
                next_observation.achieved_goal - achieved_before
            ))
            achieved_before = next_observation.achieved_goal.copy()
            states.append(state)
            raw_actions.append(raw_action)
            rewards.append(float(reward))
            dones.append(float(done))
            logps.append(float(output["logp"]))
            values.append(float(output["value"]))
            distances.append(float(info["tracking_distance"]))
            successes.append(float(info["tracking_success"]))
            requested_actions.append(requested)
            executed_actions.append(np.asarray(info["executed_action"]))
            forces.append(np.asarray(info["force"]))
            observation = next_observation
            features.update(observation)
            if done:
                break
        parameter_count = int(sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ))
        row = _episode_row(
            method=method,
            seed=seed,
            rewards=rewards,
            distances=distances,
            successes=successes,
            requested_actions=requested_actions,
            executed_actions=executed_actions,
            forces=forces,
            path_length=path_length,
            terminated=terminated,
            truncated=truncated,
            subgoal_distances=[],
            subgoals=[],
            intrinsic_rewards=[],
            lower_option_boundaries=0,
            upper_decisions=0,
            parameter_count=parameter_count,
            parameter_budget=parameter_budget,
            time_scale=time_scale,
            maximum_subgoal_delta=0.0,
            task=task,
            protocol_version=protocol_version,
            algorithm_path=algorithm_path,
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
        task.environment.close()


def rollout_hrl_pointmaze_exogenous(
    model: GoalConditionedActorCriticPPO,
    *,
    env_id: str,
    seed: int,
    horizon: int,
    sample: bool,
    parameter_budget: int,
    time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float,
    target_speed: float,
    force_rms: float,
    force_period_seconds: tuple[float, float],
    method: str = "hrl_exogenous_history",
    representation: str = "history",
    protocol_version: str = POINTMAZE_EXOGENOUS_PROTOCOL_VERSION,
    algorithm_path: str = POINTMAZE_EXOGENOUS_ALGORITHM_PATH,
) -> tuple[Any, dict[str, Any]]:
    task = _make_task(
        env_id=env_id,
        seed=seed,
        horizon=horizon,
        target_speed=target_speed,
        force_rms=force_rms,
        force_period_seconds=force_period_seconds,
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
        features = PointMazeExogenousFeatureBuilder(time_scale=time_scale)
        features.reset(observation)
        model.reset_recurrent_inference()
        builder = HierarchicalRolloutBuilder(gamma=float(model.config.gamma))
        rewards: list[float] = []
        distances: list[float] = []
        successes: list[float] = []
        requested_actions: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        subgoal_distances: list[float] = []
        subgoals: list[np.ndarray] = []
        intrinsic_rewards: list[float] = []
        path_length = 0.0
        terminated = truncated = False
        upper_decisions = 0
        lower_option_boundaries = 0
        last_lower_terminal = False
        achieved_before = observation.achieved_goal.copy()
        subgoal = achieved_before.copy()
        upper_period_steps = int(time_scale.upper_period_steps)
        for step in range(int(horizon)):
            if step % upper_period_steps == 0:
                upper_state = features.upper_state(
                    observation, representation=representation
                )
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
            lower_state = features.lower_state(
                observation,
                subgoal=subgoal,
                representation=representation,
            )
            lower_output = model.act_conditioned(lower_state, sample=sample)
            raw_action = np.asarray(
                lower_output["action"], dtype=np.float32
            ).reshape(-1)
            requested = squash_box_action(
                raw_action, task.action_low, task.action_high
            )
            next_observation, task_reward, terminated, truncated, info = task.step(
                requested
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
            option_boundary = (step + 1) % upper_period_steps == 0
            lower_terminal = bool(done or option_boundary)
            last_lower_terminal = lower_terminal
            lower_option_boundaries += int(lower_terminal)
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
            path_length += float(np.linalg.norm(
                next_observation.achieved_goal - achieved_before
            ))
            rewards.append(float(task_reward))
            distances.append(float(info["tracking_distance"]))
            successes.append(float(info["tracking_success"]))
            requested_actions.append(requested)
            executed_actions.append(np.asarray(info["executed_action"]))
            forces.append(np.asarray(info["force"]))
            subgoal_distances.append(subgoal_distance)
            intrinsic_rewards.append(float(intrinsic_reward))
            achieved_before = next_observation.achieved_goal.copy()
            observation = next_observation
            features.update(observation)
            if done:
                break
        if rewards and not last_lower_terminal:
            lower_option_boundaries += 1
        builder.finish(terminal=True)
        batch = builder.build() if sample else None
        row = _episode_row(
            method=method,
            seed=seed,
            rewards=rewards,
            distances=distances,
            successes=successes,
            requested_actions=requested_actions,
            executed_actions=executed_actions,
            forces=forces,
            path_length=path_length,
            terminated=terminated,
            truncated=truncated,
            subgoal_distances=subgoal_distances,
            subgoals=subgoals,
            intrinsic_rewards=intrinsic_rewards,
            lower_option_boundaries=lower_option_boundaries,
            upper_decisions=upper_decisions,
            parameter_count=model.trainable_parameter_count,
            parameter_budget=parameter_budget,
            time_scale=time_scale,
            maximum_subgoal_delta=maximum_subgoal_delta,
            task=task,
            protocol_version=protocol_version,
            algorithm_path=algorithm_path,
        )
        return batch, row
    finally:
        task.environment.close()


def build_pointmaze_exogenous_model(
    *,
    method: str,
    dimensions: PointMazeExogenousDimensions,
    reference_hidden_dim: int,
    learning_rate: float,
    optimizer_seed: int,
) -> tuple[Any, dict[str, Any]]:
    name = str(method)
    if name not in POINTMAZE_EXOGENOUS_METHODS:
        raise ValueError(f"unknown external PointMaze method: {name}")
    target_parameters = flat_actor_critic_parameter_count(
        dimensions.flat,
        dimensions.action,
        int(reference_hidden_dim),
    )
    torch.manual_seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed) % (2**32 - 1))
    if name == "flat_exogenous_history":
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
    if model.trainable_parameter_count != expected:
        raise RuntimeError("external PointMaze parameter accounting changed")
    return model, {
        "reference_parameter_budget": target_parameters,
        "actual_parameter_count": model.trainable_parameter_count,
        "parameter_budget_ratio": ratio,
        "hidden_dim": hidden_dim,
        **model.mainline_contract(),
    }


def pointmaze_exogenous_checkpoint_rank_spec(
    mode: str,
) -> tuple[tuple[str, str], str]:
    name = str(mode)
    if name not in POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODES:
        raise ValueError(f"unknown external PointMaze checkpoint rank mode: {name}")
    return (
        _POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_NAMES[name],
        _POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_CONTRACTS[name],
    )


def pointmaze_exogenous_checkpoint_rank(
    rows: list[dict[str, Any]],
    mode: str = DEFAULT_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODE,
) -> tuple[float, float]:
    if not rows:
        raise ValueError("external PointMaze checkpoint requires rows")
    success = float(np.mean([
        float(row["tracking_success_rate"]) for row in rows
    ]))
    episode_return = float(np.mean([
        float(row["episode_return"]) for row in rows
    ]))
    if not np.isfinite(success) or not np.isfinite(episode_return):
        raise ValueError("external PointMaze checkpoint rank must be finite")
    pointmaze_exogenous_checkpoint_rank_spec(mode)
    if mode == "return_then_success":
        return episode_return, success
    return success, episode_return


def compact_pointmaze_exogenous_history(
    history: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key in POINTMAZE_EXOGENOUS_HISTORY_FIELDS
        }
        for row in history
    ]


def train_pointmaze_exogenous_cell(
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
    target_speed: float = DEFAULT_TARGET_SPEED,
    force_rms: float = DEFAULT_FORCE_RMS,
    force_period_seconds: tuple[float, float] = DEFAULT_FORCE_PERIOD_SECONDS,
    reference_hidden_dim: int = 128,
    learning_rate: float = 3e-4,
    checkpoint_evaluation_interval: int = 16,
    checkpoint_rank_mode: str = (
        DEFAULT_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODE
    ),
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    probe = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        contract = goal_environment_contract(probe)
        world_low, world_high = pointmaze_goal_bounds(probe)
    finally:
        probe.close()
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=float(contract["env_dt_seconds"]),
        upper_period_seconds=float(upper_period_seconds),
        history_seconds=float(history_seconds),
        fast_period_seconds=float(fast_period_seconds),
    )
    if time_scale.upper_period_steps >= int(horizon):
        raise ValueError("external PointMaze upper period must be below horizon")
    dimensions = pointmaze_exogenous_dimensions(
        env_id=env_id,
        horizon=horizon,
        time_scale=time_scale,
        target_speed=target_speed,
        force_rms=force_rms,
        force_period_seconds=force_period_seconds,
    )
    checkpoint_rank_names, checkpoint_rank_contract = (
        pointmaze_exogenous_checkpoint_rank_spec(checkpoint_rank_mode)
    )
    model, capacity = build_pointmaze_exogenous_model(
        method=method,
        dimensions=dimensions,
        reference_hidden_dim=reference_hidden_dim,
        learning_rate=learning_rate,
        optimizer_seed=optimizer_seed,
    )
    parameter_budget = int(capacity["reference_parameter_budget"])
    common_metadata = {
        "protocol_version": POINTMAZE_EXOGENOUS_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_EXOGENOUS_ALGORITHM_PATH,
        "environment_id": str(env_id),
        "optimizer_seed": int(optimizer_seed),
        "task_reward_contract": "fixed_horizon_exp_negative_dynamic_target_distance_v1",
        "primary_endpoint": "tracking_success_rate",
        "checkpoint_objective": checkpoint_rank_contract,
        "checkpoint_rank_mode": str(checkpoint_rank_mode),
        "state_contract": "current_physical_z_plus_separate_external_x_history_v1",
        "external_stream_contract": (
            "action_independent_slow_route_target_plus_fast_measured_force_v1"
        ),
        "goal_semantics": "upper_relative_xy_waypoint_lower_physical_acceleration",
        "lower_final_target_visibility": (
            "no explicit target error; unmasked raw external history baseline"
        ),
        "projector": "disabled",
        "promotion": "disabled",
        "leakage_loss": "disabled",
        "responsibility_gauge": "disabled",
        "dimensions": dimensions.__dict__,
        "capacity": capacity,
        "environment_contract": contract,
        "runtime_versions": pointmaze_runtime_versions(),
        "world_low": world_low.tolist(),
        "world_high": world_high.tolist(),
        "time_scale": time_scale.metadata(),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "target_speed_world_per_second": float(target_speed),
        "force_rms_per_axis": float(force_rms),
        "force_period_seconds": list(map(float, force_period_seconds)),
        "lower_intrinsic_reward_contract": (
            "waypoint_distance_progress_minus_action_cost_v1"
            if str(method) == "hrl_exogenous_history" else "not_applicable"
        ),
        "lower_credit_boundary_contract": (
            "waypoint_change_or_episode_end_v1"
            if str(method) == "hrl_exogenous_history" else "not_applicable"
        ),
    }
    common_rollout = {
        "env_id": str(env_id),
        "horizon": int(horizon),
        "parameter_budget": parameter_budget,
        "time_scale": time_scale,
        "target_speed": float(target_speed),
        "force_rms": float(force_rms),
        "force_period_seconds": tuple(map(float, force_period_seconds)),
    }
    seed_fn = lambda root, iteration: _training_seed(
        optimizer_seed=optimizer_seed,
        rollout_root=root,
        iteration=iteration,
    )
    trainer_kwargs = {
        "train_seeds": training,
        "selection_seeds": selection,
        "eval_seeds": evaluation,
        "iterations": int(iterations),
        "objective_fn": lambda row: float(row["episode_return"]),
        "summary_fn": summarize_numeric_rows,
        "training_seed_fn": seed_fn,
        "policy": str(method),
        "domain": "pointmaze_exogenous_control",
        "metadata": common_metadata,
        "checkpoint_score_contract": "mean_dense_episode_return",
        "checkpoint_rank_fn": lambda rows: pointmaze_exogenous_checkpoint_rank(
            rows, mode=checkpoint_rank_mode
        ),
        "checkpoint_rank_names": checkpoint_rank_names,
        "checkpoint_rank_contract": checkpoint_rank_contract,
        "checkpoint_minimum_iteration": 0,
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
    }
    if str(method) == "hrl_exogenous_history":
        rollout_fn = lambda policy, seed, sample: rollout_hrl_pointmaze_exogenous(
            policy,
            seed=seed,
            sample=sample,
            maximum_subgoal_delta=maximum_subgoal_delta,
            **common_rollout,
        )
    elif str(method) == "flat_exogenous_history":
        rollout_fn = lambda policy, seed, sample: rollout_flat_pointmaze_exogenous(
            policy,
            seed=seed,
            sample=sample,
            **common_rollout,
        )
    else:
        raise ValueError(f"unknown external PointMaze method: {method}")
    untrained_rows = [
        rollout_fn(model, int(seed), False)[1] for seed in evaluation
    ]
    for row in untrained_rows:
        row["training_replicate_seed"] = int(optimizer_seed)

    if str(method) == "hrl_exogenous_history":
        payload, rows, trained = train_frequency_separated_ppo(
            model=model,
            rollout_fn=rollout_fn,
            **trainer_kwargs,
        )
        payload["training_core"] = payload["trainer"]
        payload["trainer"] = "goal_conditioned_smdp_ppo_v1"
        payload["trajectory_contract"]["lower"] = (
            "one primitive transition with GAE terminated at waypoint change "
            "or episode end"
        )
    elif str(method) == "flat_exogenous_history":
        payload, rows, trained = train_joint_ppo(
            model=model,
            rollout_fn=rollout_fn,
            **trainer_kwargs,
        )
    for row in rows:
        row["training_replicate_seed"] = int(optimizer_seed)
    payload["history"] = compact_pointmaze_exogenous_history(payload["history"])
    payload["history_schema"] = "pointmaze_exogenous_compact_training_history_v1"
    payload["optimizer_seed"] = int(optimizer_seed)
    payload["untrained_evaluation_rows"] = untrained_rows
    payload["evaluation_rows"] = rows
    return payload, rows, trained


def resolved_pointmaze_exogenous_protocol(
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
    target_speed: float,
    force_rms: float,
    force_period_seconds: tuple[float, float],
    reference_hidden_dim: int,
    learning_rate: float,
    checkpoint_evaluation_interval: int,
    checkpoint_rank_mode: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
) -> dict[str, Any]:
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    names = list(map(str, methods))
    if not names or any(name not in POINTMAZE_EXOGENOUS_METHODS for name in names):
        raise ValueError("external PointMaze protocol has an unknown method")
    _, checkpoint_rank_contract = pointmaze_exogenous_checkpoint_rank_spec(
        checkpoint_rank_mode
    )
    return {
        "protocol_version": POINTMAZE_EXOGENOUS_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_EXOGENOUS_ALGORITHM_PATH,
        "methods": names,
        "environment_id": str(env_id),
        "iterations": int(iterations),
        "horizon": int(horizon),
        "optimizer_seed": int(optimizer_seed),
        "upper_period_seconds": float(upper_period_seconds),
        "history_seconds": float(history_seconds),
        "fast_period_seconds": float(fast_period_seconds),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "target_speed_world_per_second": float(target_speed),
        "force_rms_per_axis": float(force_rms),
        "force_period_seconds": list(map(float, force_period_seconds)),
        "reference_hidden_dim": int(reference_hidden_dim),
        "learning_rate": float(learning_rate),
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
        "checkpoint_rank_mode": str(checkpoint_rank_mode),
        "checkpoint_rank_contract": checkpoint_rank_contract,
        "primary_endpoint": "tracking_success_rate",
        "train_seeds": training,
        "selection_seeds": selection,
        "eval_seeds": evaluation,
        "disabled_legacy_mechanisms": [
            "action_spectrum_projector",
            "promotion",
            "leakage_loss",
            "responsibility_gauge",
            "projection_consistency",
            "multiscale_routing",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the separate-exogenous-stream PointMaze HRL gate."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=POINTMAZE_EXOGENOUS_METHODS,
        default=list(POINTMAZE_EXOGENOUS_METHODS),
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--iterations", type=int, default=768)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--optimizer-seed", type=int, default=134007)
    parser.add_argument("--upper-period-seconds", type=float, default=0.25)
    parser.add_argument("--history-seconds", type=float, default=0.32)
    parser.add_argument("--fast-period-seconds", type=float, default=0.04)
    parser.add_argument("--maximum-subgoal-delta", type=float, default=0.75)
    parser.add_argument("--target-speed", type=float, default=DEFAULT_TARGET_SPEED)
    parser.add_argument("--force-rms", type=float, default=DEFAULT_FORCE_RMS)
    parser.add_argument(
        "--force-period-seconds",
        nargs=2,
        type=float,
        default=list(DEFAULT_FORCE_PERIOD_SECONDS),
    )
    parser.add_argument("--reference-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=16)
    parser.add_argument(
        "--checkpoint-rank-mode",
        choices=POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODES,
        default=DEFAULT_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODE,
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selection-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    force_periods = tuple(map(float, args.force_period_seconds))
    protocol = resolved_pointmaze_exogenous_protocol(
        methods=args.methods,
        env_id=args.env_id,
        iterations=args.iterations,
        horizon=args.horizon,
        optimizer_seed=args.optimizer_seed,
        upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds,
        maximum_subgoal_delta=args.maximum_subgoal_delta,
        target_speed=args.target_speed,
        force_rms=args.force_rms,
        force_period_seconds=force_periods,
        reference_hidden_dim=args.reference_hidden_dim,
        learning_rate=args.learning_rate,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        checkpoint_rank_mode=args.checkpoint_rank_mode,
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
            payload, _, _ = train_pointmaze_exogenous_cell(
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
                target_speed=args.target_speed,
                force_rms=args.force_rms,
                force_period_seconds=force_periods,
                reference_hidden_dim=args.reference_hidden_dim,
                learning_rate=args.learning_rate,
                checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
                checkpoint_rank_mode=args.checkpoint_rank_mode,
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
