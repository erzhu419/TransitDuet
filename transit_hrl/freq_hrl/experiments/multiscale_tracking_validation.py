"""Stage-1 four-grid validation for multiscale goal-conditioned control."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.core import (
    CausalHaarMultiscaleEncoder,
    MultiscaleSnapshot,
    PhysicalTimeScaleContract,
)
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from freq_hrl.domains.tracking import (
    TRACKING_SCENARIOS,
    MultiTimescaleTrackingEnv,
    TrackingObservation,
    tracking_scenario,
)
from freq_hrl.rl import (
    GoalConditionedActorCriticPPO,
    GoalConditionedPPOConfig,
    HierarchicalRolloutBuilder,
    JointActorCriticPPO,
    JointPPOConfig,
    JointTrajectoryBatch,
    summarize_numeric_rows,
    flat_actor_critic_parameter_count,
    matched_hierarchical_hidden_dim,
    train_frequency_separated_ppo,
    train_joint_ppo,
)


MULTISCALE_GOAL_PROTOCOL_VERSION = "multiscale_goal_control_stage1_v1"
STAGE1_CORE_METHODS = (
    "flat_history",
    "flat_multiscale",
    "hrl_history",
    "hrl_multiscale",
)
STAGE1_METHODS = (*STAGE1_CORE_METHODS, "flat_causal_filter")
DEFAULT_TRAIN_SEEDS = (41011, 41017, 41023, 41039)
DEFAULT_SELECTION_SEEDS = (42013, 42017, 42019, 42043)
DEFAULT_EVAL_SEEDS = (43003, 43013, 43019, 43037, 43049, 43051, 43063, 43067)


def _representation(method: str) -> str:
    name = str(method)
    if name not in STAGE1_METHODS:
        raise ValueError(f"unknown stage-1 method: {name}")
    if name == "flat_causal_filter":
        return "filtered"
    return "multiscale" if name.endswith("multiscale") else "history"


def _is_hierarchical(method: str) -> bool:
    name = str(method)
    if name not in STAGE1_METHODS:
        raise ValueError(f"unknown stage-1 method: {name}")
    return name.startswith("hrl_")


@dataclass(frozen=True)
class TrackingFeatureDimensions:
    flat: int
    upper: int
    lower: int
    raw_history: int
    slow: int
    mid: int
    high: int


class TrackingFeatureBuilder:
    """Build fair history/multiscale states from one shared causal window."""

    def __init__(self, *, time_scale: PhysicalTimeScaleContract) -> None:
        self.encoder = CausalHaarMultiscaleEncoder(
            feature_dim=1,
            time_scale=time_scale,
        )
        self._snapshot: MultiscaleSnapshot | None = None

    @property
    def snapshot(self) -> MultiscaleSnapshot:
        if self._snapshot is None:
            raise RuntimeError("feature builder must be reset before use")
        return self._snapshot

    def reset(self, observation: TrackingObservation) -> MultiscaleSnapshot:
        self._validate_observation(observation)
        self._snapshot = self.encoder.reset(observation.task_measurement)
        return self._snapshot

    def update(self, observation: TrackingObservation) -> MultiscaleSnapshot:
        self._validate_observation(observation)
        self._snapshot = self.encoder.update(observation.task_measurement)
        return self._snapshot

    @staticmethod
    def _validate_observation(observation: TrackingObservation) -> None:
        if np.asarray(observation.task_measurement).reshape(-1).shape != (1,):
            raise ValueError(
                "stage-1 protocol requires one actor-visible task channel"
            )

    def dimensions(self, observation: TrackingObservation) -> TrackingFeatureDimensions:
        physical_dim = int(np.asarray(observation.physical).size)
        snapshot = self.snapshot
        return TrackingFeatureDimensions(
            flat=physical_dim + int(snapshot.history.size),
            upper=physical_dim + int(snapshot.slow.size) + int(snapshot.slow_energy.size),
            lower=(
                physical_dim
                + int(observation.achieved_goal.size)
                + int(snapshot.mid.size)
                + int(snapshot.high.size)
            ),
            raw_history=int(snapshot.history.size),
            slow=int(snapshot.slow.size),
            mid=int(snapshot.mid.size),
            high=int(snapshot.high.size),
        )

    def flat_state(self, observation: TrackingObservation, *, representation: str) -> np.ndarray:
        encodings = {
            "history": self.snapshot.history,
            "filtered": self.snapshot.filtered,
            "multiscale": self.snapshot.multiscale,
        }
        if str(representation) not in encodings:
            raise ValueError("unknown tracking representation")
        encoded = encodings[str(representation)]
        return np.concatenate((observation.physical, encoded)).astype(
            np.float32, copy=False
        )

    def upper_state(self, observation: TrackingObservation, *, representation: str) -> np.ndarray:
        if str(representation) == "history":
            task_features = self.snapshot.history
        elif str(representation) == "multiscale":
            task_features = np.concatenate((
                self.snapshot.slow,
                self.snapshot.slow_energy,
            ))
        else:
            raise ValueError("unknown tracking representation")
        return np.concatenate((observation.physical, task_features)).astype(
            np.float32, copy=False
        )

    def lower_state(
        self,
        observation: TrackingObservation,
        *,
        subgoal: np.ndarray,
        representation: str,
    ) -> np.ndarray:
        goal_error = np.asarray(
            subgoal - observation.achieved_goal,
            dtype=np.float32,
        ).reshape(-1)
        if str(representation) == "history":
            task_features = self.snapshot.history
        elif str(representation) == "multiscale":
            task_features = np.concatenate((
                self.snapshot.mid,
                self.snapshot.high,
            ))
        else:
            raise ValueError("unknown tracking representation")
        return np.concatenate((
            observation.physical,
            goal_error,
            task_features,
        )).astype(np.float32, copy=False)


def _episode_row(
    *,
    method: str,
    scenario: str,
    seed: int,
    rewards: list[float],
    tracking_errors: list[float],
    actions: list[float],
    goal_errors: list[float],
    upper_goal_values: list[float],
    environment: MultiTimescaleTrackingEnv,
    parameter_count: int,
    parameter_budget: int,
    upper_decisions: int,
) -> dict[str, Any]:
    reward_array = np.asarray(rewards, dtype=np.float64)
    error_array = np.asarray(tracking_errors, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    goal_error_array = np.asarray(goal_errors, dtype=np.float64)
    upper_goal_array = np.asarray(upper_goal_values, dtype=np.float64)
    signal_metrics = environment.signal_diagnostics()
    time_metrics = environment.time_scale.metadata(
        response_seconds=environment.response_seconds
    )
    observability = environment.observability_contract
    return {
        "protocol_version": MULTISCALE_GOAL_PROTOCOL_VERSION,
        "algorithm_path": "multiscale_goal_hrl_mainline",
        "method": str(method),
        "scenario": str(scenario),
        "seed": int(seed),
        "episode_return": float(np.sum(reward_array)),
        "reward_mean": float(np.mean(reward_array)),
        "tracking_rmse": float(np.sqrt(np.mean(np.square(error_array)))),
        "tracking_mae": float(np.mean(np.abs(error_array))),
        "action_rms": float(np.sqrt(np.mean(np.square(action_array)))),
        "action_saturation_rate": float(np.mean(np.abs(action_array) >= 0.99)),
        "goal_tracking_rmse": (
            float(np.sqrt(np.mean(np.square(goal_error_array))))
            if goal_error_array.size else 0.0
        ),
        "persistent_goal_error_rate": (
            float(np.mean(np.abs(goal_error_array) > 0.5))
            if goal_error_array.size else 0.0
        ),
        "upper_goal_rms": (
            float(np.sqrt(np.mean(np.square(upper_goal_array))))
            if upper_goal_array.size else 0.0
        ),
        "episode_length": int(reward_array.size),
        "upper_decision_count": int(upper_decisions),
        "lower_transition_count": int(reward_array.size),
        "trainable_parameter_count": int(parameter_count),
        "parameter_budget": int(parameter_budget),
        "parameter_budget_ratio": float(parameter_count / parameter_budget),
        "projector_enabled": 0.0,
        "promotion_enabled": 0.0,
        "leakage_loss_enabled": 0.0,
        "responsibility_gauge_enabled": 0.0,
        "task_measurement_available_before_action": float(
            bool(observability["task_measurement_available_before_action"])
        ),
        "true_target_available_to_actor": float(
            bool(observability["true_target_available_to_actor"])
        ),
        "measurement_noise_truth_available_to_actor": float(
            bool(observability["measurement_noise_truth_available_to_actor"])
        ),
        "dynamics_force_available_before_action": float(
            bool(observability["dynamics_force_available_before_action"])
        ),
        "dynamics_force_injection_location": str(
            observability["dynamics_force_injection_location"]
        ),
        "measurement_noise_injection_location": str(
            observability["measurement_noise_injection_location"]
        ),
        "protocol_valid": float(
            reward_array.size == environment.horizon
            and np.all(np.isfinite(reward_array))
            and np.all(np.isfinite(error_array))
            and np.all(np.isfinite(action_array))
            and (
                not _is_hierarchical(method)
                or 0 < int(upper_decisions) < reward_array.size
            )
        ),
        **signal_metrics,
        **time_metrics,
    }


def _make_environment(
    *,
    scenario: str,
    seed: int,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
) -> MultiTimescaleTrackingEnv:
    return MultiTimescaleTrackingEnv(
        time_scale=time_scale,
        scenario=tracking_scenario(scenario, time_scale=time_scale),
        horizon=int(horizon),
        seed=int(seed),
    )


def tracking_feature_dimensions(
    *,
    time_scale: PhysicalTimeScaleContract,
    scenario: str = "clean",
) -> dict[str, TrackingFeatureDimensions]:
    environment = _make_environment(
        scenario=scenario,
        seed=0,
        horizon=max(8, time_scale.history_steps),
        time_scale=time_scale,
    )
    observation = environment.reset()
    builder = TrackingFeatureBuilder(time_scale=time_scale)
    builder.reset(observation)
    history = builder.dimensions(observation)
    return {
        "history": TrackingFeatureDimensions(
            flat=history.flat,
            upper=int(observation.physical.size + builder.snapshot.history.size),
            lower=int(
                observation.physical.size
                + observation.achieved_goal.size
                + builder.snapshot.history.size
            ),
            raw_history=history.raw_history,
            slow=history.slow,
            mid=history.mid,
            high=history.high,
        ),
        "filtered": TrackingFeatureDimensions(
            flat=history.flat,
            upper=int(observation.physical.size + builder.snapshot.history.size),
            lower=int(
                observation.physical.size
                + observation.achieved_goal.size
                + builder.snapshot.history.size
            ),
            raw_history=history.raw_history,
            slow=history.slow,
            mid=history.mid,
            high=history.high,
        ),
        "multiscale": history,
    }


def rollout_flat_tracking(
    model: JointActorCriticPPO,
    *,
    method: str,
    scenario: str,
    seed: int,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    sample: bool,
    parameter_budget: int,
) -> tuple[JointTrajectoryBatch | None, dict[str, Any]]:
    if _is_hierarchical(method):
        raise ValueError("flat rollout requires a flat stage-1 method")
    representation = _representation(method)
    environment = _make_environment(
        scenario=scenario,
        seed=seed,
        horizon=horizon,
        time_scale=time_scale,
    )
    observation = environment.reset()
    features = TrackingFeatureBuilder(time_scale=time_scale)
    features.reset(observation)
    model.reset_recurrent_inference()
    states: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[float] = []
    logps: list[float] = []
    values: list[float] = []
    tracking_errors: list[float] = []
    executed_actions: list[float] = []
    for _ in range(int(horizon)):
        state = features.flat_state(observation, representation=representation)
        output = model.act(state, sample=sample)
        raw_action = np.asarray(output["action"], dtype=np.float32).reshape(-1)
        executed = np.tanh(raw_action).astype(np.float32, copy=False)
        next_observation, reward, terminated, truncated, info = environment.step(
            executed
        )
        done = bool(terminated or truncated)
        states.append(state)
        raw_actions.append(raw_action)
        rewards.append(float(reward))
        dones.append(float(done))
        logps.append(float(output["logp"]))
        values.append(float(output["value"]))
        tracking_errors.append(float(info["tracking_error"]))
        executed_actions.append(float(executed[0]))
        observation = next_observation
        features.update(observation)
        if done:
            break
    parameter_count = int(sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    ))
    row = _episode_row(
        method=method,
        scenario=scenario,
        seed=seed,
        rewards=rewards,
        tracking_errors=tracking_errors,
        actions=executed_actions,
        goal_errors=[],
        upper_goal_values=[],
        environment=environment,
        parameter_count=parameter_count,
        parameter_budget=parameter_budget,
        upper_decisions=0,
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


def rollout_hierarchical_tracking(
    model: GoalConditionedActorCriticPPO,
    *,
    method: str,
    scenario: str,
    seed: int,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    sample: bool,
    parameter_budget: int,
) -> tuple[Any, dict[str, Any]]:
    if not _is_hierarchical(method):
        raise ValueError("hierarchical rollout requires an HRL stage-1 method")
    representation = _representation(method)
    environment = _make_environment(
        scenario=scenario,
        seed=seed,
        horizon=horizon,
        time_scale=time_scale,
    )
    observation = environment.reset()
    features = TrackingFeatureBuilder(time_scale=time_scale)
    features.reset(observation)
    model.reset_recurrent_inference()
    subgoal_adapter = RelativeSubgoalAdapter(
        maximum_delta=np.asarray([1.5], dtype=np.float32),
        action_cost=0.01,
    )
    builder = HierarchicalRolloutBuilder(gamma=float(model.config.gamma))
    rewards: list[float] = []
    tracking_errors: list[float] = []
    executed_actions: list[float] = []
    goal_errors: list[float] = []
    upper_goal_values: list[float] = []
    upper_decisions = 0
    subgoal = observation.achieved_goal.copy()
    for step in range(int(horizon)):
        if step % int(time_scale.upper_period_steps) == 0:
            upper_state = features.upper_state(
                observation,
                representation=representation,
            )
            upper_output = model.plan_goal(upper_state, sample=sample)
            raw_goal = np.asarray(
                upper_output["action"], dtype=np.float32
            ).reshape(-1)
            subgoal = subgoal_adapter.decode(
                raw_goal,
                observation.achieved_goal,
            )
            builder.begin_upper(
                state=upper_state,
                action=raw_goal,
                logp=float(upper_output["logp"]),
                value=float(upper_output["value"]),
            )
            upper_goal_values.append(float(subgoal[0]))
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
        executed = np.tanh(raw_action).astype(np.float32, copy=False)
        achieved_before = observation.achieved_goal.copy()
        next_observation, task_reward, terminated, truncated, info = environment.step(
            executed
        )
        done = bool(terminated or truncated)
        intrinsic_reward = subgoal_adapter.intrinsic_reward(
            achieved_before=achieved_before,
            achieved_after=next_observation.achieved_goal,
            subgoal=subgoal,
            action=executed,
        ) - 0.1 * float(np.mean(np.square(
            subgoal - next_observation.achieved_goal
        )))
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
        rewards.append(float(task_reward))
        tracking_errors.append(float(info["tracking_error"]))
        executed_actions.append(float(executed[0]))
        goal_errors.append(float(
            next_observation.achieved_goal[0] - subgoal[0]
        ))
        observation = next_observation
        features.update(observation)
        if done:
            break
    builder.finish(terminal=True)
    batch = builder.build() if sample else None
    row = _episode_row(
        method=method,
        scenario=scenario,
        seed=seed,
        rewards=rewards,
        tracking_errors=tracking_errors,
        actions=executed_actions,
        goal_errors=goal_errors,
        upper_goal_values=upper_goal_values,
        environment=environment,
        parameter_count=model.trainable_parameter_count,
        parameter_budget=parameter_budget,
        upper_decisions=upper_decisions,
    )
    return batch, row


def _training_seed(
    *,
    optimizer_seed: int,
    rollout_root: int,
    iteration: int,
) -> int:
    return int((
        int(rollout_root) * 1_664_525
        + int(optimizer_seed) * 1_013_904_223
        + (int(iteration) + 1) * 69_069
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
        raise ValueError("each seed role must be non-empty and unique")
    if set(roles[0]) & set(roles[1]) or set(roles[0]) & set(roles[2]) or set(roles[1]) & set(roles[2]):
        raise ValueError("train, selection, and evaluation seeds must be disjoint")
    return roles[0], roles[1], roles[2]


def build_stage1_model(
    *,
    method: str,
    dimensions: dict[str, TrackingFeatureDimensions],
    reference_hidden_dim: int,
    learning_rate: float,
    optimizer_seed: int,
) -> tuple[Any, dict[str, Any]]:
    name = str(method)
    representation = _representation(name)
    target_parameters = flat_actor_critic_parameter_count(
        dimensions["history"].flat,
        1,
        int(reference_hidden_dim),
    )
    torch.manual_seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed) % (2**32 - 1))
    if not _is_hierarchical(name):
        model = JointActorCriticPPO(JointPPOConfig(
            state_dim=dimensions[representation].flat,
            action_dim=1,
            hidden_dim=int(reference_hidden_dim),
            learning_rate=float(learning_rate),
            epochs=4,
            minibatch_size=512,
            init_log_std=-0.7,
        ))
        actual = int(sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ))
        return model, {
            "reference_parameter_budget": target_parameters,
            "actual_parameter_count": actual,
            "parameter_budget_ratio": float(actual / target_parameters),
            "hidden_dim": int(reference_hidden_dim),
        }
    selected = dimensions[representation]
    hidden_dim, expected, ratio = matched_hierarchical_hidden_dim(
        target_parameter_count=target_parameters,
        upper_state_dim=selected.upper,
        lower_state_dim=selected.lower,
        goal_dim=1,
        action_dim=1,
    )
    model = GoalConditionedActorCriticPPO(GoalConditionedPPOConfig(
        upper_state_dim=selected.upper,
        lower_state_dim=selected.lower,
        goal_dim=1,
        action_dim=1,
        hidden_dim=hidden_dim,
        learning_rate=float(learning_rate),
    ))
    actual = model.trainable_parameter_count
    if actual != expected:
        raise RuntimeError("hierarchical parameter accounting changed")
    return model, {
        "reference_parameter_budget": target_parameters,
        "actual_parameter_count": actual,
        "parameter_budget_ratio": ratio,
        "hidden_dim": hidden_dim,
        **model.mainline_contract(),
    }


def train_stage1_cell(
    *,
    method: str,
    scenario: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
    iterations: int,
    horizon: int,
    optimizer_seed: int,
    time_scale: PhysicalTimeScaleContract,
    reference_hidden_dim: int = 64,
    learning_rate: float = 3e-4,
    checkpoint_evaluation_interval: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    if str(scenario) not in TRACKING_SCENARIOS:
        raise ValueError(f"unknown tracking scenario: {scenario}")
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    dimensions = tracking_feature_dimensions(
        time_scale=time_scale,
        scenario=scenario,
    )
    model, capacity = build_stage1_model(
        method=method,
        dimensions=dimensions,
        reference_hidden_dim=reference_hidden_dim,
        learning_rate=learning_rate,
        optimizer_seed=optimizer_seed,
    )
    parameter_budget = int(capacity["reference_parameter_budget"])
    common_metadata = {
        "protocol_version": MULTISCALE_GOAL_PROTOCOL_VERSION,
        "algorithm_path": "multiscale_goal_hrl_mainline",
        "scenario": str(scenario),
        "optimizer_seed": int(optimizer_seed),
        "representation": _representation(method),
        "history_information_contract": (
            "same_fixed_trailing_samples_raw_causal_lowpass_or_orthonormal_haar"
        ),
        "checkpoint_objective": "mean_episode_return",
        "projector": "disabled",
        "promotion": "disabled",
        "leakage_loss": "disabled",
        "responsibility_gauge": "disabled",
        "feature_dimensions": {
            key: value.__dict__ for key, value in dimensions.items()
        },
        "band_layout": CausalHaarMultiscaleEncoder(
            feature_dim=1,
            time_scale=time_scale,
        ).band_layout(),
        "time_scale": time_scale.metadata(),
        "observability_contract": _make_environment(
            scenario=scenario,
            seed=0,
            horizon=max(8, time_scale.history_steps),
            time_scale=time_scale,
        ).observability_contract,
        "capacity": capacity,
    }
    rollout_kwargs = {
        "method": str(method),
        "scenario": str(scenario),
        "horizon": int(horizon),
        "time_scale": time_scale,
        "parameter_budget": parameter_budget,
    }
    seed_fn = lambda root, iteration: _training_seed(
        optimizer_seed=optimizer_seed,
        rollout_root=root,
        iteration=iteration,
    )
    if _is_hierarchical(method):
        payload, rows, trained = train_frequency_separated_ppo(
            model=model,
            train_seeds=training,
            selection_seeds=selection,
            eval_seeds=evaluation,
            iterations=int(iterations),
            rollout_fn=lambda policy, seed, sample: rollout_hierarchical_tracking(
                policy,
                seed=seed,
                sample=sample,
                **rollout_kwargs,
            ),
            objective_fn=lambda row: float(row["episode_return"]),
            summary_fn=summarize_numeric_rows,
            training_seed_fn=seed_fn,
            policy=str(method),
            domain="identifiable_tracking",
            metadata=common_metadata,
            checkpoint_score_contract="mean_episode_return",
            checkpoint_evaluation_interval=int(
                checkpoint_evaluation_interval
            ),
        )
    else:
        payload, rows, trained = train_joint_ppo(
            model=model,
            train_seeds=training,
            selection_seeds=selection,
            eval_seeds=evaluation,
            iterations=int(iterations),
            rollout_fn=lambda policy, seed, sample: rollout_flat_tracking(
                policy,
                seed=seed,
                sample=sample,
                **rollout_kwargs,
            ),
            objective_fn=lambda row: float(row["episode_return"]),
            summary_fn=summarize_numeric_rows,
            training_seed_fn=seed_fn,
            policy=str(method),
            domain="identifiable_tracking",
            metadata=common_metadata,
            checkpoint_score_contract="mean_episode_return",
            checkpoint_evaluation_interval=int(
                checkpoint_evaluation_interval
            ),
        )
    if _is_hierarchical(method):
        payload["training_core"] = payload["trainer"]
        payload["trainer"] = "goal_conditioned_smdp_ppo_v1"
    for row in rows:
        row["training_replicate_seed"] = int(optimizer_seed)
    payload["optimizer_seed"] = int(optimizer_seed)
    payload["evaluation_rows"] = rows
    return payload, rows, trained


def resolved_stage1_protocol(
    *,
    methods: Iterable[str],
    scenarios: Iterable[str],
    iterations: int,
    horizon: int,
    optimizer_seed: int,
    time_scale: PhysicalTimeScaleContract,
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
    scenario_names = list(map(str, scenarios))
    if not method_names or any(name not in STAGE1_METHODS for name in method_names):
        raise ValueError("resolved protocol contains an unknown or empty method set")
    if not scenario_names or any(name not in TRACKING_SCENARIOS for name in scenario_names):
        raise ValueError("resolved protocol contains an unknown or empty scenario set")
    return {
        "protocol_version": MULTISCALE_GOAL_PROTOCOL_VERSION,
        "methods": method_names,
        "factorial_core_methods": list(STAGE1_CORE_METHODS),
        "auxiliary_baselines": ["flat_causal_filter"],
        "scenarios": scenario_names,
        "iterations": int(iterations),
        "horizon": int(horizon),
        "optimizer_seed": int(optimizer_seed),
        "reference_hidden_dim": int(reference_hidden_dim),
        "learning_rate": float(learning_rate),
        "checkpoint_evaluation_interval": int(
            checkpoint_evaluation_interval
        ),
        "train_seeds": training,
        "selection_seeds": selection,
        "eval_seeds": evaluation,
        "time_scale": time_scale.metadata(),
        "disabled_legacy_mechanisms": [
            "action_spectrum_projector",
            "promotion",
            "leakage_loss",
            "responsibility_gauge",
            "projection_consistency",
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
        description="Run the Freq-HRL multiscale goal-control stage-1 matrix."
    )
    parser.add_argument("--methods", nargs="+", choices=STAGE1_METHODS, default=list(STAGE1_METHODS))
    parser.add_argument("--scenarios", nargs="+", choices=TRACKING_SCENARIOS, default=list(TRACKING_SCENARIOS))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--optimizer-seed", type=int, default=44017)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--selection-seeds", nargs="+", type=int, default=list(DEFAULT_SELECTION_SEEDS))
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--dt-seconds", type=float, default=0.05)
    parser.add_argument("--upper-period-seconds", type=float, default=0.8)
    parser.add_argument("--history-seconds", type=float, default=3.2)
    parser.add_argument("--fast-period-seconds", type=float, default=0.2)
    parser.add_argument("--reference-hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=args.dt_seconds,
        upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds,
    )
    protocol = resolved_stage1_protocol(
        methods=args.methods,
        scenarios=args.scenarios,
        iterations=args.iterations,
        horizon=args.horizon,
        optimizer_seed=args.optimizer_seed,
        time_scale=time_scale,
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
        for scenario in args.scenarios:
            for method in args.methods:
                payload, _, _ = train_stage1_cell(
                    method=method,
                    scenario=scenario,
                    train_seeds=args.train_seeds,
                    selection_seeds=args.selection_seeds,
                    eval_seeds=args.eval_seeds,
                    iterations=args.iterations,
                    horizon=args.horizon,
                    optimizer_seed=args.optimizer_seed,
                    time_scale=time_scale,
                    reference_hidden_dim=args.reference_hidden_dim,
                    learning_rate=args.learning_rate,
                    checkpoint_evaluation_interval=(
                        args.checkpoint_evaluation_interval
                    ),
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
