"""Stage-3 PointMaze factorial for causal multiscale Freq-HRL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.core import CausalHaarMultiscaleEncoder, PhysicalTimeScaleContract
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

from .pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    POINTMAZE_CHECKPOINT_RANK_CONTRACT,
    POINTMAZE_CHECKPOINT_RANK_NAMES,
    POINTMAZE_CONTINUING_TASK,
    POINTMAZE_LOWER_ACTION_COST,
    compact_pointmaze_history,
    make_pointmaze_environment,
    pointmaze_checkpoint_rank,
    pointmaze_goal_bounds,
    pointmaze_runtime_versions,
    squash_box_action,
)


POINTMAZE_MULTISCALE_PROTOCOL_V1 = "pointmaze_multiscale_goal_stage3_v1"
POINTMAZE_MULTISCALE_PROTOCOL_V2 = "pointmaze_multiscale_goal_stage3_v2"
POINTMAZE_MULTISCALE_PROTOCOL_VERSION = POINTMAZE_MULTISCALE_PROTOCOL_V2
POINTMAZE_MULTISCALE_CORE_METHODS = (
    "flat_history",
    "flat_multiscale",
    "hrl_history",
    "hrl_multiscale",
)
POINTMAZE_MULTISCALE_AUXILIARY_METHODS = ("flat_causal_filter",)
POINTMAZE_MULTISCALE_METHODS = (
    *POINTMAZE_MULTISCALE_CORE_METHODS,
    *POINTMAZE_MULTISCALE_AUXILIARY_METHODS,
)
POINTMAZE_MULTISCALE_SCENARIOS = (
    "clean",
    "fast_observation_noise",
    "slow_drift_fast_action",
    "persistent_action_shift",
)
POINTMAZE_MULTISCALE_ALGORITHM_PATH = "pointmaze_multiscale_goal_hrl_mainline"


def _representation(method: str) -> str:
    name = str(method)
    if name not in POINTMAZE_MULTISCALE_METHODS:
        raise ValueError(f"unknown PointMaze multiscale method: {name}")
    if name == "flat_causal_filter":
        return "filtered"
    return "multiscale" if name.endswith("multiscale") else "history"


def _is_hierarchical(method: str) -> bool:
    name = str(method)
    if name not in POINTMAZE_MULTISCALE_METHODS:
        raise ValueError(f"unknown PointMaze multiscale method: {name}")
    return name.startswith("hrl_")


@dataclass(frozen=True)
class PointMazeStressSpec:
    position_noise_std: float
    velocity_noise_std: float
    slow_action_std: float
    slow_action_time_constant_seconds: float
    fast_action_noise_std: float
    persistent_action_shift_amplitude: float
    persistent_action_shift_onset_fraction: float


POINTMAZE_STRESS_SPECS = {
    "clean": PointMazeStressSpec(
        position_noise_std=0.0,
        velocity_noise_std=0.0,
        slow_action_std=0.0,
        slow_action_time_constant_seconds=1.0,
        fast_action_noise_std=0.0,
        persistent_action_shift_amplitude=0.0,
        persistent_action_shift_onset_fraction=0.35,
    ),
    "fast_observation_noise": PointMazeStressSpec(
        position_noise_std=0.06,
        velocity_noise_std=0.08,
        slow_action_std=0.0,
        slow_action_time_constant_seconds=1.0,
        fast_action_noise_std=0.0,
        persistent_action_shift_amplitude=0.0,
        persistent_action_shift_onset_fraction=0.35,
    ),
    "slow_drift_fast_action": PointMazeStressSpec(
        position_noise_std=0.0,
        velocity_noise_std=0.0,
        slow_action_std=0.12,
        slow_action_time_constant_seconds=1.0,
        fast_action_noise_std=0.08,
        persistent_action_shift_amplitude=0.0,
        persistent_action_shift_onset_fraction=0.35,
    ),
    "persistent_action_shift": PointMazeStressSpec(
        position_noise_std=0.0,
        velocity_noise_std=0.0,
        slow_action_std=0.0,
        slow_action_time_constant_seconds=1.0,
        fast_action_noise_std=0.0,
        persistent_action_shift_amplitude=0.18,
        persistent_action_shift_onset_fraction=0.35,
    ),
}


class CausalPointMazeStress:
    """Paired hidden stress with no actor access to disturbance truth."""

    def __init__(
        self,
        *,
        scenario: str,
        seed: int,
        dt_seconds: float,
        physical_dim: int,
        goal_dim: int,
        action_dim: int,
        horizon_steps: int,
    ) -> None:
        if str(scenario) not in POINTMAZE_STRESS_SPECS:
            raise ValueError(f"unknown PointMaze stress scenario: {scenario}")
        if physical_dim < goal_dim or action_dim < 1:
            raise ValueError("invalid PointMaze stress dimensions")
        self.scenario = str(scenario)
        self.spec = POINTMAZE_STRESS_SPECS[self.scenario]
        self.physical_dim = int(physical_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.horizon_steps = int(horizon_steps)
        if self.horizon_steps < 2:
            raise ValueError("PointMaze stress horizon must span at least two steps")
        self.dt_seconds = float(dt_seconds)
        if not np.isfinite(self.dt_seconds) or self.dt_seconds <= 0.0:
            raise ValueError("PointMaze stress dt must be positive and finite")
        seed_value = int(seed) % (2**32 - 1)
        self._observation_rng = np.random.default_rng(
            (seed_value * 1_000_003 + 17_171) % (2**32 - 1)
        )
        self._action_rng = np.random.default_rng(
            (seed_value * 1_000_033 + 31_337) % (2**32 - 1)
        )
        self._slow_rho = float(np.exp(
            -self.dt_seconds / self.spec.slow_action_time_constant_seconds
        ))
        self._slow_action = self._action_rng.normal(
            0.0,
            self.spec.slow_action_std,
            size=self.action_dim,
        ).astype(np.float32)
        direction = self._action_rng.normal(size=self.action_dim)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            direction = np.ones(self.action_dim, dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction))
        self._persistent_direction = np.asarray(
            direction / direction_norm, dtype=np.float32
        )
        self._persistent_onset_step = int(np.clip(
            round(
                self.spec.persistent_action_shift_onset_fraction
                * self.horizon_steps
            ),
            1,
            self.horizon_steps - 1,
        ))
        self._action_step = 0

    def observe(self, truth: Any) -> tuple[dict[str, np.ndarray], np.ndarray]:
        parsed = parse_goal_observation(truth)
        if parsed.physical.size != self.physical_dim:
            raise ValueError("PointMaze physical observation dimension changed")
        position_noise = self._observation_rng.normal(
            0.0, self.spec.position_noise_std, size=self.goal_dim
        )
        velocity_noise = self._observation_rng.normal(
            0.0,
            self.spec.velocity_noise_std,
            size=self.physical_dim - self.goal_dim,
        )
        measurement_noise = np.concatenate(
            (position_noise, velocity_noise)
        ).astype(np.float32, copy=False)
        visible_physical = (
            parsed.physical.astype(np.float32, copy=True) + measurement_noise
        )
        visible = {
            "observation": visible_physical,
            "achieved_goal": visible_physical[: self.goal_dim].copy(),
            "desired_goal": parsed.desired_goal.astype(np.float32, copy=True),
        }
        return visible, measurement_noise

    def execute(
        self,
        requested_action: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        requested = np.asarray(requested_action, dtype=np.float32).reshape(-1)
        if requested.shape != (self.action_dim,):
            raise ValueError("PointMaze requested action dimension changed")
        innovation_std = self.spec.slow_action_std * np.sqrt(
            max(0.0, 1.0 - self._slow_rho**2)
        )
        self._slow_action = (
            self._slow_rho * self._slow_action
            + self._action_rng.normal(
                0.0, innovation_std, size=self.action_dim
            )
        ).astype(np.float32, copy=False)
        fast_action = self._action_rng.normal(
            0.0, self.spec.fast_action_noise_std, size=self.action_dim
        ).astype(np.float32, copy=False)
        persistent_action = (
            self.spec.persistent_action_shift_amplitude
            * self._persistent_direction
            if self._action_step >= self._persistent_onset_step
            else np.zeros(self.action_dim, dtype=np.float32)
        ).astype(np.float32, copy=False)
        executed = np.clip(
            requested + self._slow_action + fast_action + persistent_action,
            np.asarray(low, dtype=np.float32),
            np.asarray(high, dtype=np.float32),
        ).astype(np.float32, copy=False)
        self._action_step += 1
        return (
            executed,
            self._slow_action.copy(),
            fast_action,
            persistent_action.copy(),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            **asdict(self.spec),
            "stress_visibility": "hidden_from_actor",
            "measurement_timing": "current_truth_plus_current_noise_before_action",
            "action_stress_timing": "current_hidden_disturbance_after_actor_action",
            "slow_action_ar_coefficient": self._slow_rho,
            "persistent_action_shift_onset_step": self._persistent_onset_step,
            "persistent_action_shift_onset_seconds": (
                self._persistent_onset_step * self.dt_seconds
            ),
        }


@dataclass(frozen=True)
class PointMazeMultiscaleDimensions:
    flat: int
    upper: int
    lower: int
    physical: int
    goal: int
    action: int
    history: int
    slow: int
    mid: int
    high: int
    slow_energy: int


class PointMazeFeatureBuilder:
    """Construct raw-history or causal Haar states from one shared window."""

    def __init__(
        self,
        *,
        physical_dim: int,
        time_scale: PhysicalTimeScaleContract,
    ) -> None:
        self.encoder = CausalHaarMultiscaleEncoder(
            feature_dim=int(physical_dim),
            time_scale=time_scale,
        )
        self._snapshot = None

    @property
    def snapshot(self):
        if self._snapshot is None:
            raise RuntimeError("PointMaze feature builder must be reset")
        return self._snapshot

    def reset(self, observation: Any):
        self._snapshot = self.encoder.reset(
            parse_goal_observation(observation).physical
        )
        return self._snapshot

    def update(self, observation: Any):
        self._snapshot = self.encoder.update(
            parse_goal_observation(observation).physical
        )
        return self._snapshot

    def dimensions(
        self,
        observation: Any,
        *,
        action_dim: int,
        representation: str,
    ) -> PointMazeMultiscaleDimensions:
        parsed = parse_goal_observation(observation)
        snapshot = self.snapshot
        physical = int(parsed.physical.size)
        goal = int(parsed.achieved_goal.size)
        if representation in ("history", "filtered"):
            flat = physical + goal + int(snapshot.history.size)
            upper = physical + goal + int(snapshot.history.size)
            lower = physical + goal + int(snapshot.history.size)
        elif representation in ("multiscale", "multiscale_routed"):
            flat = physical + goal + int(snapshot.multiscale.size)
            upper = (
                physical
                + goal
                + int(snapshot.slow.size)
                + int(snapshot.mid.size)
            )
            lower = (
                physical + goal + int(snapshot.mid.size) + int(snapshot.high.size)
            )
        elif representation in (
            "multiscale_all",
            "multiscale_routed_masked",
            "multiscale_swapped_masked",
        ):
            flat = physical + goal + int(snapshot.multiscale.size)
            upper = physical + goal + int(snapshot.multiscale.size)
            lower = physical + goal + int(snapshot.multiscale.size)
        elif representation == "multiscale_swapped":
            flat = physical + goal + int(snapshot.multiscale.size)
            upper = (
                physical + goal + int(snapshot.mid.size) + int(snapshot.high.size)
            )
            lower = (
                physical + goal + int(snapshot.slow.size) + int(snapshot.mid.size)
            )
        else:
            raise ValueError("unknown PointMaze feature representation")
        return PointMazeMultiscaleDimensions(
            flat=flat,
            upper=upper,
            lower=lower,
            physical=physical,
            goal=goal,
            action=int(action_dim),
            history=int(snapshot.history.size),
            slow=int(snapshot.slow.size),
            mid=int(snapshot.mid.size),
            high=int(snapshot.high.size),
            slow_energy=int(snapshot.slow_energy.size),
        )

    def flat_state(self, observation: Any, *, representation: str) -> np.ndarray:
        parsed = parse_goal_observation(observation)
        encoded = (
            self.snapshot.history
            if representation == "history"
            else self.snapshot.filtered
            if representation == "filtered"
            else self.snapshot.multiscale
            if representation.startswith("multiscale")
            else None
        )
        if encoded is None:
            raise ValueError("unknown PointMaze flat representation")
        return np.concatenate(
            (parsed.physical, parsed.goal_error, encoded)
        ).astype(np.float32, copy=False)

    def upper_state(self, observation: Any, *, representation: str) -> np.ndarray:
        parsed = parse_goal_observation(observation)
        if representation == "history":
            task_features = self.snapshot.history
        elif representation == "filtered":
            task_features = self.snapshot.filtered
        elif representation in ("multiscale", "multiscale_routed"):
            task_features = np.concatenate(
                (self.snapshot.slow, self.snapshot.mid)
            )
        elif representation == "multiscale_all":
            task_features = self.snapshot.multiscale
        elif representation == "multiscale_routed_masked":
            task_features = np.concatenate((
                self.snapshot.slow,
                self.snapshot.mid,
                np.zeros_like(self.snapshot.high),
            ))
        elif representation == "multiscale_swapped_masked":
            task_features = np.concatenate((
                np.zeros_like(self.snapshot.slow),
                self.snapshot.mid,
                self.snapshot.high,
            ))
        elif representation == "multiscale_swapped":
            task_features = np.concatenate(
                (self.snapshot.mid, self.snapshot.high)
            )
        else:
            raise ValueError("unknown PointMaze upper representation")
        return np.concatenate(
            (parsed.physical, parsed.goal_error, task_features)
        ).astype(np.float32, copy=False)

    def lower_state(
        self,
        observation: Any,
        *,
        subgoal: np.ndarray,
        representation: str,
    ) -> np.ndarray:
        parsed = parse_goal_observation(observation)
        subgoal_error = np.asarray(
            np.asarray(subgoal, dtype=np.float32).reshape(-1)
            - parsed.achieved_goal,
            dtype=np.float32,
        )
        if subgoal_error.shape != parsed.achieved_goal.shape:
            raise ValueError("PointMaze subgoal dimension mismatch")
        if representation == "history":
            task_features = self.snapshot.history
        elif representation == "filtered":
            task_features = self.snapshot.filtered
        elif representation in ("multiscale", "multiscale_routed"):
            task_features = np.concatenate(
                (self.snapshot.mid, self.snapshot.high)
            )
        elif representation == "multiscale_all":
            task_features = self.snapshot.multiscale
        elif representation == "multiscale_routed_masked":
            task_features = np.concatenate((
                np.zeros_like(self.snapshot.slow),
                self.snapshot.mid,
                self.snapshot.high,
            ))
        elif representation == "multiscale_swapped_masked":
            task_features = np.concatenate((
                self.snapshot.slow,
                self.snapshot.mid,
                np.zeros_like(self.snapshot.high),
            ))
        elif representation == "multiscale_swapped":
            task_features = np.concatenate(
                (self.snapshot.slow, self.snapshot.mid)
            )
        else:
            raise ValueError("unknown PointMaze lower representation")
        return np.concatenate(
            (parsed.physical, subgoal_error, task_features)
        ).astype(np.float32, copy=False)


def pointmaze_multiscale_dimensions(
    *,
    env_id: str,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    representations: Iterable[str] = ("history", "filtered", "multiscale"),
) -> dict[str, PointMazeMultiscaleDimensions]:
    environment = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        observation, _ = environment.reset(seed=0)
        parsed = parse_goal_observation(observation)
        action_dim = int(np.prod(environment.action_space.shape))
        builder = PointMazeFeatureBuilder(
            physical_dim=int(parsed.physical.size),
            time_scale=time_scale,
        )
        builder.reset(observation)
        dimensions = {
            representation: builder.dimensions(
                observation,
                action_dim=action_dim,
                representation=representation,
            )
            for representation in map(str, representations)
        }
    finally:
        environment.close()
    if not dimensions or len({value.flat for value in dimensions.values()}) != 1:
        raise RuntimeError(
            "raw, causal-filter, and Haar flat states must have equal size"
        )
    return dimensions


def _episode_row(
    *,
    method: str,
    scenario: str,
    seed: int,
    rewards: list[float],
    goal_distances: list[float],
    requested_actions: list[np.ndarray],
    executed_actions: list[np.ndarray],
    measurement_noise: list[np.ndarray],
    slow_action_stress: list[np.ndarray],
    fast_action_stress: list[np.ndarray],
    persistent_action_stress: list[np.ndarray],
    path_length: float,
    success: bool,
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
    protocol_version: str = POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
    algorithm_path: str = POINTMAZE_MULTISCALE_ALGORITHM_PATH,
    representation_override: str | None = None,
    hierarchical_override: bool | None = None,
) -> dict[str, Any]:
    reward_array = np.asarray(rewards, dtype=np.float64)
    distance_array = np.asarray(goal_distances, dtype=np.float64)
    requested_array = np.asarray(requested_actions, dtype=np.float64)
    executed_array = np.asarray(executed_actions, dtype=np.float64)
    measurement_array = np.asarray(measurement_noise, dtype=np.float64)
    slow_array = np.asarray(slow_action_stress, dtype=np.float64)
    fast_array = np.asarray(fast_action_stress, dtype=np.float64)
    persistent_array = np.asarray(persistent_action_stress, dtype=np.float64)
    subgoal_distance_array = np.asarray(subgoal_distances, dtype=np.float64)
    subgoal_array = np.asarray(subgoals, dtype=np.float64)
    intrinsic_array = np.asarray(intrinsic_rewards, dtype=np.float64)
    hierarchical = (
        _is_hierarchical(method)
        if hierarchical_override is None
        else bool(hierarchical_override)
    )
    representation = (
        _representation(method)
        if representation_override is None
        else str(representation_override)
    )
    protocol_valid = bool(
        reward_array.size > 0
        and reward_array.size
        == distance_array.size
        == requested_array.shape[0]
        == executed_array.shape[0]
        == measurement_array.shape[0]
        == slow_array.shape[0]
        == fast_array.shape[0]
        == persistent_array.shape[0]
        and all(np.all(np.isfinite(array)) for array in (
            reward_array,
            distance_array,
            requested_array,
            executed_array,
            measurement_array,
            slow_array,
            fast_array,
            persistent_array,
        ))
        and (
            not hierarchical
            or (
                0 < upper_decisions < reward_array.size
                and intrinsic_array.size == reward_array.size
                and np.all(np.isfinite(intrinsic_array))
                and 0 < lower_option_boundaries <= upper_decisions
            )
        )
    )
    execution_delta = executed_array - requested_array
    return {
        "protocol_version": str(protocol_version),
        "algorithm_path": str(algorithm_path),
        "method": str(method),
        "scenario": str(scenario),
        "representation": representation,
        "hierarchical": float(hierarchical),
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
        "requested_action_rms": float(np.sqrt(np.mean(np.square(requested_array)))),
        "executed_action_rms": float(np.sqrt(np.mean(np.square(executed_array)))),
        "action_saturation_rate": float(np.mean(np.abs(executed_array) >= 0.99)),
        "execution_delta_rms": float(np.sqrt(np.mean(np.square(execution_delta)))),
        "measurement_noise_rms": float(np.sqrt(np.mean(np.square(measurement_array)))),
        "slow_action_stress_rms": float(np.sqrt(np.mean(np.square(slow_array)))),
        "fast_action_stress_rms": float(np.sqrt(np.mean(np.square(fast_array)))),
        "persistent_action_stress_rms": float(
            np.sqrt(np.mean(np.square(persistent_array)))
        ),
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
        "lower_intrinsic_return": (
            float(np.sum(intrinsic_array)) if intrinsic_array.size else 0.0
        ),
        "lower_intrinsic_reward_mean": (
            float(np.mean(intrinsic_array)) if intrinsic_array.size else 0.0
        ),
        "lower_option_boundary_count": int(lower_option_boundaries),
        **time_scale.metadata(),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "trainable_parameter_count": int(parameter_count),
        "parameter_budget": int(parameter_budget),
        "parameter_budget_ratio": float(parameter_count / parameter_budget),
        "stress_truth_available_to_actor": 0.0,
        "projector_enabled": 0.0,
        "promotion_enabled": 0.0,
        "leakage_loss_enabled": 0.0,
        "responsibility_gauge_enabled": 0.0,
        "protocol_valid": float(protocol_valid),
    }


def _stress_for_episode(
    *,
    scenario: str,
    seed: int,
    timing: Any,
    observation: Any,
    action_dim: int,
    horizon: int,
) -> CausalPointMazeStress:
    parsed = parse_goal_observation(observation)
    return CausalPointMazeStress(
        scenario=scenario,
        seed=seed,
        dt_seconds=float(timing.control_dt_seconds),
        physical_dim=int(parsed.physical.size),
        goal_dim=int(parsed.achieved_goal.size),
        action_dim=int(action_dim),
        horizon_steps=int(horizon),
    )


def rollout_flat_pointmaze_multiscale(
    model: JointActorCriticPPO,
    *,
    method: str,
    scenario: str,
    env_id: str,
    seed: int,
    horizon: int,
    sample: bool,
    parameter_budget: int,
    time_scale: PhysicalTimeScaleContract,
) -> tuple[JointTrajectoryBatch | None, dict[str, Any]]:
    if _is_hierarchical(method):
        raise ValueError("flat PointMaze rollout received an HRL method")
    representation = _representation(method)
    environment = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        truth, _ = environment.reset(seed=int(seed))
        timing = environment_timing(environment)
        action_low = np.asarray(environment.action_space.low, dtype=np.float32)
        action_high = np.asarray(environment.action_space.high, dtype=np.float32)
        stress = _stress_for_episode(
            scenario=scenario,
            seed=seed,
            timing=timing,
            observation=truth,
            action_dim=int(action_low.size),
            horizon=horizon,
        )
        visible, _ = stress.observe(truth)
        features = PointMazeFeatureBuilder(
            physical_dim=int(parse_goal_observation(visible).physical.size),
            time_scale=time_scale,
        )
        features.reset(visible)
        model.reset_recurrent_inference()
        states: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[float] = []
        logps: list[float] = []
        values: list[float] = []
        distances: list[float] = []
        requested_actions: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        measurement_noise: list[np.ndarray] = []
        slow_action_stress: list[np.ndarray] = []
        fast_action_stress: list[np.ndarray] = []
        persistent_action_stress: list[np.ndarray] = []
        path_length = 0.0
        success = False
        terminated = truncated = False
        achieved_before_true = parse_goal_observation(truth).achieved_goal
        for _ in range(int(horizon)):
            state = features.flat_state(visible, representation=representation)
            output = model.act(state, sample=sample)
            raw_action = np.asarray(output["action"], dtype=np.float32).reshape(-1)
            requested = squash_box_action(raw_action, action_low, action_high)
            (
                executed,
                slow_stress,
                fast_stress,
                persistent_stress,
            ) = stress.execute(requested, action_low, action_high)
            next_truth, reward, terminated, truncated, info = environment.step(executed)
            done = bool(terminated or truncated)
            next_visible, noise = stress.observe(next_truth)
            parsed_next_true = parse_goal_observation(next_truth)
            distance = float(np.linalg.norm(parsed_next_true.goal_error))
            path_length += float(np.linalg.norm(
                parsed_next_true.achieved_goal - achieved_before_true
            ))
            achieved_before_true = parsed_next_true.achieved_goal
            success = bool(success or info.get("success", False))
            states.append(state)
            raw_actions.append(raw_action)
            rewards.append(float(reward))
            dones.append(float(done))
            logps.append(float(output["logp"]))
            values.append(float(output["value"]))
            distances.append(distance)
            requested_actions.append(requested)
            executed_actions.append(executed)
            measurement_noise.append(noise)
            slow_action_stress.append(slow_stress)
            fast_action_stress.append(fast_stress)
            persistent_action_stress.append(persistent_stress)
            truth, visible = next_truth, next_visible
            features.update(visible)
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
            goal_distances=distances,
            requested_actions=requested_actions,
            executed_actions=executed_actions,
            measurement_noise=measurement_noise,
            slow_action_stress=slow_action_stress,
            fast_action_stress=fast_action_stress,
            persistent_action_stress=persistent_action_stress,
            path_length=path_length,
            success=success,
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


def rollout_hrl_pointmaze_multiscale(
    model: GoalConditionedActorCriticPPO,
    *,
    method: str,
    scenario: str,
    env_id: str,
    seed: int,
    horizon: int,
    sample: bool,
    parameter_budget: int,
    time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float,
    representation_override: str | None = None,
    protocol_version: str = POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
    algorithm_path: str = POINTMAZE_MULTISCALE_ALGORITHM_PATH,
) -> tuple[Any, dict[str, Any]]:
    if representation_override is None:
        if not _is_hierarchical(method):
            raise ValueError("HRL PointMaze rollout received a flat method")
        representation = _representation(method)
    else:
        if not str(method).startswith("hrl_"):
            raise ValueError("HRL PointMaze rollout method must be hierarchical")
        representation = str(representation_override)
    environment = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        truth, _ = environment.reset(seed=int(seed))
        timing = environment_timing(environment)
        action_low = np.asarray(environment.action_space.low, dtype=np.float32)
        action_high = np.asarray(environment.action_space.high, dtype=np.float32)
        world_low, world_high = pointmaze_goal_bounds(environment)
        parsed_truth = parse_goal_observation(truth)
        goal_dim = int(parsed_truth.achieved_goal.size)
        maximum_delta = np.full(
            goal_dim, float(maximum_subgoal_delta), dtype=np.float32
        )
        subgoal_adapter = RelativeSubgoalAdapter(
            maximum_delta=maximum_delta,
            world_low=world_low,
            world_high=world_high,
            action_cost=POINTMAZE_LOWER_ACTION_COST,
        )
        stress = _stress_for_episode(
            scenario=scenario,
            seed=seed,
            timing=timing,
            observation=truth,
            action_dim=int(action_low.size),
            horizon=horizon,
        )
        visible, _ = stress.observe(truth)
        features = PointMazeFeatureBuilder(
            physical_dim=int(parse_goal_observation(visible).physical.size),
            time_scale=time_scale,
        )
        features.reset(visible)
        model.reset_recurrent_inference()
        builder = HierarchicalRolloutBuilder(gamma=float(model.config.gamma))
        rewards: list[float] = []
        distances: list[float] = []
        requested_actions: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        measurement_noise: list[np.ndarray] = []
        slow_action_stress: list[np.ndarray] = []
        fast_action_stress: list[np.ndarray] = []
        persistent_action_stress: list[np.ndarray] = []
        subgoal_distances: list[float] = []
        subgoals: list[np.ndarray] = []
        intrinsic_rewards: list[float] = []
        path_length = 0.0
        success = False
        terminated = truncated = False
        upper_decisions = 0
        lower_option_boundaries = 0
        last_lower_terminal = False
        achieved_before_true = parsed_truth.achieved_goal
        achieved_before_visible = parse_goal_observation(visible).achieved_goal
        subgoal = achieved_before_visible.copy()
        upper_period_steps = int(time_scale.upper_period_steps)
        for step in range(int(horizon)):
            if step % upper_period_steps == 0:
                upper_state = features.upper_state(
                    visible, representation=representation
                )
                upper_output = model.plan_goal(upper_state, sample=sample)
                raw_goal = np.asarray(
                    upper_output["action"], dtype=np.float32
                ).reshape(-1)
                subgoal = subgoal_adapter.decode(
                    raw_goal, achieved_before_visible
                )
                builder.begin_upper(
                    state=upper_state,
                    action=raw_goal,
                    logp=float(upper_output["logp"]),
                    value=float(upper_output["value"]),
                )
                subgoals.append(subgoal.copy())
                upper_decisions += 1
            lower_state = features.lower_state(
                visible,
                subgoal=subgoal,
                representation=representation,
            )
            lower_output = model.act_conditioned(lower_state, sample=sample)
            raw_action = np.asarray(
                lower_output["action"], dtype=np.float32
            ).reshape(-1)
            requested = squash_box_action(raw_action, action_low, action_high)
            (
                executed,
                slow_stress,
                fast_stress,
                persistent_stress,
            ) = stress.execute(requested, action_low, action_high)
            next_truth, task_reward, terminated, truncated, info = environment.step(executed)
            done = bool(terminated or truncated)
            next_visible, noise = stress.observe(next_truth)
            parsed_next_true = parse_goal_observation(next_truth)
            parsed_next_visible = parse_goal_observation(next_visible)
            subgoal_distance = float(np.linalg.norm(
                (subgoal - parsed_next_visible.achieved_goal) / maximum_delta
            ) / np.sqrt(goal_dim))
            intrinsic_reward = subgoal_adapter.intrinsic_reward(
                achieved_before=achieved_before_visible,
                achieved_after=parsed_next_visible.achieved_goal,
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
            goal_distance = float(np.linalg.norm(parsed_next_true.goal_error))
            path_length += float(np.linalg.norm(
                parsed_next_true.achieved_goal - achieved_before_true
            ))
            success = bool(success or info.get("success", False))
            rewards.append(float(task_reward))
            distances.append(goal_distance)
            requested_actions.append(requested)
            executed_actions.append(executed)
            measurement_noise.append(noise)
            slow_action_stress.append(slow_stress)
            fast_action_stress.append(fast_stress)
            persistent_action_stress.append(persistent_stress)
            subgoal_distances.append(subgoal_distance)
            intrinsic_rewards.append(float(intrinsic_reward))
            achieved_before_true = parsed_next_true.achieved_goal
            achieved_before_visible = parsed_next_visible.achieved_goal
            truth, visible = next_truth, next_visible
            features.update(visible)
            if done:
                break
        if rewards and not last_lower_terminal:
            lower_option_boundaries += 1
        builder.finish(terminal=True)
        batch = builder.build() if sample else None
        row = _episode_row(
            method=method,
            scenario=scenario,
            seed=seed,
            rewards=rewards,
            goal_distances=distances,
            requested_actions=requested_actions,
            executed_actions=executed_actions,
            measurement_noise=measurement_noise,
            slow_action_stress=slow_action_stress,
            fast_action_stress=fast_action_stress,
            persistent_action_stress=persistent_action_stress,
            path_length=path_length,
            success=success,
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
            protocol_version=protocol_version,
            algorithm_path=algorithm_path,
            representation_override=representation,
            hierarchical_override=True,
        )
        return batch, row
    finally:
        environment.close()


def build_pointmaze_multiscale_model(
    *,
    method: str,
    dimensions: dict[str, PointMazeMultiscaleDimensions],
    reference_hidden_dim: int,
    learning_rate: float,
    optimizer_seed: int,
) -> tuple[Any, dict[str, Any]]:
    name = str(method)
    representation = _representation(name)
    reference = dimensions["history"]
    target_parameters = flat_actor_critic_parameter_count(
        reference.flat,
        reference.action,
        int(reference_hidden_dim),
    )
    torch.manual_seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed) % (2**32 - 1))
    selected = dimensions[representation]
    if not _is_hierarchical(name):
        model = JointActorCriticPPO(JointPPOConfig(
            state_dim=selected.flat,
            action_dim=selected.action,
            hidden_dim=int(reference_hidden_dim),
            learning_rate=float(learning_rate),
            epochs=4,
            minibatch_size=1024,
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
    hidden_dim, expected, ratio = matched_hierarchical_hidden_dim(
        target_parameter_count=target_parameters,
        upper_state_dim=selected.upper,
        lower_state_dim=selected.lower,
        goal_dim=selected.goal,
        action_dim=selected.action,
    )
    model = GoalConditionedActorCriticPPO(GoalConditionedPPOConfig(
        upper_state_dim=selected.upper,
        lower_state_dim=selected.lower,
        goal_dim=selected.goal,
        action_dim=selected.action,
        hidden_dim=hidden_dim,
        learning_rate=float(learning_rate),
        epochs=4,
        minibatch_size=1024,
        init_log_std=-0.7,
    ))
    actual = model.trainable_parameter_count
    if actual != expected:
        raise RuntimeError("PointMaze multiscale parameter accounting changed")
    return model, {
        "reference_parameter_budget": target_parameters,
        "actual_parameter_count": actual,
        "parameter_budget_ratio": ratio,
        "hidden_dim": hidden_dim,
        **model.mainline_contract(),
    }


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
        raise ValueError("each PointMaze stage-3 seed role must be non-empty and unique")
    if any(set(roles[i]) & set(roles[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("PointMaze stage-3 seed roles must be disjoint")
    return roles[0], roles[1], roles[2]


def _training_seed(
    *, optimizer_seed: int, rollout_root: int, iteration: int
) -> int:
    return int((
        int(rollout_root) * 22_695_477
        + int(optimizer_seed) * 1_103_515_245
        + (int(iteration) + 1) * 12_345
    ) % (2**32 - 1))


def train_pointmaze_multiscale_cell(
    *,
    method: str,
    scenario: str,
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
    reference_hidden_dim: int = 128,
    learning_rate: float = 3e-4,
    checkpoint_evaluation_interval: int = 96,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    if str(scenario) not in POINTMAZE_MULTISCALE_SCENARIOS:
        raise ValueError(f"unknown PointMaze stage-3 scenario: {scenario}")
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    probe = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        observation, _ = probe.reset(seed=0)
        contract = goal_environment_contract(probe)
        timing = environment_timing(probe)
        parsed = parse_goal_observation(observation)
        stress_probe = _stress_for_episode(
            scenario=scenario,
            seed=0,
            timing=timing,
            observation=observation,
            action_dim=int(np.prod(probe.action_space.shape)),
            horizon=horizon,
        )
    finally:
        probe.close()
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=float(contract["env_dt_seconds"]),
        upper_period_seconds=float(upper_period_seconds),
        history_seconds=float(history_seconds),
        fast_period_seconds=float(fast_period_seconds),
    )
    if time_scale.upper_period_steps >= int(horizon):
        raise ValueError("PointMaze upper period must be shorter than the horizon")
    dimensions = pointmaze_multiscale_dimensions(
        env_id=env_id,
        horizon=horizon,
        time_scale=time_scale,
    )
    model, capacity = build_pointmaze_multiscale_model(
        method=method,
        dimensions=dimensions,
        reference_hidden_dim=reference_hidden_dim,
        learning_rate=learning_rate,
        optimizer_seed=optimizer_seed,
    )
    parameter_budget = int(capacity["reference_parameter_budget"])
    encoder = CausalHaarMultiscaleEncoder(
        feature_dim=int(parsed.physical.size),
        time_scale=time_scale,
    )
    common_metadata = {
        "protocol_version": POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_MULTISCALE_ALGORITHM_PATH,
        "environment_id": str(env_id),
        "scenario": str(scenario),
        "optimizer_seed": int(optimizer_seed),
        "representation": _representation(method),
        "hierarchical": _is_hierarchical(method),
        "reward_type": "dense",
        "continuing_task": POINTMAZE_CONTINUING_TASK,
        "reset_target": False,
        "task_reward_contract": "fixed_horizon_exp_negative_goal_distance_v1",
        "success_is_primary_endpoint": True,
        "checkpoint_objective": POINTMAZE_CHECKPOINT_RANK_CONTRACT,
        "goal_semantics": "upper_relative_xy_waypoint_lower_physical_acceleration",
        "lower_final_goal_visibility": "hidden; lower receives only waypoint error",
        "history_information_contract": (
            "same_fixed_trailing_actor_visible_samples_raw_causal_filter_or_"
            "orthonormal_haar"
        ),
        "current_physical_feedback_contract": (
            "both_hierarchy_levels_retain_full_current_actor_visible_physical_state"
        ),
        "frequency_routing_contract": (
            "flat_multiscale_gets_all_bands; hrl_multiscale_adds_slow_mid_to_"
            "upper_and_mid_high_to_lower_without_removing_current_physical_state"
        ),
        "stress_observability_contract": (
            "actor_never_receives_measurement_or_action_disturbance_truth"
        ),
        "stress": stress_probe.metadata(),
        "feature_dimensions": {
            key: asdict(value) for key, value in dimensions.items()
        },
        "band_layout": encoder.band_layout(),
        "time_scale": time_scale.metadata(),
        "capacity": capacity,
        "environment_contract": contract,
        "runtime_versions": pointmaze_runtime_versions(),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "lower_intrinsic_reward_contract": (
            "actor_visible_waypoint_progress_minus_requested_action_cost_v1"
            if _is_hierarchical(method) else "not_applicable"
        ),
        "lower_credit_boundary_contract": (
            "waypoint_change_or_episode_end_v1"
            if _is_hierarchical(method) else "not_applicable"
        ),
        "projector": "disabled",
        "promotion": "disabled",
        "leakage_loss": "disabled",
        "responsibility_gauge": "disabled",
    }
    seed_fn = lambda root, iteration: _training_seed(
        optimizer_seed=optimizer_seed,
        rollout_root=root,
        iteration=iteration,
    )
    rollout_kwargs = {
        "method": str(method),
        "scenario": str(scenario),
        "env_id": str(env_id),
        "horizon": int(horizon),
        "parameter_budget": parameter_budget,
        "time_scale": time_scale,
    }
    if _is_hierarchical(method):
        payload, rows, trained = train_frequency_separated_ppo(
            model=model,
            train_seeds=training,
            selection_seeds=selection,
            eval_seeds=evaluation,
            iterations=int(iterations),
            rollout_fn=lambda policy, seed, sample: rollout_hrl_pointmaze_multiscale(
                policy,
                seed=seed,
                sample=sample,
                maximum_subgoal_delta=maximum_subgoal_delta,
                **rollout_kwargs,
            ),
            objective_fn=lambda row: float(row["episode_return"]),
            summary_fn=summarize_numeric_rows,
            training_seed_fn=seed_fn,
            policy=str(method),
            domain="pointmaze_multiscale_goal_control",
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
        payload["trajectory_contract"]["lower"] = (
            "one primitive transition with GAE terminated at waypoint change "
            "or episode end"
        )
    else:
        payload, rows, trained = train_joint_ppo(
            model=model,
            train_seeds=training,
            selection_seeds=selection,
            eval_seeds=evaluation,
            iterations=int(iterations),
            rollout_fn=lambda policy, seed, sample: rollout_flat_pointmaze_multiscale(
                policy,
                seed=seed,
                sample=sample,
                **rollout_kwargs,
            ),
            objective_fn=lambda row: float(row["episode_return"]),
            summary_fn=summarize_numeric_rows,
            training_seed_fn=seed_fn,
            policy=str(method),
            domain="pointmaze_multiscale_goal_control",
            metadata=common_metadata,
            checkpoint_score_contract="mean_dense_episode_return",
            checkpoint_rank_fn=pointmaze_checkpoint_rank,
            checkpoint_rank_names=POINTMAZE_CHECKPOINT_RANK_NAMES,
            checkpoint_rank_contract=POINTMAZE_CHECKPOINT_RANK_CONTRACT,
            checkpoint_minimum_iteration=0,
            checkpoint_evaluation_interval=int(checkpoint_evaluation_interval),
        )
    for row in rows:
        row["training_replicate_seed"] = int(optimizer_seed)
    payload["history"] = compact_pointmaze_history(payload["history"])
    payload["history_schema"] = "pointmaze_compact_training_history_v1"
    payload["optimizer_seed"] = int(optimizer_seed)
    payload["evaluation_rows"] = rows
    return payload, rows, trained


def resolved_pointmaze_multiscale_protocol(
    *,
    methods: Iterable[str],
    scenarios: Iterable[str],
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
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
) -> dict[str, Any]:
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    method_names = list(map(str, methods))
    scenario_names = list(map(str, scenarios))
    if not method_names or any(
        name not in POINTMAZE_MULTISCALE_METHODS for name in method_names
    ):
        raise ValueError("PointMaze stage-3 protocol has an invalid method set")
    if not scenario_names or any(
        name not in POINTMAZE_MULTISCALE_SCENARIOS for name in scenario_names
    ):
        raise ValueError("PointMaze stage-3 protocol has an invalid scenario set")
    return {
        "protocol_version": POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
        "methods": method_names,
        "factorial_core_methods": list(POINTMAZE_MULTISCALE_CORE_METHODS),
        "auxiliary_baselines": list(POINTMAZE_MULTISCALE_AUXILIARY_METHODS),
        "scenarios": scenario_names,
        "primary_stress_scenarios": [
            "fast_observation_noise",
            "slow_drift_fast_action",
        ],
        "secondary_stress_scenario": "persistent_action_shift",
        "clean_noninferiority_margin_success": 0.10,
        "primary_endpoint": "success_rate",
        "supportive_endpoints": ["episode_return", "final_goal_distance"],
        "interaction_definition": (
            "(hrl_multiscale-hrl_history)-"
            "(flat_multiscale-flat_history)"
        ),
        "environment_id": str(env_id),
        "reward_type": "dense",
        "continuing_task": POINTMAZE_CONTINUING_TASK,
        "reset_target": False,
        "iterations": int(iterations),
        "horizon": int(horizon),
        "optimizer_seed": int(optimizer_seed),
        "upper_period_seconds": float(upper_period_seconds),
        "history_seconds": float(history_seconds),
        "fast_period_seconds": float(fast_period_seconds),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "reference_hidden_dim": int(reference_hidden_dim),
        "learning_rate": float(learning_rate),
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
        "stress_specs": {
            name: asdict(POINTMAZE_STRESS_SPECS[name])
            for name in scenario_names
        },
        "train_seeds": training,
        "selection_seeds": selection,
        "eval_seeds": evaluation,
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
        description="Run the PointMaze causal multiscale factorial."
    )
    parser.add_argument("--methods", nargs="+", choices=POINTMAZE_MULTISCALE_METHODS, default=list(POINTMAZE_MULTISCALE_METHODS))
    parser.add_argument("--scenarios", nargs="+", choices=POINTMAZE_MULTISCALE_SCENARIOS, default=list(POINTMAZE_MULTISCALE_SCENARIOS))
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--iterations", type=int, default=768)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--optimizer-seed", type=int, default=84007)
    parser.add_argument("--upper-period-seconds", type=float, default=0.25)
    parser.add_argument("--history-seconds", type=float, default=0.32)
    parser.add_argument("--fast-period-seconds", type=float, default=0.04)
    parser.add_argument("--maximum-subgoal-delta", type=float, default=0.75)
    parser.add_argument("--reference-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=96)
    parser.add_argument("--train-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selection-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = resolved_pointmaze_multiscale_protocol(
        methods=args.methods,
        scenarios=args.scenarios,
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
                payload, _, _ = train_pointmaze_multiscale_cell(
                    method=method,
                    scenario=scenario,
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
