"""Goal-conditioned MuJoCo adapters for the multiscale HRL mainline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


GOAL_CONTROL_MAINLINE_CONTRACT = (
    "upper_goal_lower_physical_control_no_action_spectrum_projection_v1"
)


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size < 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return vector.astype(np.float32, copy=False)


@dataclass(frozen=True)
class GoalObservation:
    """Explicitly separate physical state, achieved goal, and task goal."""

    physical: np.ndarray
    achieved_goal: np.ndarray
    desired_goal: np.ndarray

    @property
    def goal_error(self) -> np.ndarray:
        return np.asarray(
            self.desired_goal - self.achieved_goal,
            dtype=np.float32,
        )


def parse_goal_observation(observation: Mapping[str, Any]) -> GoalObservation:
    """Parse the Gymnasium-Robotics goal-observation contract."""

    if not isinstance(observation, Mapping):
        raise TypeError("goal-conditioned environments require a mapping observation")
    required = {"observation", "achieved_goal", "desired_goal"}
    missing = required.difference(observation)
    if missing:
        raise ValueError(
            "goal observation is missing: " + ", ".join(sorted(missing))
        )
    parsed = GoalObservation(
        physical=_finite_vector(observation["observation"], name="observation"),
        achieved_goal=_finite_vector(
            observation["achieved_goal"], name="achieved_goal"
        ),
        desired_goal=_finite_vector(
            observation["desired_goal"], name="desired_goal"
        ),
    )
    if parsed.achieved_goal.shape != parsed.desired_goal.shape:
        raise ValueError("achieved_goal and desired_goal must have equal dimensions")
    return parsed


@dataclass(frozen=True)
class EnvironmentTiming:
    """Resolved physical control interval and its MuJoCo provenance."""

    control_dt_seconds: float
    model_timestep_seconds: float
    frame_skip: int
    source: str

    def __post_init__(self) -> None:
        if (
            not np.isfinite(float(self.control_dt_seconds))
            or float(self.control_dt_seconds) <= 0.0
            or not np.isfinite(float(self.model_timestep_seconds))
            or float(self.model_timestep_seconds) <= 0.0
            or int(self.frame_skip) < 1
        ):
            raise ValueError("environment timing values must be positive")


def environment_timing(environment: Any) -> EnvironmentTiming:
    """Resolve dt for standard MuJoCo and Gymnasium-Robotics maze envs."""

    unwrapped = getattr(environment, "unwrapped", environment)
    candidates: list[tuple[str, Any]] = [("env", unwrapped)]
    for name in ("point_env", "ant_env"):
        nested = getattr(unwrapped, name, None)
        if nested is not None:
            candidates.append((name, nested))

    for source, candidate in candidates:
        control_dt = getattr(candidate, "dt", None)
        frame_skip = int(getattr(candidate, "frame_skip", 1) or 1)
        model = getattr(candidate, "model", None)
        timestep = (
            getattr(getattr(model, "opt", None), "timestep", None)
            if model is not None else None
        )
        if control_dt is not None and np.isfinite(float(control_dt)):
            model_timestep = (
                float(timestep)
                if timestep is not None and np.isfinite(float(timestep))
                else float(control_dt) / frame_skip
            )
            return EnvironmentTiming(
                control_dt_seconds=float(control_dt),
                model_timestep_seconds=model_timestep,
                frame_skip=frame_skip,
                source=source,
            )
    for source, candidate in candidates:
        frame_skip = int(getattr(candidate, "frame_skip", 1) or 1)
        model = getattr(candidate, "model", None)
        timestep = (
            getattr(getattr(model, "opt", None), "timestep", None)
            if model is not None else None
        )
        if timestep is not None and np.isfinite(float(timestep)):
            return EnvironmentTiming(
                control_dt_seconds=float(timestep) * frame_skip,
                model_timestep_seconds=float(timestep),
                frame_skip=frame_skip,
                source=f"{source}.model",
            )
    raise ValueError("could not resolve a physical control interval from environment")


@dataclass(frozen=True)
class RelativeSubgoalAdapter:
    """Decode an upper action into a genuine state-space subgoal."""

    maximum_delta: np.ndarray
    world_low: np.ndarray | None = None
    world_high: np.ndarray | None = None
    action_cost: float = 0.0

    def __post_init__(self) -> None:
        delta = _finite_vector(self.maximum_delta, name="maximum_delta")
        if np.any(delta <= 0.0):
            raise ValueError("maximum_delta must be strictly positive")
        object.__setattr__(self, "maximum_delta", delta)
        if (self.world_low is None) != (self.world_high is None):
            raise ValueError("world_low and world_high must be provided together")
        if self.world_low is not None:
            low = _finite_vector(self.world_low, name="world_low")
            high = _finite_vector(self.world_high, name="world_high")
            if low.shape != delta.shape or high.shape != delta.shape:
                raise ValueError("world bounds must match the goal dimension")
            if np.any(high <= low):
                raise ValueError("world_high must be greater than world_low")
            object.__setattr__(self, "world_low", low)
            object.__setattr__(self, "world_high", high)
        if not np.isfinite(float(self.action_cost)) or float(self.action_cost) < 0.0:
            raise ValueError("action_cost must be finite and non-negative")

    @property
    def goal_dim(self) -> int:
        return int(self.maximum_delta.size)

    def decode(self, raw_goal: np.ndarray, achieved_goal: np.ndarray) -> np.ndarray:
        raw = _finite_vector(raw_goal, name="raw_goal")
        achieved = _finite_vector(achieved_goal, name="achieved_goal")
        if raw.shape != self.maximum_delta.shape or achieved.shape != raw.shape:
            raise ValueError("raw and achieved goals must match maximum_delta")
        subgoal = achieved + self.maximum_delta * np.tanh(raw)
        if self.world_low is not None and self.world_high is not None:
            subgoal = np.clip(subgoal, self.world_low, self.world_high)
        return np.asarray(subgoal, dtype=np.float32)

    def lower_goal_features(
        self,
        achieved_goal: np.ndarray,
        subgoal: np.ndarray,
    ) -> np.ndarray:
        achieved = _finite_vector(achieved_goal, name="achieved_goal")
        goal = _finite_vector(subgoal, name="subgoal")
        if achieved.shape != (self.goal_dim,) or goal.shape != achieved.shape:
            raise ValueError("lower goal features have an invalid dimension")
        return np.asarray(goal - achieved, dtype=np.float32)

    def intrinsic_reward(
        self,
        *,
        achieved_before: np.ndarray,
        achieved_after: np.ndarray,
        subgoal: np.ndarray,
        action: np.ndarray,
    ) -> float:
        before = _finite_vector(achieved_before, name="achieved_before")
        after = _finite_vector(achieved_after, name="achieved_after")
        goal = _finite_vector(subgoal, name="subgoal")
        control = _finite_vector(action, name="action")
        if before.shape != (self.goal_dim,) or after.shape != before.shape:
            raise ValueError("achieved goals must match the subgoal dimension")
        progress = float(np.linalg.norm(goal - before) - np.linalg.norm(goal - after))
        return progress - float(self.action_cost) * float(np.mean(np.square(control)))


def goal_environment_contract(environment: Any) -> dict[str, Any]:
    """Validate and summarize a Gymnasium-Robotics goal environment."""

    observation, _ = environment.reset(seed=0)
    parsed = parse_goal_observation(observation)
    action_space = getattr(environment, "action_space", None)
    shape = getattr(action_space, "shape", None)
    low = np.asarray(getattr(action_space, "low", ()), dtype=np.float64).reshape(-1)
    high = np.asarray(getattr(action_space, "high", ()), dtype=np.float64).reshape(-1)
    if (
        shape is None
        or len(shape) != 1
        or low.shape != high.shape
        or low.size != int(shape[0])
        or not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or np.any(high <= low)
    ):
        raise ValueError("goal environment requires a finite one-dimensional Box action")
    timing = environment_timing(environment)
    return {
        "contract": GOAL_CONTROL_MAINLINE_CONTRACT,
        "physical_state_dim": int(parsed.physical.size),
        "goal_dim": int(parsed.achieved_goal.size),
        "action_dim": int(low.size),
        "action_low": low.astype(float).tolist(),
        "action_high": high.astype(float).tolist(),
        "env_dt_seconds": float(timing.control_dt_seconds),
        "model_timestep_seconds": float(timing.model_timestep_seconds),
        "frame_skip": int(timing.frame_skip),
        "timing_source": timing.source,
    }
