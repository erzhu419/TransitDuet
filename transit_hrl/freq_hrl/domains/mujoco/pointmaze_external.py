"""Action-independent exogenous task stream for PointMaze tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .goal_adapter import environment_timing, parse_goal_observation


POINTMAZE_U_ROUTE_CELLS = (
    (3, 1),
    (3, 2),
    (3, 3),
    (2, 3),
    (1, 3),
    (1, 2),
    (1, 1),
)
POINTMAZE_U_ROUTE_XY = np.asarray(
    (
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
    ),
    dtype=np.float64,
)


def _finite_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (int(size),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {int(size)} finite values")
    return vector


def _matched_rms_sine(
    *,
    rng: np.random.Generator,
    count: int,
    scored_count: int,
    dt_seconds: float,
    period_seconds: float,
    rms: float,
) -> np.ndarray:
    if float(rms) == 0.0:
        return np.zeros(int(count), dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    time = np.arange(int(count), dtype=np.float64) * float(dt_seconds)
    signal = np.sin(2.0 * np.pi * time / float(period_seconds) + phase)
    signal -= float(np.mean(signal[: int(scored_count)]))
    observed = float(np.sqrt(np.mean(np.square(signal[: int(scored_count)]))))
    if observed <= 1e-12:
        raise ValueError("external-force horizon cannot resolve its period")
    return signal * (float(rms) / observed)


@dataclass(frozen=True)
class PointMazeExternalObservation:
    """Current physical feedback and a separately named exogenous stream."""

    physical: np.ndarray
    achieved_goal: np.ndarray
    task_measurement: np.ndarray
    target: np.ndarray
    force: np.ndarray

    @property
    def target_error(self) -> np.ndarray:
        return np.asarray(self.target - self.achieved_goal, dtype=np.float32)


class PointMazeExternalDriver:
    """Pre-generate a slow route target and fast measured action disturbance."""

    def __init__(
        self,
        *,
        seed: int,
        horizon: int,
        dt_seconds: float,
        target_speed: float = 1.0,
        force_rms: float = 0.12,
        force_period_seconds: tuple[float, float] = (0.04, 0.04),
    ) -> None:
        if int(horizon) < 2:
            raise ValueError("external PointMaze horizon must be at least two")
        positive = {
            "dt_seconds": dt_seconds,
            "target_speed": target_speed,
        }
        if any(
            not np.isfinite(float(value)) or float(value) <= 0.0
            for value in positive.values()
        ):
            raise ValueError("driver time and target speed must be positive")
        if not np.isfinite(float(force_rms)) or float(force_rms) < 0.0:
            raise ValueError("force_rms must be finite and non-negative")
        periods = _finite_vector(
            force_period_seconds, size=2, name="force_period_seconds"
        )
        if np.any(periods <= 2.0 * float(dt_seconds)):
            raise ValueError("force periods must exceed the Nyquist limit")

        self.seed = int(seed)
        self.horizon = int(horizon)
        self.dt_seconds = float(dt_seconds)
        self.target_speed = float(target_speed)
        self.force_rms = float(force_rms)
        self.force_period_seconds = periods

        route_seed, force_seed = np.random.SeedSequence(self.seed).spawn(2)
        route_rng = np.random.default_rng(route_seed)
        self.start_vertex = int(
            route_rng.integers(0, len(POINTMAZE_U_ROUTE_XY))
        )
        if self.start_vertex == 0:
            self.direction = 1
        elif self.start_vertex == len(POINTMAZE_U_ROUTE_XY) - 1:
            self.direction = -1
        else:
            self.direction = 1 if float(route_rng.random()) < 0.5 else -1
        count = self.horizon + 1
        progress = (
            float(self.start_vertex)
            + self.direction
            * self.target_speed
            * self.dt_seconds
            * np.arange(count, dtype=np.float64)
        )
        self._route_progress = self._reflect_progress(progress)
        self._targets = np.stack(
            [self._interpolate_route(value) for value in self._route_progress]
        )
        force_rng = np.random.default_rng(force_seed)
        self._forces = np.column_stack([
            _matched_rms_sine(
                rng=force_rng,
                count=count,
                scored_count=self.horizon,
                dt_seconds=self.dt_seconds,
                period_seconds=float(period),
                rms=self.force_rms,
            )
            for period in self.force_period_seconds
        ])

    @staticmethod
    def _reflect_progress(progress: np.ndarray) -> np.ndarray:
        route_length = float(len(POINTMAZE_U_ROUTE_XY) - 1)
        wrapped = np.mod(np.asarray(progress, dtype=np.float64), 2.0 * route_length)
        return np.where(wrapped <= route_length, wrapped, 2.0 * route_length - wrapped)

    @staticmethod
    def _interpolate_route(progress: float) -> np.ndarray:
        route_length = len(POINTMAZE_U_ROUTE_XY) - 1
        value = float(np.clip(progress, 0.0, float(route_length)))
        left = min(int(np.floor(value)), route_length - 1)
        fraction = value - float(left)
        return (
            (1.0 - fraction) * POINTMAZE_U_ROUTE_XY[left]
            + fraction * POINTMAZE_U_ROUTE_XY[left + 1]
        )

    @property
    def reset_cell(self) -> tuple[int, int]:
        return POINTMAZE_U_ROUTE_CELLS[self.start_vertex]

    def sample(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        index = int(step)
        if index < 0 or index > self.horizon:
            raise IndexError("external driver step is outside the episode")
        return (
            self._targets[index].astype(np.float32, copy=True),
            self._forces[index].astype(np.float32, copy=True),
        )

    def diagnostics(self) -> dict[str, Any]:
        scored_forces = self._forces[: self.horizon]
        return {
            "external_stream_action_independent": True,
            "target_route": "pointmaze_u_centerline_reflecting",
            "target_start_vertex": int(self.start_vertex),
            "target_initial_direction": int(self.direction),
            "target_speed_world_per_second": float(self.target_speed),
            "target_round_trip_period_seconds": float(
                2.0
                * (len(POINTMAZE_U_ROUTE_XY) - 1)
                / self.target_speed
            ),
            "force_x_period_seconds": float(self.force_period_seconds[0]),
            "force_y_period_seconds": float(self.force_period_seconds[1]),
            "force_x_rms": float(np.sqrt(np.mean(np.square(scored_forces[:, 0])))),
            "force_y_rms": float(np.sqrt(np.mean(np.square(scored_forces[:, 1])))),
        }


class PointMazeExternalTask:
    """Track the external target while compensating its measured fast force."""

    success_distance = 0.45

    def __init__(
        self,
        environment: Any,
        *,
        seed: int,
        horizon: int,
        target_speed: float = 1.0,
        force_rms: float = 0.12,
        force_period_seconds: tuple[float, float] = (0.04, 0.04),
    ) -> None:
        self.environment = environment
        self.seed = int(seed)
        self.horizon = int(horizon)
        timing = environment_timing(environment)
        self.driver = PointMazeExternalDriver(
            seed=self.seed,
            horizon=self.horizon,
            dt_seconds=timing.control_dt_seconds,
            target_speed=target_speed,
            force_rms=force_rms,
            force_period_seconds=force_period_seconds,
        )
        self.action_low = np.asarray(
            environment.action_space.low, dtype=np.float32
        ).reshape(-1)
        self.action_high = np.asarray(
            environment.action_space.high, dtype=np.float32
        ).reshape(-1)
        if self.action_low.shape != (2,) or self.action_high.shape != (2,):
            raise ValueError("external PointMaze task requires a 2D Box action")
        self._step = 0
        self._truth: Any | None = None

    @property
    def observability_contract(self) -> dict[str, bool | str]:
        return {
            "current_physical_state_visible_to_both_levels": True,
            "external_stream_visible_before_action": True,
            "external_future_visible_to_actor": False,
            "external_stream_action_independent": True,
            "force_injection_location": "normalized_action_before_environment_step",
            "reward_target_timing": "post_transition_state_vs_pre_action_target",
        }

    def _set_visual_goal(self, target: np.ndarray) -> None:
        unwrapped = getattr(self.environment, "unwrapped", self.environment)
        unwrapped.goal = np.asarray(target, dtype=np.float64).copy()
        update = getattr(unwrapped, "update_target_site_pos", None)
        if callable(update):
            update()

    def _observation(self) -> PointMazeExternalObservation:
        if self._truth is None:
            raise RuntimeError("external PointMaze task must be reset")
        parsed = parse_goal_observation(self._truth)
        target, force = self.driver.sample(self._step)
        return PointMazeExternalObservation(
            physical=parsed.physical.copy(),
            achieved_goal=parsed.achieved_goal.copy(),
            task_measurement=np.concatenate((target, force)).astype(
                np.float32, copy=False
            ),
            target=target,
            force=force,
        )

    def reset(self) -> PointMazeExternalObservation:
        alternate_goal = (
            POINTMAZE_U_ROUTE_CELLS[-1]
            if self.driver.reset_cell != POINTMAZE_U_ROUTE_CELLS[-1]
            else POINTMAZE_U_ROUTE_CELLS[0]
        )
        self._truth, _ = self.environment.reset(
            seed=self.seed,
            options={
                "reset_cell": self.driver.reset_cell,
                "goal_cell": alternate_goal,
            },
        )
        self._step = 0
        target, _ = self.driver.sample(0)
        self._set_visual_goal(target)
        return self._observation()

    def step(
        self,
        requested_action: np.ndarray,
    ) -> tuple[PointMazeExternalObservation, float, bool, bool, dict[str, Any]]:
        if self._truth is None:
            raise RuntimeError("external PointMaze task must be reset")
        if self._step >= self.horizon:
            raise RuntimeError("external PointMaze episode already ended")
        requested = _finite_vector(
            requested_action, size=2, name="requested_action"
        ).astype(np.float32)
        target, force = self.driver.sample(self._step)
        executed = np.clip(
            requested + force, self.action_low, self.action_high
        ).astype(np.float32, copy=False)
        next_truth, _, terminated, truncated, _ = self.environment.step(executed)
        parsed_next = parse_goal_observation(next_truth)
        distance = float(np.linalg.norm(parsed_next.achieved_goal - target))
        reward = float(np.exp(-distance))
        self._step += 1
        self._truth = next_truth
        next_target, _ = self.driver.sample(self._step)
        self._set_visual_goal(next_target)
        return self._observation(), reward, bool(terminated), bool(truncated), {
            "target": target.copy(),
            "force": force.copy(),
            "requested_action": requested.copy(),
            "executed_action": executed.copy(),
            "tracking_distance": distance,
            "tracking_success": float(distance <= self.success_distance),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.driver.diagnostics(),
            **self.observability_contract,
            "tracking_success_distance": float(self.success_distance),
        }
