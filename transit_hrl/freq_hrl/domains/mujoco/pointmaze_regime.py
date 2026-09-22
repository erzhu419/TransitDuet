"""Hidden persistent-regime task for plan-validity qualification.

The actor observes the current target, measured transient force, and an
irrelevant distractor.  The persistent target-motion regime and its switch
times remain hidden.  A privileged current-regime context is exposed only to
explicit oracle-reference experiments; it contains no future information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .goal_adapter import environment_timing, parse_goal_observation
from .pointmaze_external import (
    POINTMAZE_U_ROUTE_CELLS,
    POINTMAZE_U_ROUTE_XY,
)


POINTMAZE_REGIME_SPEEDS = (-1.25, -0.55, 0.55, 1.25)


def _finite_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (int(size),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {int(size)} finite values")
    return vector


def _seconds_to_steps(seconds: float, dt_seconds: float) -> int:
    if not np.isfinite(float(seconds)) or float(seconds) <= 0.0:
        raise ValueError("physical durations must be positive and finite")
    return max(1, int(round(float(seconds) / float(dt_seconds))))


@dataclass(frozen=True)
class PointMazeRegimeObservation:
    """Observable state with the latent regime deliberately omitted."""

    physical: np.ndarray
    achieved_goal: np.ndarray
    task_measurement: np.ndarray
    target: np.ndarray
    force: np.ndarray
    distractor: np.ndarray

    @property
    def target_error(self) -> np.ndarray:
        return np.asarray(self.target - self.achieved_goal, dtype=np.float32)


class PointMazeRegimeDriver:
    """Generate persistent hidden target regimes and independent nuisances."""

    def __init__(
        self,
        *,
        seed: int,
        horizon: int,
        dt_seconds: float,
        regime_dwell_seconds: tuple[float, float] = (0.80, 1.60),
        target_speed_modes: tuple[float, ...] = POINTMAZE_REGIME_SPEEDS,
        force_pulse_amplitude: float = 0.18,
        force_pulse_duration_seconds: tuple[float, float] = (0.04, 0.10),
        force_pulse_gap_seconds: tuple[float, float] = (0.45, 1.10),
        distractor_amplitude: float = 0.50,
        distractor_dwell_seconds: tuple[float, float] = (0.35, 0.85),
    ) -> None:
        if int(horizon) < 2:
            raise ValueError("regime PointMaze horizon must be at least two")
        if not np.isfinite(float(dt_seconds)) or float(dt_seconds) <= 0.0:
            raise ValueError("dt_seconds must be positive and finite")
        dwell = _finite_vector(
            regime_dwell_seconds, size=2, name="regime_dwell_seconds"
        )
        pulse_duration = _finite_vector(
            force_pulse_duration_seconds,
            size=2,
            name="force_pulse_duration_seconds",
        )
        pulse_gap = _finite_vector(
            force_pulse_gap_seconds, size=2, name="force_pulse_gap_seconds"
        )
        distractor_dwell = _finite_vector(
            distractor_dwell_seconds,
            size=2,
            name="distractor_dwell_seconds",
        )
        for name, bounds in (
            ("regime_dwell_seconds", dwell),
            ("force_pulse_duration_seconds", pulse_duration),
            ("force_pulse_gap_seconds", pulse_gap),
            ("distractor_dwell_seconds", distractor_dwell),
        ):
            if np.any(bounds <= 0.0) or float(bounds[0]) > float(bounds[1]):
                raise ValueError(f"{name} must be positive ordered bounds")
        speeds = np.asarray(target_speed_modes, dtype=np.float64).reshape(-1)
        if (
            speeds.size < 2
            or not np.all(np.isfinite(speeds))
            or np.any(np.abs(speeds) <= 0.0)
            or len(set(map(float, speeds))) != int(speeds.size)
            or not np.any(speeds < 0.0)
            or not np.any(speeds > 0.0)
        ):
            raise ValueError(
                "target_speed_modes must contain distinct positive and negative speeds"
            )
        nonnegative = {
            "force_pulse_amplitude": force_pulse_amplitude,
            "distractor_amplitude": distractor_amplitude,
        }
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in nonnegative.values()
        ):
            raise ValueError("amplitudes must be finite and non-negative")

        self.seed = int(seed)
        self.horizon = int(horizon)
        self.dt_seconds = float(dt_seconds)
        self.regime_dwell_seconds = tuple(map(float, dwell))
        self.target_speed_modes = tuple(map(float, speeds))
        self.force_pulse_amplitude = float(force_pulse_amplitude)
        self.force_pulse_duration_seconds = tuple(map(float, pulse_duration))
        self.force_pulse_gap_seconds = tuple(map(float, pulse_gap))
        self.distractor_amplitude = float(distractor_amplitude)
        self.distractor_dwell_seconds = tuple(map(float, distractor_dwell))

        route_seed, regime_seed, force_seed, distractor_seed = (
            np.random.SeedSequence(self.seed).spawn(4)
        )
        route_rng = np.random.default_rng(route_seed)
        self.start_vertex = int(
            route_rng.integers(0, len(POINTMAZE_U_ROUTE_XY))
        )
        count = self.horizon + 1
        self._regime_ids, self._regime_change_steps = self._build_regimes(
            np.random.default_rng(regime_seed), count=count
        )
        self._targets = self._build_targets(count=count)
        self._forces, self._pulse_start_steps = self._build_force_pulses(
            np.random.default_rng(force_seed), count=count
        )
        self._distractors, self._distractor_change_steps = (
            self._build_distractors(
                np.random.default_rng(distractor_seed), count=count
            )
        )

    @property
    def reset_cell(self) -> tuple[int, int]:
        return POINTMAZE_U_ROUTE_CELLS[self.start_vertex]

    @staticmethod
    def _reflect_progress(value: float) -> float:
        route_length = float(len(POINTMAZE_U_ROUTE_XY) - 1)
        wrapped = float(np.mod(float(value), 2.0 * route_length))
        return wrapped if wrapped <= route_length else 2.0 * route_length - wrapped

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

    def _random_steps(
        self,
        rng: np.random.Generator,
        bounds_seconds: tuple[float, float],
    ) -> int:
        low = _seconds_to_steps(bounds_seconds[0], self.dt_seconds)
        high = _seconds_to_steps(bounds_seconds[1], self.dt_seconds)
        return int(rng.integers(low, high + 1))

    def _build_regimes(
        self,
        rng: np.random.Generator,
        *,
        count: int,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        modes = np.empty(int(count), dtype=np.int64)
        current = int(rng.integers(0, len(self.target_speed_modes)))
        modes[:] = current
        changes: list[int] = []
        step = self._random_steps(rng, self.regime_dwell_seconds)
        while step < int(count) - 1:
            candidates = [
                index
                for index, speed in enumerate(self.target_speed_modes)
                if index != current
                and (
                    np.sign(speed)
                    != np.sign(self.target_speed_modes[current])
                )
            ]
            if not candidates:
                candidates = [
                    index
                    for index in range(len(self.target_speed_modes))
                    if index != current
                ]
            current = int(candidates[int(rng.integers(0, len(candidates)))])
            modes[step:] = current
            changes.append(int(step))
            step += self._random_steps(rng, self.regime_dwell_seconds)
        return modes, tuple(changes)

    def _build_targets(self, *, count: int) -> np.ndarray:
        unwrapped = np.empty(int(count), dtype=np.float64)
        unwrapped[0] = float(self.start_vertex)
        for step in range(1, int(count)):
            previous_mode = int(self._regime_ids[step - 1])
            unwrapped[step] = (
                unwrapped[step - 1]
                + self.target_speed_modes[previous_mode] * self.dt_seconds
            )
        reflected = np.asarray([
            self._reflect_progress(value) for value in unwrapped
        ])
        return np.stack([
            self._interpolate_route(value) for value in reflected
        ])

    def _build_force_pulses(
        self,
        rng: np.random.Generator,
        *,
        count: int,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        force = np.zeros((int(count), 2), dtype=np.float64)
        starts: list[int] = []
        step = self._random_steps(rng, self.force_pulse_gap_seconds)
        while step < int(count) - 1:
            duration = self._random_steps(
                rng, self.force_pulse_duration_seconds
            )
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            vector = self.force_pulse_amplitude * np.asarray(
                [np.cos(angle), np.sin(angle)], dtype=np.float64
            )
            stop = min(int(count), step + duration)
            force[step:stop] = vector
            starts.append(int(step))
            step = stop + self._random_steps(rng, self.force_pulse_gap_seconds)
        return force, tuple(starts)

    def _build_distractors(
        self,
        rng: np.random.Generator,
        *,
        count: int,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        distractor = np.empty((int(count), 2), dtype=np.float64)
        current = rng.uniform(-1.0, 1.0, size=2) * self.distractor_amplitude
        distractor[:] = current
        changes: list[int] = []
        step = self._random_steps(rng, self.distractor_dwell_seconds)
        while step < int(count) - 1:
            current = (
                rng.uniform(-1.0, 1.0, size=2)
                * self.distractor_amplitude
            )
            distractor[step:] = current
            changes.append(int(step))
            step += self._random_steps(rng, self.distractor_dwell_seconds)
        return distractor, tuple(changes)

    @property
    def regime_change_steps(self) -> tuple[int, ...]:
        return self._regime_change_steps

    @property
    def distractor_change_steps(self) -> tuple[int, ...]:
        return self._distractor_change_steps

    @property
    def pulse_start_steps(self) -> tuple[int, ...]:
        return self._pulse_start_steps

    def regime_id(self, step: int) -> int:
        index = int(step)
        if index < 0 or index > self.horizon:
            raise IndexError("regime driver step is outside the episode")
        return int(self._regime_ids[index])

    def privileged_context(self, step: int) -> np.ndarray:
        context = np.zeros(len(self.target_speed_modes), dtype=np.float32)
        context[self.regime_id(step)] = 1.0
        return context

    def sample(
        self,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = int(step)
        if index < 0 or index > self.horizon:
            raise IndexError("regime driver step is outside the episode")
        return (
            self._targets[index].astype(np.float32, copy=True),
            self._forces[index].astype(np.float32, copy=True),
            self._distractors[index].astype(np.float32, copy=True),
        )

    def diagnostics(self) -> dict[str, Any]:
        change_steps = np.asarray(self.regime_change_steps, dtype=np.int64)
        boundaries = np.concatenate((
            np.asarray([0], dtype=np.int64),
            change_steps,
            np.asarray([self.horizon], dtype=np.int64),
        ))
        dwell = np.diff(boundaries).astype(np.float64) * self.dt_seconds
        return {
            "external_stream_action_independent": True,
            "target_route": "pointmaze_u_centerline_hidden_signed_speed_regime",
            "target_start_vertex": int(self.start_vertex),
            "target_speed_modes_world_per_second": list(self.target_speed_modes),
            "regime_change_count": len(self.regime_change_steps),
            "regime_change_steps": list(self.regime_change_steps),
            "regime_dwell_seconds_min_realized": float(np.min(dwell)),
            "regime_dwell_seconds_max_realized": float(np.max(dwell)),
            "force_pulse_count": len(self.pulse_start_steps),
            "force_pulse_start_steps": list(self.pulse_start_steps),
            "force_pulse_amplitude": float(self.force_pulse_amplitude),
            "distractor_change_count": len(self.distractor_change_steps),
            "distractor_change_steps": list(self.distractor_change_steps),
            "distractor_amplitude": float(self.distractor_amplitude),
        }


class PointMazeRegimeTask:
    """Track a target whose persistent motion regime is not directly observed."""

    success_distance = 0.45

    def __init__(
        self,
        environment: Any,
        *,
        seed: int,
        horizon: int,
        regime_dwell_seconds: tuple[float, float] = (0.80, 1.60),
        target_speed_modes: tuple[float, ...] = POINTMAZE_REGIME_SPEEDS,
        force_pulse_amplitude: float = 0.18,
        force_pulse_duration_seconds: tuple[float, float] = (0.04, 0.10),
        force_pulse_gap_seconds: tuple[float, float] = (0.45, 1.10),
        distractor_amplitude: float = 0.50,
        distractor_dwell_seconds: tuple[float, float] = (0.35, 0.85),
    ) -> None:
        self.environment = environment
        self.seed = int(seed)
        self.horizon = int(horizon)
        timing = environment_timing(environment)
        self.dt_seconds = float(timing.control_dt_seconds)
        self.driver = PointMazeRegimeDriver(
            seed=self.seed,
            horizon=self.horizon,
            dt_seconds=self.dt_seconds,
            regime_dwell_seconds=regime_dwell_seconds,
            target_speed_modes=target_speed_modes,
            force_pulse_amplitude=force_pulse_amplitude,
            force_pulse_duration_seconds=force_pulse_duration_seconds,
            force_pulse_gap_seconds=force_pulse_gap_seconds,
            distractor_amplitude=distractor_amplitude,
            distractor_dwell_seconds=distractor_dwell_seconds,
        )
        self.action_low = np.asarray(
            environment.action_space.low, dtype=np.float32
        ).reshape(-1)
        self.action_high = np.asarray(
            environment.action_space.high, dtype=np.float32
        ).reshape(-1)
        if self.action_low.shape != (2,) or self.action_high.shape != (2,):
            raise ValueError("regime PointMaze task requires a 2D Box action")
        self._step = 0
        self._truth: Any | None = None

    @property
    def observability_contract(self) -> dict[str, bool | str]:
        return {
            "current_physical_state_visible_to_both_levels": True,
            "current_target_force_distractor_visible_before_action": True,
            "regime_label_visible_to_candidate": False,
            "regime_switch_time_visible_to_candidate": False,
            "external_future_visible_to_actor": False,
            "external_stream_action_independent": True,
            "privileged_reference_current_regime_only": True,
            "force_injection_location": (
                "normalized_action_before_environment_step"
            ),
            "reward_target_timing": (
                "post_transition_state_vs_pre_action_target"
            ),
        }

    @property
    def step_index(self) -> int:
        return int(self._step)

    def privileged_context(self) -> np.ndarray:
        return self.driver.privileged_context(self._step)

    def _set_visual_goal(self, target: np.ndarray) -> None:
        unwrapped = getattr(self.environment, "unwrapped", self.environment)
        unwrapped.goal = np.asarray(target, dtype=np.float64).copy()
        update = getattr(unwrapped, "update_target_site_pos", None)
        if callable(update):
            update()

    def _observation(self) -> PointMazeRegimeObservation:
        if self._truth is None:
            raise RuntimeError("regime PointMaze task must be reset")
        parsed = parse_goal_observation(self._truth)
        target, force, distractor = self.driver.sample(self._step)
        return PointMazeRegimeObservation(
            physical=parsed.physical.copy(),
            achieved_goal=parsed.achieved_goal.copy(),
            task_measurement=np.concatenate((
                target, force, distractor
            )).astype(np.float32, copy=False),
            target=target,
            force=force,
            distractor=distractor,
        )

    def reset(self) -> PointMazeRegimeObservation:
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
        target, _, _ = self.driver.sample(0)
        self._set_visual_goal(target)
        return self._observation()

    def step(
        self,
        requested_action: np.ndarray,
    ) -> tuple[PointMazeRegimeObservation, float, bool, bool, dict[str, Any]]:
        if self._truth is None:
            raise RuntimeError("regime PointMaze task must be reset")
        if self._step >= self.horizon:
            raise RuntimeError("regime PointMaze episode already ended")
        requested = _finite_vector(
            requested_action, size=2, name="requested_action"
        ).astype(np.float32)
        action_step = int(self._step)
        target, force, distractor = self.driver.sample(action_step)
        executed = np.clip(
            requested + force, self.action_low, self.action_high
        ).astype(np.float32, copy=False)
        next_truth, _, terminated, truncated, _ = self.environment.step(executed)
        parsed_next = parse_goal_observation(next_truth)
        distance = float(np.linalg.norm(parsed_next.achieved_goal - target))
        reward = float(np.exp(-distance))
        regime_id = self.driver.regime_id(action_step)
        self._step += 1
        self._truth = next_truth
        next_target, _, _ = self.driver.sample(self._step)
        self._set_visual_goal(next_target)
        return self._observation(), reward, bool(terminated), bool(truncated), {
            "target": target.copy(),
            "force": force.copy(),
            "distractor": distractor.copy(),
            "requested_action": requested.copy(),
            "executed_action": executed.copy(),
            "tracking_distance": distance,
            "tracking_success": float(distance <= self.success_distance),
            "regime_id": int(regime_id),
            "regime_changed": float(
                action_step in self.driver.regime_change_steps
            ),
            "distractor_changed": float(
                action_step in self.driver.distractor_change_steps
            ),
            "force_pulse_started": float(
                action_step in self.driver.pulse_start_steps
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.driver.diagnostics(),
            **self.observability_contract,
            "tracking_success_distance": float(self.success_distance),
        }
