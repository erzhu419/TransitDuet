"""Piecewise-linear polygonal trajectory tracking on Reacher-v5 2-DOF arm dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pointmaze_regime import PointMazeRegimeDriver
from .reacher_tracking import (
    REACHER_POSITION_SCALE_METRES,
    REACHER_VELOCITY_SCALE,
    ReacherSubgoalAdapter,
    ReacherTrackingObservation,
)
from .goal_adapter import environment_timing


def _pentagon_vertices(radius: float = 0.72) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num=5, endpoint=False) + 0.15
    return float(radius) * np.column_stack((np.cos(angles), np.sin(angles)))


REACHER_PENTAGON_XY = _pentagon_vertices(0.72)
REACHER_POLYGONAL_SPEEDS = (-1.35, -0.65, 0.65, 1.35)


class ReacherPolygonalDriver(PointMazeRegimeDriver):
    """Polygonal chord trajectory with sharp 72-degree corners and hidden speed switches."""

    def __init__(
        self,
        *,
        seed: int,
        horizon: int,
        dt_seconds: float,
        target_speed_modes: tuple[float, ...] = REACHER_POLYGONAL_SPEEDS,
        **kwargs,
    ) -> None:
        super().__init__(
            seed=seed,
            horizon=horizon,
            dt_seconds=dt_seconds,
            target_speed_modes=target_speed_modes,
            **kwargs,
        )

    @staticmethod
    def _interpolate_polygon(progress: float) -> np.ndarray:
        n_edges = len(REACHER_PENTAGON_XY)
        wrapped = float(np.mod(progress, float(n_edges)))
        idx = min(int(np.floor(wrapped)), n_edges - 1)
        frac = wrapped - float(idx)
        nxt = (idx + 1) % n_edges
        return (1.0 - frac) * REACHER_PENTAGON_XY[idx] + frac * REACHER_PENTAGON_XY[nxt]

    def _build_targets(self, *, count: int) -> np.ndarray:
        n_edges = len(REACHER_PENTAGON_XY)
        progress = np.empty(int(count), dtype=np.float64)
        progress[0] = float(self.start_vertex % n_edges) + 0.25
        steps = np.asarray(self.target_speed_modes, dtype=np.float64)[self._regime_ids[:-1]]
        progress[1:] = progress[0] + np.cumsum(steps * self.dt_seconds)
        return np.stack([self._interpolate_polygon(val) for val in progress])

    def diagnostics(self) -> dict:
        shared = super().diagnostics()
        for key in ("target_route", "target_start_vertex", "target_speed_modes_world_per_second"):
            shared.pop(key, None)
        return {
            **shared,
            "target_route": "pentagon_chord_annulus_r0p72",
            "target_chord_speed_modes_per_second": list(self.target_speed_modes),
            "polygon_vertices": int(len(REACHER_PENTAGON_XY)),
        }


class ReacherPolygonalTrackingTask:
    """2-DOF articulated arm tracking a hidden-regime polygonal target trajectory."""

    success_distance = 0.35

    def __init__(self, *, seed: int, horizon: int, **driver_options) -> None:
        import gymnasium as gym

        self.seed, self.horizon = int(seed), int(horizon)
        self.environment = gym.make("Reacher-v5", frame_skip=1, max_episode_steps=horizon)
        self.dt_seconds = environment_timing(self.environment).control_dt_seconds
        if not np.isclose(self.dt_seconds, 0.01):
            self.environment.close()
            raise ValueError("Reacher polygonal tracking requires the 10ms control clock")
        self.driver = ReacherPolygonalDriver(
            seed=seed,
            horizon=horizon,
            dt_seconds=self.dt_seconds,
            **driver_options,
        )
        self.action_low = np.asarray(self.environment.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.environment.action_space.high, dtype=np.float32)
        self._step = 0

    def reset(self) -> ReacherTrackingObservation:
        self.environment.reset(seed=self.seed)
        self._step = 0
        self._set_visual_target()
        return self._observation()

    def _set_visual_target(self) -> None:
        import mujoco

        native = self.environment.unwrapped
        target, _, _ = self.driver.sample(self._step)
        native.data.qpos[-2:] = target * REACHER_POSITION_SCALE_METRES
        native.data.qvel[-2:] = 0.0
        native.goal = target * REACHER_POSITION_SCALE_METRES
        mujoco.mj_forward(native.model, native.data)

    def _observation(self) -> ReacherTrackingObservation:
        native = self.environment.unwrapped
        angles = np.arctan2(np.sin(native.data.qpos[:2]), np.cos(native.data.qpos[:2]))
        physical = np.r_[angles / np.pi, native.data.qvel[:2] / REACHER_VELOCITY_SCALE]
        target, force, distractor = self.driver.sample(self._step)
        return ReacherTrackingObservation(
            physical=physical.astype(np.float32),
            achieved_goal=(
                native.get_body_com("fingertip")[:2] / REACHER_POSITION_SCALE_METRES
            ).astype(np.float32),
            task_measurement=np.r_[target, force, distractor].astype(np.float32),
            target=target,
            force=force,
            distractor=distractor,
        )

    def step(self, requested_action: np.ndarray):
        requested = np.asarray(requested_action, dtype=np.float32)
        if requested.shape != (2,) or not np.all(np.isfinite(requested)):
            raise ValueError("Reacher polygonal expects two finite joint torque controls")
        target, force, _ = self.driver.sample(self._step)
        executed = np.clip(requested + force, self.action_low, self.action_high).astype(np.float32)
        _, _, terminated, truncated, _ = self.environment.step(executed)
        self._step += 1
        self._set_visual_target()
        observation = self._observation()
        distance = float(np.linalg.norm(observation.achieved_goal - target))
        return observation, float(np.exp(-distance)), terminated, truncated, {
            "tracking_distance": distance,
            "tracking_success": float(distance <= self.success_distance),
            "requested_action": requested,
            "executed_action": executed,
            "target": target,
            "force": force,
        }

    def diagnostics(self) -> dict:
        return {
            **self.driver.diagnostics(),
            "plant": "Gymnasium/Reacher-v5",
            "derived_task": "ReacherPolygonalTracking-v1",
            "frame_skip": 1,
            "position_scale_metres": REACHER_POSITION_SCALE_METRES,
            "joint_angle_scale_radians": float(np.pi),
            "joint_velocity_scale_rad_per_second": REACHER_VELOCITY_SCALE,
            "actor_regime_labels_visible": False,
            "actor_future_visible": False,
        }


@dataclass(frozen=True)
class ReacherPolygonalSpec:
    """Duck-typed PlanValidityTaskSpec for Reacher polygonal tracking."""

    options: dict
    name: str = "reacher_polygonal"

    def open(self, *, seed: int, horizon: int) -> ReacherPolygonalTrackingTask:
        from freq_hrl.experiments.compute_budget import count_work

        count_work("environment_opens")
        return ReacherPolygonalTrackingTask(seed=seed, horizon=horizon, **self.options)

    def adapter(self, task: ReacherPolygonalTrackingTask, *, maximum_subgoal_delta: float):
        from freq_hrl.experiments.pointmaze_goal_validation import POINTMAZE_LOWER_ACTION_COST

        delta = np.full(2, float(maximum_subgoal_delta), dtype=np.float32)
        return ReacherSubgoalAdapter(
            maximum_delta=delta,
            action_cost=POINTMAZE_LOWER_ACTION_COST,
        )
