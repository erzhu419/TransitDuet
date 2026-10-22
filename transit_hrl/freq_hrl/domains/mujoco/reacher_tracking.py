"""Moving-target tracking on the standard Reacher-v5 MuJoCo arm dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .goal_adapter import RelativeSubgoalAdapter, environment_timing
from .pointmaze_regime import PointMazeRegimeDriver


REACHER_POSITION_SCALE_METRES = 0.21
REACHER_VELOCITY_SCALE = 10.0


class ReacherTrackingDriver(PointMazeRegimeDriver):
    """Reuse the hidden-regime/nuisance clocks, not the maze route geometry."""

    def _build_targets(self, *, count):
        phase = np.empty(count)
        phase[0] = 2 * np.pi * self.start_vertex / 7
        phase[1:] = phase[0] + np.cumsum(
            np.asarray(self.target_speed_modes)[self._regime_ids[:-1]] * self.dt_seconds)
        return 0.65 * np.column_stack((np.cos(phase), 0.75 * np.sin(phase)))

    def diagnostics(self):
        shared = super().diagnostics()
        for key in ("target_route", "target_start_vertex", "target_speed_modes_world_per_second"):
            shared.pop(key)
        return {**shared, "target_route": "normalized_reacher_ellipse",
                "target_angular_speed_modes_rad_per_second": list(self.target_speed_modes),
                "target_initial_phase_radians": 2 * np.pi * self.start_vertex / 7}


@dataclass(frozen=True)
class ReacherTrackingObservation:
    physical: np.ndarray
    achieved_goal: np.ndarray
    task_measurement: np.ndarray
    target: np.ndarray
    force: np.ndarray
    distractor: np.ndarray

    @property
    def target_error(self):
        return np.asarray(self.target - self.achieved_goal, dtype=np.float32)


class ReacherSubgoalAdapter(RelativeSubgoalAdapter):
    """Relative Cartesian plans projected into the arm's reachable annulus."""

    def decode(self, raw_goal, achieved_goal):
        goal = super().decode(raw_goal, achieved_goal)
        radius = float(np.linalg.norm(goal))
        # The stock joint1 limit is +/-3, giving a nonzero minimum reach.
        if radius < 1e-8:
            goal = np.asarray([0.1, 0.0], dtype=np.float32)
        else:
            goal = goal * (np.clip(radius, 0.1, 0.95) / radius)
        return goal.astype(np.float32)


class ReacherTrackingTask:
    """Normalized target/endpoint coordinates; no hidden regime in actor state."""

    success_distance = 0.45

    def __init__(self, *, seed, horizon, **driver_options):
        import gymnasium as gym

        self.seed, self.horizon = int(seed), int(horizon)
        self.environment = gym.make("Reacher-v5", frame_skip=1, max_episode_steps=horizon)
        self.dt_seconds = environment_timing(self.environment).control_dt_seconds
        if not np.isclose(self.dt_seconds, 0.01):
            self.environment.close()
            raise ValueError("Reacher tracking must preserve the 10ms timing contract")
        self.driver = ReacherTrackingDriver(seed=seed, horizon=horizon,
                                            dt_seconds=self.dt_seconds, **driver_options)
        self.action_low = np.asarray(self.environment.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.environment.action_space.high, dtype=np.float32)
        self._step = 0

    def reset(self):
        self.environment.reset(seed=self.seed)
        self._step = 0
        self._set_visual_target()
        return self._observation()

    def _set_visual_target(self):
        import mujoco

        native = self.environment.unwrapped
        target, _, _ = self.driver.sample(self._step)
        native.data.qpos[-2:] = target * REACHER_POSITION_SCALE_METRES
        native.data.qvel[-2:] = 0
        native.goal = target * REACHER_POSITION_SCALE_METRES
        mujoco.mj_forward(native.model, native.data)

    def _observation(self):
        native = self.environment.unwrapped
        angles = np.arctan2(np.sin(native.data.qpos[:2]), np.cos(native.data.qpos[:2]))
        physical = np.r_[angles / np.pi, native.data.qvel[:2] / REACHER_VELOCITY_SCALE]
        target, force, distractor = self.driver.sample(self._step)
        return ReacherTrackingObservation(
            physical=physical.astype(np.float32),
            achieved_goal=(native.get_body_com("fingertip")[:2]
                           / REACHER_POSITION_SCALE_METRES).astype(np.float32),
            task_measurement=np.r_[target, force, distractor].astype(np.float32),
            target=target, force=force, distractor=distractor,
        )

    def step(self, requested_action):
        requested = np.asarray(requested_action, dtype=np.float32)
        if requested.shape != (2,) or not np.all(np.isfinite(requested)):
            raise ValueError("Reacher expects two finite joint controls")
        target, force, _ = self.driver.sample(self._step)
        executed = np.clip(requested + force, self.action_low, self.action_high).astype(np.float32)
        _, _, terminated, truncated, _ = self.environment.step(executed)
        self._step += 1
        self._set_visual_target()
        observation = self._observation()
        # Match Stage9: score the post-action endpoint against the PRE-action target.
        distance = float(np.linalg.norm(observation.achieved_goal - target))
        return observation, float(np.exp(-distance)), terminated, truncated, {
            "tracking_distance": distance, "tracking_success": float(distance <= self.success_distance),
            "requested_action": requested, "executed_action": executed,
            "target": target, "force": force,
        }

    def diagnostics(self):
        return {**self.driver.diagnostics(), "plant": "Gymnasium/Reacher-v5",
                "derived_task": "ReacherTracking-v1", "frame_skip": 1,
                "position_scale_metres": REACHER_POSITION_SCALE_METRES,
                "joint_angle_scale_radians": float(np.pi),
                "joint_velocity_scale_rad_per_second": REACHER_VELOCITY_SCALE,
                "actor_regime_labels_visible": False, "actor_future_visible": False}
