"""MuJoCo cart-pole with a causal, piecewise constant cart position command."""

from dataclasses import dataclass

import numpy as np

DT = .01
MAX_GOAL_DELTA = .5


def command_path(seed, horizon):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 631]))
    target = np.empty(horizon + 1, dtype=np.float32)
    start, direction = 0, rng.choice([-1, 1])
    while start <= horizon:
        stop = start + int(rng.integers(100, 181))
        target[start:stop] = direction * rng.uniform(.07, .12)
        start, direction = stop, -direction
    return target


@dataclass(frozen=True)
class CartpoleObservation:
    physical: np.ndarray
    achieved_goal: np.ndarray
    target: np.ndarray
    force: np.ndarray
    task_measurement: np.ndarray

    @property
    def target_error(self):
        return self.target - self.achieved_goal


class CartpoleSetpointTask:
    def __init__(self, *, seed, horizon):
        import gymnasium as gym

        self.seed, self.horizon = seed, horizon
        self.environment = gym.make("InvertedPendulum-v5", frame_skip=1, max_episode_steps=horizon)
        self.environment.unwrapped.model.opt.timestep = DT
        self.dt_seconds = float(self.environment.unwrapped.dt)
        self.action_low = self.environment.action_space.low
        self.action_high = self.environment.action_space.high
        self.targets = command_path(seed, horizon)
        phase = np.random.default_rng(np.random.SeedSequence([seed, 632])).uniform(0, 2 * np.pi)
        self.force = .008 * np.sin(2 * np.pi * np.arange(horizon + 1) / 8 + phase)
        self.index = 0

    def _observation(self, x):
        target = np.array([self.targets[self.index], 0.], dtype=np.float32)
        force = np.array([self.force[self.index], 0.], dtype=np.float32)
        nuisance = np.zeros(2, dtype=np.float32)
        physical = np.asarray(x, dtype=np.float32)
        return CartpoleObservation(physical=physical, achieved_goal=physical[:2], target=target,
            force=force, task_measurement=np.r_[target, force, nuisance])

    def reset(self):
        self.index = 0
        x, _ = self.environment.reset(seed=self.seed)
        return self._observation(x)

    def step(self, action):
        target = float(self.targets[self.index])
        native = np.clip(action + self.force[self.index], self.action_low, self.action_high)
        x, _, terminated, truncated, _ = self.environment.step(native)
        self.index += 1
        distance = float(abs(x[0] - target))
        return self._observation(x), float(np.exp(-distance)), terminated, truncated, {
            "tracking_distance": distance}
