"""A small control system with separately controlled information time scales."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.multiscale import PhysicalTimeScaleContract


TRACKING_SCENARIOS = (
    "clean",
    "slow_target_fast_force",
    "slow_signal_fast_observation_noise",
    "band_swap",
)


@dataclass(frozen=True)
class TrackingScenarioSpec:
    name: str
    target_period_seconds: float
    target_rms: float
    dynamics_force_period_seconds: float
    dynamics_force_rms: float
    measurement_noise_period_seconds: float
    measurement_noise_rms: float
    dynamics_force_observable_before_action: bool = False

    def __post_init__(self) -> None:
        if self.name not in TRACKING_SCENARIOS:
            raise ValueError(f"unknown tracking scenario: {self.name}")
        for label in (
            "target_period_seconds",
            "dynamics_force_period_seconds",
            "measurement_noise_period_seconds",
        ):
            value = float(getattr(self, label))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
        for label in (
            "target_rms",
            "dynamics_force_rms",
            "measurement_noise_rms",
        ):
            value = float(getattr(self, label))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")


def tracking_scenario(
    name: str,
    *,
    time_scale: PhysicalTimeScaleContract,
) -> TrackingScenarioSpec:
    """Return the frozen stage-1 scenarios with RMS-matched band swaps."""

    scenario = str(name)
    if scenario not in TRACKING_SCENARIOS:
        raise ValueError(f"unknown tracking scenario: {scenario}")
    slow_period = 8.0 * time_scale.resolved_upper_period_seconds
    fast_period = max(
        2.0 * time_scale.dt_seconds,
        time_scale.resolved_fast_period_seconds,
    )
    common = {
        "target_rms": 0.75,
        "dynamics_force_period_seconds": fast_period,
        "measurement_noise_period_seconds": fast_period,
    }
    if scenario == "clean":
        return TrackingScenarioSpec(
            name=scenario,
            target_period_seconds=slow_period,
            dynamics_force_rms=0.0,
            measurement_noise_rms=0.0,
            **common,
        )
    if scenario == "slow_target_fast_force":
        return TrackingScenarioSpec(
            name=scenario,
            target_period_seconds=slow_period,
            dynamics_force_rms=0.20,
            measurement_noise_rms=0.0,
            **common,
        )
    if scenario == "slow_signal_fast_observation_noise":
        return TrackingScenarioSpec(
            name=scenario,
            target_period_seconds=slow_period,
            dynamics_force_rms=0.0,
            measurement_noise_rms=0.20,
            **common,
        )
    return TrackingScenarioSpec(
        name=scenario,
        target_period_seconds=fast_period,
        dynamics_force_period_seconds=slow_period,
        dynamics_force_rms=0.0,
        measurement_noise_period_seconds=slow_period,
        measurement_noise_rms=0.20,
        target_rms=0.75,
    )


@dataclass(frozen=True)
class TrackingObservation:
    """Actor-visible fields plus evaluator-only ground truth."""

    physical: np.ndarray
    task_measurement: np.ndarray
    achieved_goal: np.ndarray
    true_target: np.ndarray


def _matched_rms_signal(
    *,
    rng: np.random.Generator,
    count: int,
    dt_seconds: float,
    period_seconds: float,
    rms: float,
) -> np.ndarray:
    if float(rms) == 0.0:
        return np.zeros(int(count), dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    time = np.arange(int(count), dtype=np.float64) * float(dt_seconds)
    signal = np.sin(2.0 * np.pi * time / float(period_seconds) + phase)
    signal -= float(np.mean(signal))
    observed_rms = float(np.sqrt(np.mean(np.square(signal))))
    if observed_rms <= 1e-12:
        raise ValueError("signal horizon cannot resolve the requested period")
    return signal * (float(rms) / observed_rms)


class MultiTimescaleTrackingEnv:
    """Damped point-mass tracking with explicit causal information contracts."""

    def __init__(
        self,
        *,
        time_scale: PhysicalTimeScaleContract,
        scenario: TrackingScenarioSpec,
        horizon: int,
        seed: int,
        acceleration_limit: float = 3.0,
        velocity_damping: float = 0.5,
    ) -> None:
        if int(horizon) < 2:
            raise ValueError("tracking horizon must be at least two")
        if not np.isfinite(float(acceleration_limit)) or acceleration_limit <= 0.0:
            raise ValueError("acceleration_limit must be positive and finite")
        if not np.isfinite(float(velocity_damping)) or velocity_damping <= 0.0:
            raise ValueError("velocity_damping must be positive and finite")
        self.time_scale = time_scale
        self.scenario = scenario
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.acceleration_limit = float(acceleration_limit)
        self.velocity_damping = float(velocity_damping)
        self.response_seconds = 1.0 / self.velocity_damping
        self._step = 0
        self._position = 0.0
        self._velocity = 0.0
        self._target = np.empty(0, dtype=np.float64)
        self._force = np.empty(0, dtype=np.float64)
        self._measurement_noise = np.empty(0, dtype=np.float64)

    @property
    def observability_contract(self) -> dict[str, bool | str]:
        return {
            "task_measurement_available_before_action": True,
            "true_target_available_to_actor": False,
            "measurement_noise_truth_available_to_actor": False,
            "dynamics_force_available_before_action": bool(
                self.scenario.dynamics_force_observable_before_action
            ),
            "dynamics_force_injection_location": "state_transition",
            "measurement_noise_injection_location": "task_observation",
        }

    def reset(self) -> TrackingObservation:
        seed_sequence = np.random.SeedSequence(self.seed)
        state_seed, target_seed, force_seed, noise_seed = seed_sequence.spawn(4)
        state_rng = np.random.default_rng(state_seed)
        count = self.horizon + 1
        self._target = _matched_rms_signal(
            rng=np.random.default_rng(target_seed),
            count=count,
            dt_seconds=self.time_scale.dt_seconds,
            period_seconds=self.scenario.target_period_seconds,
            rms=self.scenario.target_rms,
        )
        self._force = _matched_rms_signal(
            rng=np.random.default_rng(force_seed),
            count=count,
            dt_seconds=self.time_scale.dt_seconds,
            period_seconds=self.scenario.dynamics_force_period_seconds,
            rms=self.scenario.dynamics_force_rms,
        )
        self._measurement_noise = _matched_rms_signal(
            rng=np.random.default_rng(noise_seed),
            count=count,
            dt_seconds=self.time_scale.dt_seconds,
            period_seconds=self.scenario.measurement_noise_period_seconds,
            rms=self.scenario.measurement_noise_rms,
        )
        self._position = float(state_rng.normal(0.0, 0.05))
        self._velocity = float(state_rng.normal(0.0, 0.02))
        self._step = 0
        return self._observation()

    def _observation(self) -> TrackingObservation:
        measured_target = self._target[self._step] + self._measurement_noise[self._step]
        actor_task = [measured_target]
        if self.scenario.dynamics_force_observable_before_action:
            actor_task.append(float(self._force[self._step]))
        return TrackingObservation(
            physical=np.asarray(
                [self._position, self._velocity], dtype=np.float32
            ),
            task_measurement=np.asarray(actor_task, dtype=np.float32),
            achieved_goal=np.asarray([self._position], dtype=np.float32),
            true_target=np.asarray([self._target[self._step]], dtype=np.float32),
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[TrackingObservation, float, bool, bool, dict[str, float]]:
        control = np.asarray(action, dtype=np.float64).reshape(-1)
        if control.shape != (1,) or not np.all(np.isfinite(control)):
            raise ValueError("tracking action must contain one finite value")
        clipped = float(np.clip(control[0], -1.0, 1.0))
        force = float(self._force[self._step])
        acceleration = (
            self.acceleration_limit * clipped
            + force
            - self.velocity_damping * self._velocity
        )
        dt = float(self.time_scale.dt_seconds)
        self._velocity += dt * acceleration
        self._position += dt * self._velocity
        self._step += 1
        target = float(self._target[self._step])
        error = self._position - target
        reward = (
            1.0
            - error * error
            - 0.02 * self._velocity * self._velocity
            - 0.01 * clipped * clipped
        )
        terminated = bool(self._step >= self.horizon)
        return self._observation(), float(reward), terminated, False, {
            "true_target": target,
            "tracking_error": float(error),
            "dynamics_force": force,
            "measurement_noise": float(self._measurement_noise[self._step]),
            "action_saturated": float(abs(control[0]) > 1.0),
        }

    def signal_diagnostics(self) -> dict[str, float]:
        return {
            "target_rms": float(np.sqrt(np.mean(np.square(self._target)))),
            "target_period_seconds": float(
                self.scenario.target_period_seconds
            ),
            "dynamics_force_rms": float(
                np.sqrt(np.mean(np.square(self._force)))
            ),
            "dynamics_force_period_seconds": float(
                self.scenario.dynamics_force_period_seconds
            ),
            "measurement_noise_rms": float(
                np.sqrt(np.mean(np.square(self._measurement_noise)))
            ),
            "measurement_noise_period_seconds": float(
                self.scenario.measurement_noise_period_seconds
            ),
        }
