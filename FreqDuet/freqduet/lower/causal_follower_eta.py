"""Causal follower ETA estimation and historical target calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class AVLVehicleSnapshot:
    bus_id: int
    direction: bool
    on_route: bool
    progress_m: float
    launch_time_s: float
    current_speed_mps: float
    route_speed_mps: float
    trip_id: int | None = None


@dataclass(frozen=True)
class FollowerDepartureEstimate:
    departure_gap_s: float | None
    eta_s: float | None
    spatial_gap_m: float | None
    speed_mps: float | None
    follower_bus_id: int | None
    follower_trip_id: int | None
    source: str
    valid: bool


def freeze_avl_vehicle_snapshots(
    vehicles: Iterable[Any],
) -> tuple[AVLVehicleSnapshot, ...]:
    """Freeze one simulation tick before any vehicle mutates its state."""
    snapshots = []
    for vehicle in vehicles:
        try:
            route_speed = float(vehicle.current_route.speed_limit)
        except (AttributeError, KeyError, TypeError, ValueError):
            route_speed = 0.0
        try:
            progress = float(vehicle.travel_distance)
        except (AttributeError, TypeError, ValueError):
            progress = 0.0
        trip_id = getattr(vehicle, "trip_id", None)
        snapshots.append(AVLVehicleSnapshot(
            bus_id=int(getattr(vehicle, "bus_id", -1)),
            direction=bool(getattr(vehicle, "direction", True)),
            on_route=bool(getattr(vehicle, "on_route", False)),
            progress_m=progress,
            launch_time_s=float(getattr(vehicle, "launch_time", 0.0)),
            current_speed_mps=float(getattr(
                vehicle, "current_speed", 0.0)),
            route_speed_mps=route_speed,
            trip_id=None if trip_id is None else int(trip_id),
        ))
    return tuple(snapshots)


@dataclass(frozen=True)
class FollowerTargetCalibrationResult:
    base_departure_gap_s: float | None
    calibrated_departure_gap_s: float | None
    base_target_action_s: float | None
    calibrated_target_action_s: float | None
    requested_adjustment_s: float
    effective_target_adjustment_s: float
    features: tuple[float, ...] | None
    active: bool
    history_episodes: int


class HistoricalFollowerTargetCalibrator:
    """Fit a compact action-equivalent gap prior from completed service days.

    Predictions made during episode ``d`` only use matched follower outcomes
    from episodes strictly before ``d``.  The runner updates the sufficient
    statistics once the entire training episode has completed and freezes the
    object during evaluation.  This makes the correction a deployable
    historical prior rather than a same-day look-ahead signal.
    """

    SCHEMA = "freqduet-historical-follower-target-calibrator-v1"
    UPDATE_SOURCE = "completed_learned_training_days_v1"
    DISABLED = "disabled"
    BIAS = "historical_target_bias_v1"
    CONTEXT_RIDGE = "historical_target_ridge_v1"
    CONTEXT_FEATURE_NAMES = (
        "intercept",
        "base_target_norm",
        "gap_imbalance_norm",
        "base_gap_norm",
        "forward_gap_norm",
        "eta_norm",
        "service_dwell_norm",
        "speed_norm",
        "route_progress",
        "route_progress_sq",
        "station_phase",
        "station_sin",
        "station_cos",
        "time_sin",
        "time_cos",
        "direction_sign",
        "source_current_speed",
        "source_route_speed",
    )

    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = CONTEXT_RIDGE,
        min_history_episodes: int = 5,
        min_samples_per_episode: int = 128,
        history_alpha: float = 0.2,
        ridge: float = 0.05,
        residual_clip_s: float = 30.0,
        adjustment_cap_s: float = 10.0,
        time_period_s: float = 14.0 * 3600.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode).strip().lower() if self.enabled else self.DISABLED
        if self.mode not in {self.DISABLED, self.BIAS, self.CONTEXT_RIDGE}:
            raise ValueError("unknown follower forecast calibration mode")
        self.min_history_episodes = int(min_history_episodes)
        self.min_samples_per_episode = int(min_samples_per_episode)
        self.history_alpha = float(history_alpha)
        self.ridge = float(ridge)
        self.residual_clip_s = float(residual_clip_s)
        self.adjustment_cap_s = float(adjustment_cap_s)
        self.time_period_s = float(time_period_s)
        if self.min_history_episodes < 1:
            raise ValueError("min_history_episodes must be positive")
        if self.min_samples_per_episode < 1:
            raise ValueError("min_samples_per_episode must be positive")
        if not 0.0 < self.history_alpha <= 1.0:
            raise ValueError("history_alpha must lie in (0, 1]")
        if not np.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        if not np.isfinite(self.residual_clip_s) \
                or self.residual_clip_s <= 0.0:
            raise ValueError("residual_clip_s must be finite and positive")
        if not np.isfinite(self.adjustment_cap_s) \
                or self.adjustment_cap_s <= 0.0:
            raise ValueError("adjustment_cap_s must be finite and positive")
        if not np.isfinite(self.time_period_s) or self.time_period_s <= 0.0:
            raise ValueError("time_period_s must be finite and positive")

        dimension = len(self.feature_names)
        self._gram = np.zeros((dimension, dimension), dtype=np.float64)
        self._rhs = np.zeros(dimension, dtype=np.float64)
        self._coefficients = np.zeros(dimension, dtype=np.float64)
        self.history_episodes = 0
        self.history_samples = 0
        self.last_update_episode: int | None = None

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None
    ) -> "HistoricalFollowerTargetCalibrator":
        cfg = dict(config or {})
        return cls(
            enabled=cfg.get("enable", False),
            mode=cfg.get("mode", cls.CONTEXT_RIDGE),
            min_history_episodes=cfg.get("min_history_episodes", 5),
            min_samples_per_episode=cfg.get(
                "min_samples_per_episode", 128),
            history_alpha=cfg.get("history_alpha", 0.2),
            ridge=cfg.get("ridge", 0.05),
            residual_clip_s=cfg.get("residual_clip_s", 30.0),
            adjustment_cap_s=cfg.get("adjustment_cap_s", 10.0),
            time_period_s=cfg.get("time_period_s", 14.0 * 3600.0),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.mode == self.CONTEXT_RIDGE:
            return self.CONTEXT_FEATURE_NAMES
        return ("intercept",)

    @property
    def active(self) -> bool:
        return bool(
            self.enabled
            and self.history_episodes >= self.min_history_episodes
            and np.isfinite(self._coefficients).all()
        )

    def calibrate(
        self,
        *,
        base_departure_gap_s: float | None,
        forward_departure_gap_s: float | None,
        target_headway_s: float,
        action_cap_s: float,
        eta_s: float | None,
        speed_mps: float | None,
        service_dwell_s: float,
        route_progress: float,
        station_phase: float,
        current_time_s: float,
        direction: bool,
        source: str,
    ) -> FollowerTargetCalibrationResult:
        """Return a bounded correction using the already-fitted history."""
        base_gap = _finite_nonnegative_optional(base_departure_gap_s)
        forward_gap = _finite_nonnegative_optional(forward_departure_gap_s)
        target_headway = _finite_positive(target_headway_s)
        action_cap = _finite_positive(action_cap_s)
        if any(value is None for value in (
                base_gap, forward_gap, target_headway, action_cap)):
            return FollowerTargetCalibrationResult(
                base_departure_gap_s=base_gap,
                calibrated_departure_gap_s=base_gap,
                base_target_action_s=None,
                calibrated_target_action_s=None,
                requested_adjustment_s=0.0,
                effective_target_adjustment_s=0.0,
                features=None,
                active=False,
                history_episodes=int(self.history_episodes),
            )

        base_target = _two_sided_target(
            base_gap, forward_gap, action_cap)
        features = self._features(
            base_departure_gap_s=base_gap,
            forward_departure_gap_s=forward_gap,
            base_target_action_s=base_target,
            target_headway_s=target_headway,
            action_cap_s=action_cap,
            eta_s=eta_s,
            speed_mps=speed_mps,
            service_dwell_s=service_dwell_s,
            route_progress=route_progress,
            station_phase=station_phase,
            current_time_s=current_time_s,
            direction=direction,
            source=source,
        )
        requested = 0.0
        is_active = self.active and str(source).startswith("same_time_avl_")
        if is_active:
            requested = float(np.clip(
                np.dot(self._coefficients, features),
                -self.adjustment_cap_s,
                self.adjustment_cap_s,
            ))

        # The response is the unclipped half-gap residual.  Two seconds of gap
        # correction move the unconstrained balancing action by one second, so
        # the mapping remains exact even when the final action target clips.
        calibrated_gap = max(base_gap + 2.0 * requested, 0.0)
        calibrated_target = _two_sided_target(
            calibrated_gap, forward_gap, action_cap)
        return FollowerTargetCalibrationResult(
            base_departure_gap_s=float(base_gap),
            calibrated_departure_gap_s=float(calibrated_gap),
            base_target_action_s=float(base_target),
            calibrated_target_action_s=float(calibrated_target),
            requested_adjustment_s=float(requested),
            effective_target_adjustment_s=float(
                calibrated_target - base_target),
            features=tuple(float(value) for value in features),
            active=bool(is_active),
            history_episodes=int(self.history_episodes),
        )

    def update_episode(
        self, rows: Iterable[Mapping[str, Any]], *, episode: int
    ) -> dict[str, float | int]:
        """Update equal-day sufficient statistics after a completed episode."""
        if not self.enabled:
            return self.diagnostics()
        episode_value = int(episode)
        if (self.last_update_episode is not None
                and episode_value <= self.last_update_episode):
            raise ValueError(
                "follower calibration episodes must update monotonically")

        features: list[np.ndarray] = []
        residuals: list[float] = []
        for row in rows:
            values = row.get("calibration_features")
            base_gap = row.get("base_predicted_follower_gap_s")
            actual_gap = row.get("actual_follower_gap_s")
            if values is None or base_gap is None or actual_gap is None:
                continue
            vector = np.asarray(values, dtype=np.float64).reshape(-1)
            if vector.size != len(self.feature_names) \
                    or not np.isfinite(vector).all():
                continue
            # Half-gap residual is the globally linear action-equivalent
            # correction before target clipping.  Multiplying its prediction
            # by two therefore remains exact at both action boundaries.
            residual = 0.5 * (float(actual_gap) - float(base_gap))
            if not np.isfinite(residual):
                continue
            features.append(vector)
            residuals.append(float(np.clip(
                residual, -self.residual_clip_s, self.residual_clip_s)))

        accepted = len(features)
        if accepted < self.min_samples_per_episode:
            return {
                **self.diagnostics(),
                "episode_samples_accepted": int(accepted),
                "episode_updated": 0,
            }

        x = np.vstack(features)
        y = np.asarray(residuals, dtype=np.float64)
        episode_gram = (x.T @ x) / float(accepted)
        episode_rhs = (x.T @ y) / float(accepted)
        if self.history_episodes == 0:
            self._gram = episode_gram
            self._rhs = episode_rhs
        else:
            alpha = self.history_alpha
            self._gram = (
                (1.0 - alpha) * self._gram + alpha * episode_gram)
            self._rhs = (
                (1.0 - alpha) * self._rhs + alpha * episode_rhs)
        self.history_episodes += 1
        self.history_samples += accepted
        self.last_update_episode = episode_value
        self._fit()
        return {
            **self.diagnostics(),
            "episode_samples_accepted": int(accepted),
            "episode_updated": 1,
        }

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "enabled": int(self.enabled),
            "active": int(self.active),
            "history_episodes": int(self.history_episodes),
            "history_samples": int(self.history_samples),
            "coefficient_norm": float(np.linalg.norm(self._coefficients)),
            "intercept_s": float(self._coefficients[0]),
            "last_update_episode": (
                -1 if self.last_update_episode is None
                else int(self.last_update_episode)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "feature_names": self.feature_names,
            "contract": {
                "min_history_episodes": self.min_history_episodes,
                "min_samples_per_episode": self.min_samples_per_episode,
                "history_alpha": self.history_alpha,
                "ridge": self.ridge,
                "residual_clip_s": self.residual_clip_s,
                "adjustment_cap_s": self.adjustment_cap_s,
                "time_period_s": self.time_period_s,
                "update_source": self.UPDATE_SOURCE,
            },
            "gram": self._gram.copy(),
            "rhs": self._rhs.copy(),
            "coefficients": self._coefficients.copy(),
            "history_episodes": int(self.history_episodes),
            "history_samples": int(self.history_samples),
            "last_update_episode": self.last_update_episode,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.SCHEMA:
            raise ValueError("follower calibration checkpoint schema mismatch")
        if bool(state.get("enabled")) != self.enabled \
                or str(state.get("mode")) != self.mode:
            raise ValueError("follower calibration checkpoint mode mismatch")
        if tuple(state.get("feature_names", ())) != self.feature_names:
            raise ValueError("follower calibration feature contract mismatch")
        expected_contract = self.state_dict()["contract"]
        if dict(state.get("contract", {})) != expected_contract:
            raise ValueError("follower calibration parameter contract mismatch")
        dimension = len(self.feature_names)
        gram = np.asarray(state.get("gram"), dtype=np.float64)
        rhs = np.asarray(state.get("rhs"), dtype=np.float64)
        coefficients = np.asarray(
            state.get("coefficients"), dtype=np.float64)
        if gram.shape != (dimension, dimension) \
                or rhs.shape != (dimension,) \
                or coefficients.shape != (dimension,):
            raise ValueError("follower calibration checkpoint shape mismatch")
        if not all(np.isfinite(value).all() for value in (
                gram, rhs, coefficients)):
            raise ValueError("follower calibration checkpoint is non-finite")
        self._gram = gram.copy()
        self._rhs = rhs.copy()
        self._coefficients = coefficients.copy()
        self.history_episodes = int(state.get("history_episodes", 0))
        self.history_samples = int(state.get("history_samples", 0))
        last_episode = state.get("last_update_episode")
        self.last_update_episode = (
            None if last_episode is None else int(last_episode))

    def _fit(self) -> None:
        regularizer = np.eye(len(self.feature_names), dtype=np.float64)
        regularizer[0, 0] = 0.0
        system = self._gram + self.ridge * regularizer
        try:
            coefficients = np.linalg.solve(system, self._rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(
                system, self._rhs, rcond=None)[0]
        if not np.isfinite(coefficients).all():
            raise ValueError("follower calibration fit produced non-finite values")
        self._coefficients = coefficients

    def _features(
        self,
        *,
        base_departure_gap_s: float,
        forward_departure_gap_s: float,
        base_target_action_s: float,
        target_headway_s: float,
        action_cap_s: float,
        eta_s: float | None,
        speed_mps: float | None,
        service_dwell_s: float,
        route_progress: float,
        station_phase: float,
        current_time_s: float,
        direction: bool,
        source: str,
    ) -> np.ndarray:
        if self.mode != self.CONTEXT_RIDGE:
            return np.ones(1, dtype=np.float64)

        target = max(float(target_headway_s), 1.0)
        action_cap = max(float(action_cap_s), 1.0)
        eta = _finite_nonnegative_optional(eta_s) or 0.0
        speed = _finite_positive_optional(speed_mps) or 0.25
        dwell = _finite_nonnegative_optional(service_dwell_s) or 0.0
        progress = float(np.clip(route_progress, 0.0, 1.0))
        station = float(np.clip(station_phase, 0.0, 1.0))
        day_phase = (
            float(current_time_s) % self.time_period_s) / self.time_period_s
        time_angle = 2.0 * np.pi * day_phase
        station_angle = 2.0 * np.pi * station
        source_value = str(source).strip().lower()
        return np.asarray([
            1.0,
            np.clip(base_target_action_s / action_cap, 0.0, 1.0),
            np.clip(
                (base_departure_gap_s - forward_departure_gap_s) / target,
                -3.0,
                3.0,
            ),
            np.clip(base_departure_gap_s / target, 0.0, 3.0),
            np.clip(forward_departure_gap_s / target, 0.0, 3.0),
            np.clip(eta / target, 0.0, 3.0),
            np.clip(dwell / action_cap, 0.0, 2.0),
            np.clip(speed / 15.0, 0.0, 2.0),
            progress,
            progress * progress,
            station,
            np.sin(station_angle),
            np.cos(station_angle),
            np.sin(time_angle),
            np.cos(time_angle),
            1.0 if bool(direction) else -1.0,
            float("current_speed_eta" in source_value),
            float("route_speed_eta" in source_value),
        ], dtype=np.float64)


def estimate_follower_departure_gap(
    *,
    current_bus_id: int,
    current_direction: bool,
    current_progress_m: float,
    current_time_s: float,
    service_dwell_proxy_s: float,
    vehicles: Iterable[AVLVehicleSnapshot],
) -> FollowerDepartureEstimate:
    """Estimate the nearest physical follower using only current AVL state.

    The follower must be active, travel in the same direction, and be spatially
    behind the controlled bus. Its observed journey-average speed is preferred
    because it already reflects upstream dwell and holding. Current speed and a
    conservative fraction of the static segment speed are causal fallbacks.
    The returned gap includes a proxy for mandatory service dwell, so it ends at
    the follower's next action-ready time. It intentionally excludes the
    follower's future discretionary holding action.
    """

    progress = _finite_nonnegative(current_progress_m)
    now = _finite_nonnegative(current_time_s)
    dwell = _finite_nonnegative(service_dwell_proxy_s)
    if progress is None or now is None or dwell is None:
        return _invalid("invalid_current_avl")

    candidates: list[tuple[float, AVLVehicleSnapshot]] = []
    for vehicle in vehicles:
        if int(vehicle.bus_id) == int(current_bus_id):
            continue
        if not bool(vehicle.on_route):
            continue
        if bool(vehicle.direction) != bool(current_direction):
            continue
        follower_progress = _finite_nonnegative(vehicle.progress_m)
        if follower_progress is None:
            continue
        gap = progress - follower_progress
        if gap > 1e-6:
            candidates.append((float(gap), vehicle))

    if not candidates:
        return _invalid("no_same_direction_avl_follower")

    spatial_gap, follower = min(candidates, key=lambda item: item[0])
    elapsed = now - float(follower.launch_time_s)
    journey_speed = (
        float(follower.progress_m) / elapsed
        if np.isfinite(elapsed) and elapsed > 1e-6
        and float(follower.progress_m) > 0.0
        else None
    )
    current_speed = _finite_positive(follower.current_speed_mps)
    route_speed = _finite_positive(follower.route_speed_mps)

    if journey_speed is not None and np.isfinite(journey_speed) \
            and journey_speed > 0.25:
        speed = float(journey_speed)
        source = "same_time_avl_journey_speed_eta"
    elif current_speed is not None:
        speed = float(current_speed)
        source = "same_time_avl_current_speed_eta"
    elif route_speed is not None:
        speed = 0.5 * float(route_speed)
        source = "same_time_avl_route_speed_eta"
    else:
        return _invalid("follower_speed_unavailable")

    eta = spatial_gap / max(speed, 0.25)
    departure_gap = eta + dwell
    if not np.isfinite(departure_gap) or departure_gap < 0.0:
        return _invalid("invalid_follower_eta")
    return FollowerDepartureEstimate(
        departure_gap_s=float(departure_gap),
        eta_s=float(eta),
        spatial_gap_m=float(spatial_gap),
        speed_mps=float(speed),
        follower_bus_id=int(follower.bus_id),
        follower_trip_id=(
            None if follower.trip_id is None else int(follower.trip_id)),
        source=source,
        valid=True,
    )


def _invalid(source: str) -> FollowerDepartureEstimate:
    return FollowerDepartureEstimate(
        departure_gap_s=None,
        eta_s=None,
        spatial_gap_m=None,
        speed_mps=None,
        follower_bus_id=None,
        follower_trip_id=None,
        source=str(source),
        valid=False,
    )


def _finite_nonnegative(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) and result >= 0.0 else None


def _finite_nonnegative_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value)


def _finite_positive(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) and result > 0.0 else None


def _finite_positive_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return _finite_positive(value)


def _two_sided_target(
    follower_gap_s: float, forward_gap_s: float, action_cap_s: float
) -> float:
    return float(np.clip(
        0.5 * (float(follower_gap_s) - float(forward_gap_s)),
        0.0,
        float(action_cap_s),
    ))
